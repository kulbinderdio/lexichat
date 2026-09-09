import { useState, useEffect, useRef, useCallback, useMemo, KeyboardEvent, ChangeEvent, Component, ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Settings, RotateCcw, Bug, Paperclip, Info, Clock, PanelLeft, BarChart3, Brain, Pencil, RefreshCw } from "lucide-react";
import { JobsPanel } from "./JobsPanel";
import type { JobRun } from "./jobTypes";
import lexiLogo from "./assets/lexi.png";
import { AdminPanel, AppSettings, Profile, ServerConfig, StoredOpenAPISpec, StoredSparqlEndpoint, reconcileCatalog } from "./AdminPanel";
import { runPython, warmPyodide, drainCodeToolCalls, abortPyodideRun, PyFile, PyDataFile } from "./pyodide/runner";
import { dedupeRegistry } from "./profileIO";
import { ChatParamsButton, ChatParams, DEFAULT_CHAT_PARAMS, resolveParams } from "./ChatParamsPanel";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { DebugPanel } from "./DebugPanel";
import { UsageRail, UsageHistoryModal } from "./UsagePanel";
import { MaskEditor } from "./MaskEditor";
import { HistoryPanel, ConversationMeta } from "./HistoryPanel";
import { WikiGraphPanel } from "./WikiGraphPanel";
import "./App.css";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ToolCall { name: string; args: string; startedAt?: number; durationMs?: number; }

// MCP Apps (SEP-1865) UI payload attached to a tool result (see Rust ToolUiPayload).
interface ToolUi {
  server_id: string;
  html?: string;
  uri?: string;
  structured?: unknown;
  content?: unknown;    // raw tool-result content array (forwarded to the app)
  meta?: unknown;       // raw tool-result _meta
  arguments?: unknown;  // arguments the tool was called with
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool-result" | "error" | "notice";
  text: string;
  streaming?: boolean;
  status?: string;           // transient phase label shown with the thinking dots (e.g. "Selecting tools…")
  toolCalls?: ToolCall[];
  toolName?: string;
  toolArgs?: string;
  imageDataUrls?: string[];  // base64 data URIs for attached images
  ui?: ToolUi;               // MCP-App interactive UI to render in a sandboxed iframe
  toolImages?: string[];     // base64 data: image URLs from a tool result (e.g. a Mapbox map)
  artifact?: { title: string; html: string }; // model-authored HTML artifact (create_artifact)
  savePrompt?: string[];     // run_python output files awaiting a folder — rendered with a Save button
  fullResult?: string;       // FULL untruncated tool result (connector viewer) — NOT persisted
  fullTruncated?: boolean;   // true if even fullResult hit the display cap
  wireBase?: number;         // backend wire length just before this user message — the truncation
                             // anchor for edit/regenerate (user messages only)
}

// A conversation autosaved mid-stream (e.g. the app was interrupted, or Stop landed between saves)
// can carry transient run state on its messages — `streaming`/`status`. On reopen nothing will ever
// finish those messages, so the flags would render eternal "thinking" dots. Clear them on load, and
// drop an assistant message that was interrupted with nothing renderable to show (it would otherwise
// be a blank bubble). Saves are also stripped of these fields (see saveActiveConversation), so this
// is mostly a repair for conversations saved before that.
export function sanitizeLoadedMessages(msgs: ChatMessage[]): ChatMessage[] {
  const isDeadStream = (m: ChatMessage) =>
    m.role === "assistant" && !m.text
    && !(m.toolCalls && m.toolCalls.length)
    && !(m.toolImages && m.toolImages.length)
    && !(m.imageDataUrls && m.imageDataUrls.length)
    && !m.artifact && !m.ui && !(m.savePrompt && m.savePrompt.length);
  return msgs
    .filter(m => !isDeadStream(m))
    .map(m => (m.streaming || m.status) ? { ...m, streaming: undefined, status: undefined } : m);
}

// MCP servers approved to render/interact with apps this session (frontend mirror
// of the backend apps_allowed set; gates iframe mounting).
const approvedMcpApps = new Set<string>();

interface ToolSchema {
  type: string;
  function: { name: string; description: string; parameters: unknown; };
}

const uid = () => Math.random().toString(36).slice(2);

// Only the rules that hold no matter which tools are live. Anything naming a specific tool
// belongs in `toolGuidanceSuffix` below, which is gated on that tool actually being enabled —
// otherwise the prompt instructs the model to call tools it has not been given, which is how
// "call list_files right away" survived into profiles with file tools switched off.
const BASE_SYSTEM_PROMPT = `You are Lexi, a personal AI assistant running locally for a single authorised user.
Be proactive — use the tools you have immediately rather than asking the user for paths or clarification.
- ALWAYS write a helpful text response after using tools: summarise what you found, list the results, or answer the question directly. Never leave the chat blank after a tool call.
- If asked about your own tools, capabilities, or what you can do, answer from your own knowledge — do not call a tool to find out.
- Work only from the tools actually available to you in this turn. If a request needs one you do not have, say so plainly rather than guessing or inventing the result.`;

const SUGGESTIONS = [
  { icon: "🔍", title: "Search the web",        prompt: "What are the latest developments in AI?" },
  { icon: "📊", title: "Analyse & chart data",  prompt: "Chart this data and describe the trend: Jan 120, Feb 150, Mar 300, Apr 210, May 260" },
  { icon: "🖼️", title: "Make a presentation",   prompt: "Create a 5-slide presentation introducing a local coffee shop" },
  { icon: "✍️", title: "Draft an email",        prompt: "Draft a polite follow-up email chasing an unpaid invoice" },
];

const ALL_BUILTIN_TOOLS: ToolSchema[] = [
  { type: "function", function: { name: "list_files", description: "List files and directories at a path.", parameters: { type: "object", properties: { path: { type: "string", description: "Directory path." } }, required: [] } } },
  { type: "function", function: { name: "read_file",  description: "Read a local file. Supports plain text, PDF, and DOCX (Word) — text is extracted automatically.", parameters: { type: "object", properties: { path: { type: "string", description: "Absolute file path." }, offset: { type: "integer", description: "Start line (optional)." }, limit: { type: "integer", description: "Max lines (optional)." } }, required: ["path"] } } },
  { type: "function", function: { name: "get_file_info", description: "Get metadata for a file or directory: size, type, modification date.", parameters: { type: "object", properties: { path: { type: "string", description: "Absolute path to the file or directory." } }, required: ["path"] } } },
  { type: "function", function: { name: "search_files", description: "Find files by name pattern (glob).", parameters: { type: "object", properties: { pattern: { type: "string", description: "Glob pattern e.g. '*.pdf'" }, directory: { type: "string", description: "Directory to search in." } }, required: ["pattern"] } } },
  { type: "function", function: { name: "search_in_files", description: "Search for text inside files.", parameters: { type: "object", properties: { query: { type: "string", description: "Text to search for." }, directory: { type: "string", description: "Directory to search in." }, file_pattern: { type: "string", description: "Glob filter e.g. '*.py'" } }, required: ["query"] } } },
  { type: "function", function: { name: "list_directory_tree", description: "Show a recursive directory tree.", parameters: { type: "object", properties: { path: { type: "string", description: "Root directory path." }, max_depth: { type: "integer", description: "Depth limit (default 3)." } }, required: ["path"] } } },
  { type: "function", function: { name: "write_file", description: "Create or overwrite a file. Supports plain text (.txt, .md, etc.), PDF (.pdf), and Word (.docx). The extension determines the output format.", parameters: { type: "object", properties: { path: { type: "string", description: "Absolute file path. Use .pdf for PDF, .docx for Word, .txt or .md for plain text." }, content: { type: "string", description: "Text content to write." } }, required: ["path","content"] } } },
  { type: "function", function: { name: "create_directory", description: "Create a directory.", parameters: { type: "object", properties: { path: { type: "string", description: "Directory path to create." } }, required: ["path"] } } },
  { type: "function", function: { name: "move_file", description: "Move or rename a file.", parameters: { type: "object", properties: { source: { type: "string", description: "Source path." }, destination: { type: "string", description: "Destination path." } }, required: ["source","destination"] } } },
  { type: "function", function: { name: "delete_file", description: "Delete a file.", parameters: { type: "object", properties: { path: { type: "string", description: "File path to delete." } }, required: ["path"] } } },
  { type: "function", function: { name: "find_old_files", description: "Find files not modified in N days.", parameters: { type: "object", properties: { directory: { type: "string", description: "Directory to search." }, older_than_days: { type: "integer", description: "Days threshold." }, pattern: { type: "string", description: "Optional glob filter." } }, required: ["directory","older_than_days"] } } },
  { type: "function", function: { name: "web_search", description: "Search the web for current information.", parameters: { type: "object", properties: { query: { type: "string", description: "Search query." } }, required: ["query"] } } },
  { type: "function", function: { name: "compose_email", description: "Build a base64url-encoded RFC 2822 email ready for the Gmail API. Returns ONLY the raw base64url string — use the entire return value as the 'raw' field in gmail_sendmessage, with no modification.", parameters: { type: "object", properties: { to: { type: "string", description: "Recipient email address(es), comma-separated." }, from: { type: "string", description: "Sender email address (optional)." }, subject: { type: "string", description: "Email subject line." }, body: { type: "string", description: "Plain text email body." }, reply_to_message_id: { type: "string", description: "Message-ID to reply to, for threading (optional)." } }, required: ["to","subject","body"] } } },
  { type: "function", function: { name: "fetch_webpage", description: "Fetch a URL. Web pages come back as readable text; DATA FILES (CSV, TSV, JSON, XML) come back raw — so to download a CSV or an export endpoint, just call this with its URL; you do NOT need browser automation and should not claim you can only summarise. If a page has an Export/Download link, find its URL in the page HTML and fetch that directly. Large results are truncated for display and saved to a file for run_python (the result gives the path). To scrape a structured listing or find a link's exact URL, set raw:true for unstripped HTML and parse it in run_python with BeautifulSoup. A cookie session persists across calls, so you can fetch a page then a link it set up.", parameters: { type: "object", properties: { url: { type: "string", description: "Full URL of the page or data file to fetch, must start with http:// or https://" }, raw: { type: "boolean", description: "Return the page's raw, unstripped HTML instead of readability text — for parsing listings/tables or finding links with run_python + BeautifulSoup. Default false." } }, required: ["url"] } } },
  { type: "function", function: { name: "get_current_datetime", description: "Get the current local date and time. Returns human-readable, ISO 8601, filename-safe, and Unix timestamp formats. Use whenever you need today's date or a timestamp for a filename.", parameters: { type: "object", properties: {}, required: [] } } },
  { type: "function", function: { name: "run_python", description: "Execute real Python (CPython) in a secure, offline sandbox: compute, analyse data, and create charts. Available: the standard library plus numpy, pandas, matplotlib, scipy, sympy, openpyxl, beautifulsoup4, geopandas/shapely/pyproj, python-pptx, python-docx, Pillow. These are the ONLY third-party packages and there is NO network, so importing anything else (requests, plotly, scikit-learn…) fails. print() for text output. FILES: /work/uploads/ holds the user's attachments — PDFs and Word docs are ALREADY extracted to text, so just open() them; images are real files, editable with Pillow. Save anything the user should keep to /work/out/ — the tool result reports the real disk path it was copied to, so quote THAT to the user, never /work/out (they cannot open it). Bulk data for a create_artifact page goes to /work/artifacts/<name>.json and is referenced in the HTML as {{data:<name>.json}} — never retype values into the HTML. /work/artifacts persists across messages; the rest of /work persists across calls within a turn and resets on the next message. CHARTS: build a matplotlib figure and it renders inline automatically — do not savefig unless the user wants a file, and never hand-draw ASCII or SVG. For a plain read of a document with no computation, prefer read_file. Do not read or write outside /work.", parameters: { type: "object", properties: { code: { type: "string", description: "The Python source code to execute." } }, required: ["code"] } } },
  { type: "function", function: { name: "create_artifact", description: "Render a self-contained HTML page inline in the chat, with a Save button. Use it for polished deliverables — reports, dashboards, styled tables, simple interactive views — when markdown is not enough. Inline ALL CSS and JS; external URLs are blocked, EXCEPT Leaflet from unpkg/jsdelivr and OpenStreetMap/Mapbox tiles, so real street maps with plotted points do work. To include an image you generated this turn use <img src=\"{{figure:1}}\"> (1 = order created); for one the user attached, {{upload:1}}. For bulk data produced in run_python, write it to /work/artifacts/<name>.json and reference it as {{data:<name>.json}} — do not paste the values in. Any other image must be a data: URI. Put a short summary in chat and the rich content here — not your whole answer. To show ANY HTML you MUST call this tool; pasting raw HTML, <script> or <iframe> into a chat reply renders as source text, not a page.", parameters: { type: "object", properties: { title: { type: "string", description: "Short title for the artifact (used as the saved filename and header)." }, html: { type: "string", description: "A complete, self-contained HTML document (or fragment) with all CSS/JS inlined and no external resources." } }, required: ["title", "html"] } } },
  { type: "function", function: { name: "generate_image", description: "Generate an image from a text description, or edit one the user attached, using the local offline image model. Use it whenever the user asks to create, draw, illustrate or paint something — and also to edit, restyle or repaint an attached photo, by passing source_image. The result displays inline automatically; refer to it as \"shown above\" and do not output a URL or markdown image. To put it in a deck or document, embed <img src=\"{{figure:N}}\"> in a create_artifact page, or read /work/data/generated_image_N.png in run_python. Do not re-generate an image to reuse it. source_image edits the WHOLE image toward the prompt — good for restyling while keeping composition, but not pixel-exact; for precise edits (exact colour swap, crop, overlay text) use run_python with Pillow instead.", parameters: { type: "object", properties: { prompt: { type: "string", description: "A detailed description of the image to create — or, when editing, of the desired end result (describe the whole scene as it should look after the edit, e.g. 'a street with the foreground building painted pink')." }, negative_prompt: { type: "string", description: "Things to avoid in the image (optional)." }, source_image: { type: "string", description: "To EDIT an attached image instead of creating a new one: the /work/uploads/<filename> path of an image the user attached ANYWHERE in this conversation — the current message or an earlier one. Earlier attachments stay editable; do not claim a photo is no longer attached. Omit to generate from scratch." }, strength: { type: "number", description: "Edit strength for source_image, 0.0–1.0 (optional, default 0.6, or 0.85 with mask_regions). Lower stays closer to the original; higher diverges more. Ignored without source_image." }, mask_regions: { type: "string", description: "To change ONLY part of source_image and keep the rest pixel-identical (e.g. 'the building', 'the sky'): region(s) as normalized (0..1) shapes separated by ';' — 'rect x y w h' or 'ellipse cx cy rx ry'. Estimate the region from the image you can see, e.g. 'rect 0 0.35 0.45 0.65'. If the user painted a region on the image it is used automatically (omit this). Omit to edit the whole image." }, size: { type: "integer", description: "New images: square size in px (512/768/1024). When editing: caps the longer edge; original aspect ratio is kept (optional)." }, steps: { type: "integer", description: "Sampling steps; Turbo models want ~4 (optional)." }, seed: { type: "integer", description: "Seed for reproducibility (optional)." } }, required: ["prompt"] } } },
];

// Built-in tools a chat gets when NO profile is active: read-only / no-side-effect only. Mutating
// file tools, email, code execution, and all registered OpenAPI/MCP/SPARQL integrations require an
// explicit profile. (Product decision — the no-profile default must not expose everything.)
// create_artifact is included: it only renders self-contained HTML in a sandboxed frame (no disk,
// network, or app access; the Save button is user-initiated), so a no-profile chat can still show an
// inline report/deck/dashboard instead of silently falling back to a file-only result.
const SAFE_DEFAULT_BUILTINS = new Set([
  "read_file", "list_files", "search_files", "search_in_files", "get_file_info",
  "list_directory_tree", "web_search", "fetch_webpage", "get_current_datetime",
  "create_artifact",
]);

const WIKI_TOOLS: ToolSchema[] = [
  { type: "function", function: { name: "wiki_list", description: "List all pages in the persistent wiki.", parameters: { type: "object", properties: {}, required: [] } } },
  { type: "function", function: { name: "wiki_search", description: "Search wiki pages for a keyword or phrase. Always search before writing to avoid duplicates.", parameters: { type: "object", properties: { query: { type: "string", description: "Keyword or phrase to search for." } }, required: ["query"] } } },
  { type: "function", function: { name: "wiki_read", description: "Read the full contents of a wiki page.", parameters: { type: "object", properties: { path: { type: "string", description: "Page path e.g. 'people/alice.md' or 'projects'. .md extension optional." } }, required: ["path"] } } },
  { type: "function", function: { name: "wiki_write", description: "Create or overwrite a wiki page with markdown content. Search first to avoid duplicates.", parameters: { type: "object", properties: { path: { type: "string", description: "Page path e.g. 'people/alice.md'." }, content: { type: "string", description: "Full markdown content." } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "wiki_patch", description: "Update part of a wiki page by replacing the first occurrence of a specific string.", parameters: { type: "object", properties: { path: { type: "string", description: "Page path." }, find: { type: "string", description: "Exact text to find." }, replace: { type: "string", description: "Replacement text." } }, required: ["path", "find", "replace"] } } },
  { type: "function", function: { name: "wiki_delete", description: "Permanently delete a wiki page.", parameters: { type: "object", properties: { path: { type: "string", description: "Page path to delete." } }, required: ["path"] } } },
  { type: "function", function: { name: "wiki_append", description: "Append content to a wiki page without overwriting it. Use this for log.md entries. Creates the page if it doesn't exist.", parameters: { type: "object", properties: { path: { type: "string", description: "Page path e.g. 'log.md'." }, content: { type: "string", description: "Content to append." } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "wiki_lint", description: "Run a health check on the wiki: finds empty pages, pages missing from index.md, broken index links, and log.md freshness. Call this periodically.", parameters: { type: "object", properties: {}, required: [] } } },
];

const WIKI_SYSTEM_PROMPT_BLOCK = `

You have access to a persistent personal wiki that stores knowledge across conversations. This is your long-term memory — treat it as your source of truth about the user.

Wiki tools: wiki_list, wiki_search, wiki_read, wiki_write, wiki_patch, wiki_delete, wiki_append, wiki_lint.

MANDATORY retrieval rules — follow these before answering:
- For ANY question involving dates, plans, events, anniversaries, birthdays, or "what's coming up": call wiki_search with the relevant keywords (e.g. "birthday", "july", "anniversary") BEFORE answering. Never guess from context alone.
- For ANY question involving a person's name, project, preference, or past conversation: call wiki_search with their name or the topic BEFORE answering.
- When the wiki is non-empty (index.md exists), call wiki_read("index.md") at the start of a new conversation to orient yourself.
- If wiki_search returns results, read the relevant pages with wiki_read before composing your answer.

Storage rules:
- After learning any durable fact (name, date, preference, project, goal), store it immediately without being asked.
- Always wiki_search before writing to avoid duplicates — update existing pages with wiki_patch rather than creating a new page.
- Use clear structured markdown with ## headings.
- Paths are relative like "people/alice.md" or "events/birthdays.md" — no leading slash. .md is optional.
- Keep index.md current: after creating or significantly updating a page, update index.md with a one-line entry for that page.

Logging (log.md):
- After any wiki_write or wiki_patch, also call wiki_append("log.md", "## [YYYY-MM-DD] action | detail") to record what changed and why.
- Use today's date in ISO format. Keep log entries to one short sentence.

Ingest workflow — when the user shares a large block of information to remember:
1. wiki_search for each key topic to avoid overwriting existing knowledge.
2. wiki_write or wiki_patch the relevant pages.
3. Update index.md.
4. Append a log entry summarising what was ingested.

Maintenance:
- Call wiki_lint occasionally (e.g. at the start of a session after reading index.md) to surface empty pages, missing index entries, or broken links, then fix any issues found.`;

// ── Built-in OpenAPI specs ────────────────────────────────────────────────────

const BUILTIN_OPENAPI_SPECS: StoredOpenAPISpec[] = [
  {
    id: "builtin-wikipedia",
    title: "Wikipedia",
    base_url: "https://en.wikipedia.org",
    enabled: true,
    spec_json: JSON.stringify({
      openapi: "3.0.3",
      info: {
        title: "Wikipedia",
        version: "1.0.0",
        description: "Wikipedia search, article summaries, historical events, and featured content. Use searchWikipedia or searchWikipediaFullText to find articles, then getArticleSummary to read them.",
      },
      servers: [{ url: "https://en.wikipedia.org" }],
      paths: {
        "/w/api.php": {
          get: {
            operationId: "searchWikipedia",
            summary: "Search Wikipedia for article titles",
            description: "Returns a list of matching article titles and short descriptions. Use this to find the exact title before calling getArticleSummary.",
            parameters: [
              { name: "action",    in: "query", required: true,  description: "Must be 'opensearch'", schema: { type: "string" } },
              { name: "search",    in: "query", required: true,  description: "Search query e.g. 'Albert Einstein', 'black holes'", schema: { type: "string" } },
              { name: "limit",     in: "query", required: false, description: "Number of results to return (default 5, max 20)", schema: { type: "integer" } },
              { name: "format",    in: "query", required: false, description: "Must be 'json'", schema: { type: "string" } },
              { name: "namespace", in: "query", required: false, description: "0 for main articles (default)", schema: { type: "integer" } },
            ],
            responses: { "200": { description: "Search results as [query, titles[], descriptions[], urls[]]" } },
          },
        },
        "/w/rest.php/v1/search/page": {
          get: {
            operationId: "searchWikipediaFullText",
            summary: "Full-text search Wikipedia with snippets",
            description: "Searches the full text of Wikipedia articles and returns matching pages with relevant text snippets. Prefer this over searchWikipedia when you need context about why an article matches, or when searchWikipedia returns no results.",
            parameters: [
              { name: "q",      in: "query", required: true,  description: "Search query", schema: { type: "string" } },
              { name: "limit",  in: "query", required: false, description: "Max results to return (default 10, max 100)", schema: { type: "integer" } },
              { name: "offset", in: "query", required: false, description: "Number of results to skip for pagination", schema: { type: "integer" } },
            ],
            responses: { "200": { description: "Array of matching pages with title, description, and a highlighted text snippet" } },
          },
        },
        "/api/rest_v1/page/summary/{title}": {
          get: {
            operationId: "getArticleSummary",
            summary: "Get a Wikipedia article summary",
            description: "Returns the introduction of a Wikipedia article as plain text. Includes description, thumbnail URL, and page URL.",
            parameters: [
              { name: "title",    in: "path",  required: true,  description: "Exact Wikipedia article title (use searchWikipedia to find it), e.g. 'Python_(programming_language)'", schema: { type: "string" } },
              { name: "redirect", in: "query", required: false, description: "Set to 'true' to follow redirects (recommended)", schema: { type: "string" } },
            ],
            responses: { "200": { description: "Article summary with extract, description, and thumbnail" } },
          },
        },
        "/api/rest_v1/feed/onthisday/{type}/{month}/{day}": {
          get: {
            operationId: "getOnThisDay",
            summary: "Get historical events for a date",
            description: "Returns historical events, births, deaths, holidays, or all of the above for a given month and day. Useful for answering 'what happened on this day in history' questions.",
            parameters: [
              { name: "type",  in: "path", required: true, description: "Type of events: 'selected' (curated highlights), 'births', 'deaths', 'events', 'holidays', or 'all'", schema: { type: "string" } },
              { name: "month", in: "path", required: true, description: "Two-digit month e.g. '03' for March", schema: { type: "string" } },
              { name: "day",   in: "path", required: true, description: "Two-digit day e.g. '14'", schema: { type: "string" } },
            ],
            responses: { "200": { description: "List of historical events/births/deaths with year, text, and related article links" } },
          },
        },
        "/api/rest_v1/feed/featured/{year}/{month}/{day}": {
          get: {
            operationId: "getWikipediaFeaturedContent",
            summary: "Get Wikipedia featured content for a date",
            description: "Returns the featured article, most-read articles, featured image, and in-the-news stories for a given date. Use today's date for current content.",
            parameters: [
              { name: "year",  in: "path", required: true, description: "Four-digit year e.g. '2025'", schema: { type: "string" } },
              { name: "month", in: "path", required: true, description: "Two-digit month e.g. '04'", schema: { type: "string" } },
              { name: "day",   in: "path", required: true, description: "Two-digit day e.g. '16'", schema: { type: "string" } },
            ],
            responses: { "200": { description: "Featured article summary, most-read article list, featured image, and news stories" } },
          },
        },
      },
    }),
  },
];

// ── Built-in SPARQL endpoints ─────────────────────────────────────────────────

const BUILTIN_SPARQL_ENDPOINTS: StoredSparqlEndpoint[] = [
  {
    id: "builtin-landregistry",
    title: "HM Land Registry",
    endpoint_url: "https://landregistry.data.gov.uk/landregistry/query",
    enabled: true,
    read_only: true,
    usage_hint: "UK house/property sold prices, price-paid transactions by postcode/town/street, and the UK House Price Index",
    prefixes: [
      "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
      "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
      "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>",
      "PREFIX ukhpi: <http://landregistry.data.gov.uk/def/ukhpi/>",
      "PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>",
      "PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>",
    ].join("\n"),
    schema_summary:
      "HM Land Registry open linked data. Two main datasets:\n" +
      "- UK House Price Index (ukhpi:): monthly price statistics per region. Key properties: ukhpi:refRegion, ukhpi:refMonth, ukhpi:averagePrice, ukhpi:housePriceIndex.\n" +
      "- Price Paid (lrppi:): individual residential property transactions. A lrppi:Transaction has lrppi:pricePaid, lrppi:transactionDate, lrppi:propertyType, and lrppi:propertyAddress (an lrcommon:Address with lrcommon:postcode, lrcommon:town, lrcommon:street).\n" +
      "PROPERTY TYPE IS AVAILABLE — the Price Paid data DOES record property type via lrppi:propertyType. Its values are the URIs lrcommon:detached, lrcommon:semi-detached, lrcommon:terraced, lrcommon:flat-maisonette, lrcommon:otherPropertyType. To answer \"average DETACHED price\" filter on `lrppi:propertyType lrcommon:detached` directly — do NOT claim the dataset lacks property type, and do NOT fall back to scraping property portals (Rightmove/Zoopla), which disagree with the official record.\n" +
      "FILTER SERVER-SIDE, don't bulk-pull: constrain by town/postcode + date range + propertyType IN THE QUERY (with a small LIMIT or an AVG/COUNT aggregate). Do NOT pull thousands of unfiltered rows and filter in Python — a tight filtered query is faster and the endpoint is slow for wide scans. Use xsd:date bounds for a year, e.g. FILTER(?date >= \"2023-01-01\"^^xsd:date && ?date <= \"2023-12-31\"^^xsd:date).\n" +
      "MATCH CONVENTIONS (wrong forms return 0 rows, do not just retry): lrcommon:postcode must be the FULL postcode WITH its space, e.g. \"DA11 0NA\" — a partial postcode like \"DA11\" matches nothing. lrcommon:town values are stored UPPERCASE, e.g. \"GRAVESEND\" (mixed case matches nothing). Prefer the full postcode for a postcode request; use an UPPERCASE town for a wider area.",
    example_queries: [
      {
        label: "Recent Price Paid records for a postcode",
        query:
          "PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>\n" +
          "PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>\n" +
          "SELECT ?date ?price ?street ?town WHERE {\n" +
          "  ?txn lrppi:propertyAddress ?addr ;\n" +
          "       lrppi:pricePaid ?price ;\n" +
          "       lrppi:transactionDate ?date .\n" +
          "  ?addr lrcommon:postcode \"PL6 8RU\" .\n" +
          "  OPTIONAL { ?addr lrcommon:street ?street }\n" +
          "  OPTIONAL { ?addr lrcommon:town ?town }\n" +
          "} ORDER BY DESC(?date) LIMIT 20",
      },
      {
        label: "Average price for ONE property type in a town, in a year (server-side filtered)",
        query:
          "PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>\n" +
          "PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>\n" +
          "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" +
          "SELECT (COUNT(?price) AS ?sales) (AVG(?price) AS ?avgPrice) WHERE {\n" +
          "  ?txn lrppi:pricePaid ?price ;\n" +
          "       lrppi:transactionDate ?date ;\n" +
          "       lrppi:propertyType lrcommon:detached ;\n" +
          "       lrppi:propertyAddress ?addr .\n" +
          "  ?addr lrcommon:town \"GRAVESEND\" .\n" +
          "  FILTER(?date >= \"2023-01-01\"^^xsd:date && ?date <= \"2023-12-31\"^^xsd:date)\n" +
          "}",
      },
    ],
  },
  {
    id: "builtin-opendatacommunities",
    title: "OpenDataCommunities (MHCLG)",
    endpoint_url: "https://opendatacommunities.org/sparql",
    enabled: true,
    read_only: true,
    usage_hint: "English official statistics from MHCLG: housing, homelessness, deprivation (IMD), local authority and community data",
    prefixes: [
      "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
      "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>",
      "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>",
      "PREFIX qb: <http://purl.org/linked-data/cube#>",
    ].join("\n"),
    schema_summary:
      "Official statistics from MHCLG (England & Wales), SPARQL 1.1. Much of the data uses the RDF Data Cube vocabulary (qb:). " +
      "Use the schema tool or introspect with `SELECT DISTINCT ?type WHERE { ?s a ?type } LIMIT 100` to find datasets and dimensions. Docs: https://opendatacommunities.org/help",
    example_queries: [
      {
        label: "List available classes",
        query: "SELECT DISTINCT ?type (COUNT(?s) AS ?n) WHERE { ?s a ?type } GROUP BY ?type ORDER BY DESC(?n) LIMIT 50",
      },
    ],
  },
];

// Builtin endpoints that have been retired — pruned from any user's stored list on load.
// (statistics.data.gov.uk / ONS PublishMyData was decommissioned on 31 March 2025.)
const REMOVED_BUILTIN_SPARQL_IDS = ["builtin-ons-stats"];

function injectBuiltinSparql(endpoints: StoredSparqlEndpoint[]): StoredSparqlEndpoint[] {
  // Drop any retired builtins the user may still have stored.
  let result = endpoints.filter(e => !REMOVED_BUILTIN_SPARQL_IDS.includes(e.id));
  for (const builtin of BUILTIN_SPARQL_ENDPOINTS) {
    const existing = result.find(e => e.id === builtin.id);
    if (!existing) {
      result = [...result, builtin];
    } else {
      // Refresh definition but preserve the user's enabled choice.
      result = result.map(e => e.id === builtin.id ? { ...builtin, enabled: e.enabled } : e);
    }
  }
  return result;
}

// ── Settings ──────────────────────────────────────────────────────────────────

const DEFAULT_SETTINGS: AppSettings = {
  servers: [{ id: "default-ollama", name: "Ollama", provider: "ollama", baseUrl: "http://localhost:11434" }],
  host: "http://localhost:11434",
  provider: "ollama",
  maxTools: 30,
  webSearchResults: 10,
  maxSteps: 12,
  models: [],
  enabledTools: { read_file: true, list_files: true, web_search: true },
  toolRegistry: { mcpServers: [], openapiSpecs: [], sparqlEndpoints: [] },
  profiles: [],
  activeProfileId: null,
  allowedDirs: undefined,
};

function injectBuiltinSpecs(specs: StoredOpenAPISpec[]): StoredOpenAPISpec[] {
  let result = specs;
  for (const builtin of BUILTIN_OPENAPI_SPECS) {
    const existing = result.find(sp => sp.id === builtin.id);
    if (!existing) {
      result = [builtin, ...result];
    } else {
      // Always refresh spec_json from the latest builtin definition,
      // but preserve the user's enabled/disabled choice.
      result = result.map(sp =>
        sp.id === builtin.id ? { ...builtin, enabled: sp.enabled } : sp
      );
    }
  }
  return result;
}

// The model dropdown encodes both the server id and the model name in one option value, since
// model names aren't globally unique across servers. A control char that never appears in a
// server id or model name keeps decoding unambiguous.
const MODEL_SEP = "";
const encModel = (serverId: string, model: string) => `${serverId}${MODEL_SEP}${model}`;
const decModel = (v: string): { serverId: string; model: string } => {
  const i = v.indexOf(MODEL_SEP);
  return i < 0 ? { serverId: "", model: v } : { serverId: v.slice(0, i), model: v.slice(i + 1) };
};

/// Which server a (serverId, model) selection routes to: the named server if it exists, else the
/// first server that lists the model, else the first server.
function serverForModel(servers: ServerConfig[], serverId: string | undefined, model: string): ServerConfig | undefined {
  if (serverId) { const s = servers.find(x => x.id === serverId); if (s) return s; }
  return servers.find(s => (s.models ?? []).includes(model)) ?? servers[0];
}

function dedupeById<T extends { id: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter(item => { if (seen.has(item.id)) return false; seen.add(item.id); return true; });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function migrateToRegistry(raw: any): any {
  // Already migrated — toolRegistry present in saved data
  if (raw && raw.toolRegistry) return raw;

  const legacyMcp:   StoredOpenAPISpec[] = raw?.mcpServers   ?? [];
  const legacySpecs: StoredOpenAPISpec[] = raw?.openapiSpecs ?? [];
  const profiles: any[]                  = raw?.profiles     ?? [];

  const allMcp = dedupeById([
    ...legacyMcp,
    ...profiles.flatMap((p: any) => p.mcpServers ?? []),
  ]);
  const allSpecs = dedupeById([
    ...legacySpecs,
    ...profiles.flatMap((p: any) => p.openapiSpecs ?? []),
  ]);

  const result: any = { ...raw };
  result.toolRegistry = { mcpServers: allMcp, openapiSpecs: allSpecs, sparqlEndpoints: [] };
  delete result.mcpServers;
  delete result.openapiSpecs;
  result.profiles = profiles.map((p: any) => {
    const migrated: any = { ...p };
    migrated.enabledMcpServerIds   = (p.mcpServers   ?? []).map((s: any) => s.id);
    migrated.enabledOpenapiSpecIds = (p.openapiSpecs ?? []).map((s: any) => s.id);
    migrated.enabledSparqlEndpointIds = [];
    delete migrated.mcpServers;
    delete migrated.openapiSpecs;
    return migrated;
  });
  return result;
}

// Drop enabled-IDs that point at registry entries which no longer exist. A profile keeps only
// references (see Profile scoping), so deleting a spec/server/endpoint leaves dead ids behind in
// every profile that had it enabled — observed: a TFL profile listing 19 enabled specs of which 4
// resolved to nothing. They're invisible in the UI (nothing to render) but they make the profile
// look bigger than it is and survive every export/import round-trip. Pure cleanup: an id that
// matches nothing already contributes no tools.
function pruneDanglingEnabledIds(s: AppSettings): AppSettings {
  const live = (xs: { id: string }[] | undefined) => new Set((xs ?? []).map(x => x.id));
  const specs = live(s.toolRegistry?.openapiSpecs);
  const mcp = live(s.toolRegistry?.mcpServers);
  const sparql = live(s.toolRegistry?.sparqlEndpoints);
  return {
    ...s,
    profiles: s.profiles.map(p => ({
      ...p,
      enabledOpenapiSpecIds: p.enabledOpenapiSpecIds.filter(id => specs.has(id)),
      enabledMcpServerIds: p.enabledMcpServerIds.filter(id => mcp.has(id)),
      enabledSparqlEndpointIds: p.enabledSparqlEndpointIds.filter(id => sparql.has(id)),
    })),
  };
}

export function loadSettings(): AppSettings {
  try {
    const s = localStorage.getItem("lexi_settings");
    // Run migration on the raw parsed object BEFORE merging with defaults,
    // so the toolRegistry sentinel check is against saved data only.
    const parsed = s ? JSON.parse(s) : {};
    const migrated = migrateToRegistry(parsed);
    const loaded: AppSettings = { ...DEFAULT_SETTINGS, ...migrated };
    // Migrate the legacy single-backend fields into the server registry (once).
    if (!Array.isArray(migrated.servers) || migrated.servers.length === 0) {
      const legacyProvider = (migrated.provider ?? "ollama") as "ollama" | "openai";
      loaded.servers = [{
        id: "default-server",
        name: legacyProvider === "openai" ? "OpenAI" : "Ollama",
        provider: legacyProvider,
        baseUrl: migrated.host || "http://localhost:11434",
        apiKey: migrated.apiKey,
        models: migrated.models ?? [],
      }];
    }
    loaded.toolRegistry = {
      ...loaded.toolRegistry,
      openapiSpecs: injectBuiltinSpecs(loaded.toolRegistry.openapiSpecs ?? []),
      sparqlEndpoints: injectBuiltinSparql(loaded.toolRegistry.sparqlEndpoints ?? []),
    };
    loaded.profiles = loaded.profiles.map(p => ({
      ...p,
      enabledMcpServerIds:   p.enabledMcpServerIds   ?? [],
      enabledOpenapiSpecIds: p.enabledOpenapiSpecIds  ?? [],
      enabledSparqlEndpointIds: p.enabledSparqlEndpointIds ?? [],
    }));
    // Collapse any content-duplicate registry entries (e.g. the same API imported twice with
    // different ids) and remap profile references onto the survivor. Prune AFTER that, so an id
    // dedupeRegistry is about to remap isn't mistaken for a dead one.
    return pruneDanglingEnabledIds(dedupeRegistry(loaded));
  } catch { return { ...DEFAULT_SETTINGS, toolRegistry: { mcpServers: [], openapiSpecs: [...BUILTIN_OPENAPI_SPECS], sparqlEndpoints: [...BUILTIN_SPARQL_ENDPOINTS] } }; }
}

export function saveSettings(s: AppSettings) {
  localStorage.setItem("lexi_settings", JSON.stringify(s));
}

// ── Thinking dots ─────────────────────────────────────────────────────────────

function ThinkingDots() {
  return (
    <div className="thinking-dots">
      <div className="thinking-dot" /><div className="thinking-dot" /><div className="thinking-dot" />
    </div>
  );
}

// ── Copy + Save buttons ───────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button className={`copy-btn${copied ? " copied" : ""}`} onClick={copy}>
      {copied ? "✓ Copied" : "⧉ Copy"}
    </button>
  );
}

// Isolates one message's render. Without this, an error rendering any single message (a malformed
// artifact, tool result, or markdown in an old chat) unmounts the WHOLE app — the reported
// "scrolled an old chat and it went blank". Now a bad message shows a placeholder and the rest of
// the conversation keeps working. The error is logged so the actual culprit can be found.
class MessageErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(err: unknown, info: unknown) { console.error("[message-render] failed:", err, info); }
  render() {
    if (this.state.failed) {
      return <div className="msg-error">⚠ This message couldn’t be displayed. The rest of the chat is unaffected.</div>;
    }
    return this.props.children;
  }
}

// ── Message bubbles ───────────────────────────────────────────────────────────

function UserMessage({ text, imageDataUrls, canEdit, onEdit }: {
  text: string; imageDataUrls?: string[];
  canEdit?: boolean; onEdit?: (newText: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(text);
  const start = () => { setDraft(text); setEditing(true); };
  const save = () => {
    const t = draft.trim();
    setEditing(false);
    // Resubmit on any non-empty text — including unchanged text, which is a valid "re-run from
    // here". Previously Save silently did nothing unless the text differed.
    if (t) onEdit?.(t);
  };
  return (
    <div className="msg-user">
      {imageDataUrls && imageDataUrls.length > 0 && (
        <div className="user-image-thumbs">
          {imageDataUrls.map((src, i) => (
            <img key={i} src={src} className="user-image-thumb" alt="attached image" />
          ))}
        </div>
      )}
      {editing ? (
        <div className="user-edit">
          <textarea
            className="user-edit-input"
            autoFocus
            value={draft}
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
              if (e.key === "Escape") setEditing(false);
            }}
          />
          <div className="user-edit-actions">
            <span className="user-edit-hint">Editing discards everything after this message.</span>
            <button className="user-edit-cancel" onClick={() => setEditing(false)}>Cancel</button>
            <button className="user-edit-save" onClick={save}>Save &amp; submit</button>
          </div>
        </div>
      ) : text && (
        <div className="user-bubble-wrap">
          <div className="user-bubble">{text}</div>
          {canEdit && onEdit && (
            <button className="user-edit-btn" title="Edit & resubmit" onClick={start}>
              <Pencil size={12} />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Shared react-markdown renderers for assistant messages: links open externally, and only
// data:/blob: images render. A model that emits `![](/work/out/chart.png)` or a remote URL would
// otherwise show a broken-image icon — real chart output arrives via the inline tool-image path.
const mdComponents: Components = {
  a: ({ href, children }) => (
    <a href={href} onClick={e => { e.preventDefault(); if (href) openUrl(href); }}>{children}</a>
  ),
  img: ({ src, alt }) =>
    typeof src === "string" && (src.startsWith("data:") || src.startsWith("blob:"))
      ? <img src={src} alt={alt ?? ""} style={{ maxWidth: "100%", borderRadius: 8 }} />
      : null,
};

// Save an inline base64 image (e.g. a generated chart) to disk via the native save dialog.
async function downloadImage(dataUrl: string, base: string) {
  const mime = dataUrl.match(/^data:([^;,]+)/)?.[1] || "image/png";
  const ext = mime === "image/svg+xml" ? "svg" : mime === "image/jpeg" ? "jpg" : (mime.split("/")[1] || "png");
  const safeBase = base.replace(/[^a-z0-9_-]+/gi, "_") || "image";
  try {
    const path = await save({ title: "Save image", defaultPath: `${safeBase}.${ext}`,
      filters: [{ name: "Image", extensions: [ext] }] });
    if (!path) return;
    await invoke("save_data_url", { args: { path, data_url: dataUrl } });
  } catch { /* cancelled */ }
}

// True if this assistant message is the LAST assistant bubble in its turn — so we show a single
// "Save…" on it (Save exports the whole turn's response, even when split across bubbles).
function isLastAssistantInTurn(msgs: ChatMessage[], i: number): boolean {
  if (msgs[i].role !== "assistant") return false;
  for (let j = i + 1; j < msgs.length; j++) {
    if (msgs[j].role === "user") return true;      // next turn started → i was the last assistant
    if (msgs[j].role === "assistant") return false; // a later assistant bubble in this turn
  }
  return true; // end of conversation
}

// data: image URLs (charts/maps) produced in the CURRENT turn — i.e. tool-result images since the
// last user message. These back the `{{figure:N}}` token (1-indexed) in reports and artifacts.
function collectTurnFigures(msgs: ChatMessage[]): string[] {
  let start = 0;
  for (let i = msgs.length - 1; i >= 0; i--) { if (msgs[i].role === "user") { start = i; break; } }
  return msgs.slice(start)
    .filter(m => m.role === "tool-result")
    .flatMap(m => m.toolImages ?? [])
    .filter(u => u.startsWith("data:"));
}
// Shown for a {{figure:N}} that has no matching figure (e.g. the model referenced more charts than
// it generated inline), so an artifact renders a labelled placeholder rather than a broken image.
const MISSING_FIGURE_URL = "data:image/svg+xml," + encodeURIComponent(
  "<svg xmlns='http://www.w3.org/2000/svg' width='480' height='270'><rect width='100%' height='100%' rx='12' fill='#f1f5f9'/><text x='50%' y='50%' text-anchor='middle' dominant-baseline='middle' fill='#94a3b8' font-family='sans-serif' font-size='16'>chart unavailable</text></svg>");

// Replace {{figure:N}} tokens with figure data URLs. `asMarkdown` wraps in markdown image syntax
// (for report text); otherwise substitutes the raw URL (for artifact HTML `src="…"`).
function substituteFigures(text: string, figs: string[], asMarkdown: boolean): { out: string; used: Set<number> } {
  const used = new Set<number>();
  const out = text.replace(/\{\{figure:(\d+)\}\}/g, (whole, n) => {
    const i = Number(n) - 1;
    if (!figs[i]) return asMarkdown ? whole : MISSING_FIGURE_URL;
    used.add(i);
    return asMarkdown ? `![Figure ${n}](${figs[i]})` : figs[i];
  });
  return { out, used };
}

// data: image URLs the user ATTACHED in the current turn (the last user message), backing the
// `{{upload:N}}` token (1-indexed) in artifacts — so a model can place an attached logo/photo into a
// create_artifact deck. The attachment can't be reached any other way (CSP blocks its file path).
function collectTurnUploads(msgs: ChatMessage[]): string[] {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === "user") return (msgs[i].imageDataUrls ?? []).filter(u => u.startsWith("data:"));
  }
  return [];
}
function substituteUploads(html: string, ups: string[]): string {
  return html.replace(/\{\{upload:(\d+)\}\}/g, (_whole, n) => ups[Number(n) - 1] ?? MISSING_FIGURE_URL);
}

// Replace `{{data:name.json}}` tokens in artifact HTML with the text run_python wrote to
// /work/artifacts/name.json. This is what keeps bulk data OUT of the model's context: without it a
// model wanting a 520-point route polyline in a Leaflet map has to retype ~17,000 characters of
// coordinates into the create_artifact argument — which is both minutes of generation on a local
// model AND lossy (observed: it silently dropped two of three legs). Now it writes the file in
// Python and references it by name, so the exact bytes land in the page.
//
// The substituted text goes into a <script> block, so it must not be able to close it: a payload
// containing "</script>" would break out of the script and inject markup into the frame. Escape the
// two sequences the HTML parser scans for as `<…` — valid in both a JSON string and a JS string
// literal (unlike `<\/`, which is not legal JSON), so the data still parses either way.
function escapeForScript(text: string): string {
  return text
    .replace(/<\/(script)/gi, (_m, tag: string) => `\\u003C/${tag}`)
    .replace(/<!--/g, "\\u003C!--");
}
function substituteData(html: string, files: Map<string, PyDataFile>): string {
  return html.replace(/\{\{data:([^}]+)\}\}/g, (_whole, rawName) => {
    // Tolerate the model writing the full sandbox path rather than the bare filename.
    const name = String(rawName).trim().replace(/^\/?(work\/)?artifacts\//, "");
    const f = files.get(name);
    if (!f) {
      const known = [...files.keys()];
      return JSON.stringify({
        error: `No artifact data file named "${name}". ${known.length ? `Available: ${known.join(", ")}.` : "Write it to /work/artifacts/ in run_python first."}`,
      });
    }
    if (f.error) return JSON.stringify({ error: f.error });
    return escapeForScript(f.text);
  });
}

function fmtDuration(ms: number): string {
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

// Freeze any call still "running" (startedAt set, no durationMs) — used when a run ends or is
// stopped, so a call interrupted mid-dispatch doesn't tick forever off a stale start time.
function finalizeCallTimers(messages: ChatMessage[]): ChatMessage[] {
  const now = Date.now();
  let changed = false;
  const out = messages.map(m => {
    if (!m.toolCalls?.some(tc => tc.startedAt != null && tc.durationMs == null)) return m;
    changed = true;
    return { ...m, toolCalls: m.toolCalls.map(tc =>
      tc.startedAt != null && tc.durationMs == null ? { ...tc, durationMs: now - tc.startedAt } : tc) };
  });
  return changed ? out : messages;
}

// Small per-call timer shown under each tool call: ticks live while the call is in flight
// (startedAt set, no durationMs yet), then freezes to the final duration once its result arrives.
// Only running timers re-render (on a 150ms interval); finished ones are static.
function ToolTimer({ startedAt, durationMs }: { startedAt?: number; durationMs?: number }) {
  const running = startedAt != null && durationMs == null;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => tick(t => t + 1), 150);
    return () => clearInterval(id);
  }, [running]);
  if (durationMs != null) return <span className="tool-timer">{fmtDuration(durationMs)}</span>;
  if (running) return <span className="tool-timer running">⏱ {fmtDuration(Date.now() - startedAt!)}</span>;
  return null;
}

// One tool-call badge: name + live timer, with the (potentially long) parameters collapsed behind
// a "params ▼" toggle so they don't take up room until you want to see them.
function ToolCallBadge({ tc }: { tc: ToolCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="tool-badge">
      <div className="tool-badge-main">
        <span className="tool-badge-icon">⚡</span>
        <span className="tool-badge-name">{tc.name}</span>
        {tc.args && (
          <button className="tool-badge-args-toggle" onClick={() => setOpen(o => !o)}
            title={open ? "Hide parameters" : "Show parameters"}>
            params {open ? "▲" : "▼"}
          </button>
        )}
        <ToolTimer startedAt={tc.startedAt} durationMs={tc.durationMs} />
      </div>
      {tc.args && open && <pre className="tool-badge-args">{tc.args}</pre>}
    </div>
  );
}

function AssistantMessage({ msg, onExport, onRegenerate, thinkingAt }: { msg: ChatMessage; onExport?: (msgId: string) => void; onRegenerate?: () => void; thinkingAt?: number | null }) {
  const showThinking = msg.streaming && !msg.text && (!msg.toolCalls || msg.toolCalls.length === 0);
  return (
    <div className="msg-assistant">
      <img src={lexiLogo} className="assistant-avatar" alt="Lexi" />
      <div className="assistant-content">
        {showThinking ? (
          <div className="thinking-row">
            <ThinkingDots />
            {msg.status && <span className="thinking-status">{msg.status}</span>}
            <ToolTimer startedAt={thinkingAt ?? undefined} />
          </div>
        ) : msg.streaming ? (
          <div className="assistant-text">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{msg.text}</ReactMarkdown>
            <span className="streaming-cursor" />
          </div>
        ) : (
          msg.text && (
            <div className="assistant-text">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>{msg.text}</ReactMarkdown>
            </div>
          )
        )}


        {msg.toolCalls && msg.toolCalls.length > 0 && (
          <div className="tool-calls">
            {msg.toolCalls.map((tc, i) => <ToolCallBadge key={i} tc={tc} />)}
          </div>
        )}

        {!msg.streaming && msg.text && (
          <div style={{ display: "flex", gap: 4 }}>
            <CopyButton text={msg.text} />
            {onRegenerate && (
              <button className="copy-btn" title="Regenerate this response"
                onClick={onRegenerate}>
                <RefreshCw size={12} /> Regenerate
              </button>
            )}
            {onExport && (
              <button className="copy-btn" title="Save the full response as a report (HTML / PDF / Word)"
                onClick={() => onExport(msg.id)}>
                <span aria-hidden="true">📄</span> Save…
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── File browser (interactive tool result) ────────────────────────────────────

const FILE_ICONS: Record<string, string> = {
  pdf: "📄", txt: "📝", md: "📝", csv: "📊", json: "📋", xml: "📋",
  jpg: "🖼", jpeg: "🖼", png: "🖼", gif: "🖼", webp: "🖼", bmp: "🖼",
  mp4: "🎬", mov: "🎬", avi: "🎬", mkv: "🎬",
  mp3: "🎵", wav: "🎵", flac: "🎵",
  zip: "🗜", tar: "🗜", gz: "🗜", rar: "🗜",
  py: "🐍", js: "🟨", ts: "🟦", rs: "🦀", go: "🐹", swift: "🧡",
  html: "🌐", css: "🎨", sh: "⚙", yaml: "⚙", toml: "⚙",
  xls: "📊", xlsx: "📊", doc: "📝", docx: "📝", ppt: "📊", pptx: "📊",
};

const IMAGE_EXTS_SET  = new Set(["jpg","jpeg","png","gif","webp","bmp"]);
const TEXT_EXTS_SET   = new Set(["txt","md","py","js","ts","rs","go","swift","html","css","sh","bash","yaml","toml","json","xml","csv","log"]);
const PDF_EXT         = "pdf";
const DOCX_EXT        = "docx";
// Other formats (video, audio, archives, old Office) — info only, cannot extract text

function fileIcon(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return FILE_ICONS[ext] ?? "📄";
}

function fileActions(name: string, fullPath: string): { label: string; action: "send" | "attach"; prompt?: string; path?: string }[] {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  const getInfo = { label: "Get Info", action: "send" as const, prompt: `Use get_file_info on: ${fullPath}` };

  if (IMAGE_EXTS_SET.has(ext)) {
    return [
      { label: "Describe", action: "attach", path: fullPath },
      getInfo,
    ];
  }
  if (ext === PDF_EXT) {
    return [
      { label: "Summarise", action: "send", prompt: `Use read_file to read then summarise this PDF: ${fullPath}` },
      { label: "Key Points", action: "send", prompt: `Use read_file to read then list key points from this PDF: ${fullPath}` },
      getInfo,
    ];
  }
  if (ext === DOCX_EXT) {
    return [
      { label: "Read", action: "send", prompt: `Use read_file to read this Word document: ${fullPath}` },
      { label: "Summarise", action: "send", prompt: `Use read_file to read then summarise this Word document: ${fullPath}` },
      { label: "Key Points", action: "send", prompt: `Use read_file to read then list key points from this Word document: ${fullPath}` },
      getInfo,
    ];
  }
  if (TEXT_EXTS_SET.has(ext)) {
    return [
      { label: "Read", action: "send", prompt: `Use read_file to read: ${fullPath}` },
      { label: "Summarise", action: "send", prompt: `Use read_file to read then summarise: ${fullPath}` },
      getInfo,
    ];
  }
  // Anything else (video, audio, archives, old .doc/.xls etc.) — info only
  return [getInfo];
}

const FILE_LISTING_TOOLS = new Set(["list_files", "search_files", "find_old_files", "list_directory_tree"]);

function FileBrowserResult({
  name, result, args,
  onSend, onAttach,
}: {
  name: string; result: string; args?: string;
  onSend: (text: string) => void;
  onAttach: (path: string, prompt: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  // Parse base directory from args JSON
  const baseDir = (() => {
    try {
      const parsed = JSON.parse(args ?? "{}");
      return (parsed.path ?? parsed.directory ?? "") as string;
    } catch { return ""; }
  })();

  // Extract file/dir entries from result text
  const entries = result
    .split("\n")
    .map(l => l.trim())
    .filter(l => l && !l.startsWith("[") && !l.startsWith("No files") && !l.startsWith("Error"));

  const fileCount = entries.filter(e => !e.endsWith("/")).length;
  const dirCount  = entries.filter(e => e.endsWith("/")).length;

  const fullPath = (entry: string) => {
    const clean = entry.replace(/\/$/, "").replace(/^[└├│─\s]+/, ""); // strip tree decorators
    if (!baseDir || clean.startsWith("/")) return clean;
    return `${baseDir.replace(/\/$/, "")}/${clean}`;
  };

  const summary = `${fileCount} file${fileCount !== 1 ? "s" : ""}${dirCount ? `, ${dirCount} folder${dirCount !== 1 ? "s" : ""}` : ""}`;

  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner" onClick={() => setExpanded(e => !e)} style={{ cursor: "pointer" }}>
        <span className="tool-result-check">✓</span>
        <span className="tool-result-name">{name}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">{summary}</span>
        <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="file-browser">
          {entries.length === 0 && <div className="file-browser-empty">Empty</div>}
          {entries.map((entry, i) => {
            const isDir = entry.endsWith("/");
            const cleanName = entry.replace(/\/$/, "").replace(/^[└├│─\s]+/, "");
            const fp = fullPath(entry);
            const actions = isDir ? [] : fileActions(cleanName, fp);
            return (
              <div key={i} className="file-browser-row">
                <span className="file-browser-icon">{isDir ? "📁" : fileIcon(cleanName)}</span>
                <span className="file-browser-name">{cleanName}{isDir ? "/" : ""}</span>
                {!isDir && (
                  <div className="file-browser-actions">
                    {actions.map(a => (
                      <button
                        key={a.label}
                        className="file-action-chip"
                        onClick={e => {
                          e.stopPropagation();
                          if (a.action === "send" && a.prompt) onSend(a.prompt);
                          else if (a.action === "attach" && a.path) onAttach(a.path, `Describe this image: ${cleanName}`);
                        }}
                      >
                        {a.label}
                      </button>
                    ))}
                  </div>
                )}
                {isDir && (
                  <div className="file-browser-actions">
                    <button className="file-action-chip" onClick={e => { e.stopPropagation(); onSend(`List files in ${fp}`); }}>
                      Browse
                    </button>
                    <button className="file-action-chip" onClick={e => { e.stopPropagation(); onSend(`Show directory tree of ${fp}`); }}>
                      Tree
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── URL list (interactive tool result) ───────────────────────────────────────

function extractUrlsWithTitles(text: string): { url: string; title: string }[] {
  const lines = text.split('\n');
  const result: { url: string; title: string }[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < lines.length; i++) {
    const trimmed = lines[i].trimStart();
    const urlMatch = trimmed.match(/^(https?:\/\/[^\s"'\]>),}]+)/);
    if (!urlMatch) continue;
    const url = urlMatch[1];
    if (seen.has(url)) continue;
    seen.add(url);
    // Look back up to 3 lines for a numbered title "N. Title"
    let title = '';
    for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
      const m = lines[j].trim().match(/^\d+\.\s+(.+)$/);
      if (m) { title = m[1]; break; }
    }
    result.push({ url, title });
  }
  return result;
}

function urlLabel(url: string): string {
  try {
    const u = new URL(url);
    const path = decodeURIComponent(u.pathname + u.search).replace(/\/$/, "");
    return path ? `${u.hostname}${path}` : u.hostname;
  } catch { return url; }
}

function UrlListResult({ name, result, onSend }: { name: string; result: string; onSend?: (text: string) => void }) {
  const [expanded, setExpanded] = useState(false);
  const entries = extractUrlsWithTitles(result);

  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner" onClick={() => setExpanded(e => !e)} style={{ cursor: "pointer" }}>
        <span className="tool-result-check">✓</span>
        <span className="tool-result-name">{name}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">{entries.length} link{entries.length !== 1 ? "s" : ""}</span>
        <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="file-browser">
          {entries.map(({ url, title }, i) => (
            <div key={i} className="file-browser-row">
              <span className="file-browser-icon">🔗</span>
              <span className="file-browser-name" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={title || url}>
                {urlLabel(url)}
              </span>
              <div className="file-browser-actions">
                {onSend && (
                  <button
                    className="file-action-chip"
                    onClick={e => { e.stopPropagation(); onSend(`Use fetch_webpage to read the full content of this URL and summarise it: ${url}`); }}
                  >
                    Fetch
                  </button>
                )}
                <button
                  className="file-action-chip"
                  onClick={e => { e.stopPropagation(); openUrl(url); }}
                >
                  Open
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── MCP App (SEP-1865) sandboxed iframe + postMessage bridge ──────────────────
export function McpAppFrame({ ui, toolName, onSend }: { ui: ToolUi; toolName: string; onSend: (text: string) => void }) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  // Set by the effect; called from the iframe's onLoad to proactively deliver render data,
  // so an app whose one-shot "ready"/"initialize" announce raced ahead of our listener still
  // gets its content instead of rendering a blank frame.
  const kickRef = useRef<(() => void) | null>(null);
  const [approved, setApproved] = useState(approvedMcpApps.has(ui.server_id));

  useEffect(() => {
    if (!approved || !ui.html) return;
    const iframe = iframeRef.current;
    if (!iframe) return;

    // Surface bridge traffic in the Debug panel (window event, frontend-only).
    const logBridge = (dir: "host→app" | "app→host", msg: unknown) => {
      const m = msg as Record<string, unknown> | null;
      const label = String(m?.method ?? (m?.result ? "result" : m?.error ? "error" : m?.type) ?? "message");
      let preview = "";
      try { preview = JSON.stringify(msg).slice(0, 400); } catch { /* ignore */ }
      window.dispatchEvent(new CustomEvent("mcp-app-bridge", {
        detail: { dir, tool: toolName, label, preview },
      }));
    };

    const post = (msg: unknown) => { logBridge("host→app", msg); iframe.contentWindow?.postMessage(msg, "*"); };

    const proxyCall = async (toolNameArg: string, argsObj: unknown) =>
      invoke("mcp_ui_call_tool", { args: { server_id: ui.server_id, tool_name: toolNameArg, arguments: argsObj ?? {} } });

    // Deliver the tool's input + result to the app. Reused by the reactive handshake below
    // and by the proactive onLoad kick.
    const pushToolData = () => {
      post({ jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: ui.arguments ?? {} } });
      post({ jsonrpc: "2.0", method: "ui/notifications/tool-result", params: {
        content: ui.content ?? [],
        structuredContent: ui.structured ?? undefined,
        _meta: ui.meta ?? undefined,
      }});
    };

    // Proactive delivery covering both dialects — an app only understands one and ignores the
    // other. Fired on iframe load (and retried) so a missed one-shot announce doesn't blank it.
    const kick = () => {
      pushToolData();                                                                   // ext-apps
      post({ type: "ui-lifecycle-iframe-render-data", payload: { renderData: ui.structured ?? null } }); // MCP-UI
    };
    kickRef.current = kick;

    const onMessage = async (event: MessageEvent) => {
      if (event.source !== iframe.contentWindow) return; // only this app's iframe
      const data = event.data as Record<string, unknown> | null;
      if (!data || typeof data !== "object") return;
      logBridge("app→host", data);

      // ── ext-apps dialect: JSON-RPC over postMessage ──
      if (data.jsonrpc === "2.0" && typeof data.method === "string") {
        const { id, method } = data as { id?: unknown; method: string };
        const params = (data.params ?? {}) as Record<string, unknown>;

        try {
          if (method === "ui/initialize") {
            // Respond with the full McpUiInitializeResult shape the app SDK expects.
            post({ jsonrpc: "2.0", id, result: {
              protocolVersion: "2026-01-26",
              hostCapabilities: {},
              hostInfo: { name: "LexiChat", version: "2.4.25" },
              hostContext: {
                toolInfo: {
                  id: "1",
                  tool: { name: toolName, description: "", inputSchema: { type: "object", properties: {} } },
                },
                theme: "light",
                styles: { variables: {}, css: {} },
                displayMode: "inline",
                containerDimensions: { width: 600, height: 420 },
              },
            }});
            // Some apps render on the initialize result; others wait for the
            // initialized notification. Push tool data now as a fallback too.
            pushToolData();
          } else if (method === "ui/notifications/initialized") {
            // Spec-correct trigger: deliver tool input + result after init.
            pushToolData();
          } else if (method === "tools/call") {
            const r = await proxyCall(String(params.name ?? ""), params.arguments) as Record<string, unknown>;
            post({ jsonrpc: "2.0", id, result: {
              content: r.content ?? [{ type: "text", text: String(r.text ?? "") }],
              structuredContent: r.structured ?? undefined,
              isError: Boolean(r.isError),
            }});
          } else if (method === "ui/open-link") {
            if (params.url) openUrl(String(params.url)).catch(() => {});
            if (id != null) post({ jsonrpc: "2.0", id, result: {} });
          } else if (method === "ui/message" || method === "ui/sendMessage" || method === "sendMessage") {
            const text = String(params.text ?? params.prompt ?? "");
            if (text) onSend(text);
            if (id != null) post({ jsonrpc: "2.0", id, result: {} });
          } else if (method === "ui/request-display-mode") {
            if (id != null) post({ jsonrpc: "2.0", id, result: { displayMode: "inline" } });
          } else if (id != null) {
            // Unknown request — ack politely so the app isn't left hanging.
            post({ jsonrpc: "2.0", id, result: {} });
          }
        } catch (err) {
          if (id != null) post({ jsonrpc: "2.0", id, error: { code: -32000, message: String(err) } });
        }
        return;
      }

      // ── MCP-UI dialect: { type, payload, messageId? } ──
      const type = data.type as string | undefined;
      const messageId = data.messageId;
      const payload = (data.payload ?? {}) as Record<string, unknown>;
      const respond = (body: Record<string, unknown>) => { if (messageId != null) post({ type: "ui-message-response", messageId, payload: body }); };
      try {
        if (type === "ui-lifecycle-iframe-ready") {
          // MCP-UI app announced ready → send it the initial render data.
          post({ type: "ui-lifecycle-iframe-render-data", payload: { renderData: ui.structured ?? null } });
          return;
        }
        if (type === "tool") {
          const r = await proxyCall(String(payload.toolName ?? ""), payload.params);
          respond({ response: r });
        } else if (type === "prompt") {
          if (payload.prompt) onSend(String(payload.prompt));
          respond({ response: "ok" });
        } else if (type === "link") {
          if (payload.url) openUrl(String(payload.url)).catch(() => {});
          respond({ response: "ok" });
        } else if (type != null) {
          respond({ response: "ok" });
        }
      } catch (err) {
        respond({ error: String(err) });
      }
    };

    window.addEventListener("message", onMessage);
    return () => { window.removeEventListener("message", onMessage); kickRef.current = null; };
  }, [approved, ui, onSend]);

  if (!ui.html) return null;

  if (!approved) {
    return (
      <div className="msg-tool-result">
        <div style={{ padding: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>🔒 Interactive app from “{toolName}”</div>
          <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 8 }}>
            This MCP server wants to display an interactive UI that can call its tools. Only allow apps from servers you trust.
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn primary" onClick={async () => {
              try { await invoke("approve_mcp_app", { args: { server_id: ui.server_id } }); } catch { /* ignore */ }
              approvedMcpApps.add(ui.server_id);
              setApproved(true);
            }}>Allow app</button>
            <button className="btn" onClick={() => { invoke("skip_mcp_app").catch(() => {}); }}>Skip</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner">
        <span className="tool-result-check">✓</span>
        <span className="tool-result-name">{toolName}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">interactive app</span>
      </div>
      <iframe
        ref={iframeRef}
        title={`mcp-app-${toolName}`}
        sandbox="allow-scripts allow-forms"
        srcDoc={ui.html}
        onLoad={() => {
          // The app's one-shot ready/init announce may have raced ahead of our message
          // listener (→ blank frame). Proactively deliver render data now, and retry a
          // couple of times in case the app's own listener isn't attached yet.
          const k = kickRef.current;
          if (!k) return;
          k();
          setTimeout(() => kickRef.current?.(), 150);
          setTimeout(() => kickRef.current?.(), 500);
        }}
        style={{ width: "100%", height: 420, border: "1px solid var(--border)", borderRadius: 8, background: "#fff", marginTop: 6 }}
      />
    </div>
  );
}

// Error-reporting shim injected into every artifact. A model-authored page that throws used to
// fail SILENTLY — the frame just sat there blank white with no clue why, for the user or for us
// (real case: `L.marker(-0.0037)` threw "Cannot read properties of null (reading 'lat')", which
// killed the script before fitBounds, and all anyone saw was an empty box). The frame is sandboxed
// without allow-same-origin, so its origin is opaque and we can't read into it — but postMessage
// out still works. Capture phase catches failed <script>/<img>/<link> loads too (those don't
// bubble), which is how a blocked CDN or tile host announces itself.
// Messages are matched by a per-frame TOKEN, not by comparing e.source to the iframe's
// contentWindow: for an opaque-origin frame some engines report e.source as null, and an identity
// check then silently drops every report — the exact failure mode that makes this shim look like
// it "found nothing". The token also keeps several artifacts on screen from crossing wires.
function artifactShim(token: string): string {
  const t = JSON.stringify(token);
  return `<script>(function(){
  var sent = 0;
  function post(kind, msg, extra) {
    if (sent++ > 8) return;                       // a loop mustn't flood the parent
    try { parent.postMessage({ __lexiArtifact: ${t}, kind: kind, message: String(msg == null ? "" : msg), detail: extra || "" }, "*"); } catch (e) {}
  }
  // Announce that scripts run here at all. Absence of this is itself the diagnosis: the page's
  // JavaScript never executed, so a scripted artifact renders as a static skeleton.
  post("ready", "", "");
  window.addEventListener("error", function (e) {
    var t = e.target;
    if (t && t !== window && t.tagName) {         // a subresource failed to load
      post("resource", (t.tagName || "").toLowerCase() + " failed to load", t.src || t.href || "");
    } else {
      post("error", e.message, e.lineno ? "line " + e.lineno : "");
    }
  }, true);
  window.addEventListener("unhandledrejection", function (e) {
    post("promise", (e.reason && (e.reason.message || e.reason)) || "unhandled promise rejection", "");
  });
  // Leaflet caches the container size at construction; if the frame is resized afterwards (the
  // artifact frame is user-resizable) the map keeps its old size and paints nothing new. Nudge
  // every map instance whenever the document resizes.
  var maps = [];
  function hookLeaflet() {
    if (!window.L || !window.L.Map || !window.L.Map.addInitHook) return false;
    window.L.Map.addInitHook(function () { var m = this; maps.push(m); setTimeout(function(){ try { m.invalidateSize(); } catch (e) {} }, 0); });
    return true;
  }
  if (!hookLeaflet()) window.addEventListener("load", hookLeaflet);
  function refresh() { for (var i = 0; i < maps.length; i++) { try { maps[i].invalidateSize(); } catch (e) {} } }
  window.addEventListener("resize", refresh);
  if (window.ResizeObserver) { try { new ResizeObserver(refresh).observe(document.documentElement); } catch (e) {} }
})();</` + `script>`;
}

// An exception thrown inside a cross-origin script (Leaflet, from the CDN) is sanitised by the
// browser to a bare "Script error." with no message or line — useless. Both CDNs the CSP allows
// send `access-control-allow-origin: *`, so tagging those script elements `crossorigin="anonymous"`
// opts them into full error reporting. Scoped to exactly those two hosts: adding it to a host that
// does NOT send CORS headers would block the script instead.
const CORS_SCRIPT_RE = /<script\b(?![^>]*\bcrossorigin=)([^>]*\bsrc=["'](?:https:)?\/\/(?:unpkg\.com|cdn\.jsdelivr\.net)\/[^"']*["'][^>]*)>/gi;
export function withCorsScripts(html: string): string {
  return html.replace(CORS_SCRIPT_RE, (_m, attrs) => `<script crossorigin="anonymous"${attrs}>`);
}

// Put the shim first inside <head> (or <body>) so it is installed before the page's own scripts and
// external tags run. Never before <!doctype> — that would flip the document into quirks mode and
// break the `height:100%` layout most map artifacts rely on.
export function withErrorShim(rawHtml: string, token = "probe"): string {
  const ARTIFACT_ERROR_SHIM = artifactShim(token);
  const html = withCorsScripts(rawHtml);
  const head = html.match(/<head[^>]*>/i);
  if (head) return html.replace(head[0], head[0] + ARTIFACT_ERROR_SHIM);
  const body = html.match(/<body[^>]*>/i);
  if (body) return html.replace(body[0], body[0] + ARTIFACT_ERROR_SHIM);
  const htmlTag = html.match(/<html[^>]*>/i);
  if (htmlTag) return html.replace(htmlTag[0], htmlTag[0] + ARTIFACT_ERROR_SHIM);
  return ARTIFACT_ERROR_SHIM + html;              // fragment, no doctype to protect
}

// The Content-Security-Policy actually in force at runtime. Tauri injects it as a meta tag from
// tauri.conf.json (and may add nonce/hash sources of its own), so this is the only way to see what
// the shipped app is really enforcing — the config file is not the whole story.
export function runtimeCsp(): string {
  const el = document.querySelector('meta[http-equiv="Content-Security-Policy" i]');
  return el?.getAttribute("content") ?? "";
}

// Which iframe delivery mechanisms can actually execute a script in THIS webview? Observed on
// macOS: a sandboxed srcDoc frame runs nothing at all — no error, no CSP violation we can see, just
// a static page — so an artifact silently loses its map/chart. The cause isn't visible from the
// config (Tauri serves the CSP as a response header on macOS, not a meta tag, so it can't even be
// read from JS), so probe the alternatives empirically and let the result pick the mechanism.
export type FrameProbe = { variant: string; ok: boolean };
async function probeFrameVariant(sandbox: string | null, useBlob: boolean): Promise<boolean> {
  return new Promise<boolean>(resolve => {
    const token = "lexi-probe-" + Math.random().toString(36).slice(2);
    const doc = `<!doctype html><html><head></head><body><script>` +
      `parent.postMessage({__lexiProbe:${JSON.stringify(token)}},"*");</` + `script></body></html>`;
    const frame = document.createElement("iframe");
    if (sandbox !== null) frame.setAttribute("sandbox", sandbox);
    frame.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;left:-9999px";
    let url = "";
    const done = (ok: boolean) => {
      window.removeEventListener("message", onMsg);
      clearTimeout(timer);
      frame.remove();
      if (url) URL.revokeObjectURL(url);
      resolve(ok);
    };
    const onMsg = (e: MessageEvent) => {
      if ((e.data as { __lexiProbe?: string })?.__lexiProbe === token) done(true);
    };
    window.addEventListener("message", onMsg);
    const timer = setTimeout(() => done(false), 10000);
    if (useBlob) {
      url = URL.createObjectURL(new Blob([doc], { type: "text/html" }));
      frame.src = url;
    } else {
      frame.srcdoc = doc;
    }
    document.body.appendChild(frame);
  });
}

/// Run every variant and report which can execute scripts. In PARALLEL and with a generous
/// timeout: run sequentially with a short one at startup, every variant "failed" — including the
/// fully permissive control, which is the signature of a starved main thread rather than a policy.
/// A control that cannot fail is included so a run where everything reports false is recognisable
/// as a broken measurement instead of being read as a finding.
export async function probeFrameVariants(): Promise<FrameProbe[]> {
  const variants: [string, string | null, boolean][] = [
    ["sandbox+srcdoc", "allow-scripts", false],          // what artifacts use today
    ["srcdoc-nosandbox", null, false],
    ["sandbox+same-origin+srcdoc", "allow-scripts allow-same-origin", false],
    ["sandbox+blob", "allow-scripts", true],
    ["blob-nosandbox", null, true],
  ];
  const runs = variants.map(async ([variant, sandbox, useBlob]) =>
    ({ variant, ok: await probeFrameVariant(sandbox, useBlob) }));
  return Promise.all(runs);
}

/// The CSP the webview is really enforcing. On macOS/Windows Tauri returns it as a response header
/// rather than a meta tag, so it can't be read from the DOM — but the page can re-fetch its own
/// URL through the same protocol handler and read the header off that response.
export async function fetchRuntimeCsp(): Promise<string> {
  try {
    const res = await fetch(location.href, { cache: "no-store" });
    return res.headers.get("content-security-policy") ?? "(no CSP response header)";
  } catch (e) {
    return "(could not read: " + String(e) + ")";
  }
}

// Do inline scripts actually RUN inside an artifact frame? A srcDoc iframe inherits the parent's
// CSP, so a policy that (for example) carries a nonce silently disables 'unsafe-inline' and every
// artifact becomes a static page: no map, no chart, and no error either — the page's own scripts
// never execute, so nothing is left to report it. Probe it once with a throwaway frame rather than
// reasoning about the policy string, and cache the answer for the session.
let artifactScriptProbe: Promise<boolean> | null = null;
export function probeArtifactScripts(): Promise<boolean> {
  if (artifactScriptProbe) return artifactScriptProbe;
  artifactScriptProbe = new Promise<boolean>(resolve => {
    if (typeof document === "undefined") { resolve(true); return; }
    const token = "lexi-probe-" + Math.random().toString(36).slice(2);
    const frame = document.createElement("iframe");
    frame.setAttribute("sandbox", "allow-scripts");
    frame.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;left:-9999px";
    const done = (ok: boolean) => {
      window.removeEventListener("message", onMsg);
      clearTimeout(timer);
      frame.remove();
      resolve(ok);
    };
    const onMsg = (e: MessageEvent) => {
      if ((e.data as { __lexiProbe?: string })?.__lexiProbe === token) done(true);
    };
    window.addEventListener("message", onMsg);
    const timer = setTimeout(() => done(false), 10000);   // generous: a busy main thread must not read as 'blocked'
    frame.srcdoc = `<!doctype html><html><head></head><body><script>` +
      `parent.postMessage({__lexiProbe:${JSON.stringify(token)}},"*");</` + `script></body></html>`;
    document.body.appendChild(frame);
  });
  return artifactScriptProbe;
}

// Model-authored HTML artifact (create_artifact) — rendered inline in a sandboxed frame with a
// Save button. Static-or-scripted HTML; sandbox allows scripts but not same-origin/network.
function ArtifactFrame({ title, html }: { title: string; html: string }) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [errors, setErrors] = useState<{ kind: string; message: string; detail: string }[]>([]);
  const [scriptsRan, setScriptsRan] = useState<boolean | null>(null);
  // One token per mounted frame, so reports are attributed by value rather than by comparing
  // window identities (see artifactShim).
  const token = useMemo(() => "lexi-art-" + Math.random().toString(36).slice(2), []);
  const shimmed = useMemo(() => withErrorShim(html, token), [html, token]);
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      const d = e.data as { __lexiArtifact?: string; kind?: string; message?: string; detail?: string };
      if (d?.__lexiArtifact !== token) return;
      if (d.kind === "ready") { setScriptsRan(true); return; }
      setErrors(prev => {
        const next = { kind: d.kind ?? "error", message: d.message ?? "", detail: d.detail ?? "" };
        if (prev.some(p => p.message === next.message && p.detail === next.detail)) return prev;
        return [...prev, next];
      });
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, [token]);
  // No "ready" ping shortly after mount means the page's scripts never executed at all.
  useEffect(() => {
    setScriptsRan(null);
    if (!/<script/i.test(html)) return;
    const t = setTimeout(() => setScriptsRan(v => (v === null ? false : v)), 12000);
    return () => clearTimeout(t);
  }, [html]);
  // A re-render with different HTML is a different page — drop the old page's errors.
  useEffect(() => { setErrors([]); }, [html]);
  // If artifact frames can't run scripts at all, a scripted page renders as a static skeleton with
  // nothing to explain it. Detect that and say so, rather than leaving a blank box.
  const [scriptsBlocked, setScriptsBlocked] = useState(false);
  useEffect(() => {
    let live = true;
    if (/<script/i.test(html)) probeArtifactScripts().then(ok => { if (live && !ok) setScriptsBlocked(true); });
    return () => { live = false; };
  }, [html]);

  const saveArtifact = async () => {
    const safe = title.replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "artifact";
    try {
      const path = await save({ title: "Save artifact", defaultPath: `${safe}.html`,
        filters: [{ name: "HTML", extensions: ["html"] }] });
      if (!path) return;
      await invoke("write_file_text", { path, content: html });
    } catch { /* cancelled */ }
  };
  return (
    <div className="msg-tool-result artifact-block">
      <div className="artifact-head">
        <span className="artifact-title">▤ {title}</span>
        <button className="artifact-save" onClick={saveArtifact}>Save HTML…</button>
      </div>
      <iframe ref={iframeRef} className="artifact-frame" sandbox="allow-scripts"
        srcDoc={shimmed} title={`artifact-${title}`} />
      {(scriptsRan === false || scriptsBlocked) && (
        <div className="artifact-errors">
          <span className="artifact-errors-title">⚠ Scripts are blocked inside artifacts</span>
          <div className="artifact-error-row">
            This page's JavaScript never ran, so anything it draws (maps, charts, interactivity) is
            missing. The frame inherits the app's content-security policy, which is currently
            disallowing inline scripts.
          </div>
          <div className="artifact-error-row"><code>{runtimeCsp()
            || "CSP is delivered as a response header on macOS/Windows, so it cannot be read here."}</code></div>
          <div className="artifact-error-hint">Saving the HTML and opening it in a browser will still work.</div>
        </div>
      )}
      {errors.length > 0 && (
        <div className="artifact-errors">
          <span className="artifact-errors-title">⚠ This page reported {errors.length === 1 ? "an error" : `${errors.length} errors`}</span>
          {errors.map((e, i) => (
            <div key={i} className="artifact-error-row">
              <code>{e.message}</code>{e.detail && <span className="artifact-error-detail"> — {e.detail}</span>}
            </div>
          ))}
          <div className="artifact-error-hint">
            Part of the page may be missing. Ask for it to be fixed and mention this message.
          </div>
        </div>
      )}
    </div>
  );
}

// ── Connector data viewer (OpenAPI / MCP / SPARQL structured results) ───────────
// Renders a tool's structured response (JSON/CSV) as a readable table/grid, with a Raw view,
// Copy, and Export — so the user can verify the data AS RETURNED, independent of what the model
// then says about it. Fed by the FULL (untruncated) result when the backend provides it.
function stripHttpPrefix(s: string): string { return s.replace(/^HTTP\s+\d{3}\s*\n/, ""); }
const RECORD_KEYS = ["records", "results", "data", "items", "rows", "obs", "features", "members", "correlations", "value"];

type ParsedResult =
  | { kind: "table"; columns: string[]; rows: Record<string, unknown>[] }
  | { kind: "object"; entries: [string, unknown][] }
  | { kind: "list"; items: unknown[] }
  | { kind: "json"; value: unknown }
  | null;

function recordArray(v: unknown): Record<string, unknown>[] | null {
  return Array.isArray(v) && v.length > 0 && v.every(x => x && typeof x === "object" && !Array.isArray(x))
    ? (v as Record<string, unknown>[]) : null;
}
function columnsOf(rows: Record<string, unknown>[]): string[] {
  const cols: string[] = [];
  for (const r of rows) for (const k of Object.keys(r)) if (!cols.includes(k)) cols.push(k);
  return cols;
}
function parseCsv(t: string): { columns: string[]; rows: Record<string, unknown>[] } | null {
  const lines = t.split(/\r?\n/).filter(l => l.length > 0);
  if (lines.length < 2) return null;
  const delim = lines[0].includes("\t") ? "\t" : lines[0].includes(",") ? "," : null;
  if (!delim) return null;
  const header = lines[0].split(delim).map(h => h.trim());
  if (header.length < 2) return null;
  const rowsRaw = lines.slice(1).map(l => l.split(delim));
  if (!rowsRaw.every(r => Math.abs(r.length - header.length) <= 1)) return null;
  const rows = rowsRaw.map(cells => { const o: Record<string, unknown> = {}; header.forEach((h, i) => o[h] = cells[i] ?? ""); return o; });
  return { columns: header, rows };
}
function parseStructured(text: string): ParsedResult {
  const t = stripHttpPrefix(text).trim();
  if (!t) return null;
  if (t.startsWith("{") || t.startsWith("[")) {
    let v: unknown;
    try { v = JSON.parse(t); } catch { return null; } // truncated/invalid JSON → let caller fall back
    const arr = recordArray(v);
    if (arr) return { kind: "table", columns: columnsOf(arr), rows: arr };
    if (Array.isArray(v)) return { kind: "list", items: v };
    if (v && typeof v === "object") {
      for (const k of RECORD_KEYS) {
        const inner = recordArray((v as Record<string, unknown>)[k]);
        if (inner) return { kind: "table", columns: columnsOf(inner), rows: inner };
      }
      return { kind: "object", entries: Object.entries(v as Record<string, unknown>) };
    }
    return { kind: "json", value: v };
  }
  const csv = parseCsv(t);
  return csv ? { kind: "table", columns: csv.columns, rows: csv.rows } : null;
}
// CHEAP structured-shape sniff for the render-path decision — no JSON.parse (which, run on every
// re-render during streaming for a large result, could choke the webview). The real parse happens
// once, memoized, inside ConnectorResult (and only when the row is expanded).
function maybeStructured(text: string): boolean {
  const t = stripHttpPrefix(text).trimStart();
  if (t[0] === "{" || t[0] === "[") return true;
  const nl = t.indexOf("\n");
  return nl > 0 && (t.slice(0, nl).includes(",") || t.slice(0, nl).includes("\t")) && t.indexOf("\n", nl + 1) > 0;
}
function cellText(v: unknown): string { return v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v); }
function toCsv(columns: string[], rows: Record<string, unknown>[]): string {
  const esc = (s: string) => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  return columns.map(esc).join(",") + "\n" +
    rows.map(r => columns.map(c => esc(cellText(r[c]))).join(",")).join("\n");
}

function ConnectorResult({ name, result, fullResult, fullTruncated, args }:
  { name: string; result: string; fullResult?: string; fullTruncated?: boolean; args?: string }) {
  const [expanded, setExpanded] = useState(false);
  const [raw, setRaw] = useState(false);
  const [copied, setCopied] = useState(false);
  const data = fullResult ?? result;
  const parsed = useMemo(() => parseStructured(data), [data]);
  const rawText = stripHttpPrefix(data);
  const modelTruncated = !fullResult && /\[truncated[:—]/.test(result);
  const partial = !!fullTruncated || modelTruncated;
  const MAX_ROWS = 1000;

  const summary = !parsed ? "data"
    : parsed.kind === "table"  ? `${parsed.rows.length} row${parsed.rows.length !== 1 ? "s" : ""}`
    : parsed.kind === "object" ? `${parsed.entries.length} field${parsed.entries.length !== 1 ? "s" : ""}`
    : parsed.kind === "list"   ? `${parsed.items.length} item${parsed.items.length !== 1 ? "s" : ""}`
    : "data";

  const copy = async () => { await navigator.clipboard.writeText(rawText); setCopied(true); setTimeout(() => setCopied(false), 1200); };
  const exportData = async () => {
    const isTable = parsed?.kind === "table";
    const ext = isTable ? "csv" : "json";
    const content = isTable ? toCsv(parsed!.columns, parsed!.rows) : rawText;
    const safe = name.replace(/[^a-z0-9_-]+/gi, "_") || "data";
    try {
      const path = await save({ title: "Export data", defaultPath: `${safe}.${ext}`,
        filters: [{ name: ext.toUpperCase(), extensions: [ext] }, { name: "All", extensions: ["*"] }] });
      if (path) await invoke("save_document", { path, content });
    } catch { /* cancelled */ }
  };

  const td: React.CSSProperties = { padding: "4px 8px", borderBottom: "1px solid var(--border)", verticalAlign: "top" };
  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner" onClick={() => setExpanded(e => !e)} style={{ cursor: "pointer" }}>
        <span className="tool-result-check">⚡</span>
        <span className="tool-result-name">{name}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">{summary}</span>
        {partial && <span style={{ marginLeft: 6, fontSize: 9, color: "var(--warn, #d97706)" }} title="Large response — capped view; use Export for everything">⚠ partial</span>}
        <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div style={{ padding: "6px 10px 10px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
            {parsed && parsed.kind !== "json" && (
              <button className="file-action-chip" onClick={() => setRaw(r => !r)}>{raw ? "Formatted" : "Raw"}</button>
            )}
            <button className="file-action-chip" onClick={copy}>{copied ? "Copied ✓" : "Copy"}</button>
            <button className="file-action-chip" onClick={exportData}>Export</button>
          </div>
          {args && (
            <div style={{ marginBottom: 6 }}>
              <div style={{ fontSize: 9, fontWeight: 600, color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 2 }}>Parameters</div>
              <pre style={{ margin: 0, padding: "6px 8px", fontSize: 10, lineHeight: 1.45, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text-secondary)", fontFamily: "'SF Mono','Fira Code',monospace", whiteSpace: "pre-wrap", wordBreak: "break-word", maxHeight: 160, overflow: "auto" }}>{args}</pre>
            </div>
          )}
          {partial && (
            <div style={{ fontSize: 10, color: "var(--warn, #d97706)", marginBottom: 6 }}>
              {fullTruncated ? "Response exceeded the display cap — Export for the complete data." : "The model saw a truncated slice; the full data is shown here."}
            </div>
          )}
          <div style={{ maxHeight: 340, overflow: "auto", border: "1px solid var(--border)", borderRadius: 6, background: "var(--surface2)" }}>
            {(!parsed || raw) ? (
              <pre style={{ margin: 0, padding: "8px 10px", fontSize: 11, lineHeight: 1.5, fontFamily: "'SF Mono','Fira Code',monospace", whiteSpace: "pre-wrap", wordBreak: "break-word", color: "var(--text)" }}>{rawText}</pre>
            ) : parsed.kind === "table" ? (
              <table style={{ borderCollapse: "collapse", fontSize: 11, width: "100%" }}>
                <thead><tr>{parsed.columns.map(c => (
                  <th key={c} style={{ position: "sticky", top: 0, background: "var(--surface3, var(--surface2))", textAlign: "left", padding: "5px 8px", borderBottom: "1px solid var(--border)", fontWeight: 600, whiteSpace: "nowrap" }}>{c}</th>
                ))}</tr></thead>
                <tbody>{parsed.rows.slice(0, MAX_ROWS).map((r, i) => (
                  <tr key={i}>{parsed.columns.map(c => (
                    <td key={c} style={{ ...td, maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={cellText(r[c])}>{cellText(r[c])}</td>
                  ))}</tr>
                ))}</tbody>
              </table>
            ) : parsed.kind === "object" ? (
              <table style={{ borderCollapse: "collapse", fontSize: 11, width: "100%" }}>
                <tbody>{parsed.entries.map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ ...td, fontWeight: 600, whiteSpace: "nowrap", color: "var(--purple)" }}>{k}</td>
                    <td style={{ ...td, wordBreak: "break-word" }}>{cellText(v)}</td>
                  </tr>
                ))}</tbody>
              </table>
            ) : parsed.kind === "list" ? (
              <ol style={{ margin: 0, padding: "8px 8px 8px 26px", fontSize: 11, lineHeight: 1.6 }}>
                {parsed.items.slice(0, MAX_ROWS).map((it, i) => <li key={i} style={{ wordBreak: "break-word" }}>{cellText(it)}</li>)}
              </ol>
            ) : (
              <pre style={{ margin: 0, padding: "8px 10px", fontSize: 11, whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "'SF Mono','Fira Code',monospace" }}>{JSON.stringify(parsed.value, null, 2)}</pre>
            )}
          </div>
          {parsed?.kind === "table" && parsed.rows.length > MAX_ROWS && (
            <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 4 }}>Showing first {MAX_ROWS} of {parsed.rows.length} rows — Export for all.</div>
          )}
        </div>
      )}
    </div>
  );
}

export function ToolResultRow({
  name, result, args, fullResult, fullTruncated, ui, images, artifact, onSend, onAttach,
}: {
  name: string; result: string; args?: string; fullResult?: string; fullTruncated?: boolean;
  ui?: ToolUi; images?: string[];
  artifact?: { title: string; html: string };
  onSend: (text: string) => void;
  onAttach: (path: string, prompt: string) => void;
}) {
  if (artifact?.html) {
    return <ArtifactFrame title={artifact.title} html={artifact.html} />;
  }
  if (ui?.html) {
    return <McpAppFrame ui={ui} toolName={name} onSend={onSend} />;
  }
  // A tool that returned image(s) (e.g. a Mapbox static map) — render them inline. Works
  // without the MCP-App flow; data: URLs are allowed by the CSP.
  if (images && images.length > 0) {
    return (
      <div className="msg-tool-result">
        {images.map((src, i) => (
          <div key={i} style={{ position: "relative", marginTop: i ? 8 : 0, display: "inline-block", maxWidth: "100%" }}>
            <img src={src} alt={`${name} image ${i + 1}`}
              style={{ maxWidth: "100%", borderRadius: 10, display: "block" }} />
            <button title="Save image" onClick={() => downloadImage(src, `${name}-${i + 1}`)}
              style={{ position: "absolute", top: 8, right: 8, width: 28, height: 28, borderRadius: 8,
                border: "none", cursor: "pointer", background: "rgba(15,23,42,0.55)", color: "#fff",
                fontSize: 15, lineHeight: "28px", textAlign: "center", padding: 0 }}>⤓</button>
          </div>
        ))}
      </div>
    );
  }
  if (FILE_LISTING_TOOLS.has(name)) {
    return <FileBrowserResult name={name} result={result} args={args} onSend={onSend} onAttach={onAttach} />;
  }
  // Structured connector data (OpenAPI/MCP/SPARQL JSON or CSV) → the readable data viewer. Checked
  // before the URL list since web_search returns plain text (not JSON), so it's unaffected.
  if (maybeStructured(fullResult ?? result)) {
    return <ConnectorResult name={name} result={result} fullResult={fullResult} fullTruncated={fullTruncated} args={args} />;
  }
  const urls = extractUrlsWithTitles(result);
  if (urls.length > 0) {
    return <UrlListResult name={name} result={result} onSend={onSend} />;
  }
  const preview = result.length > 120 ? result.slice(0, 120) + "…" : result;
  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner">
        <span className="tool-result-check">✓</span>
        <span className="tool-result-name">{name}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">{preview}</span>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedServerId, setSelectedServerId] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  // Start of the current model "Thinking…" phase (prompt-eval + reasoning before output). Reset at
  // send and after each tool result (the model thinks again for the next step); null when idle.
  // Drives the live timer on the thinking indicator, mirroring the per-tool-call timers.
  const [thinkingAt, setThinkingAt] = useState<number | null>(null);
  const [showAdmin, setShowAdmin] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showUsageLive, setShowUsageLive] = useState(false);
  const [showUsageHistory, setShowUsageHistory] = useState(false);
  const [showHistory, setShowHistory] = useState(false); // hidden on launch; toggle to open
  const [conversations, setConversations] = useState<ConversationMeta[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [showWikiGraph, setShowWikiGraph] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [btcCopied, setBtcCopied] = useState(false);
  const [view,     setView]     = useState<"chat" | "jobs">("chat");
  const [jobBadge, setJobBadge] = useState(0);
  // Pending run_python execution awaiting the user's approval.
  const [permissionRequest, setPermissionRequest] = useState<{ code: string } | null>(null);
  // Styled-report export: preview of the themed HTML before saving.
  const [reportPreview, setReportPreview] = useState<{ html: string; markdown: string; title: string } | null>(null);

  const exportReport = async (msgId: string) => {
    // Gather the WHOLE turn's response — the model may emit prose across several steps (split into
    // multiple assistant bubbles). Turn = messages between the preceding and next user message.
    const msgs = messagesRef.current;
    const idx = msgs.findIndex(m => m.id === msgId);
    if (idx < 0) return;
    let s = 0;
    for (let i = idx; i >= 0; i--) { if (msgs[i].role === "user") { s = i + 1; break; } }
    let e = msgs.length;
    for (let i = idx + 1; i < msgs.length; i++) { if (msgs[i].role === "user") { e = i; break; } }
    const turn = msgs.slice(s, e);
    const markdown = turn.filter(m => m.role === "assistant" && m.text).map(m => m.text).join("\n\n");
    if (!markdown.trim()) return;

    const m = markdown.match(/^#\s+(.+)$/m);
    const title = (m?.[1] ?? "LexiChat Report").trim();
    const subtitle = activeProfile?.name;
    // Figures generated in this turn: {{figure:N}} tokens go inline; the rest append as a section.
    const figs = turn.filter(x => x.role === "tool-result").flatMap(x => x.toolImages ?? []).filter(u => u.startsWith("data:"));
    const { out: md2, used } = substituteFigures(markdown, figs, true);
    const unused = figs.filter((_, i) => !used.has(i));
    try {
      const html = await invoke<string>("render_report_html", { args: { markdown: md2, title, subtitle, figures: unused } });
      setReportPreview({ html, markdown, title });
    } catch (err) {
      setMessages(prev => [...prev, { id: uid(), role: "error", text: `Could not render report: ${String(err)}` }]);
    }
  };

  // HTML and Word both save the exact themed HTML (Word opens HTML `.doc` with styling + inline
  // images) — so both keep the report's look and its charts.
  const saveReportAs = async (fmt: "html" | "doc") => {
    if (!reportPreview) return;
    const safe = reportPreview.title.replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "").slice(0, 60) || "report";
    const label = fmt === "doc" ? "Word Document" : "HTML report";
    try {
      const path = await save({ title: "Save report", defaultPath: `${safe}.${fmt}`,
        filters: [{ name: label, extensions: [fmt] }] });
      if (!path) return;
      await invoke("write_file_text", { path, content: reportPreview.html });
      setReportPreview(null);
      setMessages(prev => [...prev, { id: uid(), role: "notice", text: `Report saved: ${path}` }]);
    } catch (err) {
      setMessages(prev => [...prev, { id: uid(), role: "error", text: `Could not save report: ${String(err)}` }]);
    }
  };

  // Save stashed run_python output file(s) when the user clicks the Save button on the notice.
  const saveOutputFiles = async (noticeId: string) => {
    const dir = await open({ directory: true, title: "Choose a folder to save the output file(s)" }).catch(() => null);
    if (!dir || typeof dir !== "string") return;
    try {
      const saved = await invoke<string[]>("save_pending_outputs", { dir });
      setMessages(prev => prev.map(m => m.id === noticeId
        ? { ...m, text: `Saved to ${dir} (added to the sandbox): ${saved.join(", ")}`, savePrompt: undefined }
        : m));
    } catch (err) {
      setMessages(prev => [...prev, { id: uid(), role: "error", text: `Could not save: ${String(err)}` }]);
    }
  };

  // Faithful PDF: open the styled report in the browser, where Print → Save as PDF is exact.
  const printReport = async () => {
    if (!reportPreview) return;
    try {
      await invoke("open_html_in_browser", { html: reportPreview.html });
    } catch (err) {
      setMessages(prev => [...prev, { id: uid(), role: "error", text: `Could not open the report: ${String(err)}` }]);
    }
  };
  const [attachedFiles, setAttachedFiles] = useState<string[]>([]);
  // Per-attached-image edit masks (image path → mask PNG data URL). Set via the brush editor; sent
  // aligned to image_paths so the backend inpaints only the painted region.
  const [imageMasks, setImageMasks] = useState<Record<string, string>>({});
  const [maskEditorFor, setMaskEditorFor] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Active profile derived from settings
  const activeProfile: Profile | null =
    settings.profiles.find(p => p.id === settings.activeProfileId) ?? null;

  // Flat list of every (server, model) the dropdown can offer, in server order.
  const modelOptions = (settings.servers ?? []).flatMap(s =>
    (s.models ?? []).map(m => ({ serverId: s.id, serverName: s.name, model: m })));
  // Refetch models only when a server's *connection* changes (not when its model list is merged).
  const serversKey = JSON.stringify((settings.servers ?? []).map(s => [s.id, s.provider, s.baseUrl, s.apiKey ?? ""]));

  // Fetch each server's models independently and merge into that server's persisted list, so the
  // dropdown shows the union across all configured backends.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const servers = settings.servers ?? [];
      const results = await Promise.all(servers.map(async s => {
        try {
          const list = await invoke<string[]>("get_models",
            { args: { base_url: s.baseUrl, provider: s.provider, api_key: s.apiKey ?? null } });
          return { id: s.id, list };
        } catch { return { id: s.id, list: [] as string[] }; }
      }));
      if (cancelled) return;
      setSettings(prev => {
        const merged = (prev.servers ?? []).map(s => {
          const found = results.find(r => r.id === s.id);
          return found ? reconcileCatalog(s, found.list) : s;
        });
        const updated = { ...prev, servers: merged };
        saveSettings(updated);
        return updated;
      });
    })();
    return () => { cancelled = true; };
  }, [serversKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keep the selection valid: if the chosen model vanished (server removed, list changed), fall
  // back to the first available option.
  useEffect(() => {
    if (modelOptions.length === 0) return;
    const valid = modelOptions.some(o => o.serverId === selectedServerId && o.model === selectedModel);
    if (!valid) { setSelectedServerId(modelOptions[0].serverId); setSelectedModel(modelOptions[0].model); }
  }, [serversKey, modelOptions.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Seed the Rust code-exec session flag from the persisted "Always allow code execution" setting,
  // so approved users skip the run_python approval prompt across restarts. Turning it off re-locks
  // (the next run_python asks again). Fires at startup and whenever the setting is toggled.
  useEffect(() => {
    invoke("set_code_exec_unlocked", { unlocked: settings.alwaysAllowCodeExec === true }).catch(() => {});
  }, [settings.alwaysAllowCodeExec]);

  // Seed the Rust image-generation config (sd.cpp binary/model paths + defaults) from settings, so
  // the generate_image tool can find them. Fires at startup and whenever the config changes.
  useEffect(() => {
    invoke("set_image_gen_config", { args: settings.imageGen ?? {} }).catch(() => {});
  }, [settings.imageGen]);

  // ── Chat history ──────────────────────────────────────────────────────────
  // Reload the per-profile conversation list. Re-runs on profile switch since it
  // depends on activeProfileId.
  const refreshConversations = useCallback(async () => {
    try {
      const list = await invoke<ConversationMeta[]>("list_conversations", {
        args: { profile_id: settings.activeProfileId ?? null },
      });
      setConversations(Array.isArray(list) ? list : []);
    } catch { /* history unavailable */ }
  }, [settings.activeProfileId]);

  useEffect(() => { refreshConversations(); }, [refreshConversations]);

  // Full-content chat search. The backend scans message text (the index holds only metadata), so
  // this is a debounced call rather than a client-side filter. `null` hits = not searching; an
  // empty map = searched, nothing matched.
  const [searchQuery, setSearchQuery] = useState("");
  const [searchHits, setSearchHits] = useState<Map<string, string> | null>(null);
  useEffect(() => {
    const q = searchQuery.trim();
    if (!q) { setSearchHits(null); return; }
    const t = setTimeout(async () => {
      try {
        const hits = await invoke<{ id: string; snippet: string }[]>("search_conversations", {
          args: { query: q, profile_id: settings.activeProfileId ?? null },
        });
        setSearchHits(new Map(hits.map(h => [h.id, h.snippet])));
      } catch { setSearchHits(new Map()); }
    }, 200);
    return () => clearTimeout(t);
  }, [searchQuery, settings.activeProfileId]);

  // What the history panel actually shows: the full list, or — while searching — only the matches,
  // each carrying its snippet, in the existing pinned-first order.
  const displayedConversations = useMemo(() => {
    if (!searchHits) return conversations;
    return conversations
      .filter(c => searchHits.has(c.id))
      .map(c => ({ ...c, snippet: searchHits.get(c.id) }));
  }, [conversations, searchHits]);

  // Stream ownership: agent events from Rust are anonymous, so a run that's been
  // superseded (new chat / profile switch) must not have its trailing tokens land in the
  // now-visible chat. `streamEpoch` bumps on every send AND every context switch;
  // `streamOwner` is pinned to the epoch when a run starts. Events are applied only while
  // the two match — i.e. the run that owns the stream is still the one on screen.
  const streamEpoch = useRef(0);
  const streamOwner = useRef(0);
  const streamActive = () => streamOwner.current === streamEpoch.current;

  // Dev control (debug builds): the /dev/run HTTP endpoint drives runs through send() headlessly.
  // Refs keep the listener (registered once) pointed at the latest state/functions.
  const sendRef = useRef<((t: string) => Promise<void>) | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  // Per-turn usage tally: reset in send(), accumulated by the agent-* listeners, written to the
  // local usage log on agent-done (the backend merges in token counts). All on-device.
  const turnTallyRef = useRef<{ start: number; model: string; provider: string; profile: string; tools: Record<string, number>; images: number }>(
    { start: 0, model: "", provider: "", profile: "", tools: {}, images: 0 });
  const isRunningRef = useRef(false);
  const autoApproveControlRef = useRef(false);
  const settingsRef = useRef(settings);
  const selectedModelRef = useRef(selectedModel);
  const chatParamsRef = useRef<ChatParams | null>(null);
  const profileSwitchRef = useRef<((id: string) => Promise<void>) | null>(null);
  const handleResetRef = useRef<(() => Promise<void>) | null>(null);
  const forceAllowCodeToolsRef = useRef(false); // dev-control transient override for allow_code_tools
  // Text payloads run_python wrote to /work/artifacts, keyed by filename, backing the
  // `{{data:name}}` token in create_artifact HTML. Kept for the whole CONVERSATION, not one turn:
  // the backend persists the same files and re-stages them into /work/artifacts each turn, so a
  // dataset built in one message can be rendered in a later one. Cleared on reset / new chat.
  const turnDataFilesRef = useRef<Map<string, PyDataFile>>(new Map());
  // Every file attached so far in THIS conversation. Attachments are cleared from the
  // composer on send, but the files stay on disk and stay editable — "make the beard
  // lighter" three messages later must edit that photo, not reimagine it. Conversation-
  // scoped, so a new or reopened chat never inherits the previous one's files.
  const conversationFilesRef = useRef<string[]>([]);
  // Attachments a reopened chat expected but that are no longer on disk (moved or deleted
  // since it was saved). Named to the model so it says so, rather than reinventing the file.
  const missingAttachmentsRef = useRef<string[]>([]);
  // Result of the artifact inline-script probe, surfaced in /dev/state for headless diagnosis.
  const artifactProbeResultRef = useRef<boolean | null>(null);
  const artifactFrameProbesRef = useRef<FrameProbe[] | null>(null);
  messagesRef.current = messages;
  isRunningRef.current = isRunning;
  settingsRef.current = settings;
  selectedModelRef.current = selectedModel;

  // Cancel a running agent loop and supersede its stream so late events are dropped.
  const stopActiveRun = () => {
    invoke("stop_generation").catch(() => {});
    abortPyodideRun(); // kill a runaway run_python (WASM can't be interrupted — terminate + respawn)
    streamEpoch.current += 1;
    setIsRunning(false);
    setThinkingAt(null);
    // Close any streaming "Thinking…" bubble so it can't persist after Stop, and drop an empty one.
    setMessages(prev => finalizeCallTimers(prev)
      .map(m => (m.streaming ? { ...m, streaming: false, status: undefined } : m))
      .filter(m => !(m.role === "assistant" && !m.streaming && !m.text && !(m.toolCalls?.length))));
  };

  // Persist the active conversation (upserts by the Rust-side active id, so repeated calls update
  // the same record). Drops the large full connector-data blob — it's a live-session aid, not
  // history. Used both on run-completion AND incrementally mid-run, so a webview reload/crash during
  // a long run can't lose the chat (it's only ever re-created as a "New Chat" on reload otherwise).
  const saveActiveConversation = useCallback(async () => {
    const msgs = messagesRef.current;
    if (msgs.length === 0) return;
    try {
      const meta = await invoke<ConversationMeta>("save_active_conversation", {
        args: {
          // Strip transient/non-persisted fields: fullResult (huge, connector-only), and the
          // streaming/status run state — persisting streaming:true is what caused reopened chats
          // to show eternal "thinking" dots.
          display: msgs.map(m => ({ ...m, fullResult: undefined, streaming: undefined, status: undefined })),
          profile_id: settings.activeProfileId ?? null,
          model: selectedModel,
          message_count: msgs.length,
          attachments: conversationFilesRef.current,
        },
      });
      setActiveConversationId(meta.id);
      refreshConversations();
    } catch { /* empty wire — nothing to save */ }
  }, [settings.activeProfileId, selectedModel, refreshConversations]);

  // Auto-save when an agent run finishes (transition running→idle).
  const prevRunning = useRef(false);
  useEffect(() => {
    const justFinished = prevRunning.current && !isRunning;
    prevRunning.current = isRunning;
    if (justFinished && messages.length > 0) void saveActiveConversation();
  }, [isRunning]); // eslint-disable-line react-hooks/exhaustive-deps

  // Incremental autosave DURING a run (debounced) — a mid-run crash then loses nothing.
  useEffect(() => {
    if (!isRunning || messages.length === 0) return;
    const t = setTimeout(() => { void saveActiveConversation(); }, 1500);
    return () => clearTimeout(t);
  }, [messages, isRunning, saveActiveConversation]);

  // Edit/regenerate: truncate the wire back to a user message's checkpoint, drop the display from
  // that message on, and re-send. `truncate_conversation` keeps full-fidelity history before the
  // point (real tool calls/results), which counting display messages could not.
  const rerunFromUserMessage = async (userIndex: number, text: string) => {
    if (isRunningRef.current) return;
    const msgs = messagesRef.current;
    const target = msgs[userIndex];
    if (!target || target.role !== "user" || target.wireBase == null) return;
    await invoke("truncate_conversation", { args: { len: target.wireBase } }).catch(() => {});
    setMessages(msgs.slice(0, userIndex));   // drop the old turn(s); send() re-appends the user msg
    // Let the truncated state settle before send() reads conversation_len for the new stamp.
    await new Promise(r => setTimeout(r, 0));
    await sendRef.current?.(text);
  };

  const handleRegenerate = async (assistantId: string) => {
    const msgs = messagesRef.current;
    const aIdx = msgs.findIndex(m => m.id === assistantId);
    if (aIdx < 0) return;
    // The user message that prompted this response — the last user message before it.
    let uIdx = -1;
    for (let i = aIdx - 1; i >= 0; i--) { if (msgs[i].role === "user") { uIdx = i; break; } }
    if (uIdx < 0) return;
    await rerunFromUserMessage(uIdx, msgs[uIdx].text);
  };

  const handleEditUserMessage = async (userId: string, newText: string) => {
    const msgs = messagesRef.current;
    const uIdx = msgs.findIndex(m => m.id === userId);
    if (uIdx < 0 || !newText.trim()) return;
    await rerunFromUserMessage(uIdx, newText.trim());
  };

  const handleSelectConversation = async (id: string) => {
    if (isRunning) return;
    try {
      const loaded = await invoke<{
        display: ChatMessage[]; attachments: string[]; missing_attachments: string[];
      }>("load_conversation", { args: { id } });
      setMessages(sanitizeLoadedMessages(Array.isArray(loaded.display) ? loaded.display : []));
      setActiveConversationId(id);
      // Restore the chat's attachment ledger so its photos stay editable. Rust has already
      // split off any file that is no longer on disk.
      turnDataFilesRef.current = new Map();
      conversationFilesRef.current  = loaded.attachments ?? [];
      missingAttachmentsRef.current = loaded.missing_attachments ?? [];
    } catch { /* conversation missing */ }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await invoke("delete_conversation", { args: { id } });
      if (id === activeConversationId) {
        await invoke("reset_conversation");
        setActiveConversationId(null);
        setMessages([]);
        turnDataFilesRef.current = new Map();
        conversationFilesRef.current = [];
        missingAttachmentsRef.current = [];
      }
      refreshConversations();
    } catch { /* ignore */ }
  };

  const handleRenameConversation = async (id: string, title: string) => {
    try {
      await invoke("rename_conversation", { args: { id, title } });
      refreshConversations();
    } catch { /* ignore */ }
  };

  const handlePinConversation = async (id: string, pinned: boolean) => {
    try {
      await invoke("set_conversation_pinned", { args: { id, pinned } });
      refreshConversations();
    } catch { /* ignore */ }
  };

  const handleSetFolder = async (id: string, folder: string | null) => {
    try {
      await invoke("set_conversation_folder", { args: { id, folder } });
      refreshConversations();
    } catch { /* ignore */ }
  };

  // Listen to agent events from Rust
  useEffect(() => {
    const cleanup: Array<() => void> = [];

    // Pre-load the Python runtime so the first run_python (or a scheduled job) isn't cold.
    warmPyodide();
    // Run the artifact script probe once at startup so /dev/state can report it without waiting
    // for an artifact to be rendered.
    probeArtifactScripts().then(ok => { artifactProbeResultRef.current = ok; });
    // Leave the frame-mechanism findings on disk — the shipped app has no dev-control server, and
    // this can only be measured inside the real webview.
    // Deferred: at startup this competes with Pyodide warm-up and history loading, and a starved
    // main thread makes every variant look like it failed.
    const diagTimer = setTimeout(() => {
      Promise.all([probeFrameVariants(), fetchRuntimeCsp()]).then(([results, csp]) => {
        artifactFrameProbesRef.current = results;
        invoke("write_diagnostics", {
          name: "artifact-frame-probe.json",
          content: JSON.stringify(
            { at: new Date().toISOString(), ua: navigator.userAgent, csp, results }, null, 2),
        }).catch(() => {});
      });
    }, 4000);
    cleanup.push(() => clearTimeout(diagTimer));

    listen<{ delta: string }>("agent-token", e => {
      if (!streamActive()) return;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last.streaming) {
          return [...prev.slice(0, -1), { ...last, text: last.text + e.payload.delta, status: undefined }];
        }
        return [...prev, { id: uid(), role: "assistant", text: e.payload.delta, streaming: true }];
      });
    }).then(u => cleanup.push(u));

    // Phase label for the otherwise-silent stretches (tool selection, prompt eval) — shown next
    // to the thinking dots so a working run never looks hung.
    listen<{ phase: string }>("agent-status", e => {
      if (!streamActive()) return;
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last.streaming) {
          return [...prev.slice(0, -1), { ...last, status: e.payload.phase }];
        }
        return [...prev, { id: uid(), role: "assistant", text: "", streaming: true, status: e.payload.phase }];
      });
    }).then(u => cleanup.push(u));

    listen<{ name: string; args: string }>("agent-tool-call", e => {
      if (!streamActive()) return;
      turnTallyRef.current.tools[e.payload.name] = (turnTallyRef.current.tools[e.payload.name] ?? 0) + 1;
      const call = { name: e.payload.name, args: e.payload.args, startedAt: Date.now() };
      setMessages(prev => {
        const updated = prev.map(m =>
          m.role === "assistant" && m.streaming
            ? { ...m, toolCalls: [...(m.toolCalls ?? []), call] }
            : m
        );
        const hasStreaming = prev.some(m => m.role === "assistant" && m.streaming);
        if (!hasStreaming) {
          return [...updated, { id: uid(), role: "assistant", text: "", streaming: true, toolCalls: [call] }];
        }
        return updated;
      });
    }).then(u => cleanup.push(u));

    listen<{ name: string; result: string; full_result?: string; full_truncated?: boolean; ui?: ToolUi; images?: string[]; artifact?: { title: string; html: string } }>("agent-tool-result", e => {
      if (!streamActive()) return;
      turnTallyRef.current.images += e.payload.images?.length ?? 0;
      setThinkingAt(Date.now()); // tool finished — the model now thinks for the next step
      setMessages(prev => {
        const now = Date.now();
        // Locate the most recent still-running call of this name (dispatch is sequential, so there's
        // exactly one) to stamp its final duration and read its args for the result row.
        let ti = -1, tj = -1;
        outer: for (let mi = prev.length - 1; mi >= 0; mi--) {
          const tcs = prev[mi].toolCalls;
          if (!tcs) continue;
          for (let ci = tcs.length - 1; ci >= 0; ci--) {
            if (tcs[ci].name === e.payload.name && tcs[ci].startedAt != null && tcs[ci].durationMs == null) {
              ti = mi; tj = ci; break outer;
            }
          }
        }
        const matchingCall = ti >= 0 ? prev[ti].toolCalls![tj] : undefined;
        const closed = prev.map((m, mi) => {
          let mm = m;
          if (mi === ti) {
            const toolCalls = m.toolCalls!.map((tc, ci) =>
              ci === tj ? { ...tc, durationMs: now - (tc.startedAt ?? now) } : tc);
            mm = { ...mm, toolCalls };
          }
          return mm.streaming ? { ...mm, streaming: false } : mm;
        });
        // Resolve {{figure:N}} tokens (charts/images generated this turn) and {{upload:N}} tokens
        // (images the user ATTACHED this turn — e.g. a logo) in a model artifact.
        let artifact = e.payload.artifact;
        if (artifact?.html && artifact.html.includes("{{figure:")) {
          const figs = collectTurnFigures(prev);
          artifact = { ...artifact, html: substituteFigures(artifact.html, figs, false).out };
        }
        if (artifact?.html && artifact.html.includes("{{upload:")) {
          artifact = { ...artifact, html: substituteUploads(artifact.html, collectTurnUploads(prev)) };
        }
        // {{data:name}} — bulk text run_python staged to /work/artifacts this turn.
        if (artifact?.html && artifact.html.includes("{{data:")) {
          artifact = { ...artifact, html: substituteData(artifact.html, turnDataFilesRef.current) };
        }
        return [...closed, {
          id: uid(), role: "tool-result",
          text: e.payload.result,
          fullResult: e.payload.full_result || undefined,
          fullTruncated: e.payload.full_truncated,
          toolName: e.payload.name,
          toolArgs: matchingCall?.args,
          ui: e.payload.ui,
          toolImages: e.payload.images,
          artifact,
        }];
      });
    }).then(u => cleanup.push(u));

    // The step is being re-sampled: discard the partial text the failed attempt streamed,
    // otherwise the retry's tokens append to it.
    listen<{ step: number; attempt: number; error: string }>("agent-retry", e => {
      if (!streamActive()) return;
      setMessages(prev => {
        let next = prev;
        const last = next[next.length - 1];
        if (last?.role === "assistant" && last.streaming) next = next.slice(0, -1);
        // Tool-use fallback: tell the user the model can't use tools (fired once per run).
        if (e.payload.error?.includes("support tool use")) {
          next = [...next, { id: uid(), role: "notice",
            text: "This model doesn't support tools — continuing without them (no file access, web search, or other tools)." }];
        }
        return next;
      });
    }).then(u => cleanup.push(u));

    listen<{ error: string | null }>("agent-done", e => {
      // A completing run must ALWAYS release the running state, even if its stream was superseded —
      // otherwise a dropped agent-done strands isRunning=true and every later send silently no-ops
      // (send() and edit/regenerate both bail while running). Only the message mutations below are
      // gated by streamActive, so a superseded run's output can't land in the now-visible chat.
      setIsRunning(false);
      setThinkingAt(null);
      if (!streamActive()) return;
      // Record this turn's usage locally (backend merges in token counts). Best-effort, on-device.
      const t = turnTallyRef.current;
      if (t.start) {
        const steps = Object.values(t.tools).reduce((a, b) => a + b, 0);
        invoke("record_turn_usage", { args: {
          ts: Math.floor(Date.now() / 1000),
          model: t.model, provider: t.provider, profile: t.profile,
          duration_ms: Date.now() - t.start, steps, tools: t.tools, images: t.images,
          code_runs: t.tools["run_python"] ?? 0, error: !!e.payload.error,
        } }).catch(() => {});
        t.start = 0; // prevent a double-write
      }
      setMessages(prev => {
        const closed = finalizeCallTimers(prev).map(m => m.streaming ? { ...m, streaming: false } : m);
        if (e.payload.error) return [...closed, { id: uid(), role: "error", text: e.payload.error }];
        return closed;
      });
    }).then(u => cleanup.push(u));

    listen<JobRun>("job-run-done", () => {
      // Only badge if not already viewing the jobs panel
      setView(v => { if (v !== "jobs") setJobBadge(prev => prev + 1); return v; });
    }).then(u => cleanup.push(u));

    // Code-execution permission request from the run_python sandbox.
    listen<{ code: string }>("agent-permission-request", e => {
      // Dev control auto-approves so headless runs aren't blocked on the modal.
      if (autoApproveControlRef.current) {
        invoke("respond_code_permission", { approved: true }).catch(() => {});
        return;
      }
      setPermissionRequest({ code: e.payload.code });
    }).then(u => cleanup.push(u));

    // run_python execution: the backend hands us code + staged files; run them in the Pyodide
    // worker (WASM CPython in the webview) and send the result back.
    listen<{ request_id: number; code: string; files: PyFile[]; reset?: boolean }>("run-python-request", async e => {
      const res = await runPython(e.payload.code, e.payload.files ?? [], e.payload.reset !== false);
      // Anything staged to /work/artifacts stays HERE, keyed by name, for {{data:name}} in a later
      // create_artifact call this turn. Only the names go on to the backend (and so to the model) —
      // shipping the content would reintroduce exactly the context bloat this avoids.
      for (const f of res.dataFiles ?? []) turnDataFilesRef.current.set(f.name, f);
      await invoke("respond_python_result", { args: {
        request_id: e.payload.request_id,
        output: res.output, error: res.error, images: res.images, out_files: res.outFiles,
        // text is persisted to disk by the backend so the file is re-staged into /work/artifacts on
        // later turns; the model is still only told the name and size.
        data_files: (res.dataFiles ?? []).map(f => ({ name: f.name, chars: f.text.length, error: f.error ?? null, text: f.text })),
      } }).catch(() => {});
    }).then(u => cleanup.push(u));

    // Dev control (debug builds): the /dev/run HTTP endpoint drives a real chat turn headlessly.
    // Runs against the CURRENT active profile/settings, auto-approves the code-exec prompt, then
    // reports back a structured trace of the new messages. Config switching is a later phase.
    // Snapshot of the app's current config, for GET /dev/state and the config reply.
    const currentState = () => {
      const s = settingsRef.current;
      const ap = s.profiles.find(p => p.id === s.activeProfileId) ?? null;
      const cp = chatParamsRef.current;
      return {
        activeProfile: ap?.name ?? null,
        activeProfileId: s.activeProfileId,
        model: selectedModelRef.current,
        reasoning: cp?.reasoning ?? "auto",
        numCtx: cp?.numCtx ?? null,
        allowCodeTools: !!ap?.allowCodeTools || forceAllowCodeToolsRef.current,
        // Webview environment, for diagnosing artifacts that render as blank/static: the CSP that
        // is really in force, and whether an artifact frame can execute scripts under it.
        csp: runtimeCsp(),
        artifactScriptsRun: artifactProbeResultRef.current,
        artifactFrameProbes: artifactFrameProbesRef.current,
        profiles: s.profiles.map(p => ({
          id: p.id, name: p.name, model: p.model,
          allowCodeTools: !!p.allowCodeTools, maxTools: p.maxTools,
        })),
      };
    };

    listen<{ id: number }>("dev-control-state", e => {
      invoke("dev_control_report", { args: { id: e.payload.id, trace: currentState() } }).catch(() => {});
    }).then(u => cleanup.push(u));

    listen<{ id: number; params: { profile?: string; reasoning?: "on" | "off" | "auto"; numCtx?: number; model?: string; allowCodeTools?: boolean } }>("dev-control-config", async e => {
      const { id, params } = e.payload;
      if (params.profile) {
        const s = settingsRef.current;
        const target = s.profiles.find(p => p.name === params.profile || p.id === params.profile);
        if (target) await profileSwitchRef.current?.(target.id);
      }
      if (params.reasoning) setChatParams(p => ({ ...p, reasoning: params.reasoning === "auto" ? undefined : params.reasoning }));
      if (typeof params.numCtx === "number") setChatParams(p => ({ ...p, numCtx: params.numCtx }));
      if (params.model) setSelectedModel(String(params.model));
      if (typeof params.allowCodeTools === "boolean") forceAllowCodeToolsRef.current = params.allowCodeTools;
      await new Promise(r => setTimeout(r, 350)); // let profile switch / state settle
      invoke("dev_control_report", { args: { id, trace: currentState() } }).catch(() => {});
    }).then(u => cleanup.push(u));

    listen<{ id: number; params: { message?: string; reasoning?: "on" | "off" | "auto"; numCtx?: number; model?: string; allowCodeTools?: boolean; fresh?: boolean } }>("dev-control-run", async e => {
      const { id, params } = e.payload;
      const message = String(params?.message ?? "");
      const report = (trace: unknown) => invoke("dev_control_report", { args: { id, trace } }).catch(() => {});
      if (!message.trim()) { report({ error: "empty message" }); return; }
      // Fresh conversation per run by default so tests aren't contaminated by prior history
      // (reset first — it also restores chatParams, which the overrides below then re-apply).
      if (params.fresh !== false) { await handleResetRef.current?.(); await new Promise(r => setTimeout(r, 100)); }
      // Optional per-run setting overrides so an external driver can A/B speed vs. quality.
      if (params.reasoning) setChatParams(p => ({ ...p, reasoning: params.reasoning === "auto" ? undefined : params.reasoning }));
      if (typeof params.numCtx === "number") setChatParams(p => ({ ...p, numCtx: params.numCtx }));
      if (params.model) setSelectedModel(String(params.model));
      if (typeof params.allowCodeTools === "boolean") forceAllowCodeToolsRef.current = params.allowCodeTools;
      if (params.reasoning || params.numCtx != null || params.model) await new Promise(r => setTimeout(r, 250)); // let state + sendRef settle
      // send() silently returns if a run is already in flight, which used to leave this listener
      // waiting on nothing and the HTTP caller timing out after 900s with a misleading "is the
      // window open?". Wait briefly for the previous run to settle, then refuse clearly.
      if (isRunningRef.current) {
        const settleBy = Date.now() + 30_000;
        while (isRunningRef.current && Date.now() < settleBy) await new Promise(r => setTimeout(r, 200));
        if (isRunningRef.current) {
          report({ error: "busy — a run is already in flight; retry when it finishes" });
          return;
        }
      }
      drainCodeToolCalls(); // clear any stale code-tool log
      // Anchor the trace to the LAST MESSAGE ID rather than an array length. A length snapshot
      // silently truncated a trace to one message once, which read as "the model called no tools"
      // when it had in fact run five — an index is not a stable identity if the list is rebuilt.
      const anchorId = messagesRef.current[messagesRef.current.length - 1]?.id ?? null;
      const sliceFromAnchor = () => {
        const all = messagesRef.current;
        if (anchorId === null) return all;
        const i = all.findIndex(m => m.id === anchorId);
        return i === -1 ? all : all.slice(i + 1);   // anchor gone → conversation reset, take it all
      };
      autoApproveControlRef.current = true;
      const t0 = performance.now();
      // Phase timings: where a turn's wall time actually goes. The gap before the first step, and
      // the gap between a step starting and its first token (prompt evaluation), are invisible in
      // the UI and are usually the bulk of a slow turn.
      const marks: { at: number; what: string }[] = [];
      const mark = (what: string) => marks.push({ at: Math.round(performance.now() - t0), what });
      const phaseUnsubs: Array<() => void> = [];
      let sawToken = false;
      (await Promise.all([
        listen<{ step: number }>("debug-step-start", e => mark(`step ${e.payload.step} start`)),
        // The status label is transient in the UI (cleared on the first token), so record what the
        // user was actually shown while waiting.
        listen<{ phase: string }>("agent-status", e => mark(`status: ${e.payload.phase}`)),
        listen<{ total: number; system_tokens: number; tools_tokens: number; history_tokens: number;
                 schemas: { label: string; tokens: number }[] }>("debug-step-context", e => {
          const p = e.payload;
          const top = p.schemas.slice(0, 3).map(x => `${x.label} ${x.tokens}`).join(", ");
          mark(`context ${p.total} = system ${p.system_tokens} + schemas ${p.tools_tokens} + history ${p.history_tokens} | top: ${top}`);
        }),
        listen("agent-token", () => { if (!sawToken) { sawToken = true; mark("first token"); } }),
        listen<{ name: string }>("agent-tool-call", e => mark(`tool call: ${e.payload.name}`)),
        listen<{ name: string }>("agent-tool-result", e => { mark(`tool result: ${e.payload.name}`); sawToken = false; }),
        listen<{ step: number; duration_ms: number }>("debug-step-done",
          e => mark(`step ${e.payload.step} done (${e.payload.duration_ms}ms)`)),
      ])).forEach(u => phaseUnsubs.push(u));
      try {
        // NOTE: send() awaits invoke("send_message"), which runs the ENTIRE agent loop — it does
        // not return early. The settle loop below is therefore only a short backstop for isRunning
        // lagging the await; it is not what bounds the run. The real bound is the dev_await
        // timeout on the Rust side (DEV_RUN_TIMEOUT_SECS).
        await sendRef.current?.(message);
        await new Promise(r => setTimeout(r, 250)); // grace for isRunning → true
        const deadline = Date.now() + 3_500_000;
        while (isRunningRef.current && Date.now() < deadline) {
          await new Promise(r => setTimeout(r, 150));
        }
      } catch (err) {
        phaseUnsubs.forEach(u => u());   // otherwise a failed run leaks its phase listeners
        report({ error: String(err), elapsedMs: Math.round(performance.now() - t0), phases: marks });
        autoApproveControlRef.current = false; return;
      }
      autoApproveControlRef.current = false;
      const elapsedMs = Math.round(performance.now() - t0);
      phaseUnsubs.forEach(u => u());
      mark("run done");
      const codeToolCalls = drainCodeToolCalls();
      const captured = sliceFromAnchor();
      const trace = captured.map(m => ({
        role: m.role,
        text: m.text || undefined,
        toolCalls: m.toolCalls?.map(tc => ({ name: tc.name, args: tc.args })),
        toolName: m.toolName,
        toolResult: m.role === "tool-result" ? m.text : undefined,
        images: (m.toolImages?.length ?? m.imageDataUrls?.length) || undefined,
        ui: m.ui ? { server_id: m.ui.server_id, hasHtml: !!m.ui.html } : undefined,
        artifact: m.artifact ? { title: m.artifact.title, htmlLen: m.artifact.html.length } : undefined,
        status: m.status,
      }));
      const finalAnswer = [...captured].reverse()
        .find(m => m.role === "assistant" && !!m.text)?.text;
      // totalMessages lets a caller spot a suspiciously thin trace instead of reading it as
      // "nothing happened" — the mistake that made a five-tool run look like zero.
      report({ finalAnswer, elapsedMs, codeToolCalls, messages: trace, phases: marks,
               totalMessages: messagesRef.current.length, capturedMessages: captured.length });
    }).then(u => cleanup.push(u));

    // run_python produced output files but no sandbox folder is configured to save them. Ask the
    // user to pick a folder — it's added to the sandbox and the stashed files are written there.
    // We never write outside the sandbox.
    // run_python produced output file(s) but no sandbox folder is set. Show a NON-blocking notice
    // with a Save button (never a modal picker mid-chat); the files stay stashed until saved.
    listen<{ files: string[] }>("sandbox-save-request", e => {
      const names = e.payload.files ?? [];
      if (names.length === 0) return;
      setMessages(prev => [...prev, { id: uid(), role: "notice",
        text: `${names.length} output file(s) ready (${names.join(", ")}) — no sandbox folder set.`,
        savePrompt: names }]);
    }).then(u => cleanup.push(u));

    // Persist refreshed OAuth2 access tokens so they survive restarts.
    // Covers both OpenAPI specs and MCP servers across global settings and all profiles.
    listen<{ spec_id: string; access_token: string }>("openapi-token-refreshed", e => {
      const { spec_id, access_token } = e.payload;
      setSettings(prev => {
        const patchAuth = (auth: import("./AdminPanel").AuthConfig) =>
          ({ ...auth, access_token });
        const updated: typeof prev = {
          ...prev,
          toolRegistry: {
            mcpServers: prev.toolRegistry.mcpServers.map(s =>
              s.id === spec_id ? { ...s, auth: patchAuth(s.auth ?? { type: "none" as const }) } : s),
            openapiSpecs: prev.toolRegistry.openapiSpecs.map(s =>
              s.id === spec_id ? { ...s, auth: patchAuth(s.auth ?? { type: "none" as const }) } : s),
            sparqlEndpoints: prev.toolRegistry.sparqlEndpoints.map(s =>
              s.id === spec_id ? { ...s, auth: patchAuth(s.auth ?? { type: "none" as const }) } : s),
          },
          // Also patch profile-level auth overrides that reference this tool
          profiles: prev.profiles.map(p => {
            if (!p.toolAuthOverrides?.[spec_id]) return p;
            return { ...p, toolAuthOverrides: { ...p.toolAuthOverrides, [spec_id]: patchAuth(p.toolAuthOverrides[spec_id]) } };
          }),
        };
        saveSettings(updated);
        // Re-sync AppState so the Admin panel's tool dropdowns reflect the
        // updated token immediately — without this, state.openapi_specs can
        // get out of sync when a job refreshes a token while a different
        // profile is loaded in the main chat.
        syncServers(updated).catch(() => {});
        return updated;
      });

      // Also patch any scheduled jobs whose profile_context contains the refreshed spec/server
      invoke<import("./jobTypes").ScheduledJob[]>("get_jobs").then(jobs => {
        type JS = import("./jobTypes").JobOpenAPISpec;
        type JM = import("./jobTypes").JobMCPServer;
        const affected = jobs.flatMap(job => {
          if (!job.profile_context) return [];
          const inSpec = job.profile_context.openapi_specs.some((s: JS) => s.id === spec_id);
          const inMcp  = job.profile_context.mcp_servers.some((s: JM) => s.id === spec_id);
          if (!inSpec && !inMcp) return [];
          const patchedCtx = {
            ...job.profile_context,
            openapi_specs: job.profile_context.openapi_specs.map((s: JS) =>
              s.id === spec_id ? { ...s, auth: { ...(s.auth ?? { type: "none" as const }), access_token } } : s),
            mcp_servers: job.profile_context.mcp_servers.map((s: JM) =>
              s.id === spec_id ? { ...s, auth: { ...(s.auth ?? { type: "none" as const }), access_token } } : s),
          };
          return [{ ...job, profile_context: patchedCtx }];
        });
        affected.forEach(job => invoke("save_job", { job }).catch(() => {}));
      }).catch(() => {});
    }).then(u => cleanup.push(u));

    return () => cleanup.forEach(u => u());
  }, []);

  const handleAttach = async () => {
    const selected = await open({ multiple: true, title: "Attach files" }).catch(() => null);
    if (!selected) return;
    const files = Array.isArray(selected) ? selected : [selected];
    setAttachedFiles(prev => [...prev, ...files.filter(f => !prev.includes(f))]);
  };

  const isImage = (p: string) => IMAGE_EXTS_SET.has(p.split(".").pop()?.toLowerCase() ?? "");

  const send = async (text: string) => {
    text = text.trim();
    if ((!text && attachedFiles.length === 0) || isRunning || !selectedModel) return;


    // Profile overrides global settings; chatParams toggles can further restrict
    const effectiveEnabledTools = activeProfile?.enabledTools ?? settings.enabledTools;
    // run_python (code execution) is gated by a GLOBAL master switch — a security
    // capability that defaults off. Once the master is on it behaves like any other
    // tool: enabled unless a profile explicitly opts out.
    const runPythonMaster = settings.enabledTools.run_python === true;
    // generate_image is likewise gated by a GLOBAL master switch (it needs a local sd.cpp binary +
    // model configured), defaulting off. Once on, it behaves like any other tool per profile.
    const imageGenMaster = settings.enabledTools.generate_image === true;
    const enabledTools = ALL_BUILTIN_TOOLS.filter(t => {
      const name = t.function.name;
      if (!activeProfile) {
        // No active profile → conservative read-only default. Exception: run_python is allowed
        // when its global master switch is explicitly on (an opt-in security capability, still
        // gated by the per-run permission prompt) — so code execution doesn't need a profile.
        if (name === "run_python") return runPythonMaster;
        if (name === "generate_image") return imageGenMaster;
        return SAFE_DEFAULT_BUILTINS.has(name) && settings.enabledTools[name] !== false;
      }
      if (name === "run_python") return runPythonMaster && effectiveEnabledTools.run_python !== false;
      if (name === "generate_image") return imageGenMaster && effectiveEnabledTools.generate_image !== false;
      return effectiveEnabledTools[name] !== false;
    });
    // Wiki memory: the active profile can override the global default; an unset profile
    // (undefined) inherits it.
    const wikiEnabled = (activeProfile?.wikiEnabled ?? settings.wikiEnabled) === true;
    if (wikiEnabled) enabledTools.push(...WIKI_TOOLS);

    // Split attachments into images (sent via Ollama images field) and other files (appended as paths)
    const imagePaths = attachedFiles.filter(isImage);
    const otherFiles = attachedFiles.filter(f => !isImage(f));
    // Edit masks aligned to imagePaths ("" = none).
    const imageMasksAligned = imagePaths.map(p => imageMasks[p] ?? "");

    const filesNote = otherFiles.length > 0
      ? `\n\nThe user has attached the following local file(s). Use the read_file tool to read them directly (it extracts text from PDF, Word, and plain text automatically). This IS the document the user is referring to — do NOT search or fetch the web for it:\n${otherFiles.map(f => `- ${f}`).join("\n")}`
      : "";

    // Tell the model the sandbox path of each attached image so it can EDIT the real file instead of
    // regenerating the scene. Without this the model only sees the pixels (via vision) and doesn't
    // know the filename — it then says "I can't access the file" and reimagines from scratch.
    const imageNote = imagePaths.length > 0
      ? `\n\nThe user attached the image(s) shown to you, available in the sandbox at:\n${imagePaths.map((p, i) => {
          const uploaded = `/work/uploads/${p.split("/").pop()}`;
          return `- attachment #${i + 1}: ${uploaded} — reference it as {{upload:${i + 1}}} in create_artifact HTML (e.g. a logo/photo on slides), or use this path in run_python / python-pptx add_picture${imageMasks[p] ? " (the user PAINTED the exact region to change — call generate_image with this as source_image and a prompt describing the change; the painted mask is applied automatically, so DO NOT set mask_regions)" : ""}`;
        }).join("\n")}\nTo PLACE an attached image in a create_artifact deck/page use its {{upload:N}} token (do NOT try to inline its file path or base64 — that won't work). To EDIT an attached image (recolour, restyle, or change part of it) call generate_image with source_image set to its path above (add mask_regions to change only part of it), or open it in run_python with Pillow for exact pixel edits. Do NOT say you cannot access the file and do NOT regenerate the scene from scratch when the user wants THIS image changed.`
      : "";

    // Files from earlier turns are still on disk and still in the sandbox allow-list, so
    // they remain editable. Say so explicitly — otherwise the model reports the photo as
    // "not attached" and regenerates it from its own description.
    const priorFiles  = conversationFilesRef.current.filter(p => !attachedFiles.includes(p));
    const priorImages = priorFiles.filter(isImage);
    const priorOther  = priorFiles.filter(f => !isImage(f));
    const priorNote = priorFiles.length > 0
      ? `\n\nAlso still available from EARLIER in this conversation (attached to a previous message, and still editable/readable now):${
          priorImages.length > 0
            ? `\nImages — edit with generate_image using source_image, or open in run_python with Pillow:\n${
                priorImages.map(p => `- /work/uploads/${p.split("/").pop()}`).join("\n")}`
            : ""
        }${
          priorOther.length > 0
            ? `\nFiles — read with read_file:\n${priorOther.map(f => `- ${f}`).join("\n")}`
            : ""
        }\nDo NOT say these are no longer attached, and do NOT regenerate an earlier image from your description when the user wants THAT image changed.`
      : "";

    // A reopened chat stores paths, not bytes, so a file may have been moved or deleted since.
    // Name them: the right answer is to tell the user, not to reinvent the file.
    const missingNote = missingAttachmentsRef.current.length > 0
      ? `\n\nAttached earlier in this conversation but NO LONGER on disk (the user moved or deleted them):\n${
          missingAttachmentsRef.current.map(f => `- ${f}`).join("\n")}\nIf the user asks about one of these, say the file is no longer where it was and ask them to re-attach it. Do NOT invent its contents and do NOT regenerate an image to stand in for it.`
      : "";

    const fullText = `${text}${filesNote}${imageNote}${priorNote}${missingNote}`;

    // Build display text — only list non-image attachments (images shown as thumbnails)
    const displayText = otherFiles.length > 0
      ? `${text}\n\nAttached: ${otherFiles.map(f => f.split("/").pop()).join(", ")}`
      : text;

    // Load image data URIs for thumbnails (fire-and-forget before send)
    const imageDataUrls = await Promise.all(
      imagePaths.map(p => invoke<string>("read_image_data_url", { path: p }).catch(() => ""))
    ).then(urls => urls.filter(Boolean));

    // Stamp the pre-send wire length as this message's edit/regenerate anchor.
    const wireBase = await invoke<number>("conversation_len").catch(() => undefined);
    setMessages(prev => [...prev, { id: uid(), role: "user", text: displayText, imageDataUrls, wireBase }]);
    conversationFilesRef.current = [
      ...conversationFilesRef.current,
      ...attachedFiles.filter(f => !conversationFilesRef.current.includes(f)),
    ];
    setAttachedFiles([]);
    setImageMasks({});
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    // This run now owns the event stream (see streamEpoch/streamOwner).
    streamEpoch.current += 1;
    streamOwner.current = streamEpoch.current;
    setIsRunning(true);
    setThinkingAt(Date.now());

    try {
      const allowedDirs = await invoke<string[]>("get_allowed_dirs").catch(() => [] as string[]);
      const basePrompt = activeProfile?.systemPrompt ?? BASE_SYSTEM_PROMPT;

      // Build dynamic suffix describing any registered external tools
      const registry = settings.toolRegistry;
      // With no active profile, no integrations are enabled, so the prompt must not advertise them
      // (otherwise the model would try to call tools the backend doesn't have).
      const ctxOpenAPI = activeProfile
        ? registry.openapiSpecs.filter(s => activeProfile.enabledOpenapiSpecIds.includes(s.id) && s.enabled !== false)
        : [];
      const ctxMCP = activeProfile
        ? registry.mcpServers.filter(s => activeProfile.enabledMcpServerIds.includes(s.id))
        : [];
      const ctxSparql = activeProfile
        ? registry.sparqlEndpoints.filter(s => (activeProfile.enabledSparqlEndpointIds ?? []).includes(s.id) && s.enabled !== false)
        : [];
      const externalParts: string[] = [];
      if (ctxOpenAPI.length > 0)
        externalParts.push(`OpenAPI services you can call: ${ctxOpenAPI.map(s => s.title).join(", ")}.`);
      if (ctxSparql.length > 0) {
        const sparqlList = ctxSparql.map(s =>
          s.usage_hint?.trim() ? `${s.title} (best for: ${s.usage_hint.trim()})` : s.title
        ).join("; ");
        externalParts.push(`Connected SPARQL / linked-data endpoints — ${sparqlList}. To use one, call its "…_query" tool with a SPARQL query (call its "…_schema" tool first if unsure of the vocabulary). These return authoritative structured data — prefer them over web_search when the question matches their topic.`);
      }
      if (ctxMCP.length > 0)
        externalParts.push(`MCP servers connected: ${ctxMCP.map(s => s.name).join(", ")}.`);
      const externalSuffix = externalParts.length > 0
        ? `\nTOOL ROUTING: connected data tools are available — strongly prefer them over web_search whenever the user's request matches their topic, and only fall back to web_search for general open-web information they do not cover. When the user's question is about a topic a connected tool covers (e.g. crime/safety, house prices/property, planning, deprivation/demographics, health/care ratings), call the tool(s) that match WHAT THE USER ACTUALLY ASKED and answer from their real returned data — do not answer a data question from general knowledge, and do not substitute a map or geocode for the data. But stay on topic: only call the tools relevant to the question — do NOT pull in unrelated data tools just because they exist, and if a tool keeps failing, stop and answer with what you have rather than retrying it many times. ${externalParts.join(" ")}`
        : "";

      const resolved = resolveParams(chatParams);
      // Tool-specific guidance, each part gated on that tool family actually being live this
      // turn. Appended for every profile — a custom profile prompt shouldn't have to restate
      // these, and shouldn't be able to promise a tool the model wasn't given.
      const hasTool = (n: string) => enabledTools.some(t => t.function.name === n);
      const fileToolsLive = ["list_files", "read_file", "list_directory_tree", "search_files"].some(hasTool);
      const webToolsLive  = ["web_search", "fetch_webpage"].some(hasTool);
      const toolGuidanceSuffix = [
        fileToolsLive
          ? "\n\nLOCAL FILES: when asked about files or folders on this computer, call the file tools straight away using any path the user gave. Always use full absolute paths — never '.' or '~'."
          : "",
        webToolsLive
          ? "\n\nWEB: use web_search for current events, weather or live data, and fetch_webpage to read a specific URL (including one from a web_search result). Fetching a public page for this authorised user is permitted — do not decline on copyright grounds or claim you can only summarise."
          : "",
        hasTool("read_file")
          ? "\n\nATTACHED IMAGES: never call read_file on an image file — attached images are already provided to you visually, so describe what you see. If the user refers to an image and none is attached, ask them to attach it with the paperclip button."
          : "",
        externalParts.length > 0
          ? "\n\nTOOL ROUTING: external API tools (OpenAPI / MCP) are only for requests that name that service or the data it holds. Anything about files on this computer uses the local file tools."
          : "",
      ].join("");

      const effectiveBase = resolved.systemPromptOverride ?? basePrompt;

      // Latency-aware defaults for a LIGHTWEIGHT profile — one with no connected data tools
      // (OpenAPI/SPARQL/MCP), no code-mode, and run_python off. Such a profile does short,
      // conversational turns that can't accumulate large tool results, so two costly defaults are
      // pure overhead: (1) the model's pre-answer reasoning pass (thinking ON is qwen3's default and
      // added ~4.6s even to "17×23"), and (2) the 32K agentic context floor (needed only so a long
      // tool-heavy run isn't front-truncated — a documented failure at 16K). We only FILL these when
      // the user left them unset; an explicit per-profile reasoning/num_ctx/Extended-context choice
      // always wins. The turn-budget caps (web-tool cap 4, global cap 15) bound any accumulation a
      // light profile's web tools could still cause, keeping the smaller context safe.
      const lightweightProfile = externalParts.length === 0
        && !(activeProfile?.allowCodeTools || forceAllowCodeToolsRef.current)
        && !runPythonMaster;
      const effectiveThink = resolved.think ?? (lightweightProfile ? false : null);
      const effectiveNumCtx = (lightweightProfile && chatParams.numCtx === undefined && chatParams.contextSize === "short")
        ? 16384
        : resolved.numCtx;

      // Build context vars block from the active profile
      const contextVars = activeProfile?.contextVars?.filter(v => v.name.trim() && v.value.trim()) ?? [];
      const contextVarsSuffix = contextVars.length > 0
        ? `\n\nUser context (treat these as facts about the user — use them automatically when relevant, do not repeat them unless asked):\n${contextVars.map(v => `- ${v.name}: ${v.value}`).join("\n")}`
        : "";

      const wikiSuffix = wikiEnabled ? WIKI_SYSTEM_PROMPT_BLOCK : "";

      // Never emit remote image URLs — the CSP blocks them so they render as nothing; tool/chart
      // images are already shown inline. (The model kept appending mapbox/OSM image URLs.)
      const outputRulesSuffix = "\n\nOUTPUT RULES: NEVER write a markdown image or link pointing at a remote http(s):// image URL (a map, chart, tile, etc.) — remote images are blocked and will NOT display. Any map, chart, or image produced by a tool or by run_python is ALREADY shown inline in the chat; just refer to it as \"shown above\". Do not paste image/tile URLs into your answer.";

      // When image generation is enabled, tell the model it CAN make images (so it uses the tool
      // instead of refusing). enabledTools is the already-filtered per-profile list.
      const imageRulesSuffix = enabledTools.some(t => t.function.name === "generate_image")
        ? "\n\nIMAGE GENERATION: you CAN create images. When the user asks you to create, draw, generate, illustrate, paint, or make an image, picture, logo, or artwork, call the generate_image tool with a detailed prompt. The result renders inline automatically — refer to it as \"shown above\". Never claim you are unable to generate or display images.\n\nPHOTOREALISTIC IMAGES: when the user wants a PHOTO or a realistic/photorealistic image of a person, place, object or scene, write a rich PHOTOGRAPHIC prompt — describe the subject in detail and add camera/lighting cues such as \"photograph, photorealistic, 85mm, natural lighting, detailed skin texture, sharp focus, high detail, realistic\" — AND pass a negative_prompt that excludes non-photographic styles, e.g. \"illustration, cartoon, drawing, painting, sketch, 3d render, cgi, anime, plastic, doll, deformed, extra fingers, low quality, blurry\". For a NON-photographic style the user asked for (cartoon, logo, watercolour, pixel art, etc.), prompt for THAT style explicitly and do NOT add the photographic terms."
        : "";

      // The model has no clock — give it today's date so "latest/recent/this month" queries work
      // without needing a tool call, and warn that some data sources lag.
      const now = new Date();
      const dateSuffix = `\n\nTODAY'S DATE is ${now.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" })} (${now.toISOString().slice(0, 10)}). Use this whenever the user asks for the "latest", "recent", "current", "this month/year", "today", etc. — you already know today's date and do not need a tool for it. Be aware some data sources lag behind today (e.g. UK police street-crime data is usually 1–2 months old), so the "latest available" data may be for an earlier month than the current one — request the most recent month the source actually offers.`;

      // Anti-hallucination for statistics: small tool tables (e.g. a single district's census
      // religion breakdown, ~10 rows) stay inline, so the model tends to hand-map the categories
      // and eyeball the percentages — which is where the wrong figures came from. When run_python
      // and data tools are both available, require the arithmetic to be done in code from the tool's
      // own JSON. Only add it when there's actually numeric data work to do.
      const figuresRulesSuffix = (runPythonMaster && externalParts.length > 0)
        ? "\n\nEXACT FIGURES: when a data tool returns numbers (counts, populations, census tables, prices) and you need totals, percentages or a breakdown from them, compute them in run_python — load the tool's JSON, use the category/label fields it returns verbatim (never rename or re-map categories yourself), find the 'Total' row and divide by it for percentages, then report those computed values. Do NOT mentally calculate or round percentages, and never supply a figure from prior knowledge: any number not present in THIS turn's tool output must be reported as \"No data available\", not estimated."
        : "";

      // Street maps with plotted data points: a create_artifact Leaflet + OpenStreetMap map renders
      // real streets AND the markers; geopandas has no street basemap (points on blank). The #1
      // failure is the model plotting placeholder/made-up coordinates instead of the real ones.
      const mapRulesSuffix = runPythonMaster
        ? "\n\nSTREET MAPS WITH DATA POINTS: to show points on a street map (crime locations, incidents, places), create an INTERACTIVE map with create_artifact using Leaflet + OpenStreetMap tiles (the artifact may load OSM tiles and the Leaflet library) and plot the points as markers. CRITICAL: use the REAL lat/lng from the tool result for EVERY point — NEVER placeholder, example, sample, or made-up coordinates. First extract the exact points in run_python from the actual API response and WRITE them to /work/artifacts/points.json (print only a count and a first/last sanity check, never the whole array); then in the map's script write `const points = {{data:points.json}};` — LexiChat splices the exact file in. NEVER retype coordinates into the HTML: it takes minutes and drops points. For a ROUTE, write one entry per leg and draw each leg as its own coloured polyline so no leg is missing, and call fitBounds ONCE over all legs at the end (not inside the loop). COORDINATE ORDER IS CRITICAL: Leaflet takes [lat, lng] — latitude FIRST. Store points that way and pass them straight through (never [c[1], c[0]]); TfL lineString data is already [lat, lng], so do NOT convert it to GeoJSON [lng, lat]. Sanity-check in run_python before writing the file — for the UK every point needs -11 < lng < 3 and 49 < lat < 61. Use L.circleMarker rather than L.marker (the default marker icon is a remote PNG the artifact sandbox blocks on Windows/Linux). For the start/end markers take the first and last COORDINATE, not the first/last leg index: `const start = legs[0].pts[0], end = legs[legs.length-1].pts.slice(-1)[0];` — indexing pts by the leg number yields a bare number and Leaflet then throws, which kills the rest of the script and leaves the map blank. Do NOT use geopandas/matplotlib for a street map — it has no basemap and renders points on a blank background (use it only for boundary/choropleth plots). Build the whole map in ONE create_artifact call. CRITICAL: the map HTML must be passed to the create_artifact TOOL — never write HTML, a <script>, or an <iframe> into your chat message (HTML in your reply shows as raw source, NOT a rendered map), and the artifact HTML holds the Leaflet map DIRECTLY (do not wrap it in an inner <iframe>)."
        : "";

      // Code-mode: when the profile allows code to call tools, tell the model about the Python API.
      const codeToolsSuffix = (activeProfile?.allowCodeTools || forceAllowCodeToolsRef.current)
        ? "\n\nCODE-MODE TOOLS: inside run_python you can call registered tools directly. ALWAYS call `tools = await list_tools()` FIRST to get the EXACT tool names and their `parameters` schema — never guess a tool name or a group label. Each entry is {name, description, parameters}. Then `data = await call_tool(\"exact_tool_name\", {\"arg\": \"value\"})` runs one and returns a dict/list (parsed JSON) or string; build the args from the tool's parameters schema. Both are async — you MUST `await` them. Prefer this for multi-source work: fetch with call_tool, then compute/aggregate/plot with pandas/numpy/matplotlib in the same script, instead of many separate tool-call steps."
        : "";

      const systemPrompt = allowedDirs.length > 0
        ? `${effectiveBase}${toolGuidanceSuffix}${externalSuffix}${contextVarsSuffix}${wikiSuffix}${codeToolsSuffix}${figuresRulesSuffix}${mapRulesSuffix}${outputRulesSuffix}${imageRulesSuffix}${dateSuffix}\nThe user's configured folders are: ${allowedDirs.join(", ")}. Rules for file operations:\n- When reading or listing files without a specified path, use these folders immediately — do not ask for clarification.\n- When writing or saving a file without a specified path, save it to ${allowedDirs[0]} with a sensible filename derived from the content (e.g. sikhism_article.pdf). Never call write_file without a full absolute path.`
        : `${effectiveBase}${toolGuidanceSuffix}${externalSuffix}${contextVarsSuffix}${wikiSuffix}${codeToolsSuffix}${figuresRulesSuffix}${mapRulesSuffix}${outputRulesSuffix}${imageRulesSuffix}${dateSuffix}`;

      // MCP servers this profile may use. With no active profile, none are enabled (conservative
      // default) — a profile must opt in. The backend filters strictly by this list.
      const enabledMcpServerIds = activeProfile
        ? activeProfile.enabledMcpServerIds
        : [];
      const disabledMcpTools = ctxMCP.flatMap(srv =>
        Object.entries(srv.enabledTools ?? {})
          .filter(([, en]) => !en)
          .map(([name]) => name)
      );

      const targetServer = serverForModel(settings.servers ?? [], selectedServerId, selectedModel);
      // Start a fresh usage tally for this turn.
      turnTallyRef.current = {
        start: Date.now(), model: selectedModel,
        provider: targetServer?.provider ?? "ollama", profile: activeProfile?.name ?? "",
        tools: {}, images: 0,
      };
      await invoke("send_message", {
        args: {
          model: selectedModel,
          message: fullText,
          system_prompt: systemPrompt,
          base_url: targetServer?.baseUrl ?? null,
          provider: targetServer?.provider ?? null,
          api_key: targetServer?.apiKey ?? null,
          tools: enabledTools,
          image_paths: imagePaths,
          prior_file_paths: priorFiles,
          image_masks: imageMasksAligned,
          file_paths: otherFiles,
          temperature: resolved.temperature,
          top_p: resolved.topP ?? null,
          top_k: resolved.topK ?? null,
          repeat_penalty: resolved.repeatPenalty ?? null,
          seed: resolved.seed ?? null,
          num_ctx: effectiveNumCtx,
          num_predict: resolved.numPredict,
          stop: resolved.stop ?? null,
          think: effectiveThink,
          keep_alive: resolved.keepAlive ?? null,
          web_search_results: settings.webSearchResults ?? 10,
          max_steps: settings.maxSteps ?? 12,
          disabled_mcp_tools: disabledMcpTools,
          enabled_mcp_server_ids: enabledMcpServerIds,
          // null (no profile / undefined) = all skills; a list = only those the profile enables.
          enabled_skill_ids: activeProfile?.enabledSkillIds ?? null,
          max_tools: (activeProfile?.maxTools ?? settings.maxTools) || null,
          tool_result_limit: activeProfile?.toolResultLimit ?? null,
          web_tool_cap: activeProfile?.webToolCap ?? null,
          allow_code_tools: forceAllowCodeToolsRef.current || (activeProfile?.allowCodeTools ?? false),
          debug_full_context: settings.debugFullContext === true,
        }
      });
    } catch (err) {
      setIsRunning(false);
      // A failed agent run rejects here *and* emits agent-done with the same error — don't
      // render it twice. Errors thrown before the loop starts still surface.
      const text = String(err);
      setMessages(prev => {
        // Close any streaming "Thinking…" bubble so it can't strand as a zombie (no stop
        // button, dots forever). An empty closed assistant message renders nothing.
        const closed = prev
          .map(m => (m.streaming ? { ...m, streaming: false } : m))
          .filter(m => !(m.role === "assistant" && !m.streaming && !m.text && !(m.toolCalls?.length)));
        const last = closed[closed.length - 1];
        if (last?.role === "error" && last.text === text) return closed;
        return [...closed, { id: uid(), role: "error", text }];
      });
    }
  };
  sendRef.current = send; // keep the dev-control listener pointed at the latest send()

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input); }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  };

  const [debugClearKey, setDebugClearKey] = useState(0);

  // Chat-specific generation params — reset to profile/global defaults on new chat
  const defaultChatParams = (): ChatParams =>
    activeProfile?.chatParams ?? settings.chatParams ?? DEFAULT_CHAT_PARAMS;
  const [chatParams, setChatParams] = useState<ChatParams>(defaultChatParams);

  const handleReset = async () => {
    // Halt any in-flight run and supersede its stream so its output can't leak into the
    // fresh chat (also covers profile switches, which call handleReset).
    stopActiveRun();
    await invoke("reset_conversation");
    // Artifact data is conversation-scoped: a new chat must not resolve a {{data:}} token against
    // the previous conversation's dataset.
    turnDataFilesRef.current = new Map();
    conversationFilesRef.current = [];
    missingAttachmentsRef.current = [];
    setMessages([]);
    setActiveConversationId(null);
    setDebugClearKey(k => k + 1);
    setChatParams(defaultChatParams());
  };

  // Sync Rust's runtime state to whichever profile/global context is now active
  const syncServers = async (s: AppSettings) => {
    const profile  = s.profiles.find(p => p.id === s.activeProfileId) ?? null;
    const registry = s.toolRegistry;

    // Always connect ALL registry MCP servers so they're available in Rust's connection pool.
    // Profile filtering is enforced at call time via enabled_mcp_server_ids in send_message.
    // Auth overrides are still applied per-profile.
    let mcp = registry.mcpServers;
    if (profile?.toolAuthOverrides) {
      const ov = profile.toolAuthOverrides;
      mcp = mcp.map(srv => ov[srv.id] ? { ...srv, auth: ov[srv.id] } : srv);
    }

    // OpenAPI specs: still profile-filtered (they connect per-call, no persistent pool)
    let openapi: StoredOpenAPISpec[];
    if (profile) {
      openapi = registry.openapiSpecs.filter(sp => profile.enabledOpenapiSpecIds.includes(sp.id));
      if (profile.toolAuthOverrides) {
        const ov = profile.toolAuthOverrides;
        openapi = openapi.map(sp => ov[sp.id] ? { ...sp, auth: ov[sp.id] } : sp);
      }
    } else {
      openapi = []; // no profile → no registered APIs (conservative default)
    }

    // SPARQL endpoints — including built-in ones — are profile-scoped: a profile only
    // gets the endpoints it explicitly enables, so any of them can be turned off.
    let sparql: StoredSparqlEndpoint[];
    if (profile) {
      sparql = registry.sparqlEndpoints.filter(ep =>
        (profile.enabledSparqlEndpointIds ?? []).includes(ep.id));
      if (profile.toolAuthOverrides) {
        const ov = profile.toolAuthOverrides;
        sparql = sparql.map(ep => ov[ep.id] ? { ...ep, auth: ov[ep.id] } : ep);
      }
    } else {
      sparql = []; // no profile → no registered SPARQL endpoints (conservative default)
    }

    // Default backend (for the job designer + background jobs that carry no explicit server):
    // the profile's chosen server, else the first configured server.
    const defSrv = (s.servers ?? []).find(x => x.id === profile?.serverId) ?? (s.servers ?? [])[0];
    const dirs = profile?.allowedDirs ?? s.allowedDirs ?? [];
    await invoke("set_mcp_servers",   { servers: mcp }).catch(() => {});
    await invoke("set_openapi_specs", { specs: openapi.filter(sp => sp.enabled !== false) }).catch(() => {});
    await invoke("set_sparql_endpoints", { endpoints: sparql.filter(ep => ep.enabled !== false) }).catch(() => {});
    if (defSrv) {
      await invoke("set_backend", { args: { base_url: defSrv.baseUrl, provider: defSrv.provider, api_key: defSrv.apiKey ?? null } }).catch(() => {});
    }
    await invoke("set_allowed_dirs",  { dirs }).catch(() => {});
  };

  // On first mount: migrate persisted allowed_dirs from Rust if not yet in frontend settings
  useEffect(() => {
    const doInit = async () => {
      let s = settings;
      if (s.allowedDirs === undefined) {
        const persisted = await invoke<string[]>("get_allowed_dirs").catch(() => [] as string[]);
        if (persisted.length > 0) {
          s = { ...s, allowedDirs: persisted };
          saveSettings(s);
          setSettings(s);
        } else {
          s = { ...s, allowedDirs: [] };
          saveSettings(s);
          setSettings(s);
        }
      }
      await syncServers(s);
    };
    doInit();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSaveSettings = async (newSettings: AppSettings) => {
    saveSettings(newSettings);
    setSettings(newSettings);
    await syncServers(newSettings);
    const ap = newSettings.profiles.find(p => p.id === newSettings.activeProfileId);
    if (ap?.model) {
      const srv = serverForModel(newSettings.servers ?? [], ap.serverId, ap.model);
      if (srv && (srv.models ?? []).includes(ap.model)) { setSelectedServerId(srv.id); setSelectedModel(ap.model); }
    }
  };

  const handleProfileChange = async (id: string) => {
    const profile = settings.profiles.find(p => p.id === id) ?? null;
    const updated = { ...settings, activeProfileId: id || null };
    // Clear the chat FIRST. Previously the reset ran last, after `syncServers` — which is slow when
    // it drops/reconnects MCP servers (e.g. docker-backed ones) — so a prompt sent during the
    // switch got wiped by the trailing reset, snapping the view back to the new-chat state.
    await handleReset();
    // handleSaveSettings persists, syncs servers for the new profile, and applies its model — so
    // there's no separate syncServers call here (it used to run twice).
    await handleSaveSettings(updated);
    if (profile?.model) {
      const srv = serverForModel(updated.servers ?? [], profile.serverId, profile.model);
      if (srv && (srv.models ?? []).includes(profile.model)) { setSelectedServerId(srv.id); setSelectedModel(profile.model); }
    }
    setChatParams(profile?.chatParams ?? updated.chatParams ?? DEFAULT_CHAT_PARAMS);
  };
  chatParamsRef.current = chatParams;
  profileSwitchRef.current = handleProfileChange;
  handleResetRef.current = handleReset;

  const canSend = (input.trim().length > 0 || attachedFiles.length > 0) && !isRunning && !!selectedModel;

  return (
    <div className="app">
      {/* Toolbar */}
      <div className="toolbar">
        <img src={lexiLogo} style={{ width: 22, height: 22, borderRadius: 6 }} alt="LexiChat" />
        <span className="toolbar-title">
          {activeProfile ? activeProfile.name : (selectedModel || "LexiChat")}
        </span>
        {/* Profile selector */}
        {settings.profiles.length > 0 && (
          <select
            className="profile-select"
            value={settings.activeProfileId ?? ""}
            onChange={e => handleProfileChange(e.target.value)}
            title="Switch profile — starts a new chat"
          >
            <option value="">Default</option>
            {settings.profiles.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        )}
        <button className="btn icon-only" onClick={() => setShowHistory(v => !v)} title="Chat history"
          style={{ opacity: showHistory ? 1 : 0.55 }}>
          <PanelLeft size={13} />
        </button>
        <button className="btn" onClick={handleReset} disabled={isRunning}>
          <RotateCcw size={12} /> New chat
        </button>
        <button className="btn icon-only" onClick={() => setShowDebug(v => !v)} title="Debug"
          style={{ opacity: showDebug ? 1 : 0.55 }}>
          <Bug size={13} />
        </button>
        <button className="btn icon-only" onClick={() => setShowUsageLive(v => !v)} title="Usage &amp; Performance"
          style={{ opacity: showUsageLive ? 1 : 0.55 }}>
          <BarChart3 size={13} />
        </button>
        {(activeProfile?.wikiEnabled ?? settings.wikiEnabled) === true && (
          <button className="btn icon-only" onClick={() => setShowWikiGraph(true)} title="Memory Map">
            <Brain size={13} />
          </button>
        )}
        <button className="btn icon-only" onClick={() => setShowAbout(true)} title="About LexiChat">
          <Info size={13} />
        </button>
        <button className="btn icon-only" onClick={() => setShowAdmin(true)} title="Admin">
          <Settings size={13} />
        </button>
        <button
          className="btn icon-only"
          onClick={() => { setView(v => v === "jobs" ? "chat" : "jobs"); setJobBadge(0); }}
          title={view === "jobs" ? "Back to chat" : "Scheduled Jobs"}
          style={{ position: "relative", opacity: view === "jobs" ? 1 : undefined }}
        >
          <Clock size={13} />
          {jobBadge > 0 && view !== "jobs" && (
            <span className="job-badge">{jobBadge > 9 ? "9+" : jobBadge}</span>
          )}
        </button>
      </div>

      {/* Jobs view — full page, replaces chat when active */}
      {view === "jobs" && (
        <JobsPanel
          models={[...new Set((settings.servers ?? []).flatMap(s => s.models ?? []))]}
          profiles={settings.profiles}
          activeProfileId={settings.activeProfileId ?? null}
          globalOpenapiSpecs={settings.toolRegistry.openapiSpecs}
          globalMcpServers={settings.toolRegistry.mcpServers}
          globalEnabledTools={settings.enabledTools ?? {}}
          globalAllowedDirs={settings.allowedDirs ?? []}
          onClose={() => setView("chat")}
        />
      )}

      {/* Main content: history sidebar + chat + optional debug panel */}
      <div style={{ display: view === "jobs" ? "none" : "flex", flex: 1, overflow: "hidden" }}>
      {showWikiGraph && <WikiGraphPanel onClose={() => setShowWikiGraph(false)} />}

      <HistoryPanel
        visible={showHistory}
        conversations={displayedConversations}
        activeId={activeConversationId}
        onSelect={handleSelectConversation}
        onNew={handleReset}
        onDelete={handleDeleteConversation}
        onRename={handleRenameConversation}
        onPin={handlePinConversation}
        onSetFolder={handleSetFolder}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onHide={() => setShowHistory(false)}
      />
      <div style={{ display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
      {/* Chat area */}
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="welcome">
            <img src={lexiLogo} className="welcome-logo" alt="LexiChat" />
            <div className="welcome-text">
              <h2>LexiChat</h2>
              <p>Your local AI assistant with tools &amp; APIs</p>
            </div>
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s.title} className="suggestion-chip" onClick={() => send(s.prompt)}>
                  <span className="suggestion-icon">{s.icon}</span>
                  <div>
                    <div className="suggestion-title">{s.title}</div>
                    <div className="suggestion-prompt">{s.prompt}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {(() => {
              // Regenerate is offered only on the LAST assistant message (it re-runs the final turn,
              // discarding nothing after it). Edit is offered on any user message with a checkpoint.
              const lastAssistantId = [...messages].reverse().find(m => m.role === "assistant")?.id;
              return messages.map((msg, i) => {
              // Each message renders inside its own boundary, so one that throws shows a placeholder
              // instead of blanking the whole app (see MessageErrorBoundary).
              let el: ReactNode = null;
              if (msg.role === "user")        el = <UserMessage text={msg.text} imageDataUrls={msg.imageDataUrls}
                                                       canEdit={!isRunning && msg.wireBase != null}
                                                       onEdit={t => handleEditUserMessage(msg.id, t)} />;
              else if (msg.role === "assistant")   el = <AssistantMessage msg={msg} thinkingAt={thinkingAt}
                                                       onExport={isLastAssistantInTurn(messages, i) ? exportReport : undefined}
                                                       onRegenerate={!isRunning && msg.id === lastAssistantId ? () => handleRegenerate(msg.id) : undefined} />;
              else if (msg.role === "tool-result") el = (
                <ToolResultRow
                  name={msg.toolName ?? ""}
                  result={msg.text}
                  args={msg.toolArgs}
                  fullResult={msg.fullResult}
                  fullTruncated={msg.fullTruncated}
                  ui={msg.ui}
                  images={msg.toolImages}
                  artifact={msg.artifact}
                  onSend={send}
                  onAttach={(path, prompt) => { setAttachedFiles([path]); setInput(prompt); }}
                />
              );
              else if (msg.role === "error")       el = <div className="msg-error">⚠ {msg.text}</div>;
              else if (msg.role === "notice")      el = (
                <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, opacity: 0.75, fontStyle: "italic", padding: "4px 8px" }}>
                  <span>ℹ {msg.text}</span>
                  {msg.savePrompt && (
                    <button className="copy-btn" style={{ fontStyle: "normal" }} onClick={() => saveOutputFiles(msg.id)}>💾 Save…</button>
                  )}
                </div>
              );
              return el && <MessageErrorBoundary key={msg.id}>{el}</MessageErrorBoundary>;
            });
            })()}
            {isRunning && !messages.some(m => m.streaming) && (
              <div className="msg-assistant">
                <img src={lexiLogo} className="assistant-avatar" alt="Lexi" />
                <div className="assistant-content">
                  <div className="thinking-row">
                    <ThinkingDots />
                    <ToolTimer startedAt={thinkingAt ?? undefined} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-card">
          {/* Attached file chips */}
          {attachedFiles.length > 0 && (
            <div className="attach-chips">
              {attachedFiles.map(f => (
                <div key={f} className="attach-chip">
                  <Paperclip size={10} />
                  <span>{f.split("/").pop()}</span>
                  {isImage(f) && (
                    <button title="Mark a region to edit" onClick={() => setMaskEditorFor(f)}
                      style={{ color: imageMasks[f] ? "var(--accent)" : undefined }}>
                      {imageMasks[f] ? "✎ region" : "✎"}
                    </button>
                  )}
                  <button onClick={() => {
                    setAttachedFiles(prev => prev.filter(p => p !== f));
                    setImageMasks(prev => { const n = { ...prev }; delete n[f]; return n; });
                  }}>✕</button>
                </div>
              ))}
            </div>
          )}
          <textarea
            ref={textareaRef}
            className="input-textarea"
            value={input}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder="Message…"
            disabled={isRunning}
            rows={1}
          />
          <div className="input-divider" />
          <div className="input-bottom">
            <button className="attach-btn" onClick={handleAttach} disabled={isRunning} title="Attach file">
              <Paperclip size={14} />
            </button>
            <ChatParamsButton params={chatParams} onChange={setChatParams} disabled={isRunning} />
            <select
              className="model-select"
              value={selectedModel ? encModel(selectedServerId, selectedModel) : ""}
              onChange={e => { const d = decModel(e.target.value); setSelectedServerId(d.serverId); setSelectedModel(d.model); }}
              disabled={modelOptions.length === 0}
            >
              {modelOptions.length === 0
                ? <option>No models found</option>
                : (settings.servers ?? []).map(s => {
                    const ms = s.models ?? [];
                    if (ms.length === 0) return null;
                    // Only one server → skip the prefix; multiple → show "server / model".
                    const single = (settings.servers ?? []).filter(x => (x.models ?? []).length > 0).length <= 1;
                    return (
                      <optgroup key={s.id} label={s.name}>
                        {ms.map(m => <option key={s.id + m} value={encModel(s.id, m)}>{single ? m : `${s.name} / ${m}`}</option>)}
                      </optgroup>
                    );
                  })
              }
            </select>
            <div className="input-spacer" />
            {isRunning ? (
              <button className="send-circle stop" onClick={stopActiveRun}>
                <div className="stop-square" />
              </button>
            ) : (
              <button className={`send-circle ${canSend ? "active" : "inactive"}`} onClick={() => send(input)} disabled={!canSend}>
                <span className="send-arrow">↑</span>
              </button>
            )}
          </div>
        </div>
      </div>

      </div>{/* end chat column */}

      {/* Debug panel sidebar */}
      <DebugPanel visible={showDebug} clearKey={debugClearKey} />
      {/* Live usage rail — docked beside the chat so stats stay visible while you chat */}
      <UsageRail open={showUsageLive} onClose={() => setShowUsageLive(false)} onOpenHistory={() => setShowUsageHistory(true)} />

      </div>{/* end main content row */}

      <UsageHistoryModal open={showUsageHistory} onClose={() => setShowUsageHistory(false)} />

      {maskEditorFor && (
        <MaskEditor
          path={maskEditorFor}
          onSave={(dataUrl) => setImageMasks(prev => {
            const n = { ...prev };
            if (dataUrl) n[maskEditorFor] = dataUrl; else delete n[maskEditorFor];
            return n;
          })}
          onClose={() => setMaskEditorFor(null)}
        />
      )}

      {showAdmin && (
        <AdminPanel
          settings={settings}
          onSave={handleSaveSettings}
          onClose={() => setShowAdmin(false)}
        />
      )}


      {showAbout && (
        <div className="modal-overlay" onClick={() => setShowAbout(false)}>
          <div className="about-modal" onClick={e => e.stopPropagation()}>
            <img src={lexiLogo} className="about-logo" alt="LexiChat" />
            <h2 className="about-name">LexiChat</h2>
            <p className="about-tagline">Your local AI assistant</p>
            <p className="about-desc">
              Runs entirely on-device via Ollama. Reads files, searches the web,
              calls APIs, and keeps your data private.
            </p>
            <div className="about-version">Version 2.4.25</div>

            <div className="about-support">
              <div className="about-support-label">Support the project</div>
              <button
                className="donate-btn donate-bmc"
                onClick={() => openUrl("https://buymeacoffee.com/lexichat")}
              >
                <span className="donate-emoji">☕</span> Buy me a coffee
              </button>
              <button
                className="donate-btn donate-btc"
                title="bc1q4faazp4qndldfsa8ahqeens3mej0svgwtl7h4v"
                onClick={() => {
                  navigator.clipboard.writeText("bc1q4faazp4qndldfsa8ahqeens3mej0svgwtl7h4v");
                  setBtcCopied(true);
                  setTimeout(() => setBtcCopied(false), 2000);
                }}
              >
                <span className="donate-btc-top">
                  <span className="donate-emoji">₿</span>
                  {btcCopied ? "Address copied!" : "Donate Bitcoin"}
                </span>
                <span className="donate-btc-addr">bc1q4faazp4q…j0svgwtl7h4v</span>
              </button>
            </div>

            <button className="btn primary" style={{ marginTop: 8 }} onClick={() => setShowAbout(false)}>
              Close
            </button>
          </div>
        </div>
      )}

      {permissionRequest && (
        <div className="modal-overlay">
          <div className="about-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 560, textAlign: "left" }}>
            <h2 className="about-name" style={{ fontSize: 18 }}>Run Python code?</h2>
            <p className="about-desc">
              The assistant wants to execute this code in the sandbox. It can read and
              write files within your allowed folders and attached files. Approving will
              allow code execution for the rest of this session — or choose “Always allow”
              to skip this prompt on future runs.
            </p>
            <pre style={{
              background: "var(--code-bg, #1e1e1e)", color: "var(--code-fg, #e0e0e0)",
              padding: 12, borderRadius: 6, maxHeight: 280, overflow: "auto",
              fontSize: 12, whiteSpace: "pre-wrap", wordBreak: "break-word",
            }}>{permissionRequest.code}</pre>
            <div style={{ display: "flex", gap: 8, marginTop: 12, justifyContent: "flex-end" }}>
              <button className="btn" onClick={() => {
                invoke("respond_code_permission", { approved: false }).catch(() => {});
                setPermissionRequest(null);
              }}>Deny</button>
              <button className="btn" onClick={() => {
                invoke("respond_code_permission", { approved: true }).catch(() => {});
                setSettings(prev => { const upd = { ...prev, alwaysAllowCodeExec: true }; saveSettings(upd); return upd; });
                setPermissionRequest(null);
              }}>Always allow</button>
              <button className="btn primary" onClick={() => {
                invoke("respond_code_permission", { approved: true }).catch(() => {});
                setPermissionRequest(null);
              }}>Allow &amp; run</button>
            </div>
          </div>
        </div>
      )}

      {reportPreview && (
        <div className="modal-overlay" onClick={() => setReportPreview(null)}>
          <div className="report-preview-modal" onClick={e => e.stopPropagation()}>
            <div className="report-preview-head">
              <span className="report-preview-title">Report preview — {reportPreview.title}</span>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "var(--text-tertiary)" }}>Save as</span>
                <button className="btn primary" onClick={() => saveReportAs("html")} title="Styled report with charts embedded — opens in any browser">HTML</button>
                <button className="btn" onClick={printReport} title="Opens the styled report in your browser — then Print → Save as PDF for an exact copy">PDF…</button>
                <button className="btn" onClick={() => saveReportAs("doc")} title="Opens in Word with styling and charts">Word</button>
                <button className="btn" onClick={() => setReportPreview(null)}>Close</button>
              </div>
            </div>
            {/* sandbox="" → static HTML+CSS only, no scripts/same-origin — safe by construction. */}
            <iframe className="report-preview-frame" sandbox="" srcDoc={reportPreview.html} title="Report preview" />
          </div>
        </div>
      )}
    </div>
  );
}
