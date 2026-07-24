import { useState, useEffect, useRef, useCallback, KeyboardEvent, ChangeEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Settings, RotateCcw, Bug, Paperclip, Info, Clock, PanelLeft } from "lucide-react";
import { JobsPanel } from "./JobsPanel";
import type { JobRun } from "./jobTypes";
import lexiLogo from "./assets/lexi.png";
import { AdminPanel, AppSettings, Profile, ServerConfig, StoredOpenAPISpec, StoredSparqlEndpoint, reconcileCatalog } from "./AdminPanel";
import { runPython, warmPyodide, drainCodeToolCalls, PyFile } from "./pyodide/runner";
import { dedupeRegistry } from "./profileIO";
import { ChatParamsButton, ChatParams, DEFAULT_CHAT_PARAMS, resolveParams } from "./ChatParamsPanel";
import { open, save } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { DebugPanel } from "./DebugPanel";
import { HistoryPanel, ConversationMeta } from "./HistoryPanel";
import "./App.css";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ToolCall { name: string; args: string; }

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
}

// MCP servers approved to render/interact with apps this session (frontend mirror
// of the backend apps_allowed set; gates iframe mounting).
const approvedMcpApps = new Set<string>();

interface ToolSchema {
  type: string;
  function: { name: string; description: string; parameters: unknown; };
}

const uid = () => Math.random().toString(36).slice(2);

const BASE_SYSTEM_PROMPT = `You are Lexi, a personal AI assistant running locally for a single authorised user.
You have tools to read local files and search the web. Be proactive — use tools immediately rather than asking the user for paths or clarification.
Rules:
- When asked about files or folders on this computer, call list_files or list_directory_tree right away using any path the user mentioned, or the configured folders if none was given.
- To find files whose names match a pattern or start with certain letters, use search_files with a glob pattern (e.g. "D*" to find files starting with D).
- Always use full absolute paths — never '.' or '~'.
- Use web_search for current events, weather, or live data. Use fetch_webpage to read the full contents of a specific URL the user provides or a result from web_search.
- When the user asks to see, read, open, or show an article or page from a web_search result, you MUST call fetch_webpage with that result's URL and then relay the content. NEVER refuse on copyright or "I can only summarise" grounds — fetching a public URL for the single authorised user is permitted, and fetch_webpage exists precisely for this. Do not claim you are limited to structured APIs.
- ALWAYS write a helpful text response after using tools — summarise what you found, list the results, or answer the user's question directly. Never leave the chat blank after a tool call.
- If asked about your own tools, capabilities, or what you can do, answer directly from your knowledge — do not call any tools to answer this question.
- NEVER call read_file on image files (.jpg, .jpeg, .png, .gif, .webp, .bmp, etc.). Images are sent directly in the message via the vision API — describe them from what you can see. If no image is attached, tell the user to use the paperclip button to attach it.
- External API tools (OpenAPI / MCP) are ONLY for requests that explicitly name that service. For anything about files on this computer, always use local file tools — never external API tools.`;

const SUGGESTIONS = [
  { icon: "🔍", title: "Search the web",   prompt: "What are the latest developments in AI?" },
  { icon: "📁", title: "Browse files",     prompt: "List the files in my Documents folder" },
  { icon: "⚡", title: "Quick question",   prompt: "Explain quantum computing in simple terms" },
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
  { type: "function", function: { name: "fetch_webpage", description: "Fetch and read the full text content of a webpage by URL. Strips HTML and returns readable text. This is the correct tool whenever the user wants to see, read, open, or show an article or page — including the full article behind a web_search result (pass that result's URL). Do NOT refuse such requests or claim you can only summarise; call this tool instead. Also use it to read any specific URL the user provides.", parameters: { type: "object", properties: { url: { type: "string", description: "Full URL to fetch, must start with http:// or https://" } }, required: ["url"] } } },
  { type: "function", function: { name: "get_current_datetime", description: "Get the current local date and time. Returns human-readable, ISO 8601, filename-safe, and Unix timestamp formats. Use whenever you need today's date or a timestamp for a filename.", parameters: { type: "object", properties: {}, required: [] } } },
  { type: "function", function: { name: "run_python", description: "Execute real Python (CPython) in a secure, offline sandbox to compute, analyse data, and CREATE CHARTS. The full standard library plus numpy, pandas, matplotlib, scipy, sympy, openpyxl (read/write Excel .xlsx), and beautifulsoup4 (parse HTML) are available — import them normally. Use print() for text output. Files live in a virtual workspace at /work/uploads/: the user's attached files are there — documents (PDF, Word) are ALREADY extracted to plain text, so just open() and read them (do NOT try to PDF-parse); data files (CSV, Excel, JSON) are as-is for pandas. SAVE any output (files, charts) to /work/out/ (kept for the user). (For a plain read/summary of a document with no computation, prefer the read_file tool — no code or permission needed.) Use normal Python I/O — open(), pathlib, pd.read_csv('/work/uploads/data.csv'). TO SHOW A GRAPH, build a matplotlib figure (e.g. `import matplotlib.pyplot as plt; plt.plot(x, y)`) — it is rendered INLINE in the chat automatically — you do NOT need to save it (do NOT hand-draw ASCII or SVG). Only use plt.savefig('/work/out/name.png') if the user explicitly wants a saved file — /work/out is an in-memory scratch path, but anything you write there is copied to a real folder on the user's disk and the tool result reports that real absolute path. When telling the user where a file was saved, quote the real path from the tool result (the line marked SAVED TO DISK); NEVER tell the user the file is at /work/out (they cannot open that). No network access. Do not read/write paths outside /work.", parameters: { type: "object", properties: { code: { type: "string", description: "The Python source code to execute." } }, required: ["code"] } } },
  { type: "function", function: { name: "create_artifact", description: "Render a rich, self-contained HTML page inline in the chat, with a Save button (saves as a .html file the user can open in any browser). Use this for polished deliverables — formatted reports, dashboards, styled tables/cards, or simple interactive views — when plain markdown isn't enough. The HTML MUST be fully self-contained: inline all CSS in a <style> tag and any JS in a <script> tag; NO external URLs, fonts, images, or CDNs (they are blocked). To include a chart, map or image you generated earlier THIS TURN (e.g. a matplotlib chart from run_python, or a map), use the placeholder token as the image source: <img src=\"{{figure:1}}\"> for the first such image, {{figure:2}} for the second, and so on (in the order they were created) — LexiChat substitutes the real image. Do NOT paste base64 image data yourself. Any other images must be data: URIs. It renders in a sandboxed frame. Do NOT put your final prose answer inside the artifact — write a short summary in chat and put the rich content in the artifact.", parameters: { type: "object", properties: { title: { type: "string", description: "Short title for the artifact (used as the saved filename and header)." }, html: { type: "string", description: "A complete, self-contained HTML document (or fragment) with all CSS/JS inlined and no external resources." } }, required: ["title", "html"] } } },
];

// Built-in tools a chat gets when NO profile is active: read-only / no-side-effect only. Mutating
// file tools, email, code execution, and all registered OpenAPI/MCP/SPARQL integrations require an
// explicit profile. (Product decision — the no-profile default must not expose everything.)
const SAFE_DEFAULT_BUILTINS = new Set([
  "read_file", "list_files", "search_files", "search_in_files", "get_file_info",
  "list_directory_tree", "web_search", "fetch_webpage", "get_current_datetime",
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
      "- Price Paid (lrppi:): individual residential property transactions. A lrppi:Transaction has lrppi:pricePaid, lrppi:transactionDate, and lrppi:propertyAddress (an lrcommon:Address with lrcommon:postcode, lrcommon:town, lrcommon:street).",
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
  maxSteps: 20,
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
    // different ids) and remap profile references onto the survivor.
    return dedupeRegistry(loaded);
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

// ── Message bubbles ───────────────────────────────────────────────────────────

function UserMessage({ text, imageDataUrls }: { text: string; imageDataUrls?: string[] }) {
  return (
    <div className="msg-user">
      {imageDataUrls && imageDataUrls.length > 0 && (
        <div className="user-image-thumbs">
          {imageDataUrls.map((src, i) => (
            <img key={i} src={src} className="user-image-thumb" alt="attached image" />
          ))}
        </div>
      )}
      {text && <div className="user-bubble">{text}</div>}
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
// Replace {{figure:N}} tokens with figure data URLs. `asMarkdown` wraps in markdown image syntax
// (for report text); otherwise substitutes the raw URL (for artifact HTML `src="…"`).
function substituteFigures(text: string, figs: string[], asMarkdown: boolean): { out: string; used: Set<number> } {
  const used = new Set<number>();
  const out = text.replace(/\{\{figure:(\d+)\}\}/g, (whole, n) => {
    const i = Number(n) - 1;
    if (!figs[i]) return whole;
    used.add(i);
    return asMarkdown ? `![Figure ${n}](${figs[i]})` : figs[i];
  });
  return { out, used };
}

function AssistantMessage({ msg, onExport }: { msg: ChatMessage; onExport?: (msgId: string) => void }) {
  const showThinking = msg.streaming && !msg.text && (!msg.toolCalls || msg.toolCalls.length === 0);
  return (
    <div className="msg-assistant">
      <img src={lexiLogo} className="assistant-avatar" alt="Lexi" />
      <div className="assistant-content">
        {showThinking ? (
          <div className="thinking-row">
            <ThinkingDots />
            {msg.status && <span className="thinking-status">{msg.status}</span>}
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
            {msg.toolCalls.map((tc, i) => (
              <div key={i} className="tool-badge">
                <span className="tool-badge-icon">⚡</span>
                <span className="tool-badge-name">{tc.name}</span>
                {tc.args && <span className="tool-badge-args">{tc.args}</span>}
              </div>
            ))}
          </div>
        )}

        {!msg.streaming && msg.text && (
          <div style={{ display: "flex", gap: 4 }}>
            <CopyButton text={msg.text} />
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
              hostInfo: { name: "LexiChat", version: "2.0.12" },
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
          <button className="btn primary" onClick={async () => {
            try { await invoke("approve_mcp_app", { args: { server_id: ui.server_id } }); } catch { /* ignore */ }
            approvedMcpApps.add(ui.server_id);
            setApproved(true);
          }}>Allow app</button>
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

// Model-authored HTML artifact (create_artifact) — rendered inline in a sandboxed frame with a
// Save button. Static-or-scripted HTML; sandbox allows scripts but not same-origin/network.
function ArtifactFrame({ title, html }: { title: string; html: string }) {
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
      <iframe className="artifact-frame" sandbox="allow-scripts" srcDoc={html} title={`artifact-${title}`} />
    </div>
  );
}

export function ToolResultRow({
  name, result, args, ui, images, artifact, onSend, onAttach,
}: {
  name: string; result: string; args?: string; ui?: ToolUi; images?: string[];
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
  const [showAdmin, setShowAdmin] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showHistory, setShowHistory] = useState(false); // hidden on launch; toggle to open
  const [conversations, setConversations] = useState<ConversationMeta[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
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
  const isRunningRef = useRef(false);
  const autoApproveControlRef = useRef(false);
  const settingsRef = useRef(settings);
  const selectedModelRef = useRef(selectedModel);
  const chatParamsRef = useRef<ChatParams | null>(null);
  const profileSwitchRef = useRef<((id: string) => Promise<void>) | null>(null);
  const handleResetRef = useRef<(() => Promise<void>) | null>(null);
  const forceAllowCodeToolsRef = useRef(false); // dev-control transient override for allow_code_tools
  messagesRef.current = messages;
  isRunningRef.current = isRunning;
  settingsRef.current = settings;
  selectedModelRef.current = selectedModel;

  // Cancel a running agent loop and supersede its stream so late events are dropped.
  const stopActiveRun = () => {
    invoke("stop_generation").catch(() => {});
    streamEpoch.current += 1;
    setIsRunning(false);
  };

  // Auto-save the conversation when an agent run finishes (transition running→idle).
  const prevRunning = useRef(false);
  useEffect(() => {
    const justFinished = prevRunning.current && !isRunning;
    prevRunning.current = isRunning;
    if (!justFinished || messages.length === 0) return;
    (async () => {
      try {
        const meta = await invoke<ConversationMeta>("save_active_conversation", {
          args: {
            display: messages,
            profile_id: settings.activeProfileId ?? null,
            model: selectedModel,
            message_count: messages.length,
          },
        });
        setActiveConversationId(meta.id);
        refreshConversations();
      } catch { /* empty wire — nothing to save */ }
    })();
  }, [isRunning]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSelectConversation = async (id: string) => {
    if (isRunning) return;
    try {
      const display = await invoke<ChatMessage[]>("load_conversation", { args: { id } });
      setMessages(Array.isArray(display) ? display : []);
      setActiveConversationId(id);
    } catch { /* conversation missing */ }
  };

  const handleDeleteConversation = async (id: string) => {
    try {
      await invoke("delete_conversation", { args: { id } });
      if (id === activeConversationId) {
        await invoke("reset_conversation");
        setActiveConversationId(null);
        setMessages([]);
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

  // Listen to agent events from Rust
  useEffect(() => {
    const cleanup: Array<() => void> = [];

    // Pre-load the Python runtime so the first run_python (or a scheduled job) isn't cold.
    warmPyodide();

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
      setMessages(prev => {
        const updated = prev.map(m =>
          m.role === "assistant" && m.streaming
            ? { ...m, toolCalls: [...(m.toolCalls ?? []), { name: e.payload.name, args: e.payload.args }] }
            : m
        );
        const hasStreaming = prev.some(m => m.role === "assistant" && m.streaming);
        if (!hasStreaming) {
          return [...updated, { id: uid(), role: "assistant", text: "", streaming: true, toolCalls: [{ name: e.payload.name, args: e.payload.args }] }];
        }
        return updated;
      });
    }).then(u => cleanup.push(u));

    listen<{ name: string; result: string; ui?: ToolUi; images?: string[]; artifact?: { title: string; html: string } }>("agent-tool-result", e => {
      if (!streamActive()) return;
      setMessages(prev => {
        // Find args for this tool call from the most recent streaming assistant message
        const streamingMsg = [...prev].reverse().find(m => m.role === "assistant" && m.streaming);
        const matchingCall = streamingMsg?.toolCalls?.find(tc => tc.name === e.payload.name);
        const closed = prev.map(m => m.streaming ? { ...m, streaming: false } : m);
        // Resolve {{figure:N}} tokens in a model artifact against charts generated this turn.
        let artifact = e.payload.artifact;
        if (artifact?.html && artifact.html.includes("{{figure:")) {
          const figs = collectTurnFigures(prev);
          artifact = { ...artifact, html: substituteFigures(artifact.html, figs, false).out };
        }
        return [...closed, {
          id: uid(), role: "tool-result",
          text: e.payload.result,
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
      if (!streamActive()) return;
      setIsRunning(false);
      setMessages(prev => {
        const closed = prev.map(m => m.streaming ? { ...m, streaming: false } : m);
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
    listen<{ request_id: number; code: string; files: PyFile[] }>("run-python-request", async e => {
      const res = await runPython(e.payload.code, e.payload.files ?? []);
      await invoke("respond_python_result", { args: {
        request_id: e.payload.request_id,
        output: res.output, error: res.error, images: res.images, out_files: res.outFiles,
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
      drainCodeToolCalls(); // clear any stale code-tool log
      const startLen = messagesRef.current.length;
      autoApproveControlRef.current = true;
      const t0 = performance.now();
      try {
        await sendRef.current?.(message);
        // send() returns before the agent loop finishes; wait for isRunning to settle to false.
        await new Promise(r => setTimeout(r, 250)); // grace for isRunning → true
        const deadline = Date.now() + 880_000;
        while (isRunningRef.current && Date.now() < deadline) {
          await new Promise(r => setTimeout(r, 150));
        }
      } catch (err) {
        report({ error: String(err), elapsedMs: Math.round(performance.now() - t0) }); autoApproveControlRef.current = false; return;
      }
      autoApproveControlRef.current = false;
      const elapsedMs = Math.round(performance.now() - t0);
      const codeToolCalls = drainCodeToolCalls();
      const trace = messagesRef.current.slice(startLen).map(m => ({
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
      const finalAnswer = [...messagesRef.current.slice(startLen)].reverse()
        .find(m => m.role === "assistant" && !!m.text)?.text;
      report({ finalAnswer, elapsedMs, codeToolCalls, messages: trace });
    }).then(u => cleanup.push(u));

    // run_python produced output files but no sandbox folder is configured to save them. Ask the
    // user to pick a folder — it's added to the sandbox and the stashed files are written there.
    // We never write outside the sandbox.
    listen<{ files: string[] }>("sandbox-save-request", async e => {
      const names = e.payload.files ?? [];
      const dir = await open({
        directory: true,
        title: `Choose a folder to save ${names.length} file(s) and add it to the sandbox`,
      }).catch(() => null);
      if (!dir || typeof dir !== "string") return; // user cancelled → files discarded
      try {
        const saved = await invoke<string[]>("save_pending_outputs", { dir });
        setMessages(prev => [...prev, { id: uid(), role: "notice",
          text: `Saved to sandbox folder (now added to the sandbox): ${saved.join(", ")}` }]);
      } catch (err) {
        setMessages(prev => [...prev, { id: uid(), role: "error", text: `Could not save files: ${String(err)}` }]);
      }
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
    const enabledTools = ALL_BUILTIN_TOOLS.filter(t => {
      const name = t.function.name;
      if (!activeProfile) {
        // No active profile → conservative read-only default. Exception: run_python is allowed
        // when its global master switch is explicitly on (an opt-in security capability, still
        // gated by the per-run permission prompt) — so code execution doesn't need a profile.
        if (name === "run_python") return runPythonMaster;
        return SAFE_DEFAULT_BUILTINS.has(name) && settings.enabledTools[name] !== false;
      }
      if (name === "run_python") return runPythonMaster && effectiveEnabledTools.run_python !== false;
      return effectiveEnabledTools[name] !== false;
    });
    // Wiki memory: the active profile can override the global default; an unset profile
    // (undefined) inherits it.
    const wikiEnabled = (activeProfile?.wikiEnabled ?? settings.wikiEnabled) === true;
    if (wikiEnabled) enabledTools.push(...WIKI_TOOLS);

    // Split attachments into images (sent via Ollama images field) and other files (appended as paths)
    const imagePaths = attachedFiles.filter(isImage);
    const otherFiles = attachedFiles.filter(f => !isImage(f));

    const fullText = otherFiles.length > 0
      ? `${text}\n\nThe user has attached the following local file(s). Use the read_file tool to read them directly (it extracts text from PDF, Word, and plain text automatically). This IS the document the user is referring to — do NOT search or fetch the web for it:\n${otherFiles.map(f => `- ${f}`).join("\n")}`
      : text;

    // Build display text — only list non-image attachments (images shown as thumbnails)
    const displayText = otherFiles.length > 0
      ? `${text}\n\nAttached: ${otherFiles.map(f => f.split("/").pop()).join(", ")}`
      : text;

    // Load image data URIs for thumbnails (fire-and-forget before send)
    const imageDataUrls = await Promise.all(
      imagePaths.map(p => invoke<string>("read_image_data_url", { path: p }).catch(() => ""))
    ).then(urls => urls.filter(Boolean));

    setMessages(prev => [...prev, { id: uid(), role: "user", text: displayText, imageDataUrls }]);
    setAttachedFiles([]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    // This run now owns the event stream (see streamEpoch/streamOwner).
    streamEpoch.current += 1;
    streamOwner.current = streamEpoch.current;
    setIsRunning(true);

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
      const effectiveBase = resolved.systemPromptOverride ?? basePrompt;

      // Build context vars block from the active profile
      const contextVars = activeProfile?.contextVars?.filter(v => v.name.trim() && v.value.trim()) ?? [];
      const contextVarsSuffix = contextVars.length > 0
        ? `\n\nUser context (treat these as facts about the user — use them automatically when relevant, do not repeat them unless asked):\n${contextVars.map(v => `- ${v.name}: ${v.value}`).join("\n")}`
        : "";

      const wikiSuffix = wikiEnabled ? WIKI_SYSTEM_PROMPT_BLOCK : "";

      // Never emit remote image URLs — the CSP blocks them so they render as nothing; tool/chart
      // images are already shown inline. (The model kept appending mapbox/OSM image URLs.)
      const outputRulesSuffix = "\n\nOUTPUT RULES: NEVER write a markdown image or link pointing at a remote http(s):// image URL (a map, chart, tile, etc.) — remote images are blocked and will NOT display. Any map, chart, or image produced by a tool or by run_python is ALREADY shown inline in the chat; just refer to it as \"shown above\". Do not paste image/tile URLs into your answer.";

      // The model has no clock — give it today's date so "latest/recent/this month" queries work
      // without needing a tool call, and warn that some data sources lag.
      const now = new Date();
      const dateSuffix = `\n\nTODAY'S DATE is ${now.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" })} (${now.toISOString().slice(0, 10)}). Use this whenever the user asks for the "latest", "recent", "current", "this month/year", "today", etc. — you already know today's date and do not need a tool for it. Be aware some data sources lag behind today (e.g. UK police street-crime data is usually 1–2 months old), so the "latest available" data may be for an earlier month than the current one — request the most recent month the source actually offers.`;

      // Code-mode: when the profile allows code to call tools, tell the model about the Python API.
      const codeToolsSuffix = (activeProfile?.allowCodeTools || forceAllowCodeToolsRef.current)
        ? "\n\nCODE-MODE TOOLS: inside run_python you can call registered tools directly. ALWAYS call `tools = await list_tools()` FIRST to get the EXACT tool names and their `parameters` schema — never guess a tool name or a group label. Each entry is {name, description, parameters}. Then `data = await call_tool(\"exact_tool_name\", {\"arg\": \"value\"})` runs one and returns a dict/list (parsed JSON) or string; build the args from the tool's parameters schema. Both are async — you MUST `await` them. Prefer this for multi-source work: fetch with call_tool, then compute/aggregate/plot with pandas/numpy/matplotlib in the same script, instead of many separate tool-call steps."
        : "";

      const systemPrompt = allowedDirs.length > 0
        ? `${effectiveBase}${externalSuffix}${contextVarsSuffix}${wikiSuffix}${codeToolsSuffix}${outputRulesSuffix}${dateSuffix}\nThe user's configured folders are: ${allowedDirs.join(", ")}. Rules for file operations:\n- When reading or listing files without a specified path, use these folders immediately — do not ask for clarification.\n- When writing or saving a file without a specified path, save it to ${allowedDirs[0]} with a sensible filename derived from the content (e.g. sikhism_article.pdf). Never call write_file without a full absolute path.\n- Always use full absolute paths — never '.' or '~'.`
        : `${effectiveBase}${externalSuffix}${contextVarsSuffix}${wikiSuffix}${codeToolsSuffix}${outputRulesSuffix}${dateSuffix}`;

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
          file_paths: otherFiles,
          temperature: resolved.temperature,
          top_p: resolved.topP ?? null,
          top_k: resolved.topK ?? null,
          repeat_penalty: resolved.repeatPenalty ?? null,
          seed: resolved.seed ?? null,
          num_ctx: resolved.numCtx,
          num_predict: resolved.numPredict,
          stop: resolved.stop ?? null,
          think: resolved.think ?? null,
          keep_alive: resolved.keepAlive ?? null,
          web_search_results: settings.webSearchResults ?? 10,
          max_steps: settings.maxSteps ?? 20,
          disabled_mcp_tools: disabledMcpTools,
          enabled_mcp_server_ids: enabledMcpServerIds,
          max_tools: (activeProfile?.maxTools ?? settings.maxTools) || null,
          tool_result_limit: activeProfile?.toolResultLimit ?? null,
          allow_code_tools: forceAllowCodeToolsRef.current || (activeProfile?.allowCodeTools ?? false),
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
    await handleSaveSettings(updated);
    if (profile?.model) {
      const srv = serverForModel(updated.servers ?? [], profile.serverId, profile.model);
      if (srv && (srv.models ?? []).includes(profile.model)) { setSelectedServerId(srv.id); setSelectedModel(profile.model); }
    }
    setChatParams(profile?.chatParams ?? updated.chatParams ?? DEFAULT_CHAT_PARAMS);
    await syncServers(updated);
    await handleReset();
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
      <HistoryPanel
        visible={showHistory}
        conversations={conversations}
        activeId={activeConversationId}
        onSelect={handleSelectConversation}
        onNew={handleReset}
        onDelete={handleDeleteConversation}
        onRename={handleRenameConversation}
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
            {messages.map((msg, i) => {
              if (msg.role === "user")        return <UserMessage key={msg.id} text={msg.text} imageDataUrls={msg.imageDataUrls} />;
              if (msg.role === "assistant")   return <AssistantMessage key={msg.id} msg={msg} onExport={isLastAssistantInTurn(messages, i) ? exportReport : undefined} />;
              if (msg.role === "tool-result") return (
                <ToolResultRow
                  key={msg.id}
                  name={msg.toolName ?? ""}
                  result={msg.text}
                  args={msg.toolArgs}
                  ui={msg.ui}
                  images={msg.toolImages}
                  artifact={msg.artifact}
                  onSend={send}
                  onAttach={(path, prompt) => { setAttachedFiles([path]); setInput(prompt); }}
                />
              );
              if (msg.role === "error")       return <div key={msg.id} className="msg-error">⚠ {msg.text}</div>;
              if (msg.role === "notice")      return <div key={msg.id} style={{ fontSize: 12, opacity: 0.6, fontStyle: "italic", padding: "4px 8px" }}>ℹ {msg.text}</div>;
              return null;
            })}
            {isRunning && !messages.some(m => m.streaming) && (
              <div className="msg-assistant">
                <img src={lexiLogo} className="assistant-avatar" alt="Lexi" />
                <div className="assistant-content"><ThinkingDots /></div>
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
                  <button onClick={() => setAttachedFiles(prev => prev.filter(p => p !== f))}>✕</button>
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

      </div>{/* end main content row */}

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
            <div className="about-version">Version 2.0.12</div>

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
              allow code execution for the rest of this session.
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
