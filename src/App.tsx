import { useState, useEffect, useRef, useCallback, KeyboardEvent, ChangeEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Settings, RotateCcw, Bug, Paperclip, Info, Clock, PanelLeft } from "lucide-react";
import { JobsPanel } from "./JobsPanel";
import type { JobRun } from "./jobTypes";
import lexiLogo from "./assets/lexi.png";
import { AdminPanel, AppSettings, Profile, StoredOpenAPISpec, StoredSparqlEndpoint } from "./AdminPanel";
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
  role: "user" | "assistant" | "tool-result" | "error";
  text: string;
  streaming?: boolean;
  toolCalls?: ToolCall[];
  toolName?: string;
  toolArgs?: string;
  imageDataUrls?: string[];  // base64 data URIs for attached images
  ui?: ToolUi;               // MCP-App interactive UI to render in a sandboxed iframe
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
  { type: "function", function: { name: "run_python", description: "Execute Python code in a secure sandbox to compute or answer questions (math, data processing, string/logic work). Use print() to output results. Supports a subset of Python: NO class definitions and NO third-party packages (no numpy/pandas/requests). For file access use the provided functions read_file(path)->str, write_file(path, content)->int, and list_files(dir)->list — do NOT use open() or pathlib (they are disabled). To parse JSON, `import json` then json.loads(read_file(path)) — the json module (loads/dumps) IS available (but json.load(fp) is not); most other stdlib (collections, os, datetime, re) is NOT available, so use plain dict/list/set. Paths must be within the user's allowed folders or attached files.", parameters: { type: "object", properties: { code: { type: "string", description: "The Python source code to execute." } }, required: ["code"] } } },
];

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
  {
    id: "builtin-ons-stats",
    title: "ONS Statistics (decommissioned)",
    endpoint_url: "https://statistics.data.gov.uk/sparql",
    enabled: false,
    read_only: true,
    prefixes: "",
    schema_summary:
      "NOTE: the ONS PublishMyData SPARQL endpoint (statistics.data.gov.uk) was decommissioned on 31 March 2025 and is no longer live — 'Test & discover' will report it unreachable. " +
      "Data moved to portals such as the ONS Geography Portal and Explore Local Statistics. For live UK linked-data statistics use the OpenDataCommunities endpoint instead. Kept here as a reference/template.",
    example_queries: [],
  },
];

function injectBuiltinSparql(endpoints: StoredSparqlEndpoint[]): StoredSparqlEndpoint[] {
  let result = endpoints;
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
  host: "http://localhost:11434",
  maxTools: 30,
  webSearchResults: 10,
  maxSteps: 20,
  numGPULayers: null,
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
    return loaded;
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

function SaveMenu({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const saveAs = async (ext: string) => {
    setOpen(false);
    const filters = ext === "docx"
      ? [{ name: "Word Document", extensions: ["docx"] }]
      : ext === "pdf"
      ? [{ name: "PDF", extensions: ["pdf"] }]
      : [{ name: "Text File", extensions: ["txt"] }];
    const path = await save({ title: "Save response", filters });
    if (!path) return;
    const finalPath = path.endsWith(`.${ext}`) ? path : `${path}.${ext}`;
    await invoke("save_document", { path: finalPath, content: text });
  };

  return (
    <div ref={ref} style={{ position: "relative", display: "inline-block" }}>
      <button className="copy-btn" onClick={() => setOpen(o => !o)} title="Save as…">
        ···
      </button>
      {open && (
        <div className="save-menu-dropdown">
          <div className="save-menu-header">Save response as</div>
          <button className="save-menu-item" onClick={() => saveAs("txt")}>
            <span className="save-menu-ext">TXT</span>Plain text
          </button>
          <button className="save-menu-item" onClick={() => saveAs("pdf")}>
            <span className="save-menu-ext">PDF</span>PDF document
          </button>
          <button className="save-menu-item" onClick={() => saveAs("docx")}>
            <span className="save-menu-ext" style={{ background: "rgba(59,130,246,0.15)", color: "#60a5fa", borderColor: "rgba(59,130,246,0.25)" }}>DOC</span>Word document
          </button>
        </div>
      )}
    </div>
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

function AssistantMessage({ msg }: { msg: ChatMessage }) {
  const showThinking = msg.streaming && !msg.text && (!msg.toolCalls || msg.toolCalls.length === 0);
  return (
    <div className="msg-assistant">
      <img src={lexiLogo} className="assistant-avatar" alt="Lexi" />
      <div className="assistant-content">
        {showThinking ? (
          <ThinkingDots />
        ) : msg.streaming ? (
          <div className="assistant-text">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children }) => (
              <a href={href} onClick={e => { e.preventDefault(); if (href) openUrl(href); }}>{children}</a>
            )}}>{msg.text}</ReactMarkdown>
            <span className="streaming-cursor" />
          </div>
        ) : (
          msg.text && (
            <div className="assistant-text">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ a: ({ href, children }) => (
                <a href={href} onClick={e => { e.preventDefault(); if (href) openUrl(href); }}>{children}</a>
              )}}>{msg.text}</ReactMarkdown>
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
            <SaveMenu text={msg.text} />
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

    const onMessage = async (event: MessageEvent) => {
      if (event.source !== iframe.contentWindow) return; // only this app's iframe
      const data = event.data as Record<string, unknown> | null;
      if (!data || typeof data !== "object") return;
      logBridge("app→host", data);

      // ── ext-apps dialect: JSON-RPC over postMessage ──
      if (data.jsonrpc === "2.0" && typeof data.method === "string") {
        const { id, method } = data as { id?: unknown; method: string };
        const params = (data.params ?? {}) as Record<string, unknown>;

        // Push the tool's input + result so the app can render its content.
        const pushToolData = () => {
          post({ jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: ui.arguments ?? {} } });
          post({ jsonrpc: "2.0", method: "ui/notifications/tool-result", params: {
            content: ui.content ?? [],
            structuredContent: ui.structured ?? undefined,
            _meta: ui.meta ?? undefined,
          }});
        };

        try {
          if (method === "ui/initialize") {
            // Respond with the full McpUiInitializeResult shape the app SDK expects.
            post({ jsonrpc: "2.0", id, result: {
              protocolVersion: "2026-01-26",
              hostCapabilities: {},
              hostInfo: { name: "LexiChat", version: "2.0.3" },
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
    return () => window.removeEventListener("message", onMessage);
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
        style={{ width: "100%", height: 420, border: "1px solid var(--border)", borderRadius: 8, background: "#fff", marginTop: 6 }}
      />
    </div>
  );
}

export function ToolResultRow({
  name, result, args, ui, onSend, onAttach,
}: {
  name: string; result: string; args?: string; ui?: ToolUi;
  onSend: (text: string) => void;
  onAttach: (path: string, prompt: string) => void;
}) {
  if (ui?.html) {
    return <McpAppFrame ui={ui} toolName={name} onSend={onSend} />;
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
  const [isRunning, setIsRunning] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showHistory, setShowHistory] = useState(true);
  const [conversations, setConversations] = useState<ConversationMeta[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [showAbout, setShowAbout] = useState(false);
  const [btcCopied, setBtcCopied] = useState(false);
  const [view,     setView]     = useState<"chat" | "jobs">("chat");
  const [jobBadge, setJobBadge] = useState(0);
  // Pending run_python execution awaiting the user's approval.
  const [permissionRequest, setPermissionRequest] = useState<{ code: string } | null>(null);
  const [attachedFiles, setAttachedFiles] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Active profile derived from settings
  const activeProfile: Profile | null =
    settings.profiles.find(p => p.id === settings.activeProfileId) ?? null;

  const effectiveHost = activeProfile?.host || settings.host;

  // Fetch models and sync host on mount and when effective host changes
  const fetchModels = useCallback(async () => {
    try {
      await invoke("set_ollama_host", { host: effectiveHost });
      const list = await invoke<string[]>("get_models");
      setSettings(prev => {
        const merged = [...list, ...prev.models.filter(m => !list.includes(m))];
        const updated = { ...prev, models: merged };
        saveSettings(updated);
        return updated;
      });
      setSelectedModel(m => m && list.includes(m) ? m : (list[0] ?? ""));
    } catch { /* Ollama not running */ }
  }, [effectiveHost]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { fetchModels(); }, [effectiveHost]); // eslint-disable-line react-hooks/exhaustive-deps

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

    listen<{ delta: string }>("agent-token", e => {
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last.streaming) {
          return [...prev.slice(0, -1), { ...last, text: last.text + e.payload.delta }];
        }
        return [...prev, { id: uid(), role: "assistant", text: e.payload.delta, streaming: true }];
      });
    }).then(u => cleanup.push(u));

    listen<{ name: string; args: string }>("agent-tool-call", e => {
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

    listen<{ name: string; result: string; ui?: ToolUi }>("agent-tool-result", e => {
      setMessages(prev => {
        // Find args for this tool call from the most recent streaming assistant message
        const streamingMsg = [...prev].reverse().find(m => m.role === "assistant" && m.streaming);
        const matchingCall = streamingMsg?.toolCalls?.find(tc => tc.name === e.payload.name);
        const closed = prev.map(m => m.streaming ? { ...m, streaming: false } : m);
        return [...closed, {
          id: uid(), role: "tool-result",
          text: e.payload.result,
          toolName: e.payload.name,
          toolArgs: matchingCall?.args,
          ui: e.payload.ui,
        }];
      });
    }).then(u => cleanup.push(u));

    // The step is being re-sampled: discard the partial text the failed attempt streamed,
    // otherwise the retry's tokens append to it.
    listen<{ step: number; attempt: number; error: string }>("agent-retry", () => {
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "assistant" && last.streaming) return prev.slice(0, -1);
        return prev;
      });
    }).then(u => cleanup.push(u));

    listen<{ error: string | null }>("agent-done", e => {
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
      setPermissionRequest({ code: e.payload.code });
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
    const enabledTools = ALL_BUILTIN_TOOLS.filter(t =>
      t.function.name === "run_python"
        ? runPythonMaster && effectiveEnabledTools.run_python !== false
        : effectiveEnabledTools[t.function.name] !== false
    );
    // Wiki memory: the active profile can override the global default; an unset profile
    // (undefined) inherits it.
    const wikiEnabled = (activeProfile?.wikiEnabled ?? settings.wikiEnabled) === true;
    if (wikiEnabled) enabledTools.push(...WIKI_TOOLS);

    // Split attachments into images (sent via Ollama images field) and other files (appended as paths)
    const imagePaths = attachedFiles.filter(isImage);
    const otherFiles = attachedFiles.filter(f => !isImage(f));

    const fullText = otherFiles.length > 0
      ? `${text}\n\nAttached files:\n${otherFiles.map(f => `- ${f}`).join("\n")}`
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
    setIsRunning(true);

    try {
      const allowedDirs = await invoke<string[]>("get_allowed_dirs").catch(() => [] as string[]);
      const basePrompt = activeProfile?.systemPrompt ?? BASE_SYSTEM_PROMPT;

      // Build dynamic suffix describing any registered external tools
      const registry = settings.toolRegistry;
      const ctxOpenAPI = activeProfile
        ? registry.openapiSpecs.filter(s => activeProfile.enabledOpenapiSpecIds.includes(s.id) && s.enabled !== false)
        : registry.openapiSpecs.filter(s => s.enabled !== false);
      const ctxMCP = activeProfile
        ? registry.mcpServers.filter(s => activeProfile.enabledMcpServerIds.includes(s.id))
        : registry.mcpServers;
      const ctxSparql = activeProfile
        ? registry.sparqlEndpoints.filter(s => (activeProfile.enabledSparqlEndpointIds ?? []).includes(s.id) && s.enabled !== false)
        : registry.sparqlEndpoints.filter(s => s.enabled !== false);
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
        ? `\nTOOL ROUTING: connected data tools are available — strongly prefer them over web_search whenever the user's request matches their topic, and only fall back to web_search for general open-web information they do not cover. ${externalParts.join(" ")}`
        : "";

      const resolved = resolveParams(chatParams);
      const effectiveBase = resolved.systemPromptOverride ?? basePrompt;

      // Build context vars block from the active profile
      const contextVars = activeProfile?.contextVars?.filter(v => v.name.trim() && v.value.trim()) ?? [];
      const contextVarsSuffix = contextVars.length > 0
        ? `\n\nUser context (treat these as facts about the user — use them automatically when relevant, do not repeat them unless asked):\n${contextVars.map(v => `- ${v.name}: ${v.value}`).join("\n")}`
        : "";

      const wikiSuffix = wikiEnabled ? WIKI_SYSTEM_PROMPT_BLOCK : "";

      const systemPrompt = allowedDirs.length > 0
        ? `${effectiveBase}${externalSuffix}${contextVarsSuffix}${wikiSuffix}\nThe user's configured folders are: ${allowedDirs.join(", ")}. Rules for file operations:\n- When reading or listing files without a specified path, use these folders immediately — do not ask for clarification.\n- When writing or saving a file without a specified path, save it to ${allowedDirs[0]} with a sensible filename derived from the content (e.g. sikhism_article.pdf). Never call write_file without a full absolute path.\n- Always use full absolute paths — never '.' or '~'.`
        : `${effectiveBase}${externalSuffix}${contextVarsSuffix}${wikiSuffix}`;

      // MCP servers this profile may use. With no active profile, all registered servers are
      // visible; the backend filters strictly by this list, so an empty list means none.
      const enabledMcpServerIds = activeProfile
        ? activeProfile.enabledMcpServerIds
        : registry.mcpServers.map(s => s.id);
      const disabledMcpTools = ctxMCP.flatMap(srv =>
        Object.entries(srv.enabledTools ?? {})
          .filter(([, en]) => !en)
          .map(([name]) => name)
      );

      await invoke("send_message", {
        args: {
          model: selectedModel,
          message: fullText,
          system_prompt: systemPrompt,
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
          keep_alive: resolved.keepAlive ?? null,
          web_search_results: settings.webSearchResults ?? 10,
          max_steps: settings.maxSteps ?? 20,
          disabled_mcp_tools: disabledMcpTools,
          enabled_mcp_server_ids: enabledMcpServerIds,
          max_tools: (activeProfile?.maxTools ?? settings.maxTools) || null,
          tool_result_limit: activeProfile?.toolResultLimit ?? null,
        }
      });
    } catch (err) {
      setIsRunning(false);
      // A failed agent run rejects here *and* emits agent-done with the same error — don't
      // render it twice. Errors thrown before the loop starts still surface.
      const text = String(err);
      setMessages(prev => {
        const last = prev[prev.length - 1];
        if (last?.role === "error" && last.text === text) return prev;
        return [...prev, { id: uid(), role: "error", text }];
      });
    }
  };

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
      openapi = registry.openapiSpecs;
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
      sparql = registry.sparqlEndpoints;
    }

    const host = profile?.host || s.host;
    const dirs = profile?.allowedDirs ?? s.allowedDirs ?? [];
    await invoke("set_mcp_servers",   { servers: mcp }).catch(() => {});
    await invoke("set_openapi_specs", { specs: openapi.filter(sp => sp.enabled !== false) }).catch(() => {});
    await invoke("set_sparql_endpoints", { endpoints: sparql.filter(ep => ep.enabled !== false) }).catch(() => {});
    await invoke("set_ollama_host",   { host }).catch(() => {});
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
    if (ap?.model && newSettings.models.includes(ap.model)) setSelectedModel(ap.model);
  };

  const handleProfileChange = async (id: string) => {
    const profile = settings.profiles.find(p => p.id === id) ?? null;
    const updated = { ...settings, activeProfileId: id || null };
    await handleSaveSettings(updated);
    if (profile?.model && settings.models.includes(profile.model)) setSelectedModel(profile.model);
    setChatParams(profile?.chatParams ?? updated.chatParams ?? DEFAULT_CHAT_PARAMS);
    await syncServers(updated);
    await handleReset();
  };

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
          models={settings.models}
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
            {messages.map(msg => {
              if (msg.role === "user")        return <UserMessage key={msg.id} text={msg.text} imageDataUrls={msg.imageDataUrls} />;
              if (msg.role === "assistant")   return <AssistantMessage key={msg.id} msg={msg} />;
              if (msg.role === "tool-result") return (
                <ToolResultRow
                  key={msg.id}
                  name={msg.toolName ?? ""}
                  result={msg.text}
                  args={msg.toolArgs}
                  ui={msg.ui}
                  onSend={send}
                  onAttach={(path, prompt) => { setAttachedFiles([path]); setInput(prompt); }}
                />
              );
              if (msg.role === "error")       return <div key={msg.id} className="msg-error">⚠ {msg.text}</div>;
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
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              disabled={settings.models.length === 0}
            >
              {settings.models.length === 0
                ? <option>No models found</option>
                : settings.models.map(m => <option key={m}>{m}</option>)
              }
            </select>
            <div className="input-spacer" />
            {isRunning ? (
              <button className="send-circle stop" onClick={() => { invoke("reset_conversation"); setIsRunning(false); }}>
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
            <div className="about-version">Version 2.0.3</div>

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
    </div>
  );
}
