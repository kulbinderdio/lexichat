import { useState, useEffect, useRef, useCallback, KeyboardEvent, ChangeEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Settings, RotateCcw, Bug, Paperclip, Info } from "lucide-react";
import lexiLogo from "./assets/lexi.png";
import { AdminPanel, AppSettings, Profile, StoredOpenAPISpec } from "./AdminPanel";
import { open } from "@tauri-apps/plugin-dialog";
import { openUrl } from "@tauri-apps/plugin-opener";
import { DebugPanel } from "./DebugPanel";
import "./App.css";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ToolCall { name: string; args: string; }

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool-result" | "error";
  text: string;
  streaming?: boolean;
  toolCalls?: ToolCall[];
  toolName?: string;
  toolArgs?: string;
}

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
- Use web_search for current events, weather, or live data.
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
  { type: "function", function: { name: "compose_email", description: "Build a base64url-encoded RFC 2822 email string for the Gmail API. Call this first, then pass the result as the 'raw' field to gmail_sendmessage.", parameters: { type: "object", properties: { to: { type: "string", description: "Recipient email address(es), comma-separated." }, from: { type: "string", description: "Sender email address (optional)." }, subject: { type: "string", description: "Email subject line." }, body: { type: "string", description: "Plain text email body." }, reply_to_message_id: { type: "string", description: "Message-ID to reply to, for threading (optional)." } }, required: ["to","subject","body"] } } },
];

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

// ── Settings ──────────────────────────────────────────────────────────────────

const DEFAULT_SETTINGS: AppSettings = {
  host: "http://localhost:11434",
  maxTools: 30,
  numGPULayers: null,
  models: [],
  enabledTools: { read_file: true, list_files: true, web_search: true },
  mcpServers: [],
  openapiSpecs: [],
  profiles: [],
  activeProfileId: null,
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

export function loadSettings(): AppSettings {
  try {
    const s = localStorage.getItem("lexi_settings");
    const loaded: AppSettings = s ? { ...DEFAULT_SETTINGS, ...JSON.parse(s) } : { ...DEFAULT_SETTINGS };
    // Inject built-in specs at global level and into every profile
    loaded.openapiSpecs = injectBuiltinSpecs(loaded.openapiSpecs);
    loaded.profiles = loaded.profiles.map(p => ({
      ...p,
      openapiSpecs: injectBuiltinSpecs(p.openapiSpecs ?? []),
    }));
    return loaded;
  } catch { return { ...DEFAULT_SETTINGS, openapiSpecs: [...BUILTIN_OPENAPI_SPECS] }; }
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

// ── Copy button ───────────────────────────────────────────────────────────────

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

function UserMessage({ text }: { text: string }) {
  return (
    <div className="msg-user">
      <div className="user-bubble">{text}</div>
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
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
            <span className="streaming-cursor" />
          </div>
        ) : (
          msg.text && (
            <div className="assistant-text">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
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
          <div><CopyButton text={msg.text} /></div>
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

const URL_REGEX = /https?:\/\/[^\s"'\]>),}]+/g;

function extractUrls(text: string): string[] {
  const matches = text.match(URL_REGEX) ?? [];
  // Deduplicate while preserving order
  return [...new Set(matches)];
}

function urlLabel(url: string): string {
  try {
    const u = new URL(url);
    // Show host + first path segment as a readable label
    const parts = u.pathname.split("/").filter(Boolean);
    return parts.length ? `${u.hostname} / ${decodeURIComponent(parts[0])}` : u.hostname;
  } catch { return url; }
}

function UrlListResult({ name, result }: { name: string; result: string }) {
  const [expanded, setExpanded] = useState(false);
  const urls = extractUrls(result);

  return (
    <div className="msg-tool-result">
      <div className="tool-result-inner" onClick={() => setExpanded(e => !e)} style={{ cursor: "pointer" }}>
        <span className="tool-result-check">✓</span>
        <span className="tool-result-name">{name}</span>
        <span className="tool-result-dot">·</span>
        <span className="tool-result-preview">{urls.length} link{urls.length !== 1 ? "s" : ""}</span>
        <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{expanded ? "▲" : "▼"}</span>
      </div>
      {expanded && (
        <div className="file-browser">
          {urls.map((url, i) => (
            <div key={i} className="file-browser-row">
              <span className="file-browser-icon">🔗</span>
              <span className="file-browser-name" style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={url}>
                {urlLabel(url)}
              </span>
              <div className="file-browser-actions">
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

function ToolResultRow({
  name, result, args, onSend, onAttach,
}: {
  name: string; result: string; args?: string;
  onSend: (text: string) => void;
  onAttach: (path: string, prompt: string) => void;
}) {
  if (FILE_LISTING_TOOLS.has(name)) {
    return <FileBrowserResult name={name} result={result} args={args} onSend={onSend} onAttach={onAttach} />;
  }
  const urls = extractUrls(result);
  if (urls.length > 0) {
    return <UrlListResult name={name} result={result} />;
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
  const [showAbout, setShowAbout] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Active profile derived from settings
  const activeProfile: Profile | null =
    settings.profiles.find(p => p.id === settings.activeProfileId) ?? null;

  // Fetch models and sync host on mount and when settings.host changes
  const fetchModels = useCallback(async () => {
    try {
      await invoke("set_ollama_host", { host: settings.host });
      const list = await invoke<string[]>("get_models");
      setSettings(prev => {
        const merged = [...list, ...prev.models.filter(m => !list.includes(m))];
        const updated = { ...prev, models: merged };
        saveSettings(updated);
        return updated;
      });
      setSelectedModel(m => m && list.includes(m) ? m : (list[0] ?? ""));
    } catch { /* Ollama not running */ }
  }, [settings.host]);

  useEffect(() => { fetchModels(); }, [settings.host]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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

    listen<{ name: string; result: string }>("agent-tool-result", e => {
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
        }];
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

    // Profile overrides global settings
    const effectiveEnabledTools = activeProfile?.enabledTools ?? settings.enabledTools;
    const enabledTools = ALL_BUILTIN_TOOLS.filter(
      t => effectiveEnabledTools[t.function.name] !== false
    );

    // Split attachments into images (sent via Ollama images field) and other files (appended as paths)
    const imagePaths = attachedFiles.filter(isImage);
    const otherFiles = attachedFiles.filter(f => !isImage(f));

    const fullText = otherFiles.length > 0
      ? `${text}\n\nAttached files:\n${otherFiles.map(f => `- ${f}`).join("\n")}`
      : text;

    // Build display text showing all attachments
    const displayText = attachedFiles.length > 0
      ? `${text}\n\nAttached: ${attachedFiles.map(f => f.split("/").pop()).join(", ")}`
      : text;

    setMessages(prev => [...prev, { id: uid(), role: "user", text: displayText }]);
    setAttachedFiles([]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setIsRunning(true);

    try {
      const allowedDirs = await invoke<string[]>("get_allowed_dirs").catch(() => [] as string[]);
      const basePrompt = activeProfile?.systemPrompt ?? BASE_SYSTEM_PROMPT;

      // Build dynamic suffix describing any registered external tools
      const ctx = activeProfile ?? settings;
      const ctxOpenAPI = Array.isArray((ctx as typeof settings).openapiSpecs) ? (ctx as typeof settings).openapiSpecs : [];
      const ctxMCP     = Array.isArray((ctx as typeof settings).mcpServers)   ? (ctx as typeof settings).mcpServers   : [];
      const externalParts: string[] = [];
      if (ctxOpenAPI.length > 0)
        externalParts.push(`OpenAPI services you can call: ${ctxOpenAPI.map(s => s.title).join(", ")}.`);
      if (ctxMCP.length > 0)
        externalParts.push(`MCP servers connected: ${ctxMCP.map(s => s.name).join(", ")}.`);
      const externalSuffix = externalParts.length > 0
        ? `\nExternal service tools available — call these ONLY when the user explicitly names that service: ${externalParts.join(" ")}`
        : "";

      const systemPrompt = allowedDirs.length > 0
        ? `${basePrompt}${externalSuffix}\nThe user's configured folders are: ${allowedDirs.join(", ")}. When the user asks about files or folders without specifying a path, immediately use list_files on these directories — do not ask for clarification. Always use full absolute paths.`
        : `${basePrompt}${externalSuffix}`;

      await invoke("send_message", {
        args: {
          model: selectedModel,
          message: fullText,
          system_prompt: systemPrompt,
          tools: enabledTools,
          image_paths: imagePaths,
        }
      });
    } catch (err) {
      setIsRunning(false);
      setMessages(prev => [...prev, { id: uid(), role: "error", text: String(err) }]);
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

  const handleReset = async () => {
    await invoke("reset_conversation");
    setMessages([]);
    setDebugClearKey(k => k + 1);
  };

  // Sync Rust's runtime MCP/OpenAPI state to whichever context is now active
  const syncServers = async (s: AppSettings) => {
    const ctx = s.profiles.find(p => p.id === s.activeProfileId) ?? s;
    const mcp     = (ctx as { mcpServers?: unknown }).mcpServers;
    const openapi = (ctx as { openapiSpecs?: unknown }).openapiSpecs;
    const enabledOpenapi = Array.isArray(openapi)
      ? (openapi as StoredOpenAPISpec[]).filter(s => s.enabled !== false)
      : [];
    await invoke("set_mcp_servers",   { servers: Array.isArray(mcp) ? mcp : [] }).catch(() => {});
    await invoke("set_openapi_specs", { specs: enabledOpenapi }).catch(() => {});
  };

  // Sync on first mount
  useEffect(() => { syncServers(settings); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSaveSettings = async (newSettings: AppSettings) => {
    saveSettings(newSettings);
    setSettings(newSettings);
    await invoke("set_ollama_host", { host: newSettings.host });
    const ap = newSettings.profiles.find(p => p.id === newSettings.activeProfileId);
    if (ap?.model && newSettings.models.includes(ap.model)) setSelectedModel(ap.model);
  };

  const handleProfileChange = async (id: string) => {
    const profile = settings.profiles.find(p => p.id === id) ?? null;
    const updated = { ...settings, activeProfileId: id || null };
    await handleSaveSettings(updated);
    if (profile?.model && settings.models.includes(profile.model)) setSelectedModel(profile.model);
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
      </div>

      {/* Main content: chat + optional debug panel */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
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
              if (msg.role === "user")        return <UserMessage key={msg.id} text={msg.text} />;
              if (msg.role === "assistant")   return <AssistantMessage key={msg.id} msg={msg} />;
              if (msg.role === "tool-result") return (
                <ToolResultRow
                  key={msg.id}
                  name={msg.toolName ?? ""}
                  result={msg.text}
                  args={msg.toolArgs}
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
            <div className="about-version">Version 0.1.0</div>
            <button className="btn primary" style={{ marginTop: 8 }} onClick={() => setShowAbout(false)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
