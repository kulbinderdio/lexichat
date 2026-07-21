import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { buildExportEnvelope, parseImport, mergeImport } from "./profileIO";
import { ChatParams, DEFAULT_CHAT_PARAMS, ChatParamsDefaults, AdvancedParamsContent } from "./ChatParamsPanel";

// ── Types ─────────────────────────────────────────────────────────────────────

export type AuthType = "none" | "bearer" | "apikey" | "basic" | "oauth2";

export interface AuthConfig {
  type: AuthType;
  bearer_token?: string;
  api_key_header?: string;
  api_key_value?: string;
  basic_username?: string;
  basic_password?: string;
  token_url?: string;
  client_id?: string;
  client_secret?: string;
  scope?: string;
  authorization_url?: string;
  access_token?: string;
  refresh_token?: string;
}

export const DEFAULT_AUTH: AuthConfig = { type: "none" };

export interface StoredMCPServer {
  id: string;
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  auth?: AuthConfig;
  enabledTools?: Record<string, boolean>;  // prefixed tool name → enabled; absent = enabled
  enable_apps?: boolean;                    // allow this server's tools to render MCP-App UIs
}

export interface StoredOpenAPISpec {
  id: string;
  title: string;
  base_url: string;
  spec_json: string;
  auth?: AuthConfig;
  enabled?: boolean;
}

// IDs of built-in specs that ship with the app (user can disable but not delete)
export const BUILTIN_OPENAPI_SPEC_IDS = new Set(["builtin-wikipedia"]);

export interface SparqlExampleQuery {
  label: string;
  query: string;
}

export interface StoredSparqlEndpoint {
  id: string;
  title: string;
  endpoint_url: string;
  prefixes?: string;
  schema_summary?: string;
  example_queries?: SparqlExampleQuery[];
  usage_hint?: string;
  auth?: AuthConfig;
  enabled?: boolean;
  read_only?: boolean;
}

// IDs of built-in SPARQL endpoints that ship with the app (user can disable but not delete)
export const BUILTIN_SPARQL_ENDPOINT_IDS = new Set([
  "builtin-landregistry",
  "builtin-opendatacommunities",
]);

export interface ContextVar {
  name: string;
  value: string;
}

export interface ToolRegistry {
  mcpServers: StoredMCPServer[];
  openapiSpecs: StoredOpenAPISpec[];
  sparqlEndpoints: StoredSparqlEndpoint[];
}

export interface Profile {
  id: string;
  name: string;
  systemPrompt: string;
  model: string;
  enabledTools: Record<string, boolean>;
  // Registry-based tool selection (replaces per-profile mcpServers/openapiSpecs)
  enabledMcpServerIds: string[];
  enabledOpenapiSpecIds: string[];
  enabledSparqlEndpointIds: string[];
  toolAuthOverrides?: Record<string, AuthConfig>; // tool id → auth override for this profile
  // Legacy fields kept for migration only — will be undefined after first load
  mcpServers?: StoredMCPServer[];
  openapiSpecs?: StoredOpenAPISpec[];
  maxTools: number;
  chatParams?: ChatParams;
  contextVars?: ContextVar[];
  // Which registered server this profile's default model lives on. undefined = resolve by model.
  serverId?: string;
  host?: string; // legacy (pre-registry per-profile host override)
  allowedDirs?: string[];
  // Per-profile override for wiki memory. undefined = follow the global default.
  wikiEnabled?: boolean;
  // Max chars of a tool result fed back to the model. undefined = default (6000).
  toolResultLimit?: number;
}

export type ProviderKind = "ollama" | "openai";

// A configured inference server. Models from every server are shown together in the model
// dropdown (prefixed with `name`), and each chat is routed to its model's server.
export interface ServerConfig {
  id: string;
  name: string;               // short label; used as the dropdown prefix
  provider: ProviderKind;
  baseUrl: string;
  apiKey?: string;            // OpenAI-compatible only
  // Exactly what this endpoint returned on the last successful fetch — REPLACED each refresh, so
  // repointing the server at a different endpoint doesn't leave stale models behind. `undefined`
  // means "never fetched yet" (drives first-fetch auto-enable).
  catalog?: string[];
  // Models the user typed by hand (for endpoints that don't enumerate, e.g. Anthropic). Kept
  // separate from `catalog` so they survive a refetch. The Models-tab list = catalog ∪ manual.
  manualModels?: string[];
  // The curated subset actually shown in the chat dropdown. Small pools auto-enable all
  // (non-embedding); large ones start empty so the user picks. `undefined` = not curated yet.
  models?: string[];
}

// Above this many chat models in a freshly-fetched catalog, don't auto-enable — make the user
// pick (otherwise a provider like OpenRouter dumps 300+ into the dropdown).
export const AUTO_ENABLE_MAX = 20;

// Heuristic: models that can't chat (embeddings / rerankers) so they're excluded from auto-enable
// and flagged in the Models tab. Name-based so it works across providers; the user can override.
export function isEmbeddingModel(name: string): boolean {
  return /(^|[-_/.])(embed|embedding|rerank|reranker|bge|gte|minilm|e5)\d*([-_/.]|$)|text-embedding|nomic-embed|mxbai-embed/i.test(name);
}

/// The full model pool for a server = what it returned last (`catalog`) ∪ hand-typed entries.
export function serverModelPool(s: ServerConfig): string[] {
  return [...new Set([...(s.catalog ?? []), ...(s.manualModels ?? [])])];
}

/// Fold a freshly-fetched model list into a server. `catalog` is REPLACED with exactly what the
/// endpoint returned (so repointing the server drops the old endpoint's models); hand-typed
/// entries in `manualModels` are preserved. First fetch auto-enables a small chat set (or none if
/// large); later fetches keep the user's curated set, pruned to what still exists. An empty fetch
/// is a no-op (an unreachable endpoint doesn't wipe the current list).
export function reconcileCatalog(s: ServerConfig, fetched: string[]): ServerConfig {
  if (fetched.length === 0) return s;
  const pool = [...new Set([...fetched, ...(s.manualModels ?? [])])];
  let models: string[];
  if (s.catalog === undefined) {
    const chat = pool.filter(m => !isEmbeddingModel(m));
    models = chat.length <= AUTO_ENABLE_MAX ? chat : [];
  } else {
    models = (s.models ?? []).filter(m => pool.includes(m));
  }
  return { ...s, catalog: fetched, models };
}

export interface AppSettings {
  servers: ServerConfig[];
  // Legacy single-backend fields — kept for one-time migration into `servers`.
  host: string;
  provider?: ProviderKind;
  apiKey?: string;
  maxTools: number;
  webSearchResults: number;
  maxSteps: number;
  models: string[];
  enabledTools: Record<string, boolean>;
  toolRegistry: ToolRegistry;
  profiles: Profile[];
  activeProfileId: string | null;
  chatParams?: ChatParams;
  allowedDirs?: string[];
  wikiEnabled?: boolean;
  // Legacy fields kept for migration only — will be undefined after first load
  mcpServers?: StoredMCPServer[];
  openapiSpecs?: StoredOpenAPISpec[];
}

interface SpecInfo {
  id: string;
  title: string;
  base_url: string;
  tool_count: number;
  tools: { name: string; description: string; method: string; path: string }[];
}

interface SparqlInfo {
  id: string;
  title: string;
  endpoint_url: string;
  tool_count: number;
  tools: string[];
}

interface SparqlDiscovery {
  live: boolean;
  message: string;
  suggested_prefixes: string;
  suggested_schema: string;
}

interface MCPServerInfo {
  id: string;
  name: string;
  command: string;
  args: string[];
  connected: boolean;
  tool_count: number;
  tools: { name: string; description: string; has_ui?: boolean }[];
  error?: string;
  enable_apps?: boolean;
}

interface Props {
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
  onClose: () => void;
}

type Tab = "profiles" | "tools" | "models" | "openapi" | "sparql" | "mcp" | "sandbox" | "server" | "defaults";

const BUILTIN_TOOLS = [
  { name: "list_files",          label: "List Files",           icon: "📁" },
  { name: "read_file",           label: "Read File",            icon: "📄" },
  { name: "write_file",          label: "Write File",           icon: "✏️" },
  { name: "search_files",        label: "Search Files",         icon: "🔎" },
  { name: "search_in_files",     label: "Search In Files",      icon: "🔍" },
  { name: "get_file_info",       label: "Get File Info",        icon: "ℹ️" },
  { name: "list_directory_tree", label: "Directory Tree",       icon: "🌳" },
  { name: "create_directory",    label: "Create Directory",     icon: "📂" },
  { name: "move_file",           label: "Move / Rename File",   icon: "↕️" },
  { name: "delete_file",         label: "Delete File",          icon: "🗑️" },
  { name: "find_old_files",      label: "Find Old Files",       icon: "🗂️" },
  { name: "web_search",          label: "Web Search",           icon: "🌐" },
  { name: "get_current_datetime", label: "Get Date / Time",      icon: "🕐" },
  { name: "run_python",          label: "Run Python (Code Sandbox)", icon: "🐍" },
];

// Tools that are opt-in: disabled unless the user explicitly turns them on.
const OPT_IN_TOOLS = new Set(["run_python"]);
const toolEnabled = (enabled: Record<string, boolean>, name: string): boolean =>
  OPT_IN_TOOLS.has(name) ? enabled[name] === true : enabled[name] !== false;

const DEFAULT_SYSTEM_PROMPT = `You are Lexi, a personal AI assistant running locally for a single authorised user.
You have tools to read local files and search the web. Be proactive — use tools immediately rather than asking the user for paths or clarification.
Rules:
- When asked about files or folders, call list_files or list_directory_tree right away using any path the user mentioned, or the configured folders if none was given.
- Always use full absolute paths — never '.' or '~'.
- Use web_search for current events, weather, or live data.
- ALWAYS write a helpful text response after using tools — summarise what you found, list the results, or answer the user's question directly. Never leave the chat blank after a tool call.
- If asked about your own tools, capabilities, or what you can do, answer directly from your knowledge — do not call any tools to answer this question.
- NEVER call read_file on image files (.jpg, .jpeg, .png, .gif, .webp, .bmp, etc.). Images are sent directly in the message via the vision API — describe them from what you can see. If no image is attached, tell the user to use the paperclip button to attach it.`;

const uid = () => Math.random().toString(36).slice(2);

// ── Auth config form ──────────────────────────────────────────────────────────

function AuthConfigForm({ auth, onChange }: { auth: AuthConfig; onChange: (a: AuthConfig) => void }) {
  const [authorizing, setAuthorizing] = useState(false);
  const [authError, setAuthError] = useState("");

  const inp = (label: string, value: string, field: keyof AuthConfig, placeholder?: string, isPassword?: boolean) => (
    <div className="field" style={{ marginTop: 6 }}>
      <label style={{ fontSize: 11 }}>{label}</label>
      <input
        className="admin-input"
        type={isPassword ? "password" : "text"}
        value={value}
        onChange={e => onChange({ ...auth, [field]: e.target.value })}
        placeholder={placeholder}
        style={{ fontFamily: "monospace", fontSize: 11 }}
      />
    </div>
  );

  const authorize = async () => {
    setAuthorizing(true);
    setAuthError("");
    try {
      const result = await invoke<{ access_token: string; refresh_token: string }>("oauth2_authorize", {
        args: {
          authorization_url: auth.authorization_url ?? "",
          token_url: auth.token_url ?? "",
          client_id: auth.client_id ?? "",
          client_secret: auth.client_secret ?? "",
          scope: auth.scope ?? "",
        },
      });
      onChange({ ...auth, access_token: result.access_token, refresh_token: result.refresh_token });
    } catch (e) {
      setAuthError(String(e));
    }
    setAuthorizing(false);
  };

  const isAuthorized = auth.type === "oauth2" && !!auth.access_token;

  return (
    <div className="field" style={{ marginTop: 6 }}>
      <label>Authentication</label>
      <select
        className="admin-input"
        value={auth.type}
        onChange={e => onChange({ ...DEFAULT_AUTH, type: e.target.value as AuthType })}
        style={{ marginTop: 4 }}
      >
        <option value="none">None</option>
        <option value="bearer">Bearer Token</option>
        <option value="apikey">API Key (custom header)</option>
        <option value="basic">Basic Auth (username / password)</option>
        <option value="oauth2">OAuth2</option>
      </select>

      {auth.type === "bearer" && inp("Token", auth.bearer_token ?? "", "bearer_token", "sk-...")}

      {auth.type === "apikey" && <>
        {inp("Header name", auth.api_key_header ?? "", "api_key_header", "X-API-Key")}
        {inp("Header value", auth.api_key_value ?? "", "api_key_value", "your-api-key")}
      </>}

      {auth.type === "basic" && <>
        {inp("Username", auth.basic_username ?? "", "basic_username", "username")}
        {inp("Password", auth.basic_password ?? "", "basic_password", "password", true)}
      </>}

      {auth.type === "oauth2" && <>
        {inp("Authorization URL", auth.authorization_url ?? "", "authorization_url", "https://accounts.google.com/o/oauth2/auth")}
        {inp("Token URL", auth.token_url ?? "", "token_url", "https://oauth2.googleapis.com/token")}
        {inp("Client ID", auth.client_id ?? "", "client_id", "your-client-id")}
        {inp("Client Secret", auth.client_secret ?? "", "client_secret", "your-client-secret", true)}
        {inp("Scope", auth.scope ?? "", "scope", "https://www.googleapis.com/auth/drive.readonly")}

        <div style={{ marginTop: 8, display: "flex", alignItems: "center", gap: 10 }}>
          <button
            className="btn primary"
            onClick={authorize}
            disabled={authorizing || !auth.authorization_url || !auth.token_url || !auth.client_id}
          >
            {authorizing ? "Waiting for browser…" : "Authorize in Browser"}
          </button>
          {isAuthorized && (
            <span style={{ fontSize: 11, color: "#4ade80" }}>✓ Authorized</span>
          )}
          {!isAuthorized && !authorizing && auth.client_id && (
            <span style={{ fontSize: 11, opacity: 0.45 }}>Not yet authorized</span>
          )}
        </div>
        {isAuthorized && (
          <button className="link-btn" style={{ marginTop: 4, fontSize: 10 }}
            onClick={() => onChange({ ...auth, access_token: "", refresh_token: "" })}>
            Revoke / re-authorize
          </button>
        )}
        {authError && <div style={{ color: "#f87171", fontSize: 11, marginTop: 4 }}>{authError}</div>}
      </>}
    </div>
  );
}

// ── Profiles tab ──────────────────────────────────────────────────────────────

function ProfilesTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const [selectedId, setSelectedId] = useState<string | null>(
    settings.profiles[0]?.id ?? null
  );
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<Profile | null>(null);

  const selected = settings.profiles.find(p => p.id === selectedId) ?? null;

  const newProfile = () => {
    const firstServer = (settings.servers ?? []).find(s => (s.models ?? []).length > 0);
    const p: Profile = {
      id: uid(),
      name: "New Profile",
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      model: firstServer?.models?.[0] ?? "",
      serverId: firstServer?.id,
      enabledTools: { ...settings.enabledTools },
      enabledMcpServerIds: [],
      enabledOpenapiSpecIds: [],
      enabledSparqlEndpointIds: [],
      maxTools: settings.maxTools,
    };
    setDraft(p);
    setEditing(true);
    setSelectedId(p.id);
  };

  const editProfile = (p: Profile) => {
    setDraft({ ...p });
    setEditing(true);
    setSelectedId(p.id);
  };

  const saveProfile = () => {
    if (!draft) return;
    const exists = settings.profiles.some(p => p.id === draft.id);
    const profiles = exists
      ? settings.profiles.map(p => p.id === draft.id ? draft : p)
      : [...settings.profiles, draft];
    onChange({ ...settings, profiles });
    setEditing(false);
  };

  const cancelEdit = () => {
    setEditing(false);
    setDraft(null);
    if (!settings.profiles.find(p => p.id === selectedId)) {
      setSelectedId(settings.profiles[0]?.id ?? null);
    }
  };

  const deleteProfile = (id: string) => {
    const profiles = settings.profiles.filter(p => p.id !== id);
    const activeProfileId = settings.activeProfileId === id ? null : settings.activeProfileId;
    onChange({ ...settings, profiles, activeProfileId });
    setSelectedId(profiles[0]?.id ?? null);
    setEditing(false);
    setDraft(null);
  };

  const setActive = (id: string | null) => {
    onChange({ ...settings, activeProfileId: id });
  };

  const [importError, setImportError] = useState("");
  const [importSummary, setImportSummary] = useState<string[]>([]);

  const exportProfile = async (profile: Profile) => {
    const envelope = buildExportEnvelope(profile, settings.toolRegistry);
    const json = JSON.stringify(envelope, null, 2);
    const defaultName = profile.name.replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
    const path = await save({
      title: "Export Profile",
      defaultPath: `${defaultName}.json`,
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!path) return;
    await invoke("write_file_text", { path, content: json });
  };

  const importProfile = async () => {
    setImportError(""); setImportSummary([]);
    const path = await open({ multiple: false, title: "Import Profile", filters: [{ name: "JSON", extensions: ["json"] }] });
    if (typeof path !== "string") return;
    let raw: string;
    try { raw = await invoke<string>("read_file_text", { path }); } catch { setImportError("Could not read file."); return; }
    let parsed: unknown;
    try { parsed = JSON.parse(raw); } catch { setImportError("File is not valid JSON."); return; }
    const bundle = parseImport(parsed);
    if (!bundle) { setImportError("Not a valid LexiChat profile file."); return; }
    // Bundles the profile's APIs into the registry, re-links, and reports what still needs setup.
    const result = mergeImport(bundle, settings, uid());
    onChange(result.settings);
    setSelectedId(result.profileId);
    setImportSummary(result.warnings);
  };

  const d = draft;

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* Profile list */}
      <div style={{
        width: 180, flexShrink: 0, borderRight: "1px solid var(--border)",
        display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        <div style={{ flex: 1, overflowY: "auto" }}>
          {settings.profiles.length === 0 && (
            <div className="admin-empty" style={{ padding: "16px 12px", fontSize: 11 }}>
              No profiles yet. Create one to save a custom assistant configuration.
            </div>
          )}
          {settings.profiles.map(p => (
            <div
              key={p.id}
              className={`profile-list-item ${selectedId === p.id ? "active" : ""}`}
              onClick={() => { setSelectedId(p.id); setEditing(false); setDraft(null); }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: 13 }}>🤖</span>
                <span style={{ flex: 1, fontSize: 12, fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {p.name}
                </span>
                {settings.activeProfileId === p.id && (
                  <span style={{ fontSize: 9, color: "#4ade80", fontWeight: 700 }}>●</span>
                )}
              </div>
              {p.model && (
                <div style={{ fontSize: 10, opacity: 0.45, fontFamily: "monospace", marginTop: 2, paddingLeft: 19 }}>
                  {p.model.split(":")[0]}
                </div>
              )}
            </div>
          ))}
        </div>
        <div style={{ padding: 8, borderTop: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 6 }}>
          <button className="btn primary" style={{ width: "100%", fontSize: 11 }} onClick={newProfile}>
            + New Profile
          </button>
          <button className="btn" style={{ width: "100%", fontSize: 11 }} onClick={importProfile}>
            Import Profile
          </button>
          {importError && <div style={{ color: "#f87171", fontSize: 11 }}>{importError}</div>}
          {importSummary.length > 0 && (
            <div style={{ marginTop: 6, padding: "6px 8px", borderRadius: 6, fontSize: 10,
              background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.35)" }}>
              <div style={{ fontWeight: 600, marginBottom: 2 }}>Imported — a few things to finish:</div>
              <ul style={{ margin: 0, paddingLeft: 14 }}>
                {importSummary.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Detail / edit pane */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {!selected && !editing && (
          <div className="admin-empty" style={{ margin: "auto" }}>Select a profile or create a new one.</div>
        )}

        {(selected || editing) && !editing && (
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700 }}>{selected!.name}</div>
                {selected!.model && <div style={{ fontSize: 11, opacity: 0.5, fontFamily: "monospace", marginTop: 2 }}>{selected!.model}</div>}
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button className="btn" onClick={() => exportProfile(selected!)}>Export</button>
                <button className="btn" onClick={() => editProfile(selected!)}>Edit</button>
                <button
                  className="btn primary"
                  style={{ background: settings.activeProfileId === selected!.id ? "#4ade8033" : undefined,
                    borderColor: settings.activeProfileId === selected!.id ? "#4ade80" : undefined }}
                  onClick={() => setActive(settings.activeProfileId === selected!.id ? null : selected!.id)}
                >
                  {settings.activeProfileId === selected!.id ? "● Active" : "Set Active"}
                </button>
              </div>
            </div>

            <div className="field" style={{ marginBottom: 14 }}>
              <label>System Prompt</label>
              <div style={{
                background: "var(--surface2)", borderRadius: 6, padding: "8px 10px",
                fontSize: 11, fontFamily: "monospace", whiteSpace: "pre-wrap",
                lineHeight: 1.5, maxHeight: 160, overflowY: "auto", opacity: 0.8,
              }}>
                {selected!.systemPrompt}
              </div>
            </div>

            <div className="field" style={{ marginBottom: 14 }}>
              <label>Enabled Tools</label>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                {BUILTIN_TOOLS.filter(t => selected!.enabledTools[t.name] !== false).map(t => (
                  <span key={t.name} style={{
                    fontSize: 10, padding: "2px 7px", borderRadius: 10,
                    background: "var(--purple-bg)", border: "1px solid var(--purple-border)", color: "var(--purple)"
                  }}>{t.icon} {t.label}</span>
                ))}
              </div>
            </div>

            {settings.toolRegistry.openapiSpecs.filter(s => !BUILTIN_OPENAPI_SPEC_IDS.has(s.id)).length > 0 && (
              <div className="field" style={{ marginBottom: 14 }}>
                <label>OpenAPI Services</label>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {settings.toolRegistry.openapiSpecs
                    .filter(s => selected!.enabledOpenapiSpecIds.includes(s.id))
                    .map(s => (
                      <span key={s.id} style={{
                        fontSize: 10, padding: "2px 7px", borderRadius: 10,
                        background: "var(--purple-bg)", border: "1px solid var(--purple-border)", color: "var(--purple)"
                      }}>🌐 {s.title}</span>
                    ))}
                  {settings.toolRegistry.openapiSpecs.filter(s => selected!.enabledOpenapiSpecIds.includes(s.id)).length === 0 && (
                    <span style={{ fontSize: 11, opacity: 0.5 }}>None selected</span>
                  )}
                </div>
              </div>
            )}

            {settings.toolRegistry.sparqlEndpoints.length > 0 && (
              <div className="field" style={{ marginBottom: 14 }}>
                <label>SPARQL Endpoints</label>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {settings.toolRegistry.sparqlEndpoints
                    .filter(s => (selected!.enabledSparqlEndpointIds ?? []).includes(s.id))
                    .map(s => (
                      <span key={s.id} style={{
                        fontSize: 10, padding: "2px 7px", borderRadius: 10,
                        background: "var(--purple-bg)", border: "1px solid var(--purple-border)", color: "var(--purple)"
                      }}>🔗 {s.title}</span>
                    ))}
                  {(selected!.enabledSparqlEndpointIds ?? []).filter(id => settings.toolRegistry.sparqlEndpoints.some(s => s.id === id)).length === 0 && (
                    <span style={{ fontSize: 11, opacity: 0.5 }}>None selected</span>
                  )}
                </div>
              </div>
            )}

            {settings.toolRegistry.mcpServers.length > 0 && (
              <div className="field" style={{ marginBottom: 14 }}>
                <label>MCP Servers</label>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {settings.toolRegistry.mcpServers
                    .filter(s => selected!.enabledMcpServerIds.includes(s.id))
                    .map(s => (
                      <span key={s.id} style={{
                        fontSize: 10, padding: "2px 7px", borderRadius: 10,
                        background: "var(--purple-bg)", border: "1px solid var(--purple-border)", color: "var(--purple)"
                      }}>🔌 {s.name}</span>
                    ))}
                  {settings.toolRegistry.mcpServers.filter(s => selected!.enabledMcpServerIds.includes(s.id)).length === 0 && (
                    <span style={{ fontSize: 11, opacity: 0.5 }}>None selected</span>
                  )}
                </div>
              </div>
            )}

            <button className="btn" style={{ color: "#f87171", borderColor: "#f8717133", marginTop: 8 }}
              onClick={() => deleteProfile(selected!.id)}>
              Delete Profile
            </button>
          </div>
        )}

        {editing && d && (
          <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
            <div className="field" style={{ marginBottom: 12 }}>
              <label>Profile Name</label>
              <input className="admin-input" value={d.name}
                onChange={e => setDraft({ ...d, name: e.target.value })}
                placeholder="e.g. Code Assistant" />
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>System Prompt</label>
              <textarea className="admin-input" value={d.systemPrompt}
                onChange={e => setDraft({ ...d, systemPrompt: e.target.value })}
                rows={7} style={{ fontFamily: "monospace", fontSize: 11, resize: "vertical" }} />
              <button className="link-btn" style={{ marginTop: 4 }}
                onClick={() => setDraft({ ...d, systemPrompt: DEFAULT_SYSTEM_PROMPT })}>
                Reset to default
              </button>
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Default Model</label>
              <select className="admin-input"
                value={d.model ? `${d.serverId ?? ""}${d.model}` : ""}
                onChange={e => {
                  const v = e.target.value;
                  if (!v) { setDraft({ ...d, model: "", serverId: undefined }); return; }
                  const i = v.indexOf("");
                  setDraft({ ...d, serverId: v.slice(0, i), model: v.slice(i + 1) });
                }}>
                <option value="">— pick when chatting —</option>
                {(settings.servers ?? []).map(s => {
                  const ms = s.models ?? [];
                  if (ms.length === 0) return null;
                  return (
                    <optgroup key={s.id} label={s.name}>
                      {ms.map(m => <option key={s.id + m} value={`${s.id}${m}`}>{s.name} / {m}</option>)}
                    </optgroup>
                  );
                })}
              </select>
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Tools</label>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 4 }}>
                {BUILTIN_TOOLS.map(t => (
                  <label key={t.name} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12 }}>
                    <input type="checkbox"
                      checked={d.enabledTools[t.name] !== false}
                      onChange={e => setDraft({ ...d, enabledTools: { ...d.enabledTools, [t.name]: e.target.checked } })}
                      className="admin-checkbox" />
                    <span>{t.icon}</span>
                    <span>{t.label}</span>
                  </label>
                ))}
              </div>
            </div>

            {settings.toolRegistry.openapiSpecs.length > 0 && (
              <div className="field" style={{ marginBottom: 12 }}>
                <label>OpenAPI Services</label>
                <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
                  Select which services this profile can access.
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {settings.toolRegistry.openapiSpecs.map(sp => (
                    <label key={sp.id} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12 }}>
                      <input type="checkbox" className="admin-checkbox"
                        checked={d.enabledOpenapiSpecIds.includes(sp.id)}
                        onChange={e => setDraft({ ...d, enabledOpenapiSpecIds: e.target.checked
                          ? [...d.enabledOpenapiSpecIds, sp.id]
                          : d.enabledOpenapiSpecIds.filter(id => id !== sp.id) })} />
                      <span>🌐 {sp.title}</span>
                      {BUILTIN_OPENAPI_SPEC_IDS.has(sp.id) && (
                        <span style={{ fontSize: 10, opacity: 0.4 }}>(built-in)</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            )}

            {settings.toolRegistry.sparqlEndpoints.length > 0 && (
              <div className="field" style={{ marginBottom: 12 }}>
                <label>SPARQL Endpoints</label>
                <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
                  Select which linked-data endpoints this profile can query.
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {settings.toolRegistry.sparqlEndpoints.map(sp => (
                    <label key={sp.id} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12 }}>
                      <input type="checkbox" className="admin-checkbox"
                        checked={(d.enabledSparqlEndpointIds ?? []).includes(sp.id)}
                        onChange={e => setDraft({ ...d, enabledSparqlEndpointIds: e.target.checked
                          ? [...(d.enabledSparqlEndpointIds ?? []), sp.id]
                          : (d.enabledSparqlEndpointIds ?? []).filter(id => id !== sp.id) })} />
                      <span>🔗 {sp.title}</span>
                      {BUILTIN_SPARQL_ENDPOINT_IDS.has(sp.id) && (
                        <span style={{ fontSize: 10, opacity: 0.4 }}>(built-in)</span>
                      )}
                    </label>
                  ))}
                </div>
              </div>
            )}

            {settings.toolRegistry.mcpServers.length > 0 && (
              <div className="field" style={{ marginBottom: 12 }}>
                <label>MCP Servers</label>
                <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
                  Select which MCP servers this profile can use.
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {settings.toolRegistry.mcpServers.map(srv => (
                    <label key={srv.id} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 12 }}>
                      <input type="checkbox" className="admin-checkbox"
                        checked={d.enabledMcpServerIds.includes(srv.id)}
                        onChange={e => setDraft({ ...d, enabledMcpServerIds: e.target.checked
                          ? [...d.enabledMcpServerIds, srv.id]
                          : d.enabledMcpServerIds.filter(id => id !== srv.id) })} />
                      <span>🔌 {srv.name}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Max Tools per Query</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                <button className="stepper-btn" onClick={() => setDraft({ ...d, maxTools: Math.max(5, d.maxTools - 5) })}>−</button>
                <span style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 600, minWidth: 28, textAlign: "center" }}>{d.maxTools}</span>
                <button className="stepper-btn" onClick={() => setDraft({ ...d, maxTools: Math.min(100, d.maxTools + 5) })}>+</button>
              </div>
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" className="admin-checkbox"
                  checked={(d.wikiEnabled ?? settings.wikiEnabled) === true}
                  onChange={e => setDraft({ ...d, wikiEnabled: e.target.checked })} />
                <span>🧠 Persistent Wiki Memory</span>
              </label>
              <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 4 }}>
                Give this profile the wiki tools to store and recall knowledge across chats.
                {d.wikiEnabled == null && " Currently following the global default."}
              </div>
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Tool Result Limit (characters)</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                <button className="stepper-btn"
                  onClick={() => setDraft({ ...d, toolResultLimit: Math.max(2000, (d.toolResultLimit ?? 6000) - 2000) })}>−</button>
                <span style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 600, minWidth: 52, textAlign: "center" }}>{d.toolResultLimit ?? 6000}</span>
                <button className="stepper-btn"
                  onClick={() => setDraft({ ...d, toolResultLimit: Math.min(50000, (d.toolResultLimit ?? 6000) + 2000) })}>+</button>
              </div>
              <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginTop: 4 }}>
                How much of each tool/API response the model sees before it's truncated. Raise it for
                data-heavy APIs that return large JSON (e.g. Parliament membership lists); lower it to
                save context. Default 6000.
              </div>
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Chat Defaults</label>
              <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 8 }}>
                New chats in this profile start with these settings (can still be changed per chat).
              </div>
              <ChatParamsDefaults
                params={d.chatParams ?? DEFAULT_CHAT_PARAMS}
                onChange={cp => setDraft({ ...d, chatParams: cp })}
              />
              {d.chatParams && (
                <button className="link-btn" style={{ marginTop: 4 }}
                  onClick={() => setDraft({ ...d, chatParams: undefined })}>
                  Reset to global defaults
                </button>
              )}
            </div>

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Context Variables</label>
              <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 8, lineHeight: 1.5 }}>
                Name/value pairs injected into every conversation as facts about you. The AI uses them automatically when relevant — e.g. <em>location</em>, <em>timezone</em>, <em>name</em>, <em>occupation</em>.
              </div>
              <ContextVarsEditor
                vars={d.contextVars ?? []}
                onChange={vars => setDraft({ ...d, contextVars: vars })}
              />
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={cancelEdit}>Cancel</button>
              <button className="btn primary" onClick={saveProfile} disabled={!d.name.trim()}>Save Profile</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Context Vars Editor ───────────────────────────────────────────────────────

const SUGGESTED_VARS = [
  { name: "name",         placeholder: "e.g. Alex" },
  { name: "location",     placeholder: "e.g. Gravesend, UK" },
  { name: "timezone",     placeholder: "e.g. Europe/London" },
  { name: "language",     placeholder: "e.g. British English" },
  { name: "units",        placeholder: "e.g. metric" },
  { name: "currency",     placeholder: "e.g. GBP" },
  { name: "date_format",  placeholder: "e.g. DD/MM/YYYY" },
  { name: "occupation",   placeholder: "e.g. Software Developer" },
  { name: "expertise",    placeholder: "e.g. expert / intermediate / beginner" },
  { name: "os",           placeholder: "e.g. macOS Sequoia" },
  { name: "stack",        placeholder: "e.g. Rust, TypeScript, React" },
  { name: "current_project", placeholder: "e.g. LexiChat — Tauri/Rust AI app" },
  { name: "company",      placeholder: "e.g. Acme Ltd" },
  { name: "industry",     placeholder: "e.g. Technology" },
  { name: "writing_tone", placeholder: "e.g. professional but friendly" },
  { name: "interests",    placeholder: "e.g. history, cycling, jazz" },
];

function ContextVarsEditor({ vars, onChange }: { vars: ContextVar[]; onChange: (v: ContextVar[]) => void }) {
  const [showSuggestions, setShowSuggestions] = useState(false);

  const update = (i: number, field: "name" | "value", val: string) => {
    const next = vars.map((v, idx) => idx === i ? { ...v, [field]: val } : v);
    onChange(next);
  };

  const remove = (i: number) => onChange(vars.filter((_, idx) => idx !== i));

  const add = (name = "", value = "") => onChange([...vars, { name, value }]);

  const addSuggestion = (name: string) => {
    if (!vars.some(v => v.name === name)) add(name, "");
    setShowSuggestions(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {vars.map((v, i) => (
        <div key={i} style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <input
            className="admin-input"
            value={v.name}
            onChange={e => update(i, "name", e.target.value)}
            placeholder="name"
            style={{ width: 130, flexShrink: 0, fontFamily: "monospace", fontSize: 11 }}
          />
          <input
            className="admin-input"
            value={v.value}
            onChange={e => update(i, "value", e.target.value)}
            placeholder={SUGGESTED_VARS.find(s => s.name === v.name)?.placeholder ?? "value"}
            style={{ flex: 1, fontSize: 12 }}
          />
          <button
            className="icon-btn"
            onClick={() => remove(i)}
            title="Remove"
            style={{ flexShrink: 0, color: "var(--text-tertiary)" }}
          >✕</button>
        </div>
      ))}

      <div style={{ display: "flex", gap: 6, marginTop: 2, position: "relative" }}>
        <button className="btn" style={{ fontSize: 11 }} onClick={() => add()}>+ Add variable</button>
        <button className="btn" style={{ fontSize: 11 }} onClick={() => setShowSuggestions(s => !s)}>
          Suggestions ▾
        </button>
        {showSuggestions && (
          <div style={{
            position: "absolute", top: "calc(100% + 4px)", left: 0, zIndex: 50,
            background: "var(--bg-primary)", border: "1px solid var(--border)",
            borderRadius: 10, padding: "6px 0", minWidth: 200,
            boxShadow: "0 8px 24px rgba(0,0,0,0.15)", maxHeight: 260, overflowY: "auto"
          }}>
            {SUGGESTED_VARS.map(s => {
              const already = vars.some(v => v.name === s.name);
              return (
                <button
                  key={s.name}
                  onClick={() => !already && addSuggestion(s.name)}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    width: "100%", padding: "6px 12px", background: "none", border: "none",
                    cursor: already ? "default" : "pointer", textAlign: "left",
                    opacity: already ? 0.4 : 1, gap: 8,
                  }}
                >
                  <span style={{ fontFamily: "monospace", fontSize: 11 }}>{s.name}</span>
                  {already && <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>added</span>}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Tools tab ─────────────────────────────────────────────────────────────────

function ToolsTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const setToolEnabled = (name: string, val: boolean) =>
    onChange({ ...settings, enabledTools: { ...settings.enabledTools, [name]: val } });
  const setMaxTools = (v: number) => onChange({ ...settings, maxTools: v });
  const setWebSearchResults = (v: number) => onChange({ ...settings, webSearchResults: v });
  const setMaxSteps = (v: number) => onChange({ ...settings, maxSteps: v });
  const setWikiEnabled = (val: boolean) => onChange({ ...settings, wikiEnabled: val });

  return (
    <div className="admin-scroll">
      <section className="admin-section">
        <div className="admin-section-header">
          <span className="admin-section-icon">⚡</span>
          <span className="admin-section-title">BUILT-IN</span>
        </div>
        {BUILTIN_TOOLS.map(t => (
          <label key={t.name} className="admin-row" style={{ cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={toolEnabled(settings.enabledTools, t.name)}
              onChange={e => setToolEnabled(t.name, e.target.checked)}
              className="admin-checkbox"
            />
            <span className="tool-icon">{t.icon}</span>
            <div className="admin-row-text">
              <span className="admin-row-title">{t.label}</span>
              <span className="admin-row-sub">
                {t.name === "run_python"
                  ? "Global master switch. Executes LLM-written Python in a sandbox; asks for approval before the first run each session. Profiles can opt out individually."
                  : t.name}
              </span>
            </div>
          </label>
        ))}
      </section>

      <section className="admin-section">
        <div className="admin-section-header">
          <span className="admin-section-icon">📖</span>
          <span className="admin-section-title">WIKI MEMORY</span>
        </div>
        <label className="admin-row" style={{ cursor: "pointer" }}>
          <input
            type="checkbox"
            checked={settings.wikiEnabled === true}
            onChange={e => setWikiEnabled(e.target.checked)}
            className="admin-checkbox"
          />
          <span className="tool-icon">📖</span>
          <div className="admin-row-text">
            <span className="admin-row-title">Persistent Wiki Memory (global default)</span>
            <span className="admin-row-sub">Gives the model wiki_read, wiki_write, wiki_search, wiki_patch, wiki_list and wiki_delete tools to store and recall knowledge across conversations. Stored as markdown files in ~/.local/share/lexichat/wiki/. Individual profiles can override this in their settings.</span>
          </div>
        </label>
      </section>

      <section className="admin-section">
        <div className="admin-section-header">
          <span className="admin-section-icon">⚙</span>
          <span className="admin-section-title">QUERY SETTINGS</span>
        </div>
        <div className="admin-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, width: "100%" }}>
            <div style={{ flex: 1 }}>
              <div className="admin-row-title">Tools sent to model per query</div>
              <div className="admin-row-sub">Only the most relevant tools are selected when this limit is exceeded.</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button className="stepper-btn" onClick={() => setMaxTools(Math.max(5, settings.maxTools - 5))}>−</button>
              <span style={{ fontSize: 14, fontWeight: 600, fontFamily: "monospace", minWidth: 28, textAlign: "center" }}>{settings.maxTools}</span>
              <button className="stepper-btn" onClick={() => setMaxTools(Math.min(100, settings.maxTools + 5))}>+</button>
            </div>
          </div>
          {settings.maxTools !== 30 && (
            <button className="link-btn" onClick={() => setMaxTools(30)}>Reset to default (30)</button>
          )}
        </div>
        <div className="admin-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, width: "100%" }}>
            <div style={{ flex: 1 }}>
              <div className="admin-row-title">Web search results</div>
              <div className="admin-row-sub">Number of results returned per web_search call.</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button className="stepper-btn" onClick={() => setWebSearchResults(Math.max(1, settings.webSearchResults - 1))}>−</button>
              <span style={{ fontSize: 14, fontWeight: 600, fontFamily: "monospace", minWidth: 28, textAlign: "center" }}>{settings.webSearchResults}</span>
              <button className="stepper-btn" onClick={() => setWebSearchResults(Math.min(20, settings.webSearchResults + 1))}>+</button>
            </div>
          </div>
          {settings.webSearchResults !== 10 && (
            <button className="link-btn" onClick={() => setWebSearchResults(10)}>Reset to default (10)</button>
          )}
        </div>
        <div className="admin-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, width: "100%" }}>
            <div style={{ flex: 1 }}>
              <div className="admin-row-title">Max agent steps</div>
              <div className="admin-row-sub">Tool-calling rounds the agent may take before it must answer. Raise for deep multi-source research.</div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <button className="stepper-btn" onClick={() => setMaxSteps(Math.max(5, (settings.maxSteps ?? 20) - 5))}>−</button>
              <span style={{ fontSize: 14, fontWeight: 600, fontFamily: "monospace", minWidth: 28, textAlign: "center" }}>{settings.maxSteps ?? 20}</span>
              <button className="stepper-btn" onClick={() => setMaxSteps(Math.min(50, (settings.maxSteps ?? 20) + 5))}>+</button>
            </div>
          </div>
          {(settings.maxSteps ?? 20) !== 20 && (
            <button className="link-btn" onClick={() => setMaxSteps(20)}>Reset to default (20)</button>
          )}
        </div>
      </section>
    </div>
  );
}

// ── Models tab ────────────────────────────────────────────────────────────────

function ModelsTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const [newModel, setNewModel] = useState<Record<string, string>>({});
  const [search, setSearch] = useState<Record<string, string>>({});
  const [refreshing, setRefreshing] = useState("");
  const servers = settings.servers ?? [];

  const patch = (id: string, p: Partial<ServerConfig>) =>
    onChange({ ...settings, servers: servers.map(s => s.id === id ? { ...s, ...p } : s) });

  const toggle = (s: ServerConfig, m: string, on: boolean) => {
    const enabled = new Set(s.models ?? []);
    if (on) enabled.add(m); else enabled.delete(m);
    patch(s.id, { models: [...enabled] });
  };

  const addManual = (s: ServerConfig) => {
    const name = (newModel[s.id] ?? "").trim();
    if (!name) return;
    const manualModels = (s.manualModels ?? []).includes(name) ? (s.manualModels ?? []) : [...(s.manualModels ?? []), name];
    const models       = (s.models       ?? []).includes(name) ? (s.models       ?? []) : [...(s.models       ?? []), name];
    patch(s.id, { manualModels, models });
    setNewModel(n => ({ ...n, [s.id]: "" }));
  };

  const refresh = async (s: ServerConfig) => {
    setRefreshing(s.id);
    try {
      const fetched = await invoke<string[]>("get_models",
        { args: { base_url: s.baseUrl, provider: s.provider, api_key: s.apiKey ?? null } });
      const next = reconcileCatalog(s, fetched);
      patch(s.id, { catalog: next.catalog, models: next.models });
    } catch { }
    setRefreshing("");
  };

  return (
    <div className="admin-scroll" style={{ padding: "16px 20px" }}>
      {servers.length === 0 && <div className="admin-empty">No servers configured. Add one in the Server tab.</div>}
      {servers.map(s => {
        const pool = serverModelPool(s);
        const enabled = new Set(s.models ?? []);
        const q = (search[s.id] ?? "").toLowerCase();
        const shown = pool.filter(m => m.toLowerCase().includes(q));
        const shownChat = shown.filter(m => !isEmbeddingModel(m));
        return (
          <section key={s.id} className="admin-section" style={{ marginBottom: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>{s.name}</span>
              <span style={{ fontSize: 11, opacity: 0.5 }}>{enabled.size} enabled / {pool.length} available</span>
              <div style={{ flex: 1 }} />
              <button className="btn" onClick={() => refresh(s)} disabled={refreshing === s.id}>{refreshing === s.id ? "…" : "↻ Refresh"}</button>
            </div>

            {pool.length > 8 && (
              <input className="admin-input" style={{ marginBottom: 6 }} value={search[s.id] ?? ""}
                onChange={e => setSearch(v => ({ ...v, [s.id]: e.target.value }))}
                placeholder={`Search ${pool.length} models…`} />
            )}

            {pool.length > 0 && (
              <div style={{ display: "flex", gap: 12, marginBottom: 6 }}>
                <button className="link-btn" onClick={() => patch(s.id, { models: [...new Set([...(s.models ?? []), ...shownChat])] })}>
                  Enable all chat{q ? " (shown)" : ""}
                </button>
                <button className="link-btn" onClick={() => patch(s.id, { models: (s.models ?? []).filter(m => !shown.includes(m)) })}>
                  Disable {q ? "shown" : "all"}
                </button>
              </div>
            )}

            <div style={{ maxHeight: 280, overflowY: "auto", border: "1px solid var(--border-color,#333)", borderRadius: 6 }}>
              {shown.length === 0
                ? <div className="admin-empty" style={{ padding: 10 }}>{pool.length === 0 ? "No models yet — Refresh or add one below." : "No matches."}</div>
                : shown.map(m => {
                    const embed = isEmbeddingModel(m);
                    return (
                      <label key={m} className="admin-row" style={{ cursor: "pointer", opacity: embed && !enabled.has(m) ? 0.55 : 1 }}>
                        <input type="checkbox" checked={enabled.has(m)} onChange={e => toggle(s, m, e.target.checked)} />
                        <span style={{ flex: 1, fontFamily: "monospace", fontSize: 12 }}>{m}</span>
                        {embed && <span style={{ fontSize: 10, opacity: 0.6, border: "1px solid currentColor", borderRadius: 4, padding: "0 4px" }}>embedding</span>}
                      </label>
                    );
                  })
              }
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <input className="admin-input" style={{ flex: 1 }} value={newModel[s.id] ?? ""}
                onChange={e => setNewModel(n => ({ ...n, [s.id]: e.target.value }))}
                placeholder="Add a model by name (e.g. claude-opus-4-8)…"
                onKeyDown={e => e.key === "Enter" && addManual(s)} />
              <button className="btn" onClick={() => addManual(s)} disabled={!(newModel[s.id] ?? "").trim()}>Add</button>
            </div>
          </section>
        );
      })}
    </div>
  );
}

// ── OpenAPI tab ───────────────────────────────────────────────────────────────

interface SpecMeta {
  title?: string;
  baseUrl?: string;
  tokenUrl?: string;
  authorizationUrl?: string;
  scopes?: string;
}

export function extractSpecMeta(json: string): SpecMeta {
  try {
    const s = JSON.parse(json);
    let extractedTitle: string | undefined;
    let extractedBaseUrl: string | undefined;
    let extractedTokenUrl: string | undefined;
    let extractedAuthorizationUrl: string | undefined;
    let extractedScopes: string | undefined;

    // Title: info.title
    if (typeof s?.info?.title === "string" && s.info.title.trim()) {
      extractedTitle = s.info.title.trim();
    }

    // OpenAPI 3.x: servers[0].url
    if (Array.isArray(s?.servers) && s.servers.length > 0) {
      const url = s.servers[0]?.url;
      if (typeof url === "string" && url.trim()) {
        extractedBaseUrl = url.trim().replace(/\/$/, "");
      }
    }

    // Swagger 2.0 fallback: host + basePath + schemes
    if (!extractedBaseUrl && typeof s?.host === "string") {
      const scheme = Array.isArray(s.schemes) && s.schemes.length > 0 ? s.schemes[0] : "https";
      const basePath = typeof s.basePath === "string" ? s.basePath.replace(/\/$/, "") : "";
      extractedBaseUrl = `${scheme}://${s.host}${basePath}`;
    }

    // OpenAPI 3.x: components.securitySchemes — find first OAuth2 scheme
    const schemes3 = s?.components?.securitySchemes;
    if (schemes3 && typeof schemes3 === "object") {
      for (const scheme of Object.values(schemes3) as Record<string, unknown>[]) {
        if (scheme?.type !== "oauth2") continue;
        const flows = scheme.flows as Record<string, { tokenUrl?: string; scopes?: Record<string, string> }> | undefined;
        if (!flows) continue;
        // Prefer authorizationCode (needs browser), then clientCredentials, then others
        const flow = flows.authorizationCode ?? flows.clientCredentials ?? flows.password ?? flows.implicit;
        if (flow) {
          if (flow.tokenUrl) extractedTokenUrl = flow.tokenUrl;
          const authUrl = (flow as { authorizationUrl?: string }).authorizationUrl;
          if (authUrl) extractedAuthorizationUrl = authUrl;
          if (flow.scopes) extractedScopes = Object.keys(flow.scopes).join(" ");
          break;
        }
      }
    }

    // Swagger 2.0: securityDefinitions
    // Prefer flow=accessCode (has both authorizationUrl + tokenUrl) over implicit (no tokenUrl)
    if (!extractedTokenUrl) {
      const defs = s?.securityDefinitions;
      if (defs && typeof defs === "object") {
        const defValues = Object.values(defs) as Record<string, unknown>[];
        // First pass: prefer accessCode / authorizationCode flows (have tokenUrl)
        for (const def of defValues) {
          if (def?.type === "oauth2" && typeof def.tokenUrl === "string") {
            extractedTokenUrl = def.tokenUrl as string;
            if (typeof def.authorizationUrl === "string")
              extractedAuthorizationUrl = def.authorizationUrl as string;
            if (def.scopes && typeof def.scopes === "object")
              extractedScopes = Object.keys(def.scopes as object).join(" ");
            break;
          }
        }
        // Second pass: implicit flows only have authorizationUrl (no tokenUrl)
        if (!extractedAuthorizationUrl) {
          for (const def of defValues) {
            if (def?.type === "oauth2" && typeof def.authorizationUrl === "string") {
              extractedAuthorizationUrl = def.authorizationUrl as string;
              if (!extractedScopes && def.scopes && typeof def.scopes === "object")
                extractedScopes = Object.keys(def.scopes as object).join(" ");
              break;
            }
          }
        }
      }
    }

    return { title: extractedTitle, baseUrl: extractedBaseUrl, tokenUrl: extractedTokenUrl, authorizationUrl: extractedAuthorizationUrl, scopes: extractedScopes };
  } catch {
    return {};
  }
}

function OpenAPITab({ stored, onChange }: { stored: StoredOpenAPISpec[]; onChange: (s: StoredOpenAPISpec[]) => void }) {
  const [specs, setSpecs] = useState<SpecInfo[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [specJson, setSpecJson] = useState("");
  const [auth, setAuth] = useState<AuthConfig>(DEFAULT_AUTH);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    invoke<SpecInfo[]>("list_openapi_specs").then(setSpecs).catch(() => {});
  }, []);

  const resetForm = () => {
    setTitle(""); setBaseUrl(""); setSpecJson(""); setAuth(DEFAULT_AUTH); setError("");
    setEditingId(null); setShowAdd(false);
  };

  const handleSpecJson = (json: string) => {
    setSpecJson(json);
    const { title: t, baseUrl: u, tokenUrl, authorizationUrl, scopes } = extractSpecMeta(json);
    if (t && !title.trim()) setTitle(t);
    if (u && !baseUrl.trim()) setBaseUrl(u);
    if (tokenUrl) {
      setAuth(prev => ({
        ...prev,
        type: "oauth2",
        token_url: prev.token_url || tokenUrl,
        authorization_url: prev.authorization_url || authorizationUrl || "",
        scope: prev.scope || scopes || "",
      }));
    }
  };

  const startEdit = (id: string) => {
    const storedSpec = stored.find(s => s.id === id);
    if (!storedSpec) return;
    setEditingId(id);
    setTitle(storedSpec.title);
    setBaseUrl(storedSpec.base_url);
    setSpecJson(storedSpec.spec_json);
    setAuth(storedSpec.auth ?? DEFAULT_AUTH);
    setError("");
    setShowAdd(false);
  };

  const add = async () => {
    if (!title.trim() || !baseUrl.trim() || !specJson.trim()) return;
    setLoading(true);
    setError("");
    try {
      const info = await invoke<SpecInfo>("register_openapi_spec", {
        args: { title: title.trim(), base_url: baseUrl.trim(), spec_json: specJson.trim(), auth }
      });
      const entry: StoredOpenAPISpec = { id: info.id, title: title.trim(), base_url: baseUrl.trim(), spec_json: specJson.trim(), auth };
      setSpecs(prev => [...prev, info]);
      onChange([...stored, entry]);
      resetForm();
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  const saveEdit = async () => {
    if (!editingId || !title.trim() || !baseUrl.trim() || !specJson.trim()) return;
    setLoading(true);
    setError("");
    try {
      // Remove old, register new with same stored JSON but updated metadata
      await invoke("remove_openapi_spec", { id: editingId });
      const info = await invoke<SpecInfo>("register_openapi_spec", {
        args: { title: title.trim(), base_url: baseUrl.trim(), spec_json: specJson.trim(), auth }
      });
      const entry: StoredOpenAPISpec = { id: info.id, title: title.trim(), base_url: baseUrl.trim(), spec_json: specJson.trim(), auth };
      setSpecs(prev => prev.filter(s => s.id !== editingId).concat(info));
      onChange(stored.filter(s => s.id !== editingId).concat(entry));
      resetForm();
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  const remove = async (id: string) => {
    await invoke("remove_openapi_spec", { id });
    setSpecs(prev => prev.filter(s => s.id !== id));
    onChange(stored.filter(s => s.id !== id));
    if (editingId === id) resetForm();
  };

  const toggleExpand = (id: string) =>
    setExpanded(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const methodColor = (m: string) => ({ GET: "#4ade80", POST: "#60a5fa", PUT: "#fb923c", PATCH: "#facc15", DELETE: "#f87171" }[m] ?? "#888");

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {specs.length === 0 && !showAdd && (
          <div className="admin-empty">No OpenAPI specs registered. Add one to call external APIs.</div>
        )}

        {stored.map(storedSpec => {
          const isBuiltin = BUILTIN_OPENAPI_SPEC_IDS.has(storedSpec.id);
          const isEnabled = storedSpec.enabled !== false;
          const rustSpec = specs.find(s => s.id === storedSpec.id);

          const toggleEnabled = () =>
            onChange(stored.map(s => s.id === storedSpec.id ? { ...s, enabled: !isEnabled } : s));

          return (
          <section key={storedSpec.id} className="admin-section" style={{ opacity: isEnabled ? 1 : 0.55 }}>
            <div className="admin-row">
              {(isEnabled || isBuiltin) && (
                <button className="icon-btn" onClick={() => toggleExpand(storedSpec.id)} style={{ fontSize: 9 }}>
                  {expanded.has(storedSpec.id) ? "▼" : "▶"}
                </button>
              )}
              <div style={{ flex: 1 }}>
                <div className="admin-row-title">
                  {storedSpec.title}
                  {isBuiltin && (
                    <span style={{ fontSize: 9, marginLeft: 6, opacity: 0.45, fontWeight: 400 }}>built-in</span>
                  )}
                </div>
                <div className="admin-row-sub">
                  {storedSpec.base_url}
                  {rustSpec ? ` · ${rustSpec.tool_count} tools` : isEnabled ? " · loading…" : " · disabled"}
                </div>
              </div>
              {isBuiltin ? (
                <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, cursor: "pointer" }}>
                  <input type="checkbox" checked={isEnabled} onChange={toggleEnabled} />
                  {isEnabled ? "On" : "Off"}
                </label>
              ) : (
                <>
                  <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
                    onClick={() => editingId === storedSpec.id ? resetForm() : startEdit(storedSpec.id)}>
                    {editingId === storedSpec.id ? "Cancel" : "Edit"}
                  </button>
                  <button className="icon-btn danger" onClick={() => remove(storedSpec.id)}>✕</button>
                </>
              )}
            </div>

            {editingId === storedSpec.id && (
              <div style={{ padding: "8px 16px 12px", borderTop: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="field">
                  <label>OpenAPI JSON Spec</label>
                  <textarea className="admin-input" value={specJson} onChange={e => handleSpecJson(e.target.value)}
                    style={{ minHeight: 100, resize: "vertical", fontFamily: "monospace", fontSize: 11 }} />
                </div>
                <div className="field">
                  <label>Title</label>
                  <input className="admin-input" value={title} onChange={e => setTitle(e.target.value)} />
                </div>
                <div className="field">
                  <label>Base URL</label>
                  <input className="admin-input" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} />
                </div>
                <AuthConfigForm auth={auth} onChange={setAuth} />
                {error && <div style={{ color: "#f87171", fontSize: 12 }}>{error}</div>}
                <div style={{ display: "flex", gap: 8 }}>
                  <button className="btn" onClick={resetForm}>Cancel</button>
                  <button className="btn primary" onClick={saveEdit}
                    disabled={loading || !title.trim() || !baseUrl.trim() || !specJson.trim()}>
                    {loading ? "Saving…" : "Save Changes"}
                  </button>
                </div>
              </div>
            )}

            {!editingId && isEnabled && expanded.has(storedSpec.id) && rustSpec && (
              <div style={{ paddingLeft: 28 }}>
                {rustSpec.tools.map(t => (
                  <div key={t.name} className="admin-row" style={{ paddingTop: 4, paddingBottom: 4 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, color: methodColor(t.method),
                      background: methodColor(t.method) + "22", padding: "1px 5px", borderRadius: 3, minWidth: 42, textAlign: "center", flexShrink: 0 }}>
                      {t.method}
                    </span>
                    <div className="admin-row-text">
                      <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600 }}>{t.name}</span>
                      <span className="admin-row-sub">{t.path}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
          );
        })}

        {showAdd && (
          <section className="admin-section" style={{ padding: "12px 16px" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div className="field">
                <label>OpenAPI JSON Spec</label>
                <textarea className="admin-input" value={specJson} onChange={e => handleSpecJson(e.target.value)}
                  placeholder='Paste your OpenAPI JSON spec here — title and base URL will be extracted automatically'
                  style={{ minHeight: 120, resize: "vertical", fontFamily: "monospace", fontSize: 11 }} />
              </div>
              <div className="field">
                <label>Title</label>
                <input className="admin-input" value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g. My API" />
              </div>
              <div className="field">
                <label>Base URL</label>
                <input className="admin-input" value={baseUrl} onChange={e => setBaseUrl(e.target.value)} placeholder="https://api.example.com" />
              </div>
              <AuthConfigForm auth={auth} onChange={setAuth} />
              {error && <div style={{ color: "#f87171", fontSize: 12 }}>{error}</div>}
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" onClick={resetForm}>Cancel</button>
                <button className="btn primary" onClick={add} disabled={loading || !title.trim() || !baseUrl.trim() || !specJson.trim()}>
                  {loading ? "Parsing…" : "Add Spec"}
                </button>
              </div>
            </div>
          </section>
        )}
      </div>
      {!showAdd && !editingId && (
        <div className="admin-footer-bar">
          <button className="btn primary" onClick={() => setShowAdd(true)}>+ Add OpenAPI Spec</button>
        </div>
      )}
    </div>
  );
}

// ── SPARQL tab ────────────────────────────────────────────────────────────────

function SparqlTab({ stored, onChange }: { stored: StoredSparqlEndpoint[]; onChange: (s: StoredSparqlEndpoint[]) => void }) {
  const [infos, setInfos] = useState<SparqlInfo[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [endpointUrl, setEndpointUrl] = useState("");
  const [prefixes, setPrefixes] = useState("");
  const [schemaSummary, setSchemaSummary] = useState("");
  const [usageHint, setUsageHint] = useState("");
  const [examples, setExamples] = useState<SparqlExampleQuery[]>([]);
  const [readOnly, setReadOnly] = useState(true);
  const [auth, setAuth] = useState<AuthConfig>(DEFAULT_AUTH);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [discovering, setDiscovering] = useState(false);
  const [discoverMsg, setDiscoverMsg] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Compute tool info statelessly from the displayed endpoints so it shows
  // regardless of which profile is active (Rust's registered state is profile-scoped).
  useEffect(() => {
    invoke<SparqlInfo[]>("get_sparql_tools", { endpoints: stored }).then(setInfos).catch(() => {});
  }, [stored]);

  const resetForm = () => {
    setTitle(""); setEndpointUrl(""); setPrefixes(""); setSchemaSummary(""); setUsageHint("");
    setExamples([]); setReadOnly(true); setAuth(DEFAULT_AUTH);
    setError(""); setDiscoverMsg(""); setEditingId(null); setShowAdd(false);
  };

  const startEdit = (id: string) => {
    const ep = stored.find(s => s.id === id);
    if (!ep) return;
    setEditingId(id);
    setTitle(ep.title);
    setEndpointUrl(ep.endpoint_url);
    setPrefixes(ep.prefixes ?? "");
    setSchemaSummary(ep.schema_summary ?? "");
    setUsageHint(ep.usage_hint ?? "");
    setExamples(ep.example_queries ?? []);
    setReadOnly(ep.read_only !== false);
    setAuth(ep.auth ?? DEFAULT_AUTH);
    setError(""); setDiscoverMsg(""); setShowAdd(false);
  };

  const argsForRegister = () => ({
    title: title.trim(),
    endpoint_url: endpointUrl.trim(),
    prefixes,
    schema_summary: schemaSummary,
    example_queries: examples.filter(e => e.query.trim()),
    usage_hint: usageHint,
    auth,
    read_only: readOnly,
  });

  const entryFromArgs = (id: string): StoredSparqlEndpoint => ({
    id, title: title.trim(), endpoint_url: endpointUrl.trim(),
    prefixes, schema_summary: schemaSummary,
    example_queries: examples.filter(e => e.query.trim()),
    usage_hint: usageHint,
    auth, read_only: readOnly,
  });

  const discover = async () => {
    if (!endpointUrl.trim()) return;
    setDiscovering(true);
    setDiscoverMsg("");
    setError("");
    try {
      const res = await invoke<SparqlDiscovery>("discover_sparql_endpoint", {
        args: { endpoint_url: endpointUrl.trim(), auth },
      });
      setDiscoverMsg(res.message);
      if (res.live) {
        if (res.suggested_prefixes && !prefixes.trim()) setPrefixes(res.suggested_prefixes);
        if (res.suggested_schema && !schemaSummary.trim()) setSchemaSummary(res.suggested_schema);
      }
    } catch (e) {
      setDiscoverMsg(`Discovery failed: ${String(e)}`);
    }
    setDiscovering(false);
  };

  const add = async () => {
    if (!title.trim() || !endpointUrl.trim()) return;
    setLoading(true);
    setError("");
    try {
      const info = await invoke<SparqlInfo>("register_sparql_endpoint", { args: argsForRegister() });
      setInfos(prev => [...prev, info]);
      onChange([...stored, entryFromArgs(info.id)]);
      resetForm();
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  const saveEdit = async () => {
    if (!editingId || !title.trim() || !endpointUrl.trim()) return;
    setLoading(true);
    setError("");
    try {
      await invoke("remove_sparql_endpoint", { id: editingId });
      const info = await invoke<SparqlInfo>("register_sparql_endpoint", { args: argsForRegister() });
      setInfos(prev => prev.filter(s => s.id !== editingId).concat(info));
      onChange(stored.filter(s => s.id !== editingId).concat(entryFromArgs(info.id)));
      resetForm();
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  const remove = async (id: string) => {
    await invoke("remove_sparql_endpoint", { id });
    setInfos(prev => prev.filter(s => s.id !== id));
    onChange(stored.filter(s => s.id !== id));
    if (editingId === id) resetForm();
  };

  const toggleExpand = (id: string) =>
    setExpanded(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const setExample = (i: number, patch: Partial<SparqlExampleQuery>) =>
    setExamples(prev => prev.map((e, idx) => idx === i ? { ...e, ...patch } : e));

  const formBody = (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div className="field">
        <label>Title</label>
        <input className="admin-input" value={title} onChange={e => setTitle(e.target.value)} placeholder="e.g. HM Land Registry" />
      </div>
      <div className="field">
        <label>SPARQL Endpoint URL</label>
        <input className="admin-input" value={endpointUrl} onChange={e => setEndpointUrl(e.target.value)}
          placeholder="https://landregistry.data.gov.uk/landregistry/query" style={{ fontFamily: "monospace", fontSize: 11 }} />
      </div>
      <div>
        <button className="btn" onClick={discover} disabled={discovering || !endpointUrl.trim()}>
          {discovering ? "Testing…" : "Test & discover"}
        </button>
        {discoverMsg && <div style={{ fontSize: 11, marginTop: 4, opacity: 0.8 }}>{discoverMsg}</div>}
      </div>
      <div className="field">
        <label>When to use (topics)</label>
        <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 4 }}>
          What questions this endpoint answers. The model uses this to pick this tool over web search.
        </div>
        <input className="admin-input" value={usageHint} onChange={e => setUsageHint(e.target.value)}
          placeholder="e.g. UK property sold prices, house price index, address-level transactions" />
      </div>
      <div className="field">
        <label>Prefixes</label>
        <textarea className="admin-input" value={prefixes} onChange={e => setPrefixes(e.target.value)}
          placeholder={"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX ukhpi: <http://landregistry.data.gov.uk/def/ukhpi/>"}
          style={{ minHeight: 70, resize: "vertical", fontFamily: "monospace", fontSize: 11 }} />
      </div>
      <div className="field">
        <label>Schema / vocabulary summary</label>
        <textarea className="admin-input" value={schemaSummary} onChange={e => setSchemaSummary(e.target.value)}
          placeholder="Describe the key classes and properties the model should use, or paste an ontology summary."
          style={{ minHeight: 90, resize: "vertical", fontFamily: "monospace", fontSize: 11 }} />
      </div>
      <div className="field">
        <label>Example queries</label>
        <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>
          Real example queries are the most valuable context for the model.
        </div>
        {examples.map((ex, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 8, border: "1px solid var(--border)", borderRadius: 6, padding: 8 }}>
            <div style={{ display: "flex", gap: 6 }}>
              <input className="admin-input" value={ex.label} onChange={e => setExample(i, { label: e.target.value })}
                placeholder="Label (e.g. Recent price-paid records)" style={{ flex: 1 }} />
              <button className="icon-btn danger" onClick={() => setExamples(prev => prev.filter((_, idx) => idx !== i))}>✕</button>
            </div>
            <textarea className="admin-input" value={ex.query} onChange={e => setExample(i, { query: e.target.value })}
              placeholder="SELECT ?s WHERE { ?s ?p ?o } LIMIT 10"
              style={{ minHeight: 60, resize: "vertical", fontFamily: "monospace", fontSize: 11 }} />
          </div>
        ))}
        <button className="btn" onClick={() => setExamples(prev => [...prev, { label: "", query: "" }])}>+ Add example query</button>
      </div>
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer" }}>
        <input type="checkbox" checked={readOnly} onChange={e => setReadOnly(e.target.checked)} />
        Read-only (reject INSERT/DELETE/DROP and other update operations)
      </label>
      <AuthConfigForm auth={auth} onChange={setAuth} />
      {error && <div style={{ color: "#f87171", fontSize: 12 }}>{error}</div>}
    </div>
  );

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {stored.length === 0 && !showAdd && (
          <div className="admin-empty">No SPARQL endpoints registered. Add one to query linked-data services.</div>
        )}

        {stored.map(ep => {
          const isBuiltin = BUILTIN_SPARQL_ENDPOINT_IDS.has(ep.id);
          const isEnabled = ep.enabled !== false;
          const info = infos.find(s => s.id === ep.id);

          const toggleEnabled = () =>
            onChange(stored.map(s => s.id === ep.id ? { ...s, enabled: !isEnabled } : s));

          return (
          <section key={ep.id} className="admin-section" style={{ opacity: isEnabled ? 1 : 0.55 }}>
            <div className="admin-row">
              <button className="icon-btn" onClick={() => toggleExpand(ep.id)} style={{ fontSize: 9 }}>
                {expanded.has(ep.id) ? "▼" : "▶"}
              </button>
              <div style={{ flex: 1 }}>
                <div className="admin-row-title">
                  {ep.title}
                  {isBuiltin && <span style={{ fontSize: 9, marginLeft: 6, opacity: 0.45, fontWeight: 400 }}>built-in</span>}
                </div>
                <div className="admin-row-sub">
                  {ep.endpoint_url}
                  {info ? ` · ${info.tool_count} tools` : isEnabled ? " · loading…" : " · disabled"}
                </div>
              </div>
              {isBuiltin ? (
                <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, cursor: "pointer" }}>
                  <input type="checkbox" checked={isEnabled} onChange={toggleEnabled} />
                  {isEnabled ? "On" : "Off"}
                </label>
              ) : (
                <>
                  <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
                    onClick={() => editingId === ep.id ? resetForm() : startEdit(ep.id)}>
                    {editingId === ep.id ? "Cancel" : "Edit"}
                  </button>
                  <button className="icon-btn danger" onClick={() => remove(ep.id)}>✕</button>
                </>
              )}
            </div>

            {editingId === ep.id && (
              <div style={{ padding: "8px 16px 12px", borderTop: "1px solid var(--border)" }}>
                {formBody}
                <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                  <button className="btn" onClick={resetForm}>Cancel</button>
                  <button className="btn primary" onClick={saveEdit} disabled={loading || !title.trim() || !endpointUrl.trim()}>
                    {loading ? "Saving…" : "Save Changes"}
                  </button>
                </div>
              </div>
            )}

            {!editingId && expanded.has(ep.id) && (
              <div style={{ padding: "8px 16px 12px 28px", borderTop: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: 10, fontSize: 11 }}>
                {info && info.tools.length > 0 && (
                  <div>
                    <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>Generated tools</div>
                    {info.tools.map(name => (
                      <div key={name} style={{ fontFamily: "monospace", fontWeight: 600 }}>{name}</div>
                    ))}
                  </div>
                )}
                {(ep.usage_hint ?? "").trim() && (
                  <div>
                    <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>When to use</div>
                    <span style={{ opacity: 0.85 }}>{ep.usage_hint}</span>
                  </div>
                )}
                <div>
                  <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>Read-only</div>
                  <span>{ep.read_only !== false ? "Yes — updates (INSERT/DELETE/…) are rejected" : "No — update queries are allowed"}</span>
                </div>
                {(ep.prefixes ?? "").trim() && (
                  <div>
                    <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>Prefixes</div>
                    <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 11, opacity: 0.85 }}>{ep.prefixes}</pre>
                  </div>
                )}
                {(ep.schema_summary ?? "").trim() && (
                  <div>
                    <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>Schema / vocabulary</div>
                    <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 11, opacity: 0.85 }}>{ep.schema_summary}</pre>
                  </div>
                )}
                {(ep.example_queries ?? []).length > 0 && (
                  <div>
                    <div style={{ fontWeight: 700, opacity: 0.6, marginBottom: 3 }}>Example queries</div>
                    {(ep.example_queries ?? []).map((ex, i) => (
                      <div key={i} style={{ marginBottom: 6 }}>
                        <div style={{ fontWeight: 600, marginBottom: 2 }}>{ex.label || `Example ${i + 1}`}</div>
                        <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 11, opacity: 0.85, background: "var(--code-bg, rgba(127,127,127,0.08))", padding: 6, borderRadius: 4 }}>{ex.query}</pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </section>
          );
        })}

        {showAdd && (
          <section className="admin-section" style={{ padding: "12px 16px" }}>
            {formBody}
            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={resetForm}>Cancel</button>
              <button className="btn primary" onClick={add} disabled={loading || !title.trim() || !endpointUrl.trim()}>
                {loading ? "Saving…" : "Add Endpoint"}
              </button>
            </div>
          </section>
        )}
      </div>
      {!showAdd && !editingId && (
        <div className="admin-footer-bar">
          <button className="btn primary" onClick={() => setShowAdd(true)}>+ Add SPARQL Endpoint</button>
        </div>
      )}
    </div>
  );
}

// ── MCP tab ───────────────────────────────────────────────────────────────────

function MCPTab({ stored, onChange }: { stored: StoredMCPServer[]; onChange: (s: StoredMCPServer[]) => void }) {
  const [servers, setServers] = useState<MCPServerInfo[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [command, setCommand] = useState("");
  const [argsStr, setArgsStr] = useState("");
  const [envStr, setEnvStr] = useState("");
  const [enableApps, setEnableApps] = useState(false);
  const [auth, setAuth] = useState<AuthConfig>(DEFAULT_AUTH);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [reconnecting, setReconnecting] = useState<Set<string>>(new Set());

  const isHttp = command.startsWith("http");

  useEffect(() => {
    invoke<MCPServerInfo[]>("list_mcp_servers").then(setServers).catch(() => {});
  }, []);

  const closeForm = () => {
    setShowAdd(false); setEditingId(null); setError("");
    setName(""); setCommand(""); setArgsStr(""); setEnvStr(""); setEnableApps(false); setAuth(DEFAULT_AUTH);
  };

  // Open the form pre-filled with a server's config, to edit it (e.g. re-enter credentials
  // that were stripped on import, or fix a command).
  const startEdit = (srv: StoredMCPServer) => {
    setEditingId(srv.id);
    setName(srv.name);
    setCommand(srv.command);
    setArgsStr((srv.args ?? []).join(" "));
    setEnvStr(Object.entries(srv.env ?? {}).map(([k, v]) => `${k}=${v}`).join("\n"));
    setEnableApps(srv.enable_apps ?? false);
    setAuth(srv.auth ?? DEFAULT_AUTH);
    setError("");
    setShowAdd(true);
  };

  const add = async () => {
    if (!name.trim() || !command.trim()) return;
    setLoading(true);
    setError("");
    try {
      const cmdArgs = argsStr.trim() ? argsStr.split(/\s+/) : [];
      const env: Record<string,string> = {};
      for (const line of envStr.split("\n")) {
        const [k, ...v] = line.split("=");
        if (k?.trim()) env[k.trim()] = v.join("=").trim();
      }
      const effectiveAuth = isHttp ? auth : DEFAULT_AUTH;
      const info = await invoke<MCPServerInfo>("add_mcp_server", {
        args: { ...(editingId ? { id: editingId } : {}), name: name.trim(), command: command.trim(), args: cmdArgs, env, auth: effectiveAuth, enable_apps: enableApps }
      });
      // Editing preserves the per-tool enable map; adding starts fresh.
      const prevTools = editingId ? stored.find(s => s.id === editingId)?.enabledTools : undefined;
      const entry: StoredMCPServer = { id: info.id, name: name.trim(), command: command.trim(), args: cmdArgs, env, auth: effectiveAuth, enable_apps: enableApps, enabledTools: prevTools };
      if (editingId) {
        setServers(prev => prev.map(s => s.id === info.id ? info : s));
        onChange(stored.map(s => s.id === editingId ? entry : s));
      } else {
        setServers(prev => [...prev, info]);
        onChange([...stored, entry]);
      }
      closeForm();
    } catch (e) {
      setError(String(e));
    }
    setLoading(false);
  };

  const remove = async (id: string) => {
    await invoke("remove_mcp_server", { id });
    setServers(prev => prev.filter(s => s.id !== id));
    onChange(stored.filter(s => s.id !== id));
  };

  const reconnect = async (id: string) => {
    setReconnecting(prev => new Set([...prev, id]));
    try {
      const inRust = servers.some(s => s.id === id);
      let updated: MCPServerInfo;
      if (!inRust) {
        // Server was evicted from Rust state (e.g. not enabled in active profile).
        // Re-register it using the stored config and the same ID so dispatch works.
        const storedSrv = stored.find(s => s.id === id);
        if (!storedSrv) return;
        updated = await invoke<MCPServerInfo>("add_mcp_server", { args: {
          id: storedSrv.id, name: storedSrv.name, command: storedSrv.command,
          args: storedSrv.args, env: storedSrv.env ?? {}, auth: storedSrv.auth,
          enable_apps: storedSrv.enable_apps ?? false,
        }});
      } else {
        updated = await invoke<MCPServerInfo>("reconnect_mcp_server", { id });
      }
      setServers(prev => {
        const exists = prev.some(s => s.id === id);
        return exists ? prev.map(s => s.id === id ? updated : s) : [...prev, updated];
      });
    } catch { }
    setReconnecting(prev => { const s = new Set(prev); s.delete(id); return s; });
  };

  const toggleExpand = (id: string) =>
    setExpanded(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  const isToolEnabled = (storedSrv: StoredMCPServer | undefined, toolName: string) =>
    storedSrv?.enabledTools?.[toolName] !== false;

  const toggleTool = (serverId: string, toolName: string, enabled: boolean) => {
    const updated = stored.map(s =>
      s.id !== serverId ? s : { ...s, enabledTools: { ...(s.enabledTools ?? {}), [toolName]: enabled } }
    );
    onChange(updated);
  };

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {stored.length === 0 && !showAdd && (
          <div className="admin-empty">No MCP servers connected. Add a server to extend LexiChat with custom tools.</div>
        )}

        {stored.map(storedSrv => {
          // Overlay runtime connection status from Rust state (may be absent if server
          // is not enabled in the active profile — that's fine, show as "Not active").
          const srv = servers.find(s => s.id === storedSrv.id);
          const connected = srv?.connected ?? false;
          const srvError = srv?.error ?? null;
          const tools = srv?.tools ?? [];
          const tool_count = srv?.tool_count ?? 0;
          return (
          <section key={storedSrv.id} className="admin-section">
            <div className="admin-row">
              <button className="icon-btn" onClick={() => toggleExpand(storedSrv.id)} style={{ fontSize: 9 }}>
                {expanded.has(storedSrv.id) ? "▼" : "▶"}
              </button>
              <span style={{ fontSize: 14, marginRight: 2 }}>🖧</span>
              <div style={{ flex: 1 }}>
                <div className="admin-row-title">{storedSrv.name}</div>
                <div className="admin-row-sub" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontFamily: "monospace" }}>{storedSrv.command}</span>
                  <span style={{ color: connected ? "#4ade80" : srvError ? "#f87171" : "#888",
                    fontSize: 10, background: (connected ? "#4ade80" : srvError ? "#f87171" : "#888") + "22",
                    padding: "1px 6px", borderRadius: 10 }}>
                    {connected ? (() => {
                      const enabledCount = tools.filter(t => isToolEnabled(storedSrv, t.name)).length;
                      return enabledCount === tool_count
                        ? `✓ ${tool_count} tools`
                        : `✓ ${enabledCount}/${tool_count} tools`;
                    })() : srvError ? "✕ Error" : "Not active"}
                  </span>
                </div>
              </div>
              {!connected && (
                <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
                  onClick={() => reconnect(storedSrv.id)} disabled={reconnecting.has(storedSrv.id)}>
                  {reconnecting.has(storedSrv.id) ? "…" : "↻"}
                </button>
              )}
              <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
                onClick={() => startEdit(storedSrv)}>Edit</button>
              <button className="icon-btn danger" onClick={() => remove(storedSrv.id)}>✕</button>
            </div>
            {expanded.has(storedSrv.id) && (
              <div style={{ paddingLeft: 28 }}>
                {srvError && <div style={{ fontSize: 11, color: "#f87171", padding: "4px 16px" }}>{srvError}</div>}
                <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, padding: "4px 0", cursor: "pointer" }}>
                  <input type="checkbox" checked={storedSrv.enable_apps ?? false}
                    onChange={e => onChange(stored.map(s => s.id === storedSrv.id ? { ...s, enable_apps: e.target.checked } : s))} />
                  Interactive apps (UI){tools.some(t => t.has_ui) ? " · this server offers app tools" : ""}
                </label>
                {tools.length > 0 && (
                  <div style={{ paddingBottom: 4, display: "flex", gap: 8 }}>
                    <button className="btn" style={{ fontSize: 10, padding: "1px 7px" }}
                      onClick={() => onChange(stored.map(s => s.id !== storedSrv.id ? s : {
                        ...s, enabledTools: Object.fromEntries(tools.map(t => [t.name, true]))
                      }))}>All</button>
                    <button className="btn" style={{ fontSize: 10, padding: "1px 7px" }}
                      onClick={() => onChange(stored.map(s => s.id !== storedSrv.id ? s : {
                        ...s, enabledTools: Object.fromEntries(tools.map(t => [t.name, false]))
                      }))}>None</button>
                  </div>
                )}
                {tools.map(t => {
                  const enabled = isToolEnabled(storedSrv, t.name);
                  return (
                    <label key={t.name} className="admin-row" style={{ cursor: "pointer", paddingTop: 3, paddingBottom: 3 }}>
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={e => toggleTool(storedSrv.id, t.name, e.target.checked)}
                        style={{ marginRight: 6, accentColor: "var(--purple)" }}
                      />
                      <div className="admin-row-text">
                        <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600 }}>{t.name}</span>
                        {t.description && <span className="admin-row-sub">{t.description}</span>}
                      </div>
                    </label>
                  );
                })}
              </div>
            )}
          </section>
          );
        })}

        {showAdd && (
          <section className="admin-section" style={{ padding: "12px 16px" }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div className="field"><label>Server name</label>
                <input className="admin-input" value={name} onChange={e => setName(e.target.value)} placeholder="My MCP Server" />
              </div>
              <div className="field">
                <label>Command or URL</label>
                <input className="admin-input" value={command} onChange={e => setCommand(e.target.value)}
                  placeholder="https://example.com/mcp  or  npx -y @mcp/server" style={{ fontFamily: "monospace" }} />
                {command.startsWith("http") && (
                  <span style={{ fontSize: 10, color: "#4ade80", marginTop: 2 }}>✓ Remote HTTP server</span>
                )}
              </div>
              {!isHttp && (
                <div className="field"><label>Arguments (space-separated)</label>
                  <input className="admin-input" value={argsStr} onChange={e => setArgsStr(e.target.value)}
                    placeholder="-y @modelcontextprotocol/server-filesystem /path" style={{ fontFamily: "monospace" }} />
                </div>
              )}
              <div className="field"><label>Environment variables (KEY=VALUE per line, optional)</label>
                <textarea className="admin-input" value={envStr} onChange={e => setEnvStr(e.target.value)}
                  placeholder={"API_KEY=abc123\nBASE_URL=https://..."} rows={3}
                  style={{ fontFamily: "monospace", fontSize: 11, resize: "vertical" }} />
              </div>
              {isHttp && <AuthConfigForm auth={auth} onChange={setAuth} />}
              <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer" }}>
                <input type="checkbox" checked={enableApps} onChange={e => setEnableApps(e.target.checked)} />
                Enable interactive apps (UI) — lets this server render sandboxed HTML in chat
              </label>
              {error && <div style={{ color: "#f87171", fontSize: 12 }}>{error}</div>}
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" onClick={closeForm}>Cancel</button>
                <button className="btn primary" onClick={add} disabled={loading || !name.trim() || !command.trim()}>
                  {loading ? "Connecting…" : editingId ? "Save changes" : "Add Server"}
                </button>
              </div>
            </div>
          </section>
        )}
      </div>
      {!showAdd && (
        <div className="admin-footer-bar">
          <button className="btn primary" onClick={() => setShowAdd(true)}>+ Add MCP Server</button>
        </div>
      )}
    </div>
  );
}

// ── Sandbox tab ───────────────────────────────────────────────────────────────

function SandboxTab({ dirs, onChange }: { dirs: string[]; onChange: (dirs: string[]) => void }) {
  const [loading, setLoading] = useState(false);

  const addFolder = async () => {
    setLoading(true);
    try {
      const selected = await open({ directory: true, multiple: false, title: "Select a folder to allow" });
      if (typeof selected === "string" && selected && !dirs.includes(selected)) {
        onChange([...dirs, selected]);
      }
    } catch { }
    setLoading(false);
  };

  const removeDir = (path: string) => {
    onChange(dirs.filter(d => d !== path));
  };

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        <div style={{ padding: "16px 20px 8px", fontSize: 12, opacity: 0.6, lineHeight: 1.5 }}>
          File tools can only access folders listed here. Add a folder to grant access.
          {dirs.length === 0 && (
            <span style={{ color: "#fb923c", display: "block", marginTop: 6 }}>
              No folders selected — file tools are currently unrestricted.
            </span>
          )}
        </div>
        {dirs.length > 0 && (
          <section className="admin-section">
            {dirs.map(dir => (
              <div key={dir} className="admin-row">
                <span style={{ fontSize: 16, marginRight: 2, opacity: 0.7 }}>📁</span>
                <span style={{ flex: 1, fontFamily: "monospace", fontSize: 11, wordBreak: "break-all" }}>{dir}</span>
                <button className="icon-btn danger" onClick={() => removeDir(dir)} title="Remove">✕</button>
              </div>
            ))}
          </section>
        )}
      </div>
      <div className="admin-footer-bar">
        <button className="btn primary" onClick={addFolder} disabled={loading}>
          {loading ? "…" : "+ Add Folder"}
        </button>
      </div>
    </div>
  );
}

// ── Server tab ────────────────────────────────────────────────────────────────

// Starter connections for well-known inference APIs. Selecting one fills provider + URL and
// flags whether an API key is expected; every field stays editable afterwards.
interface Preset { label: string; provider: ProviderKind; baseUrl: string; needsKey: boolean }
const SERVER_PRESETS: Preset[] = [
  { label: "Ollama (local)",            provider: "ollama", baseUrl: "http://localhost:11434",        needsKey: false },
  { label: "OpenAI",                    provider: "openai", baseUrl: "https://api.openai.com/v1",     needsKey: true  },
  { label: "Anthropic (OpenAI-compat)", provider: "openai", baseUrl: "https://api.anthropic.com/v1",  needsKey: true  },
  { label: "Google Gemini (OpenAI-compat)", provider: "openai", baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai", needsKey: true },
  { label: "Groq",                      provider: "openai", baseUrl: "https://api.groq.com/openai/v1", needsKey: true  },
  { label: "Together",                  provider: "openai", baseUrl: "https://api.together.xyz/v1",    needsKey: true  },
  { label: "OpenRouter",                provider: "openai", baseUrl: "https://openrouter.ai/api/v1",   needsKey: true  },
  { label: "Mistral",                   provider: "openai", baseUrl: "https://api.mistral.ai/v1",      needsKey: true  },
  { label: "LM Studio / llama.cpp / vLLM (local)", provider: "openai", baseUrl: "http://localhost:1234/v1", needsKey: false },
];

function ServerTab({ settings, onChange }: {
  settings: AppSettings;
  onChange: (s: AppSettings) => void;
}) {
  const [testState, setTestState] = useState<Record<string, string>>({});
  const servers = settings.servers ?? [];

  const update = (id: string, patch: Partial<ServerConfig>) => {
    // Changing where a server points invalidates its fetched catalog — clear it (and the enabled
    // set) so stale models from the old endpoint don't linger; the next fetch reseeds. Hand-typed
    // models are kept.
    const resetsCatalog = "baseUrl" in patch || "provider" in patch;
    onChange({ ...settings, servers: servers.map(s => {
      if (s.id !== id) return s;
      const next = { ...s, ...patch };
      if (resetsCatalog) { next.catalog = undefined; next.models = []; }
      return next;
    }) });
  };
  const remove = (id: string) =>
    onChange({ ...settings, servers: servers.filter(s => s.id !== id) });
  const add = () =>
    onChange({ ...settings, servers: [...servers,
      { id: crypto.randomUUID(), name: `Server ${servers.length + 1}`, provider: "ollama", baseUrl: "http://localhost:11434" }] });

  const test = async (s: ServerConfig) => {
    setTestState(t => ({ ...t, [s.id]: "testing" }));
    try {
      const list = await invoke<string[]>("get_models",
        { args: { base_url: s.baseUrl, provider: s.provider, api_key: s.apiKey ?? null } });
      setTestState(t => ({ ...t, [s.id]: `ok:${list.length}` }));
    } catch (e) { setTestState(t => ({ ...t, [s.id]: `err:${String(e)}` })); }
  };

  return (
    <div className="admin-scroll" style={{ padding: "20px 20px 0" }}>
      <div className="server-section">
        <h3 className="server-heading">Inference Servers</h3>
        <p className="server-note" style={{ marginTop: 0 }}>
          Add any Ollama or OpenAI-compatible endpoint (OpenAI, Anthropic, Groq, OpenRouter, Together, Mistral,
          or local servers like LM Studio / llama.cpp / vLLM). Models from every server appear together in the
          model dropdown, prefixed with the server name, and each chat is routed to its model's server. Keys are
          stored locally.
        </p>

        {servers.map(s => {
          const st = testState[s.id] ?? "";
          return (
            <div key={s.id} style={{ border: "1px solid var(--border-color, #333)", borderRadius: 8, padding: 12, marginTop: 12 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <input className="admin-input" style={{ flex: "0 0 150px" }} value={s.name}
                  onChange={e => update(s.id, { name: e.target.value })} placeholder="Server name" />
                <select className="admin-input" style={{ flex: "0 0 140px" }} value={s.provider}
                  onChange={e => update(s.id, { provider: e.target.value as ProviderKind })}>
                  <option value="ollama">Ollama</option>
                  <option value="openai">OpenAI-compatible</option>
                </select>
                <select className="admin-input" style={{ flex: 1 }}
                  value={SERVER_PRESETS.find(p => p.provider === s.provider && p.baseUrl === s.baseUrl)?.label ?? ""}
                  onChange={e => { const p = SERVER_PRESETS.find(x => x.label === e.target.value); if (p) update(s.id, { provider: p.provider, baseUrl: p.baseUrl }); }}>
                  <option value="">Preset…</option>
                  {SERVER_PRESETS.map(p => <option key={p.label} value={p.label}>{p.label}</option>)}
                </select>
                <button className="icon-btn danger" onClick={() => remove(s.id)} title="Remove server">✕</button>
              </div>
              <input className="admin-input" style={{ fontFamily: "monospace", marginTop: 8 }} value={s.baseUrl}
                onChange={e => update(s.id, { baseUrl: e.target.value })}
                placeholder={s.provider === "openai" ? "https://api.openai.com/v1" : "http://localhost:11434"} />
              {s.provider === "openai" && (
                <input className="admin-input" type="password" style={{ fontFamily: "monospace", marginTop: 8 }} value={s.apiKey ?? ""}
                  onChange={e => update(s.id, { apiKey: e.target.value || undefined })}
                  placeholder="API key (blank for keyless local servers)" />
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 8 }}>
                <button className="btn" onClick={() => test(s)} disabled={st === "testing"}>Test</button>
                {st === "testing" && <span style={{ fontSize: 12, opacity: 0.6 }}>Testing…</span>}
                {st.startsWith("ok:") && <span style={{ fontSize: 12, color: "#4ade80" }}>✓ {st.slice(3)} models</span>}
                {st.startsWith("err:") && <span style={{ fontSize: 12, color: "#f87171" }}>✕ {st.slice(4)}</span>}
              </div>
            </div>
          );
        })}

        <button className="btn" style={{ marginTop: 12 }} onClick={add}>+ Add Server</button>
      </div>
    </div>
  );
}

// ── Defaults tab ─────────────────────────────────────────────────────────────

function DefaultsTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const params = settings.chatParams ?? DEFAULT_CHAT_PARAMS;
  const setParams = (p: ChatParams) => onChange({ ...settings, chatParams: p });

  return (
    <div className="admin-scroll" style={{ padding: "16px 20px" }}>
      <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Global Chat Defaults</div>
      <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 16, lineHeight: 1.5 }}>
        New chats use these settings. Profiles can override them, and you can always change them per-chat using the sliders icon in the input bar.
      </div>

      <section style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 12 }}>Presets</div>
        <ChatParamsDefaults params={params} onChange={setParams} />
      </section>

      <div style={{ height: 1, background: "var(--border)", marginBottom: 20 }} />

      <section>
        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 12 }}>Advanced</div>
        <AdvancedParamsContent draft={params} onChange={setParams} />
      </section>

      {settings.chatParams && (
        <button className="link-btn" style={{ marginTop: 16 }}
          onClick={() => onChange({ ...settings, chatParams: undefined })}>
          Reset all to factory defaults
        </button>
      )}
    </div>
  );
}

// ── Admin Panel modal ─────────────────────────────────────────────────────────

export function AdminPanel({ settings, onSave, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("tools");
  const [draft, setDraft] = useState<AppSettings>({ ...settings, servers: [...(settings.servers ?? [])], models: [...(settings.models ?? [])] });

  // The active profile (if any) or global settings is the "context" for MCP/OpenAPI/Sandbox/Server tabs
  const activeProfile = draft.profiles.find(p => p.id === draft.activeProfileId) ?? null;
  // Tools are now stored globally in toolRegistry; profiles reference by ID
  const ctxMCP     = draft.toolRegistry.mcpServers;
  const ctxOpenAPI = draft.toolRegistry.openapiSpecs;
  const ctxSparql  = draft.toolRegistry.sparqlEndpoints;
  const ctxDirs    = (activeProfile ? activeProfile.allowedDirs  : draft.allowedDirs)  ?? [];

  const setCtxMCP = (servers: StoredMCPServer[]) => {
    setDraft(d => {
      // Auto-enable any newly added servers in the active profile so they
      // aren't immediately excluded by syncServers after clicking Done.
      const prevIds = new Set(d.toolRegistry.mcpServers.map(s => s.id));
      const newIds = servers.filter(s => !prevIds.has(s.id)).map(s => s.id);
      const profiles = newIds.length > 0 && d.activeProfileId
        ? d.profiles.map(p => p.id === d.activeProfileId
            ? { ...p, enabledMcpServerIds: [...(p.enabledMcpServerIds ?? []), ...newIds] }
            : p)
        : d.profiles;
      return { ...d, toolRegistry: { ...d.toolRegistry, mcpServers: servers }, profiles };
    });
  };
  const setCtxOpenAPI = (specs: StoredOpenAPISpec[]) => {
    setDraft(d => {
      // Auto-enable any newly added specs in the active profile.
      const prevIds = new Set(d.toolRegistry.openapiSpecs.map(s => s.id));
      const newIds = specs.filter(s => !prevIds.has(s.id)).map(s => s.id);
      const profiles = newIds.length > 0 && d.activeProfileId
        ? d.profiles.map(p => p.id === d.activeProfileId
            ? { ...p, enabledOpenapiSpecIds: [...(p.enabledOpenapiSpecIds ?? []), ...newIds] }
            : p)
        : d.profiles;
      return { ...d, toolRegistry: { ...d.toolRegistry, openapiSpecs: specs }, profiles };
    });
  };
  const setCtxSparql = (endpoints: StoredSparqlEndpoint[]) => {
    setDraft(d => {
      // Auto-enable any newly added endpoints in the active profile.
      const prevIds = new Set(d.toolRegistry.sparqlEndpoints.map(s => s.id));
      const newIds = endpoints.filter(s => !prevIds.has(s.id)).map(s => s.id);
      const profiles = newIds.length > 0 && d.activeProfileId
        ? d.profiles.map(p => p.id === d.activeProfileId
            ? { ...p, enabledSparqlEndpointIds: [...(p.enabledSparqlEndpointIds ?? []), ...newIds] }
            : p)
        : d.profiles;
      return { ...d, toolRegistry: { ...d.toolRegistry, sparqlEndpoints: endpoints }, profiles };
    });
  };
  const setCtxDirs = (dirs: string[]) => {
    if (activeProfile) {
      setDraft(d => ({ ...d, profiles: d.profiles.map(p => p.id === activeProfile.id ? { ...p, allowedDirs: dirs } : p) }));
    } else {
      setDraft(d => ({ ...d, allowedDirs: dirs }));
    }
  };
  const tabs: { id: Tab; icon: string; label: string }[] = [
    { id: "profiles", icon: "🤖",  label: "Profiles" },
    { id: "tools",    icon: "⚡",  label: "Tools" },
    { id: "models",   icon: "🖥",  label: "Models" },
    { id: "openapi",  icon: "🌐",  label: "OpenAPI" },
    { id: "sparql",   icon: "🔗",  label: "SPARQL" },
    { id: "mcp",      icon: "🔌",  label: "MCP" },
    { id: "sandbox",  icon: "🔒",  label: "Sandbox" },
    { id: "server",   icon: "⚙️",  label: "Server" },
    { id: "defaults", icon: "🎛",  label: "Defaults" },
  ];

  const saveAndClose = () => { onSave(draft); onClose(); };

  return (
    // The panel batches edits into `draft`, so it only closes via the Done button (which
    // saves). A backdrop click is deliberately ignored so an accidental click outside can't
    // dismiss it and lose the changes.
    <div className="modal-overlay">
      <div className="admin-modal" onClick={e => e.stopPropagation()}>
        <div className="admin-header">
          <span className="admin-title">Admin</span>
          <button className="btn primary" onClick={saveAndClose}>Done</button>
        </div>
        <div className="admin-tabbar">
          {tabs.map(t => (
            <button key={t.id} className={`admin-tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
              <span>{t.icon}</span>
              <span>{t.label}</span>
            </button>
          ))}
        </div>
        <div className="admin-divider" />
        {activeProfile && ["mcp","openapi","sparql","tools","sandbox","server"].includes(tab) && (
          <div style={{ padding: "4px 16px", background: "var(--purple-bg)", borderBottom: "1px solid var(--purple-border)", fontSize: 11, color: "var(--purple)", display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
            <span>🤖</span> Profile: <strong>{activeProfile.name}</strong>
          </div>
        )}
        <div className="admin-content">
          {tab === "profiles" && <ProfilesTab settings={draft} onChange={setDraft} />}
          {tab === "tools"   && <ToolsTab   settings={draft} onChange={setDraft} />}
          {tab === "models"  && <ModelsTab  settings={draft} onChange={setDraft} />}
          {tab === "openapi" && <OpenAPITab stored={ctxOpenAPI} onChange={setCtxOpenAPI} />}
          {tab === "sparql"  && <SparqlTab stored={ctxSparql} onChange={setCtxSparql} />}
          {tab === "mcp"     && <MCPTab stored={ctxMCP} onChange={setCtxMCP} />}
          {tab === "sandbox" && <SandboxTab dirs={ctxDirs} onChange={setCtxDirs} />}
          {tab === "server"  && <ServerTab  settings={draft} onChange={setDraft} />}
          {tab === "defaults" && <DefaultsTab settings={draft} onChange={setDraft} />}
        </div>
      </div>
    </div>
  );
}
