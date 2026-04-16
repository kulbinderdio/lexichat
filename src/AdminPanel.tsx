import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

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

export interface Profile {
  id: string;
  name: string;
  systemPrompt: string;
  model: string;
  enabledTools: Record<string, boolean>;
  mcpServers: StoredMCPServer[];
  openapiSpecs: StoredOpenAPISpec[];
  maxTools: number;
}

export interface AppSettings {
  host: string;
  maxTools: number;
  numGPULayers: number | null;
  models: string[];
  enabledTools: Record<string, boolean>;
  mcpServers: StoredMCPServer[];
  openapiSpecs: StoredOpenAPISpec[];
  profiles: Profile[];
  activeProfileId: string | null;
}

interface SpecInfo {
  id: string;
  title: string;
  base_url: string;
  tool_count: number;
  tools: { name: string; description: string; method: string; path: string }[];
}

interface MCPServerInfo {
  id: string;
  name: string;
  command: string;
  args: string[];
  connected: boolean;
  tool_count: number;
  tools: { name: string; description: string }[];
  error?: string;
}

interface Props {
  settings: AppSettings;
  onSave: (s: AppSettings) => void;
  onClose: () => void;
}

type Tab = "profiles" | "tools" | "models" | "openapi" | "mcp" | "sandbox" | "server";

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
];

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
    const p: Profile = {
      id: uid(),
      name: "New Profile",
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      model: settings.models[0] ?? "",
      enabledTools: { ...settings.enabledTools },
      mcpServers: [],
      openapiSpecs: [],
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
        <div style={{ padding: 8, borderTop: "1px solid var(--border)" }}>
          <button className="btn primary" style={{ width: "100%", fontSize: 11 }} onClick={newProfile}>
            + New Profile
          </button>
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
              <select className="admin-input" value={d.model}
                onChange={e => setDraft({ ...d, model: e.target.value })}>
                <option value="">— same as global —</option>
                {settings.models.map(m => <option key={m} value={m}>{m}</option>)}
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

            <div className="field" style={{ marginBottom: 12 }}>
              <label>Max Tools per Query</label>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
                <button className="stepper-btn" onClick={() => setDraft({ ...d, maxTools: Math.max(5, d.maxTools - 5) })}>−</button>
                <span style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 600, minWidth: 28, textAlign: "center" }}>{d.maxTools}</span>
                <button className="stepper-btn" onClick={() => setDraft({ ...d, maxTools: Math.min(100, d.maxTools + 5) })}>+</button>
              </div>
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

// ── Tools tab ─────────────────────────────────────────────────────────────────

function ToolsTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const setToolEnabled = (name: string, val: boolean) =>
    onChange({ ...settings, enabledTools: { ...settings.enabledTools, [name]: val } });
  const setMaxTools = (v: number) => onChange({ ...settings, maxTools: v });

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
              checked={settings.enabledTools[t.name] !== false}
              onChange={e => setToolEnabled(t.name, e.target.checked)}
              className="admin-checkbox"
            />
            <span className="tool-icon">{t.icon}</span>
            <div className="admin-row-text">
              <span className="admin-row-title">{t.label}</span>
              <span className="admin-row-sub">{t.name}</span>
            </div>
          </label>
        ))}
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
      </section>
    </div>
  );
}

// ── Models tab ────────────────────────────────────────────────────────────────

function ModelsTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const [newModel, setNewModel] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const addModel = () => {
    const name = newModel.trim();
    if (!name || settings.models.includes(name)) return;
    onChange({ ...settings, models: [...settings.models, name] });
    setNewModel("");
  };

  const removeModel = (name: string) =>
    onChange({ ...settings, models: settings.models.filter(m => m !== name) });

  const refresh = async () => {
    setRefreshing(true);
    try {
      const fetched = await invoke<string[]>("get_models");
      const merged = [...fetched, ...settings.models.filter(m => !fetched.includes(m))];
      onChange({ ...settings, models: merged });
    } catch { }
    setRefreshing(false);
  };

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {settings.models.length === 0 ? (
          <div className="admin-empty">No models. Add one below or refresh from Ollama.</div>
        ) : (
          <section className="admin-section">
            {settings.models.map(m => (
              <div key={m} className="admin-row">
                <span style={{ fontSize: 13, opacity: 0.4 }}>🖥</span>
                <span style={{ flex: 1, fontFamily: "monospace", fontSize: 12 }}>{m}</span>
                <button className="icon-btn danger" onClick={() => removeModel(m)}>✕</button>
              </div>
            ))}
          </section>
        )}
      </div>
      <div className="admin-footer-bar">
        <input className="admin-input" value={newModel} onChange={e => setNewModel(e.target.value)}
          placeholder="Add model name…" onKeyDown={e => e.key === "Enter" && addModel()} />
        <button className="btn" onClick={addModel} disabled={!newModel.trim()}>Add</button>
        <button className="btn" onClick={refresh} disabled={refreshing}>{refreshing ? "…" : "↻ Refresh"}</button>
      </div>
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

// ── MCP tab ───────────────────────────────────────────────────────────────────

function MCPTab({ stored, onChange }: { stored: StoredMCPServer[]; onChange: (s: StoredMCPServer[]) => void }) {
  const [servers, setServers] = useState<MCPServerInfo[]>([]);
  const [showAdd, setShowAdd] = useState(false);
  const [name, setName] = useState("");
  const [command, setCommand] = useState("");
  const [argsStr, setArgsStr] = useState("");
  const [envStr, setEnvStr] = useState("");
  const [auth, setAuth] = useState<AuthConfig>(DEFAULT_AUTH);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [reconnecting, setReconnecting] = useState<Set<string>>(new Set());

  const isHttp = command.startsWith("http");

  useEffect(() => {
    invoke<MCPServerInfo[]>("list_mcp_servers").then(setServers).catch(() => {});
  }, []);

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
        args: { name: name.trim(), command: command.trim(), args: cmdArgs, env, auth: effectiveAuth }
      });
      const entry: StoredMCPServer = { id: info.id, name: name.trim(), command: command.trim(), args: cmdArgs, env, auth: effectiveAuth };
      setServers(prev => [...prev, info]);
      onChange([...stored, entry]);
      setShowAdd(false);
      setName(""); setCommand(""); setArgsStr(""); setEnvStr(""); setAuth(DEFAULT_AUTH);
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
      const updated = await invoke<MCPServerInfo>("reconnect_mcp_server", { id });
      setServers(prev => prev.map(s => s.id === id ? updated : s));
    } catch { }
    setReconnecting(prev => { const s = new Set(prev); s.delete(id); return s; });
  };

  const toggleExpand = (id: string) =>
    setExpanded(prev => { const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s; });

  return (
    <div className="admin-scroll" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1 }}>
        {servers.length === 0 && !showAdd && (
          <div className="admin-empty">No MCP servers connected. Add a server to extend LexiChat with custom tools.</div>
        )}

        {servers.map(srv => (
          <section key={srv.id} className="admin-section">
            <div className="admin-row">
              <button className="icon-btn" onClick={() => toggleExpand(srv.id)} style={{ fontSize: 9 }}>
                {expanded.has(srv.id) ? "▼" : "▶"}
              </button>
              <span style={{ fontSize: 14, marginRight: 2 }}>🖧</span>
              <div style={{ flex: 1 }}>
                <div className="admin-row-title">{srv.name}</div>
                <div className="admin-row-sub" style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontFamily: "monospace" }}>{srv.command}</span>
                  <span style={{ color: srv.connected ? "#4ade80" : srv.error ? "#f87171" : "#888",
                    fontSize: 10, background: (srv.connected ? "#4ade80" : "#f87171") + "22",
                    padding: "1px 6px", borderRadius: 10 }}>
                    {srv.connected ? `✓ ${srv.tool_count} tools` : srv.error ? "✕ Error" : "Disconnected"}
                  </span>
                </div>
              </div>
              {!srv.connected && (
                <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }}
                  onClick={() => reconnect(srv.id)} disabled={reconnecting.has(srv.id)}>
                  {reconnecting.has(srv.id) ? "…" : "↻"}
                </button>
              )}
              <button className="icon-btn danger" onClick={() => remove(srv.id)}>✕</button>
            </div>
            {expanded.has(srv.id) && (
              <div style={{ paddingLeft: 28 }}>
                {srv.error && <div style={{ fontSize: 11, color: "#f87171", padding: "4px 16px" }}>{srv.error}</div>}
                {srv.tools.map(t => (
                  <div key={t.name} className="admin-row" style={{ paddingTop: 3, paddingBottom: 3 }}>
                    <span style={{ color: "var(--purple)", fontSize: 10 }}>⚡</span>
                    <div className="admin-row-text">
                      <span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600 }}>{t.name}</span>
                      {t.description && <span className="admin-row-sub">{t.description}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        ))}

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
              {error && <div style={{ color: "#f87171", fontSize: 12 }}>{error}</div>}
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn" onClick={() => { setShowAdd(false); setError(""); }}>Cancel</button>
                <button className="btn primary" onClick={add} disabled={loading || !name.trim() || !command.trim()}>
                  {loading ? "Connecting…" : "Add Server"}
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

function SandboxTab() {
  const [dirs, setDirs] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    invoke<string[]>("get_allowed_dirs").then(setDirs).catch(() => {});
  }, []);

  const addFolder = async () => {
    setLoading(true);
    try {
      const selected = await open({ directory: true, multiple: false, title: "Select a folder to allow" });
      if (typeof selected === "string" && selected) {
        const updated = await invoke<string[]>("add_allowed_dir", { path: selected });
        setDirs(updated);
      }
    } catch { }
    setLoading(false);
  };

  const removeDir = async (path: string) => {
    const updated = await invoke<string[]>("remove_allowed_dir", { path });
    setDirs(updated);
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

function ServerTab({ settings, onChange }: { settings: AppSettings; onChange: (s: AppSettings) => void }) {
  const [testState, setTestState] = useState<"idle" | "testing" | "ok" | string>("idle");
  const gpuOn  = settings.numGPULayers !== null;
  const gpuVal = settings.numGPULayers ?? 999;

  const testConnection = async () => {
    setTestState("testing");
    try {
      await invoke<string[]>("get_models");
      setTestState("ok");
    } catch (e) { setTestState(String(e)); }
  };

  return (
    <div className="admin-scroll" style={{ padding: "20px 20px 0" }}>
      <div className="server-section">
        <h3 className="server-heading">Ollama Server</h3>
        <div className="field">
          <label>Server URL</label>
          <div style={{ display: "flex", gap: 8 }}>
            <input className="admin-input" style={{ flex: 1, fontFamily: "monospace" }}
              value={settings.host} onChange={e => onChange({ ...settings, host: e.target.value })}
              placeholder="http://localhost:11434" />
            <button className="btn" onClick={() => onChange({ ...settings, host: "http://localhost:11434" })}>Reset</button>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 10 }}>
          <button className="btn primary" onClick={testConnection} disabled={testState === "testing"}>
            Test Connection
          </button>
          {testState === "testing" && <span style={{ fontSize: 12, opacity: 0.6 }}>Testing…</span>}
          {testState === "ok" && <span style={{ fontSize: 12, color: "#4ade80" }}>✓ Connected</span>}
          {testState !== "idle" && testState !== "testing" && testState !== "ok" && (
            <span style={{ fontSize: 12, color: "#f87171" }}>✕ {testState}</span>
          )}
        </div>
        <p className="server-note">The URL of your Ollama instance. Change this to connect to a remote server.</p>
      </div>

      <div className="admin-divider" />

      <div className="server-section" style={{ paddingTop: 20 }}>
        <h3 className="server-heading">GPU Layers</h3>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <label className="toggle-row">
            <input type="checkbox" checked={gpuOn}
              onChange={e => onChange({ ...settings, numGPULayers: e.target.checked ? 999 : null })} />
            <span style={{ fontSize: 13 }}>Override GPU layer offload</span>
          </label>
          {gpuOn && (
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontSize: 13 }}>Layers:</span>
              <button className="stepper-btn" onClick={() => onChange({ ...settings, numGPULayers: Math.max(0, gpuVal - 1) })}>−</button>
              <span style={{ fontFamily: "monospace", fontSize: 14, fontWeight: 600, minWidth: 36, textAlign: "center" }}>{gpuVal}</span>
              <button className="stepper-btn" onClick={() => onChange({ ...settings, numGPULayers: Math.min(999, gpuVal + 1) })}>+</button>
              <button className="btn" onClick={() => onChange({ ...settings, numGPULayers: null })}>Reset</button>
            </div>
          )}
          {gpuOn && (
            <p className="server-note" style={{ marginTop: 0 }}>
              {gpuVal === 0 ? "CPU only." : gpuVal >= 999 ? "All layers on GPU." : `${gpuVal} layers on GPU, rest on CPU.`}
            </p>
          )}
          <p className="server-note">Leave off to let Ollama decide (recommended). Set to 0 for CPU-only, 999 for full GPU offload.</p>
        </div>
      </div>
    </div>
  );
}

// ── Admin Panel modal ─────────────────────────────────────────────────────────

export function AdminPanel({ settings, onSave, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("tools");
  const [draft, setDraft] = useState<AppSettings>({ ...settings, models: [...settings.models] });

  // The active profile (if any) or global settings is the "context" for MCP/OpenAPI tabs
  const activeProfile = draft.profiles.find(p => p.id === draft.activeProfileId) ?? null;
  const ctxMCP     = (activeProfile ? activeProfile.mcpServers   : draft.mcpServers)   ?? [];
  const ctxOpenAPI = (activeProfile ? activeProfile.openapiSpecs : draft.openapiSpecs) ?? [];

  const setCtxMCP = (servers: StoredMCPServer[]) => {
    if (activeProfile) {
      setDraft(d => ({ ...d, profiles: d.profiles.map(p => p.id === activeProfile.id ? { ...p, mcpServers: servers } : p) }));
    } else {
      setDraft(d => ({ ...d, mcpServers: servers }));
    }
  };
  const setCtxOpenAPI = (specs: StoredOpenAPISpec[]) => {
    if (activeProfile) {
      setDraft(d => ({ ...d, profiles: d.profiles.map(p => p.id === activeProfile.id ? { ...p, openapiSpecs: specs } : p) }));
    } else {
      setDraft(d => ({ ...d, openapiSpecs: specs }));
    }
  };

  const tabs: { id: Tab; icon: string; label: string }[] = [
    { id: "profiles", icon: "🤖",  label: "Profiles" },
    { id: "tools",    icon: "⚡",  label: "Tools" },
    { id: "models",   icon: "🖥",  label: "Models" },
    { id: "openapi",  icon: "🌐",  label: "OpenAPI" },
    { id: "mcp",      icon: "🔌",  label: "MCP" },
    { id: "sandbox",  icon: "🔒",  label: "Sandbox" },
    { id: "server",   icon: "⚙️",  label: "Server" },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="admin-modal" onClick={e => e.stopPropagation()}>
        <div className="admin-header">
          <span className="admin-title">Admin</span>
          <button className="btn primary" onClick={() => { onSave(draft); onClose(); }}>Done</button>
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
        {activeProfile && ["mcp","openapi","tools"].includes(tab) && (
          <div style={{ padding: "4px 16px", background: "var(--purple-bg)", borderBottom: "1px solid var(--purple-border)", fontSize: 11, color: "var(--purple)", display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
            <span>🤖</span> Profile: <strong>{activeProfile.name}</strong>
          </div>
        )}
        <div className="admin-content">
          {tab === "profiles" && <ProfilesTab settings={draft} onChange={setDraft} />}
          {tab === "tools"   && <ToolsTab   settings={draft} onChange={setDraft} />}
          {tab === "models"  && <ModelsTab  settings={draft} onChange={setDraft} />}
          {tab === "openapi" && <OpenAPITab stored={ctxOpenAPI} onChange={setCtxOpenAPI} />}
          {tab === "mcp"     && <MCPTab stored={ctxMCP} onChange={setCtxMCP} />}
          {tab === "sandbox" && <SandboxTab />}
          {tab === "server"  && <ServerTab  settings={draft} onChange={setDraft} />}
        </div>
      </div>
    </div>
  );
}
