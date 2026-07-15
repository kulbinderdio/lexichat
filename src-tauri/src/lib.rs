mod ollama;
mod tools;
mod openapi;
mod sparql;
mod mcp;
mod jobs;
mod history;
mod wiki;
mod sandbox;

use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};
use serde::{Deserialize, Serialize};
use openapi::RegisteredSpec;
use sparql::RegisteredSparqlEndpoint;
use mcp::{MCPServerConfig, MCPConnection, AuthConfig};

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    pub ollama_host:          Mutex<String>,
    pub conversation:         Mutex<Vec<ollama::WireMessage>>,
    pub openapi_specs:        Mutex<Vec<RegisteredSpec>>,
    pub sparql_endpoints:     Mutex<Vec<RegisteredSparqlEndpoint>>,
    pub mcp_servers:          Mutex<Vec<MCPServerConfig>>,
    pub mcp_connections:      tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    pub allowed_dirs:         Mutex<Vec<String>>,
    pub jobs:                 Mutex<Vec<jobs::ScheduledJob>>,
    pub job_runs:             Mutex<Vec<jobs::JobRun>>,
    pub tray:                 Mutex<Option<tauri::tray::TrayIcon<tauri::Wry>>>,
    /// Stores the last compose_email result during silent job runs so the
    /// subsequent sendmessage call can retrieve it without the model needing
    /// to pass the full base64 string through its context window.
    pub pending_email_raw:    Mutex<Option<String>>,
    /// Whether the user has approved code execution (run_python) for this session.
    /// Resets to false on app restart (session-toggle permission model).
    pub code_exec_unlocked:   Mutex<bool>,
    /// In-flight code-execution permission request: the agent loop parks a
    /// oneshot sender here while it waits for the frontend to approve/deny.
    pub pending_code_permission: Mutex<Option<tokio::sync::oneshot::Sender<bool>>>,
    /// Scratch slot: an MCP-App UI payload stashed by dispatch_tool for the agent
    /// loop to attach to the next `agent-tool-result` event.
    pub pending_tool_ui: Mutex<Option<ollama::ToolUiPayload>>,
    /// MCP server ids the user has approved to render/interact with apps this
    /// session (set by `approve_mcp_app`; reset on restart).
    pub apps_allowed: Mutex<std::collections::HashSet<String>>,
    /// Id of the saved conversation the current chat maps to, so auto-save
    /// updates the same record. `None` = a fresh chat not yet persisted.
    pub active_conversation_id: Mutex<Option<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        let saved = std::fs::read_to_string(allowed_dirs_path())
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        Self {
            ollama_host:     Mutex::new("http://localhost:11434".into()),
            conversation:    Mutex::new(Vec::new()),
            openapi_specs:   Mutex::new(Vec::new()),
            sparql_endpoints: Mutex::new(Vec::new()),
            mcp_servers:     Mutex::new(Vec::new()),
            mcp_connections: tokio::sync::Mutex::new(HashMap::new()),
            allowed_dirs:    Mutex::new(saved),
            jobs:            Mutex::new(jobs::load_jobs()),
            job_runs:        Mutex::new(jobs::load_runs()),
            tray:            Mutex::new(None),
            pending_email_raw: Mutex::new(None),
            code_exec_unlocked: Mutex::new(false),
            pending_code_permission: Mutex::new(None),
            pending_tool_ui: Mutex::new(None),
            apps_allowed: Mutex::new(std::collections::HashSet::new()),
            active_conversation_id: Mutex::new(None),
        }
    }
}

fn allowed_dirs_path() -> std::path::PathBuf {
    dirs_path().join("allowed_dirs.json")
}

pub fn dirs_path() -> std::path::PathBuf {
    let base = dirs::data_local_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let new_dir = base.join("lexichat");
    let _ = std::fs::create_dir_all(&new_dir);

    // One-time migration: copy allowed_dirs.json from the old "ai-agent-cross" data dir
    let old_file = base.join("ai-agent-cross").join("allowed_dirs.json");
    let new_file = new_dir.join("allowed_dirs.json");
    if old_file.exists() && !new_file.exists() {
        let _ = std::fs::copy(&old_file, &new_file);
    }

    new_dir
}

// ── Ollama commands ───────────────────────────────────────────────────────────

#[tauri::command]
async fn get_models(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let host = state.ollama_host.lock().unwrap().clone();
    ollama::list_models(&host).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn set_ollama_host(host: String, state: State<'_, AppState>) -> Result<(), String> {
    *state.ollama_host.lock().unwrap() = host;
    Ok(())
}

#[tauri::command]
async fn reset_conversation(state: State<'_, AppState>) -> Result<(), String> {
    state.conversation.lock().unwrap().clear();
    *state.active_conversation_id.lock().unwrap() = None;
    Ok(())
}

// ── Chat history commands ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ListConversationsArgs {
    #[serde(default)]
    pub profile_id: Option<String>,
}

/// List saved conversations scoped to the given profile (per-profile history).
/// A `None` profile matches conversations saved with no active profile.
#[tauri::command]
fn list_conversations(args: ListConversationsArgs) -> Vec<history::ConversationMeta> {
    history::load_index()
        .into_iter()
        .filter(|m| m.profile_id == args.profile_id)
        .collect()
}

#[derive(Deserialize)]
pub struct SaveConversationArgs {
    /// Opaque frontend ChatMessage[] for rendering.
    pub display: serde_json::Value,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub message_count: usize,
}

/// Persist the current chat, creating a record on first save and updating it
/// thereafter. The backend `conversation` (wire history) is the source of truth
/// for context; `display` is stored opaquely for rendering.
#[tauri::command]
fn save_active_conversation(
    args: SaveConversationArgs,
    state: State<'_, AppState>,
) -> Result<history::ConversationMeta, String> {
    let wire = state.conversation.lock().unwrap().clone();
    if wire.is_empty() {
        return Err("no conversation to save".into());
    }

    let mut active = state.active_conversation_id.lock().unwrap();
    let id = active.clone().unwrap_or_else(history::new_id);
    let now = history::now_secs();

    // Preserve created_at and any user-set title from an existing record.
    let existing = history::load_one(&id);
    let created_at = existing.as_ref().map(|c| c.meta.created_at).unwrap_or(now);
    let title = existing
        .as_ref()
        .map(|c| c.meta.title.clone())
        .unwrap_or_else(|| history::derive_title(&wire));

    let meta = history::ConversationMeta {
        id: id.clone(),
        title,
        profile_id: args.profile_id,
        model: args.model,
        created_at,
        updated_at: now,
        message_count: args.message_count,
    };
    let conv = history::Conversation { meta: meta.clone(), wire, display: args.display };
    history::save_one(&conv).map_err(|e| e.to_string())?;
    *active = Some(id);
    Ok(meta)
}

#[derive(Deserialize)]
pub struct ConversationIdArgs {
    pub id: String,
}

/// Load a saved conversation: restore its wire history as the active backend
/// context and return the display messages for the frontend to render.
#[tauri::command]
fn load_conversation(
    args: ConversationIdArgs,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let conv = history::load_one(&args.id).ok_or("conversation not found")?;
    *state.conversation.lock().unwrap() = conv.wire;
    *state.active_conversation_id.lock().unwrap() = Some(args.id);
    Ok(conv.display)
}

#[tauri::command]
fn delete_conversation(
    args: ConversationIdArgs,
    state: State<'_, AppState>,
) -> Result<(), String> {
    history::delete_one(&args.id).map_err(|e| e.to_string())?;
    let mut active = state.active_conversation_id.lock().unwrap();
    if active.as_deref() == Some(args.id.as_str()) {
        *active = None;
    }
    Ok(())
}

#[derive(Deserialize)]
pub struct RenameConversationArgs {
    pub id: String,
    pub title: String,
}

#[tauri::command]
fn rename_conversation(args: RenameConversationArgs) -> Result<(), String> {
    history::rename(&args.id, &args.title).map_err(|e| e.to_string())
}

// ── Send message ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SendMessageArgs {
    pub model: String,
    pub message: String,
    pub system_prompt: String,
    pub tools: Vec<ollama::ToolSchema>,
    #[serde(default)]
    pub image_paths: Vec<String>,
    /// Non-image files the user attached. The run_python sandbox is allowed to
    /// read/write these even if they fall outside the configured allowed dirs.
    #[serde(default)]
    pub file_paths: Vec<String>,
    // LLM generation options (all optional)
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub num_ctx: Option<i32>,
    #[serde(default)]
    pub num_predict: Option<i32>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub keep_alive: Option<String>,
    #[serde(default = "default_web_search_results")]
    pub web_search_results: usize,
    /// Max tool-calling rounds the agent loop may take before answering.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub disabled_mcp_tools: Vec<String>,
    /// Server IDs the active profile has enabled. Empty = no profile active (use all servers).
    #[serde(default)]
    pub enabled_mcp_server_ids: Vec<String>,
}

fn default_web_search_results() -> usize { 10 }
fn default_max_steps() -> usize { 20 }

#[tauri::command]
async fn send_message(
    args: SendMessageArgs,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    let host = state.ollama_host.lock().unwrap().clone();

    {
        // Base64-encode any attached images for vision models
        use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
        let images: Option<Vec<String>> = if args.image_paths.is_empty() {
            None
        } else {
            let encoded: Vec<String> = args.image_paths.iter()
                .filter_map(|p| std::fs::read(p).ok())
                .map(|bytes| B64.encode(&bytes))
                .collect();
            if encoded.is_empty() { None } else { Some(encoded) }
        };

        let mut conv = state.conversation.lock().unwrap();
        conv.push(ollama::WireMessage {
            role: "user".into(),
            content: Some(args.message.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            images,
        });
    }

    // Collect built-in tool schemas (frontend-provided: file tools, wiki, etc.)
    let builtin_tools = args.tools.clone();

    // The wiki tools are one workflow, not independent tools: the system prompt requires a
    // wiki_search before every write and a wiki_append to log.md after it. Discovery picks
    // tools per-message and would hand back only wiki_write, leaving the model unable to obey
    // its own rules — so hold them out of the pre-flight and always offer the whole group.
    let (wiki_tools, discoverable): (Vec<_>, Vec<_>) = builtin_tools
        .into_iter()
        .partition(|t| t.function.name.starts_with("wiki_"));

    // Tool discovery pre-flight: filter built-ins only — MCP/OpenAPI tools are
    // user-curated and must always be available regardless of the message content.
    let mut all_tools = ollama::discover_tools(&host, &args.model, &args.message, &discoverable).await;
    all_tools.extend(wiki_tools);

    // Always append OpenAPI tools (profile-scoped by set_openapi_specs)
    {
        let extra: Vec<ollama::ToolSchema> = state.openapi_specs.lock().unwrap().iter()
            .flat_map(|spec| spec.tools.iter()
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);
    }

    // Always append SPARQL endpoint tools (profile-scoped by set_sparql_endpoints)
    {
        let extra: Vec<ollama::ToolSchema> = state.sparql_endpoints.lock().unwrap().iter()
            .flat_map(|ep| ep.tools.iter()
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);
    }

    // Append MCP tools. Every registered server sits in the connection pool regardless of
    // profile, so `enabled_mcp_server_ids` is the sole authority on which are visible. It is
    // filtered strictly — an empty list means "none", so a profile with no MCP servers enabled
    // gets none. (The frontend passes the full set of IDs when there is no active profile.)
    {
        let connections = state.mcp_connections.lock().await;
        let extra: Vec<ollama::ToolSchema> = connections.iter()
            .filter(|(id, _)| args.enabled_mcp_server_ids.contains(*id))
            .flat_map(|(_, conn)| conn.tools.iter()
                .filter(|t| !args.disabled_mcp_tools.contains(&t.name))
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);
    }

    let specs_snapshot: Vec<openapi::RegisteredSpec> = state.openapi_specs.lock().unwrap().clone();
    let sparql_snapshot: Vec<sparql::RegisteredSparqlEndpoint> = state.sparql_endpoints.lock().unwrap().clone();
    let allowed_dirs_snapshot: Vec<String> = state.allowed_dirs.lock().unwrap().clone();

    let options = if args.temperature.is_some() || args.top_p.is_some() || args.top_k.is_some()
        || args.repeat_penalty.is_some() || args.seed.is_some()
        || args.num_ctx.is_some() || args.num_predict.is_some() || args.stop.is_some()
    {
        Some(ollama::ChatOptions {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repeat_penalty: args.repeat_penalty,
            seed: args.seed,
            num_ctx: args.num_ctx,
            num_predict: args.num_predict,
            stop: args.stop.clone(),
        })
    } else {
        None
    };

    ollama::agent_loop(
        &host,
        &args.model,
        &args.system_prompt,
        &all_tools,
        options,
        args.keep_alive.clone(),
        &state.conversation,
        specs_snapshot,
        sparql_snapshot,
        &state.mcp_connections,
        allowed_dirs_snapshot,
        args.file_paths.clone(), // sandbox may read/write attached files
        args.web_search_results,
        &app,
        false, // silent = false for interactive chat
        args.max_steps.clamp(1, 50), // configurable; default 20
    )
    .await
    .map_err(|e| e.to_string())
}

// ── Sandbox commands ──────────────────────────────────────────────────────────

/// Frontend's response to an `agent-permission-request` for run_python.
/// Resolves the oneshot the agent loop is awaiting. `approved = true` unlocks
/// code execution for the rest of the session.
#[tauri::command]
fn respond_code_permission(approved: bool, state: State<'_, AppState>) -> Result<(), String> {
    if let Some(tx) = state.pending_code_permission.lock().unwrap().take() {
        let _ = tx.send(approved);
    }
    Ok(())
}

#[tauri::command]
fn get_allowed_dirs(state: State<'_, AppState>) -> Vec<String> {
    state.allowed_dirs.lock().unwrap().clone()
}

#[tauri::command]
fn add_allowed_dir(path: String, state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let mut dirs = state.allowed_dirs.lock().unwrap();
    if !dirs.contains(&path) {
        dirs.push(path);
    }
    let result = dirs.clone();
    drop(dirs);
    persist_allowed_dirs(&state);
    Ok(result)
}

#[tauri::command]
fn remove_allowed_dir(path: String, state: State<'_, AppState>) -> Vec<String> {
    let mut dirs = state.allowed_dirs.lock().unwrap();
    dirs.retain(|d| d != &path);
    let result = dirs.clone();
    drop(dirs);
    persist_allowed_dirs(&state);
    result
}

fn persist_allowed_dirs(state: &State<'_, AppState>) {
    let dirs = state.allowed_dirs.lock().unwrap().clone();
    if let Ok(json) = serde_json::to_string(&dirs) {
        let _ = std::fs::write(allowed_dirs_path(), json);
    }
}

#[tauri::command]
fn write_file_text(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| e.to_string())
}

#[tauri::command]
fn save_document(path: String, content: String) -> Result<(), String> {
    crate::tools::save_document(&path, &content)
}

#[tauri::command]
fn read_file_text(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn read_image_data_url(path: String) -> Result<String, String> {
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let ext = std::path::Path::new(&path)
        .extension().and_then(|e| e.to_str()).unwrap_or("png").to_lowercase();
    let mime = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "gif"          => "image/gif",
        "webp"         => "image/webp",
        "bmp"          => "image/bmp",
        _              => "image/png",
    };
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(&bytes)))
}

#[tauri::command]
fn set_allowed_dirs(dirs: Vec<String>, state: State<'_, AppState>) -> Result<(), String> {
    *state.allowed_dirs.lock().unwrap() = dirs;
    Ok(())
}

/// Parse a set of OpenAPI specs and return their tool lists without touching AppState.
/// Used by the step builder to show tools for a specific profile's specs regardless
/// of which profile is currently active in the main chat.
#[tauri::command]
fn get_spec_tools(specs: Vec<serde_json::Value>) -> Vec<SpecInfo> {
    specs.iter().filter_map(|s| {
        let id    = s["id"].as_str().unwrap_or("").to_string();
        let title = s["title"].as_str().unwrap_or("").to_string();
        let base  = s["base_url"].as_str().unwrap_or("").to_string();
        let json  = s["spec_json"].as_str().unwrap_or("");
        if json.is_empty() { return None; }
        let tools = openapi::parse_spec(&title, &base, json).ok()?;
        Some(SpecInfo {
            id,
            title,
            base_url: base,
            tool_count: tools.len(),
            tools: tools.iter().map(|t| ToolInfo {
                name: t.name.clone(),
                description: t.description.clone(),
                method: t.method.clone(),
                path: t.path.clone(),
            }).collect(),
        })
    }).collect()
}

// ── Built-in tool schemas ─────────────────────────────────────────────────────

#[tauri::command]
fn get_builtin_schemas() -> Vec<serde_json::Value> {
    tools::all_builtin_schemas()
}

// ── OpenAPI commands ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RegisterSpecArgs {
    pub title: String,
    pub base_url: String,
    pub spec_json: String,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[derive(Serialize)]
pub struct SpecInfo {
    pub id: String,
    pub title: String,
    pub base_url: String,
    pub tool_count: usize,
    pub tools: Vec<ToolInfo>,
}

#[derive(Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub method: String,
    pub path: String,
}

#[tauri::command]
async fn register_openapi_spec(
    args: RegisterSpecArgs,
    state: State<'_, AppState>,
) -> Result<SpecInfo, String> {
    let tools = openapi::parse_spec(&args.title, &args.base_url, &args.spec_json)?;
    let id = uuid_v4();
    let info = SpecInfo {
        id: id.clone(),
        title: args.title.clone(),
        base_url: args.base_url.clone(),
        tool_count: tools.len(),
        tools: tools.iter().map(|t| ToolInfo {
            name: t.name.clone(),
            description: t.description.clone(),
            method: t.method.clone(),
            path: t.path.clone(),
        }).collect(),
    };
    let spec = RegisteredSpec {
        id,
        title: args.title,
        base_url: args.base_url,
        auth: args.auth,
        tools,
    };
    state.openapi_specs.lock().unwrap().push(spec);
    Ok(info)
}

#[tauri::command]
async fn remove_openapi_spec(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.openapi_specs.lock().unwrap().retain(|s| s.id != id);
    Ok(())
}

#[tauri::command]
async fn list_openapi_specs(state: State<'_, AppState>) -> Result<Vec<SpecInfo>, String> {
    let specs = state.openapi_specs.lock().unwrap();
    Ok(specs.iter().map(|s| SpecInfo {
        id: s.id.clone(),
        title: s.title.clone(),
        base_url: s.base_url.clone(),
        tool_count: s.tools.len(),
        tools: s.tools.iter().map(|t| ToolInfo {
            name: t.name.clone(),
            description: t.description.clone(),
            method: t.method.clone(),
            path: t.path.clone(),
        }).collect(),
    }).collect())
}

// ── SPARQL commands ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct RegisterSparqlArgs {
    pub title: String,
    pub endpoint_url: String,
    #[serde(default)] pub prefixes: String,
    #[serde(default)] pub schema_summary: String,
    #[serde(default)] pub example_queries: Vec<sparql::ExampleQuery>,
    #[serde(default)] pub usage_hint: String,
    #[serde(default)] pub auth: AuthConfig,
    #[serde(default = "default_read_only")] pub read_only: bool,
}

fn default_read_only() -> bool { true }

#[derive(Serialize)]
pub struct SparqlInfo {
    pub id: String,
    pub title: String,
    pub endpoint_url: String,
    pub tool_count: usize,
    pub tools: Vec<String>,
}

fn sparql_info(ep: &RegisteredSparqlEndpoint) -> SparqlInfo {
    SparqlInfo {
        id: ep.id.clone(),
        title: ep.title.clone(),
        endpoint_url: ep.endpoint_url.clone(),
        tool_count: ep.tools.len(),
        tools: ep.tools.iter().map(|t| t.name.clone()).collect(),
    }
}

fn build_endpoint(id: String, args: RegisterSparqlArgs) -> RegisteredSparqlEndpoint {
    let mut ep = RegisteredSparqlEndpoint {
        id,
        title: args.title,
        endpoint_url: args.endpoint_url,
        prefixes: args.prefixes,
        schema_summary: args.schema_summary,
        example_queries: args.example_queries,
        usage_hint: args.usage_hint,
        auth: args.auth,
        read_only: args.read_only,
        tools: Vec::new(),
    };
    ep.tools = sparql::build_tools(&ep);
    ep
}

#[tauri::command]
async fn register_sparql_endpoint(
    args: RegisterSparqlArgs,
    state: State<'_, AppState>,
) -> Result<SparqlInfo, String> {
    if args.title.trim().is_empty() || args.endpoint_url.trim().is_empty() {
        return Err("Title and endpoint URL are required".into());
    }
    let ep = build_endpoint(uuid_v4(), args);
    let info = sparql_info(&ep);
    state.sparql_endpoints.lock().unwrap().push(ep);
    Ok(info)
}

#[tauri::command]
async fn remove_sparql_endpoint(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.sparql_endpoints.lock().unwrap().retain(|e| e.id != id);
    Ok(())
}

#[tauri::command]
async fn list_sparql_endpoints(state: State<'_, AppState>) -> Result<Vec<SparqlInfo>, String> {
    Ok(state.sparql_endpoints.lock().unwrap().iter().map(sparql_info).collect())
}

/// Compute the tools for a set of endpoints without touching AppState — used by the
/// Admin SPARQL tab so tool info shows regardless of which profile is active.
#[tauri::command]
fn get_sparql_tools(endpoints: Vec<serde_json::Value>) -> Vec<SparqlInfo> {
    endpoints.iter().filter_map(|e| {
        let id    = e["id"].as_str().unwrap_or("").to_string();
        let title = e["title"].as_str().unwrap_or("").to_string();
        let url   = e["endpoint_url"].as_str().unwrap_or("").to_string();
        if title.is_empty() || url.is_empty() { return None; }
        let ep = build_endpoint(id, RegisterSparqlArgs {
            title,
            endpoint_url: url,
            prefixes: e["prefixes"].as_str().unwrap_or("").to_string(),
            schema_summary: e["schema_summary"].as_str().unwrap_or("").to_string(),
            example_queries: serde_json::from_value(e["example_queries"].clone()).unwrap_or_default(),
            usage_hint: e["usage_hint"].as_str().unwrap_or("").to_string(),
            auth: serde_json::from_value(e["auth"].clone()).unwrap_or_default(),
            read_only: e["read_only"].as_bool().unwrap_or(true),
        });
        Some(sparql_info(&ep))
    }).collect()
}

#[derive(Deserialize)]
pub struct SyncSparqlInput {
    pub id: String,
    pub title: String,
    pub endpoint_url: String,
    #[serde(default)] pub prefixes: String,
    #[serde(default)] pub schema_summary: String,
    #[serde(default)] pub example_queries: Vec<sparql::ExampleQuery>,
    #[serde(default)] pub usage_hint: String,
    #[serde(default)] pub auth: AuthConfig,
    #[serde(default = "default_read_only")] pub read_only: bool,
}

#[tauri::command]
async fn set_sparql_endpoints(
    endpoints: Vec<SyncSparqlInput>,
    state: State<'_, AppState>,
) -> Result<Vec<SparqlInfo>, String> {
    let mut registered = Vec::new();
    let mut infos = Vec::new();
    for input in endpoints {
        let ep = build_endpoint(input.id, RegisterSparqlArgs {
            title: input.title,
            endpoint_url: input.endpoint_url,
            prefixes: input.prefixes,
            schema_summary: input.schema_summary,
            example_queries: input.example_queries,
            usage_hint: input.usage_hint,
            auth: input.auth,
            read_only: input.read_only,
        });
        infos.push(sparql_info(&ep));
        registered.push(ep);
    }
    *state.sparql_endpoints.lock().unwrap() = registered;
    Ok(infos)
}

#[derive(Deserialize)]
pub struct DiscoverSparqlArgs {
    pub endpoint_url: String,
    #[serde(default)] pub auth: AuthConfig,
}

#[tauri::command]
async fn discover_sparql_endpoint(args: DiscoverSparqlArgs) -> Result<sparql::DiscoveryResult, String> {
    Ok(sparql::probe(&args.endpoint_url, &args.auth).await)
}

// ── MCP commands ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AddMCPServerArgs {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub enable_apps: bool,
    /// If provided, reuse this ID instead of generating a new one (used when
    /// re-registering an existing stored server that was evicted from Rust state).
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct MCPServerInfo {
    pub id: String,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub connected: bool,
    pub tool_count: usize,
    pub tools: Vec<MCPToolInfo>,
    pub error: Option<String>,
    pub enable_apps: bool,
}

#[derive(Serialize, Clone)]
pub struct MCPToolInfo {
    pub name: String,
    pub description: String,
    /// True if this tool declares an MCP-App UI resource.
    pub has_ui: bool,
}

#[tauri::command]
async fn add_mcp_server(
    args: AddMCPServerArgs,
    state: State<'_, AppState>,
) -> Result<MCPServerInfo, String> {
    let id = args.id.clone().unwrap_or_else(uuid_v4);
    // Remove any stale entry with this ID before re-registering
    state.mcp_servers.lock().unwrap().retain(|s| s.id != id);
    state.mcp_connections.lock().await.remove(&id);
    let config = MCPServerConfig {
        id: id.clone(),
        name: args.name.clone(),
        command: args.command.clone(),
        args: args.args.clone(),
        env: args.env.clone(),
        enabled: true,
        auth: args.auth,
        enable_apps: args.enable_apps,
    };

    state.mcp_servers.lock().unwrap().push(config.clone());

    // Try to connect
    match MCPConnection::connect(config.clone()).await {
        Ok(conn) => {
            let tools: Vec<MCPToolInfo> = conn.tools.iter().map(|t| MCPToolInfo {
                name: t.name.clone(),
                description: t.description.clone(),
                has_ui: t.ui_resource_uri.is_some(),
            }).collect();
            let tool_count = tools.len();
            state.mcp_connections.lock().await.insert(id.clone(), conn);
            Ok(MCPServerInfo { id, name: args.name, command: args.command, args: args.args, connected: true, tool_count, tools, error: None, enable_apps: config.enable_apps })
        }
        Err(e) => {
            Ok(MCPServerInfo { id, name: args.name, command: args.command, args: args.args, connected: false, tool_count: 0, tools: vec![], error: Some(e), enable_apps: config.enable_apps })
        }
    }
}

#[tauri::command]
async fn remove_mcp_server(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state.mcp_servers.lock().unwrap().retain(|s| s.id != id);
    state.mcp_connections.lock().await.remove(&id);
    Ok(())
}

#[tauri::command]
async fn list_mcp_servers(state: State<'_, AppState>) -> Result<Vec<MCPServerInfo>, String> {
    let servers = state.mcp_servers.lock().unwrap().clone();
    let connections = state.mcp_connections.lock().await;
    Ok(servers.iter().map(|s| {
        let conn = connections.get(&s.id);
        let tools: Vec<MCPToolInfo> = conn.map(|c| c.tools.iter().map(|t| MCPToolInfo {
            name: t.name.clone(), description: t.description.clone(),
            has_ui: t.ui_resource_uri.is_some(),
        }).collect()).unwrap_or_default();
        MCPServerInfo {
            id: s.id.clone(),
            name: s.name.clone(),
            command: s.command.clone(),
            args: s.args.clone(),
            connected: conn.is_some(),
            tool_count: tools.len(),
            tools,
            error: None,
            enable_apps: s.enable_apps,
        }
    }).collect())
}

#[tauri::command]
async fn reconnect_mcp_server(id: String, state: State<'_, AppState>) -> Result<MCPServerInfo, String> {
    let config = {
        let servers = state.mcp_servers.lock().unwrap();
        servers.iter().find(|s| s.id == id).cloned().ok_or("Server not found")?
    };
    state.mcp_connections.lock().await.remove(&id);
    match MCPConnection::connect(config.clone()).await {
        Ok(conn) => {
            let tools: Vec<MCPToolInfo> = conn.tools.iter().map(|t| MCPToolInfo { name: t.name.clone(), description: t.description.clone(), has_ui: t.ui_resource_uri.is_some() }).collect();
            let tool_count = tools.len();
            let enable_apps = config.enable_apps;
            state.mcp_connections.lock().await.insert(id.clone(), conn);
            Ok(MCPServerInfo { id, name: config.name, command: config.command, args: config.args, connected: true, tool_count, tools, error: None, enable_apps })
        }
        Err(e) => Ok(MCPServerInfo { id, name: config.name, command: config.command, args: config.args, connected: false, tool_count: 0, tools: vec![], error: Some(e), enable_apps: config.enable_apps })
    }
}

// ── Profile-aware sync commands ───────────────────────────────────────────────
// Called whenever the active profile changes. Replaces all runtime MCP/OpenAPI
// state with the profile's (or Default's) isolated configuration.

#[derive(Deserialize)]
pub struct SyncMCPInput {
    pub id: String,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub enable_apps: bool,
}

#[tauri::command]
async fn set_mcp_servers(
    servers: Vec<SyncMCPInput>,
    state: State<'_, AppState>,
) -> Result<Vec<MCPServerInfo>, String> {
    // Drop all existing connections and configs
    state.mcp_connections.lock().await.clear();
    state.mcp_servers.lock().unwrap().clear();

    let mut results = Vec::new();
    for srv in servers {
        let config = MCPServerConfig {
            id: srv.id.clone(), name: srv.name.clone(),
            command: srv.command.clone(), args: srv.args.clone(),
            env: srv.env.clone(), enabled: true, auth: srv.auth,
            enable_apps: srv.enable_apps,
        };
        state.mcp_servers.lock().unwrap().push(config.clone());
        match MCPConnection::connect(config).await {
            Ok(conn) => {
                let tools: Vec<MCPToolInfo> = conn.tools.iter()
                    .map(|t| MCPToolInfo { name: t.name.clone(), description: t.description.clone(), has_ui: t.ui_resource_uri.is_some() })
                    .collect();
                let tool_count = tools.len();
                state.mcp_connections.lock().await.insert(srv.id.clone(), conn);
                results.push(MCPServerInfo { id: srv.id, name: srv.name, command: srv.command,
                    args: srv.args, connected: true, tool_count, tools, error: None, enable_apps: srv.enable_apps });
            }
            Err(e) => {
                results.push(MCPServerInfo { id: srv.id, name: srv.name, command: srv.command,
                    args: srv.args, connected: false, tool_count: 0, tools: vec![], error: Some(e), enable_apps: srv.enable_apps });
            }
        }
    }
    Ok(results)
}

// ── MCP Apps (SEP-1865) ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ApproveAppArgs { pub server_id: String }

/// Grant an MCP server permission to render/interact with apps for this session.
/// Called by the frontend after the user approves the consent prompt.
#[tauri::command]
fn approve_mcp_app(args: ApproveAppArgs, state: State<'_, AppState>) -> Result<(), String> {
    state.apps_allowed.lock().unwrap().insert(args.server_id);
    Ok(())
}

#[derive(Deserialize)]
pub struct McpUiCallArgs {
    pub server_id: String,
    pub tool_name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

/// Proxy an MCP-App (iframe) initiated `tools/call` to the live MCP connection.
/// Consent-gated: the server must have been approved via `approve_mcp_app`.
#[tauri::command]
async fn mcp_ui_call_tool(
    args: McpUiCallArgs,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    if !state.apps_allowed.lock().unwrap().contains(&args.server_id) {
        return Err("This MCP app is not approved to call tools in this session.".into());
    }
    let mut conns = state.mcp_connections.lock().await;
    let conn = conns.get_mut(&args.server_id).ok_or("MCP server not connected")?;
    let rich = conn.call_tool_rich(&args.tool_name, &args.arguments).await;
    Ok(serde_json::json!({
        "text": rich.text,
        "structured": rich.structured,
        "isError": rich.is_error,
        "uiHtml": rich.ui_html,
        "uiUri": rich.ui_uri,
    }))
}

#[derive(Deserialize)]
pub struct SyncOpenAPIInput {
    pub id: String,
    pub title: String,
    pub base_url: String,
    pub spec_json: String,
    #[serde(default)]
    pub auth: AuthConfig,
}

#[tauri::command]
async fn set_openapi_specs(
    specs: Vec<SyncOpenAPIInput>,
    state: State<'_, AppState>,
) -> Result<Vec<SpecInfo>, String> {
    let mut registered = Vec::new();
    let mut infos = Vec::new();
    for input in specs {
        match openapi::parse_spec(&input.title, &input.base_url, &input.spec_json) {
            Ok(tools) => {
                infos.push(SpecInfo {
                    id: input.id.clone(), title: input.title.clone(),
                    base_url: input.base_url.clone(), tool_count: tools.len(),
                    tools: tools.iter().map(|t| ToolInfo {
                        name: t.name.clone(), description: t.description.clone(),
                        method: t.method.clone(), path: t.path.clone(),
                    }).collect(),
                });
                registered.push(openapi::RegisteredSpec {
                    id: input.id, title: input.title, base_url: input.base_url,
                    auth: input.auth, tools,
                });
            }
            Err(_) => {} // skip malformed specs silently
        }
    }
    *state.openapi_specs.lock().unwrap() = registered;
    Ok(infos)
}

// ── OAuth2 authorization-code flow ────────────────────────────────────────────

#[derive(Serialize)]
pub struct OAuth2Tokens {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Deserialize)]
pub struct OAuth2AuthorizeArgs {
    pub authorization_url: String,
    pub token_url: String,
    pub client_id: String,
    pub client_secret: String,
    pub scope: String,
}

#[tauri::command]
async fn oauth2_authorize(
    args: OAuth2AuthorizeArgs,
    app: AppHandle,
) -> Result<OAuth2Tokens, String> {
    let OAuth2AuthorizeArgs { authorization_url, token_url, client_id, client_secret, scope } = args;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // Bind loopback on a random port
    let listener = TcpListener::bind("127.0.0.1:0").await.map_err(|e| e.to_string())?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let redirect_uri = format!("http://127.0.0.1:{port}/callback");

    // Build the browser URL
    let auth_url = format!(
        "{}?client_id={}&redirect_uri={}&scope={}&response_type=code&access_type=offline&prompt=consent",
        authorization_url,
        urlencoding::encode(&client_id),
        urlencoding::encode(&redirect_uri),
        urlencoding::encode(&scope),
    );

    // Open the user's browser via the opener plugin
    use tauri_plugin_opener::OpenerExt as _;
    app.opener()
        .open_url(&auth_url, None::<String>)
        .map_err(|e| format!("Could not open browser: {e}"))?;

    // Wait for the redirect (with a 5-minute timeout)
    let code = tokio::time::timeout(std::time::Duration::from_secs(300), async move {
        loop {
            let (mut stream, _) = listener.accept().await.map_err(|e| e.to_string())?;
            let mut buf = vec![0u8; 8192];
            let n = stream.read(&mut buf).await.map_err(|e| e.to_string())?;
            let request = String::from_utf8_lossy(&buf[..n]).to_string();

            // Parse code from first request line: "GET /callback?code=xxx HTTP/1.1"
            let code = request.lines().next()
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|path| {
                    let qs = path.splitn(2, '?').nth(1).unwrap_or("");
                    url::form_urlencoded::parse(qs.as_bytes())
                        .find(|(k, _)| k == "code")
                        .map(|(_, v)| v.into_owned())
                });

            // Respond to the browser
            let html = if code.is_some() {
                "<html><body style='font-family:sans-serif;text-align:center;padding:60px'><h2>✓ Authorized!</h2><p>You can close this tab and return to LexiChat.</p></body></html>"
            } else {
                "<html><body style='font-family:sans-serif;text-align:center;padding:60px'><h2>⚠ No code received</h2><p>Authorization failed. Close this tab and try again.</p></body></html>"
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                html.len(), html
            );
            let _ = stream.write_all(response.as_bytes()).await;

            if let Some(c) = code { return Ok(c); }
        }
    })
    .await
    .map_err(|_| "Authorization timed out after 5 minutes".to_string())?
    .map_err(|e: String| e)?;

    // Exchange code for tokens
    let client = reqwest::Client::new();
    let resp = client.post(&token_url)
        .form(&[
            ("code",          code.as_str()),
            ("client_id",     client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("redirect_uri",  redirect_uri.as_str()),
            ("grant_type",    "authorization_code"),
        ])
        .send().await.map_err(|e| format!("Token exchange failed: {e}"))?;

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("Token parse failed: {e}"))?;

    if let Some(err) = json["error"].as_str() {
        let desc = json["error_description"].as_str().unwrap_or("");
        return Err(format!("OAuth2 error: {err} — {desc}"));
    }

    Ok(OAuth2Tokens {
        access_token:  json["access_token"].as_str().unwrap_or("").to_string(),
        refresh_token: json["refresh_token"].as_str().unwrap_or("").to_string(),
    })
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().subsec_nanos();
    format!("{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        t, t >> 16, t & 0xfff, (t >> 4) & 0x3fff | 0x8000, t as u64 * 0x1000193)
}

// ── Menu ─────────────────────────────────────────────────────────────────────

fn build_menu(app: &tauri::App) -> tauri::Result<()> {
    use tauri::menu::{AboutMetadata, MenuBuilder, PredefinedMenuItem, SubmenuBuilder};

    let icon = app.default_window_icon().cloned();

    let about_item = PredefinedMenuItem::about(
        app,
        Some("About LexiChat"),
        Some(AboutMetadata {
            name:      Some("LexiChat".into()),
            version:   Some(env!("CARGO_PKG_VERSION").into()),
            copyright: Some("© 2024 LexiChat".into()),
            icon,
            ..Default::default()
        }),
    )?;

    let app_submenu = SubmenuBuilder::new(app, "LexiChat")
        .item(&about_item)
        .separator()
        .item(&PredefinedMenuItem::services(app, None)?)
        .separator()
        .item(&PredefinedMenuItem::hide(app, Some("Hide LexiChat"))?)
        .item(&PredefinedMenuItem::hide_others(app, None)?)
        .item(&PredefinedMenuItem::show_all(app, None)?)
        .separator()
        .item(&PredefinedMenuItem::quit(app, Some("Quit LexiChat"))?)
        .build()?;

    let edit_submenu = SubmenuBuilder::new(app, "Edit")
        .item(&PredefinedMenuItem::undo(app, None)?)
        .item(&PredefinedMenuItem::redo(app, None)?)
        .separator()
        .item(&PredefinedMenuItem::cut(app, None)?)
        .item(&PredefinedMenuItem::copy(app, None)?)
        .item(&PredefinedMenuItem::paste(app, None)?)
        .item(&PredefinedMenuItem::select_all(app, None)?)
        .build()?;

    let menu = MenuBuilder::new(app)
        .item(&app_submenu)
        .item(&edit_submenu)
        .build()?;

    app.set_menu(menu)?;
    Ok(())
}

// ── Scheduled job commands ────────────────────────────────────────────────────

#[tauri::command]
fn get_jobs(state: State<'_, AppState>) -> Vec<jobs::ScheduledJob> {
    state.jobs.lock().unwrap().clone()
}

#[tauri::command]
fn save_job(job: jobs::ScheduledJob, state: State<'_, AppState>, app: AppHandle) -> Result<(), String> {
    let mut stored = state.jobs.lock().unwrap();
    if let Some(pos) = stored.iter().position(|j| j.id == job.id) {
        stored[pos] = job.clone();
    } else {
        stored.push(job.clone());
    }
    let list = stored.clone();
    drop(stored);
    let r = jobs::save_jobs(&list).map_err(|e| e.to_string());
    update_tray_tooltip(&app);
    r
}

#[tauri::command]
fn delete_job(id: String, state: State<'_, AppState>, app: AppHandle) -> Result<(), String> {
    let mut stored = state.jobs.lock().unwrap();
    stored.retain(|j| j.id != id);
    let list = stored.clone();
    drop(stored);
    let r = jobs::save_jobs(&list).map_err(|e| e.to_string());
    update_tray_tooltip(&app);
    r
}

#[tauri::command]
async fn run_job_now(
    id: String,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<jobs::JobRun, String> {
    let job = state.jobs.lock().unwrap()
        .iter()
        .find(|j| j.id == id)
        .cloned()
        .ok_or_else(|| "Job not found".to_string())?;

    let (started_at, output, trace, error) = jobs::execute_job(&job, &state, &app).await;
    let finished_at = chrono::Utc::now();
    let duration_ms = (finished_at - started_at).num_milliseconds().max(0) as u64;

    let run = jobs::JobRun {
        id:           uuid_v4(),
        job_id:       job.id.clone(),
        job_name:     job.name.clone(),
        profile_name: job.profile_name.clone(),
        started_at,
        finished_at,
        duration_ms,
        status:       if error.is_none() { jobs::RunStatus::Success } else { jobs::RunStatus::Error },
        output,
        error,
        trace,
        _ran_at_legacy: None,
    };

    jobs::append_run(run.clone()).map_err(|e| e.to_string())?;

    {
        let mut stored = state.jobs.lock().unwrap();
        if let Some(j) = stored.iter_mut().find(|j| j.id == id) {
            j.last_run_at = Some(chrono::Utc::now());
        }
        let list = stored.clone();
        drop(stored);
        jobs::save_jobs(&list).map_err(|e| e.to_string())?;
    }

    use tauri::Emitter;
    let _ = app.emit("job-run-done", &run);

    Ok(run)
}

#[tauri::command]
fn get_job_runs(job_id: Option<String>) -> Vec<jobs::JobRun> {
    let runs = jobs::load_runs();
    match job_id {
        Some(id) => runs.into_iter().filter(|r| r.job_id == id).collect(),
        None     => runs,
    }
}

#[tauri::command]
fn clear_job_runs(job_id: Option<String>) -> Result<(), String> {
    let mut runs = jobs::load_runs();
    match &job_id {
        Some(id) => runs.retain(|r| &r.job_id != id),
        None     => runs.clear(),
    }
    let json = serde_json::to_string_pretty(&runs).map_err(|e| e.to_string())?;
    std::fs::write(jobs::runs_path(), json).map_err(|e| e.to_string())
}

// ── Tray ──────────────────────────────────────────────────────────────────────

fn toggle_window(app: &AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        if w.is_visible().unwrap_or(false) {
            let _ = w.hide();
        } else {
            let _ = w.show();
            let _ = w.set_focus();
        }
    }
}

pub fn update_tray_tooltip(app: &AppHandle) {
    let job_count = app.try_state::<AppState>()
        .map(|s| s.jobs.lock().map(|j| j.iter().filter(|x| x.enabled).count()).unwrap_or(0))
        .unwrap_or(0);
    let tip = if job_count == 0 {
        "LexiChat".to_string()
    } else {
        format!("LexiChat — {job_count} scheduled job{}", if job_count == 1 { "" } else { "s" })
    };
    if let Some(state) = app.try_state::<AppState>() {
        if let Ok(guard) = state.tray.lock() {
            if let Some(tray) = guard.as_ref() {
                let _ = tray.set_tooltip(Some(tip.as_str()));
            }
        }
    }
}

fn setup_tray(app: &mut tauri::App) -> tauri::Result<()> {
    use tauri::{
        menu::{Menu, MenuItem, PredefinedMenuItem},
        tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    };

    let open = MenuItem::with_id(app, "open", "Open LexiChat", true, None::<&str>)?;
    let sep  = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit LexiChat", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &sep, &quit])?;

    let tray = TrayIconBuilder::with_id("main")
        .icon(app.default_window_icon().unwrap().clone())
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("LexiChat")
        .on_menu_event(|app: &AppHandle, event| match event.id.as_ref() {
            "open" => {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.set_focus();
                }
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray: &tauri::tray::TrayIcon<tauri::Wry>, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                toggle_window(tray.app_handle());
            }
        })
        .build(app)?;

    // Store in AppState so update_tray_tooltip can reach it
    app.state::<AppState>().tray.lock().unwrap().replace(tray);

    Ok(())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            build_menu(app)?;
            jobs::spawn_scheduler(app.handle().clone());
            setup_tray(app)?;

            // Hide window on close instead of quitting — keeps scheduler alive
            if let Some(window) = app.get_webview_window("main") {
                let w = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = w.hide();
                    }
                });
            }

            update_tray_tooltip(app.handle());
            Ok(())
        })
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_models,
            set_ollama_host,
            reset_conversation,
            send_message,
            get_builtin_schemas,
            get_spec_tools,
            register_openapi_spec,
            remove_openapi_spec,
            list_openapi_specs,
            register_sparql_endpoint,
            remove_sparql_endpoint,
            list_sparql_endpoints,
            get_sparql_tools,
            set_sparql_endpoints,
            discover_sparql_endpoint,
            add_mcp_server,
            remove_mcp_server,
            list_mcp_servers,
            reconnect_mcp_server,
            set_mcp_servers,
            approve_mcp_app,
            mcp_ui_call_tool,
            set_openapi_specs,
            oauth2_authorize,
            get_allowed_dirs,
            add_allowed_dir,
            remove_allowed_dir,
            set_allowed_dirs,
            respond_code_permission,
            write_file_text,
            read_file_text,
            read_image_data_url,
            save_document,
            get_jobs,
            save_job,
            delete_job,
            run_job_now,
            get_job_runs,
            clear_job_runs,
            list_conversations,
            save_active_conversation,
            load_conversation,
            delete_conversation,
            rename_conversation,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            // On macOS, clicking the Dock icon while all windows are hidden restores the window
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen { has_visible_windows, .. } = &event {
                if !has_visible_windows {
                    if let Some(w) = app.get_webview_window("main") {
                        let _ = w.show();
                        let _ = w.set_focus();
                    }
                }
            }
            let _ = event; // suppress unused warning on non-mac
        });
}
