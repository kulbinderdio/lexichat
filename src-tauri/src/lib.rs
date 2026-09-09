mod ollama;
mod tools;
mod report;
mod openapi;
mod sparql;
mod mcp;
mod jobs;
mod job_designer;
mod history;
mod wiki;
mod wiki_index;
mod skills;
mod image_gen;
mod usage;

use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, State};
use serde::{Deserialize, Serialize};
use openapi::RegisteredSpec;
use sparql::RegisteredSparqlEndpoint;
use mcp::{MCPServerConfig, MCPConnection, AuthConfig};

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    /// Active inference backend (Ollama or an OpenAI-compatible endpoint) + its base URL/key.
    pub backend:              Mutex<ollama::Backend>,
    pub conversation:         Mutex<Vec<ollama::WireMessage>>,
    pub openapi_specs:        Mutex<Vec<RegisteredSpec>>,
    pub sparql_endpoints:     Mutex<Vec<RegisteredSparqlEndpoint>>,
    pub mcp_servers:          Mutex<Vec<MCPServerConfig>>,
    pub mcp_connections:      tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    pub allowed_dirs:         Mutex<Vec<String>>,
    pub skills:               Mutex<Vec<skills::RegisteredSkill>>,
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
    /// In-flight `run_python` executions, keyed by request id: the agent loop parks a oneshot
    /// here while the Pyodide worker in the webview runs the code; `respond_python_result` routes
    /// the answer back by id. A map (not a single slot) so a background job and an interactive
    /// chat can each have a run in flight without clobbering each other. See `ollama.rs`.
    pub pending_python_result: Mutex<HashMap<u64, tokio::sync::oneshot::Sender<ollama::PyResult>>>,
    /// Monotonic id source for `run_python` requests.
    pub python_request_seq: std::sync::atomic::AtomicU64,
    /// Scratch slot: an MCP-App UI payload stashed by dispatch_tool for the agent
    /// loop to attach to the next `agent-tool-result` event.
    pub pending_tool_ui: Mutex<Option<ollama::ToolUiPayload>>,
    /// Base64 image `data:` URLs from the last tool result, rendered inline regardless of the
    /// MCP-App/approval flow. Taken by the agent loop when it emits the tool-result event.
    pub pending_tool_images: Mutex<Vec<String>>,
    /// Model-authored HTML artifact (from `create_artifact`) stashed by dispatch for the agent
    /// loop to attach to the next tool-result event (rendered inline in a sandboxed iframe).
    pub pending_artifact: Mutex<Option<ollama::ArtifactPayload>>,
    /// Local image-generation settings (sd.cpp binary + model paths + defaults), pushed from the
    /// frontend. Read by dispatch when the model calls `generate_image`.
    pub image_gen_config: Mutex<image_gen::ImageGenConfig>,
    /// Set true to cancel an in-flight model download.
    pub image_model_download_cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Token accumulator for the current turn (prompt, completion), summed across the turn's model
    /// calls by the stream parser and read back by the agent loop when it writes the usage record.
    pub turn_tokens: Mutex<(u64, u64)>,
    /// Count of images generated this turn — gives each a stable name (generated_image_N.png) the
    /// model can reference in run_python / a deck. Reset at the start of the agent loop.
    pub turn_image_count: Mutex<u32>,
    /// User-brushed edit masks for this turn's attached images: (image basename → mask PNG bytes,
    /// white = edit). Set by `send_message` from the frontend; read by the generate_image dispatch
    /// so "edit the part I painted" inpaints exactly that region. Overwritten each send.
    pub attachment_masks: Mutex<Vec<(String, Vec<u8>)>>,
    /// Persistent sysinfo handle for the Usage panel's Live tab — kept between polls so CPU% is a
    /// real delta over the poll interval rather than a cold-start zero.
    pub sys: Mutex<sysinfo::System>,
    /// MCP server ids the user has approved to render/interact with apps this
    /// session (set by `approve_mcp_app`; reset on restart).
    pub apps_allowed: Mutex<std::collections::HashSet<String>>,
    /// The agent loop parks a sender here while it waits for the user to Allow/Skip an unapproved
    /// MCP-app before continuing, so a model can't stack more prompts (or keep flailing) while a
    /// deliverable awaits your decision. Resolved by `approve_mcp_app` (true) / `skip_mcp_app` (false).
    pub pending_app_approval: Mutex<Option<tokio::sync::oneshot::Sender<bool>>>,
    /// Id of the saved conversation the current chat maps to, so auto-save
    /// updates the same record. `None` = a fresh chat not yet persisted.
    pub active_conversation_id: Mutex<Option<String>>,
    /// The current run's cancel token — a PER-RUN `Arc<AtomicBool>` held in a swappable slot, not
    /// a single shared flag. Each `send_message` cancels the previous token and installs a fresh
    /// one, so stopping a run and starting another can never revive the old run by resetting a
    /// shared boolean (which is exactly what leaked a stopped run's answer into a new chat).
    /// `stop_generation` and `reset_conversation` set the current token; the loop checks it between
    /// steps and while streaming.
    pub cancel: std::sync::Mutex<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Files `run_python` wrote to /work/out that couldn't be saved because no sandbox folder is
    /// configured. Stashed (name, bytes) while the user is asked to add a folder; written by
    /// `save_pending_outputs`. We never write outside the sandbox.
    pub pending_output_files: Mutex<Vec<(String, Vec<u8>)>>,
    /// Whether the active run's profile allows code (`run_python`) to call registered tools via
    /// `call_tool` (code-mode). Set at `send_message` start. Off unless the profile opts in.
    pub code_tools_allowed: Mutex<bool>,
    /// Capture the verbatim context in the debug panel as well as its sizes. Off by default: a long
    /// research turn would hold tens of MB of history in the renderer, and it puts every tool result
    /// and file the run touched on screen.
    pub debug_full_context: Mutex<bool>,
    /// Built-in tool names the active run may call from code (the profile's enabled built-ins,
    /// minus run_python itself). Used to gate `call_tool_from_code`.
    pub run_callable_builtins: Mutex<std::collections::HashSet<String>>,
    /// Dev control server (debug builds): in-flight `/dev/run` requests, keyed by id. The server
    /// parks a oneshot here and the frontend resolves it via `dev_control_report` with the trace.
    pub pending_dev_run: Mutex<HashMap<u64, tokio::sync::oneshot::Sender<serde_json::Value>>>,
    pub dev_run_seq: std::sync::atomic::AtomicU64,
}

impl Default for AppState {
    fn default() -> Self {
        let saved = std::fs::read_to_string(allowed_dirs_path())
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        skills::seed_builtin_skills();
        Self {
            backend:         Mutex::new(ollama::Backend::ollama("http://localhost:11434")),
            conversation:    Mutex::new(Vec::new()),
            openapi_specs:   Mutex::new(Vec::new()),
            sparql_endpoints: Mutex::new(Vec::new()),
            mcp_servers:     Mutex::new(Vec::new()),
            mcp_connections: tokio::sync::Mutex::new(HashMap::new()),
            allowed_dirs:    Mutex::new(saved),
            skills:          Mutex::new(skills::load_skills()),
            jobs:            Mutex::new(jobs::load_jobs()),
            job_runs:        Mutex::new(jobs::load_runs()),
            tray:            Mutex::new(None),
            pending_email_raw: Mutex::new(None),
            code_exec_unlocked: Mutex::new(false),
            pending_code_permission: Mutex::new(None),
            pending_python_result: Mutex::new(HashMap::new()),
            python_request_seq: std::sync::atomic::AtomicU64::new(0),
            pending_tool_ui: Mutex::new(None),
            pending_tool_images: Mutex::new(Vec::new()),
            pending_artifact: Mutex::new(None),
            image_gen_config: Mutex::new(image_gen::ImageGenConfig::default()),
            image_model_download_cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            turn_tokens: Mutex::new((0, 0)),
            turn_image_count: Mutex::new(0),
            attachment_masks: Mutex::new(Vec::new()),
            sys: Mutex::new(sysinfo::System::new()),
            apps_allowed: Mutex::new(std::collections::HashSet::new()),
            pending_app_approval: Mutex::new(None),
            active_conversation_id: Mutex::new(None),
            cancel: std::sync::Mutex::new(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false))),
            pending_output_files: Mutex::new(Vec::new()),
            code_tools_allowed: Mutex::new(false),
            debug_full_context: Mutex::new(false),
            run_callable_builtins: Mutex::new(std::collections::HashSet::new()),
            pending_dev_run: Mutex::new(HashMap::new()),
            dev_run_seq: std::sync::atomic::AtomicU64::new(0),
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

/// Build a `Backend` from the frontend's loosely-typed connection fields.
fn backend_from(provider: Option<String>, base_url: String, api_key: Option<String>) -> ollama::Backend {
    let kind = match provider.as_deref() {
        Some("openai") => ollama::ProviderKind::OpenAI,
        Some("anthropic") => ollama::ProviderKind::Anthropic,
        _ => ollama::ProviderKind::Ollama,
    };
    ollama::Backend { kind, base_url, api_key: api_key.filter(|k| !k.is_empty()) }
}

#[derive(Deserialize)]
struct BackendArgs {
    base_url: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
}

/// List models from a specific server. Each configured server is queried independently so the UI
/// can show the union. Uses the `args`-struct IPC convention for verbatim snake_case fields.
#[tauri::command]
async fn get_models(args: BackendArgs) -> Result<Vec<String>, String> {
    let backend = backend_from(args.provider, args.base_url, args.api_key);
    ollama::list_models(&backend).await.map_err(|e| e.to_string())
}

// ── Model management ──────────────────────────────────────────────────────────

/// One model for the manager UI. `size`/`loaded` are meaningful only for Ollama; `manageable` tells
/// the UI whether to offer install/delete (true only for a local Ollama runtime).
#[derive(Serialize)]
struct ManagedModel { name: String, size: u64, loaded: bool, manageable: bool }

/// List a server's models for the manager: Ollama returns sizes + loaded status and is manageable;
/// hosted providers return their catalog (names only), read-only.
#[tauri::command]
async fn list_managed_models(args: BackendArgs) -> Result<Vec<ManagedModel>, String> {
    let is_ollama = !matches!(args.provider.as_deref(), Some("openai") | Some("anthropic"));
    if is_ollama {
        let detailed = ollama::list_ollama_detailed(&args.base_url).await.map_err(|e| e.to_string())?;
        // Loaded status is best-effort — an older Ollama without /api/ps just yields none.
        let loaded = ollama::ollama_loaded(&args.base_url).await.unwrap_or_default();
        Ok(detailed.into_iter().map(|m| ManagedModel {
            loaded: loaded.contains(&m.name), name: m.name, size: m.size, manageable: true,
        }).collect())
    } else {
        let backend = backend_from(args.provider, args.base_url, args.api_key);
        let names = ollama::list_models(&backend).await.map_err(|e| e.to_string())?;
        Ok(names.into_iter().map(|name| ManagedModel { name, size: 0, loaded: false, manageable: false }).collect())
    }
}

#[derive(Deserialize)]
struct OllamaModelArgs { base_url: String, name: String }

/// Pull an Ollama model, streaming progress on `model-pull` so the UI can show a bar. Reuses the
/// image-download cancel flag (only one heavy download runs at a time in practice).
#[tauri::command]
async fn pull_ollama_model(args: OllamaModelArgs, app: AppHandle) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    use tauri::Emitter as _;
    let cancel = {
        let state = app.state::<AppState>();
        state.image_model_download_cancel.store(false, Ordering::SeqCst);
        state.image_model_download_cancel.clone()
    };
    let app2 = app.clone();
    let name = args.name.clone();
    let mut last: u64 = 0;
    let result = ollama::pull_ollama(&args.base_url, &args.name, &cancel, |status, completed, total| {
        // Throttle: emit on phase change (status), every ~16 MB, and at completion of a phase.
        if completed == 0 || completed.saturating_sub(last) >= 16_000_000 || (total > 0 && completed >= total) {
            last = completed;
            let _ = app2.emit("model-pull", serde_json::json!({
                "name": name, "status": status, "completed": completed, "total": total, "done": false,
            }));
        }
    }).await;
    match result {
        Ok(()) => { let _ = app.emit("model-pull", serde_json::json!({ "name": args.name, "done": true })); Ok(()) }
        Err(e) => {
            let msg = e.to_string();
            let _ = app.emit("model-pull", serde_json::json!({ "name": args.name, "done": true, "error": msg }));
            Err(msg)
        }
    }
}

#[tauri::command]
async fn delete_ollama_model(args: OllamaModelArgs) -> Result<(), String> {
    ollama::delete_ollama(&args.base_url, &args.name).await.map_err(|e| e.to_string())
}

/// Cancel an in-flight model pull (shares the image-download cancel flag).
#[tauri::command]
fn cancel_ollama_pull(state: State<'_, AppState>) {
    state.image_model_download_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Set the active *default* backend — used by the job designer and background jobs that carry no
/// explicit server. Interactive chat routes per-message via `send_message` instead.
#[tauri::command]
async fn set_backend(args: BackendArgs, state: State<'_, AppState>) -> Result<(), String> {
    *state.backend.lock().unwrap() = backend_from(args.provider, args.base_url, args.api_key);
    Ok(())
}

#[tauri::command]
async fn reset_conversation(state: State<'_, AppState>) -> Result<(), String> {
    // Cancel any in-flight run so its late events (e.g. a slow run_python) can't bleed into the
    // fresh chat. This cancels the CURRENT run's token; the next send_message installs its own.
    state.cancel.lock().unwrap().store(true, std::sync::atomic::Ordering::SeqCst);
    state.conversation.lock().unwrap().clear();
    *state.active_conversation_id.lock().unwrap() = None;
    Ok(())
}

/// Cancel the running agent loop. It stops at the next step boundary or token, whichever
/// comes first, and emits `agent-done`. Conversation history is left intact.
#[tauri::command]
fn stop_generation(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel.lock().unwrap().store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(())
}

/// Cancel the run token currently in `slot` and install a fresh, uncancelled one, returning it.
/// The old token stays cancelled forever — that is the whole point: because each run holds a
/// distinct token, a newly-started run can never revive a stopped one by clearing a shared flag
/// (the bug that leaked a stopped run's answer into the next chat).
fn supersede_run(slot: &std::sync::Mutex<std::sync::Arc<std::sync::atomic::AtomicBool>>)
    -> std::sync::Arc<std::sync::atomic::AtomicBool>
{
    let mut g = slot.lock().unwrap();
    g.store(true, std::sync::atomic::Ordering::SeqCst);
    let fresh = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    *g = fresh.clone();
    fresh
}

// ── Chat history commands ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ListConversationsArgs {
    #[serde(default)]
    pub profile_id: Option<String>,
}

/// Index entry plus its on-disk footprint. Size is derived, so it is attached here rather than
/// stored in index.json where it would go stale.
#[derive(Serialize)]
struct ConversationListItem {
    #[serde(flatten)]
    meta: history::ConversationMeta,
    size_bytes: u64,
}

/// List saved conversations scoped to the given profile (per-profile history).
/// A `None` profile matches conversations saved with no active profile.
#[tauri::command]
fn list_conversations(args: ListConversationsArgs) -> Vec<ConversationListItem> {
    let mut items: Vec<ConversationListItem> = history::load_index()
        .into_iter()
        .filter(|m| m.profile_id == args.profile_id)
        .map(|m| { let size_bytes = history::disk_size(&m.id); ConversationListItem { meta: m, size_bytes } })
        .collect();
    // Pinned chats float to the top; within each group, most-recently-updated first.
    items.sort_by(|a, b| b.meta.pinned.cmp(&a.meta.pinned)
        .then(b.meta.updated_at.cmp(&a.meta.updated_at)));
    items
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
    /// Files attached during this chat, so a reopened conversation can still edit them.
    #[serde(default)]
    pub attachments: Vec<String>,
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
        // Preserved from the index by save_one; the frontend save payload never carries these.
        pinned: false,
        folder: None,
    };
    let conv = history::Conversation {
        meta: meta.clone(), wire, display: args.display, attachments: args.attachments,
    };
    history::save_one(&conv).map_err(|e| e.to_string())?;
    *active = Some(id);
    Ok(meta)
}

/// Working files for a chat — offloaded tool results (`data`) and `/work/artifacts` payloads
/// (`artifacts`) — kept under the conversation so they last exactly as long as it does.
///
/// Previously both lived in a shared temp dir swept after 6 hours, which quietly broke the
/// reasonable assumption that reopening a saved chat gives you back everything it produced: the
/// artifact HTML survived, but the underlying data did not, and the conversation still contained
/// instructions telling the model to read files that no longer existed.
///
/// Mints the conversation id when the chat has not been saved yet, so the very first turn already
/// has a home; `save_active_conversation` reuses whatever id is set.
pub fn conversation_files_dir(state: &AppState, sub: &str) -> Option<std::path::PathBuf> {
    let mut active = state.active_conversation_id.lock().unwrap();
    let id = match active.as_ref() {
        Some(id) => id.clone(),
        None => { let id = history::new_id(); *active = Some(id.clone()); id }
    };
    let dir = dirs_path().join("conversations").join(format!("{id}-files")).join(sub);
    std::fs::create_dir_all(&dir).ok().map(|_| dir)
}

/// Remove working-file directories whose conversation no longer exists — chats deleted outside the
/// app, or a first turn that was never saved.
fn prune_orphan_conversation_files() {
    let convs = dirs_path().join("conversations");
    let Ok(entries) = std::fs::read_dir(&convs) else { return };
    for e in entries.flatten() {
        let Some(name) = e.file_name().to_str().map(String::from) else { continue };
        let Some(id) = name.strip_suffix("-files") else { continue };
        if !convs.join(format!("{id}.json")).exists() {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

/// Delete one wiki page. Irreversible — the file is removed from disk, not moved to a bin —
/// so the UI confirms first. Path traversal is rejected by the wiki resolver, and the removal
/// is appended to log.md like any other change.
#[tauri::command]
fn delete_wiki_page(path: String) -> String {
    wiki::wiki_delete(&serde_json::json!({ "path": path }))
}

/// The wiki as a graph: pages, the links between them, and pages that are merely about
/// similar things. `min_score` filters the semantic edges — the UI exposes it as a slider,
/// since the right threshold depends on how varied the wiki is.
/// One wiki page's raw Markdown, for the graph's reading pane. Path traversal is rejected by
/// the same resolver the wiki tools use, so this cannot read outside the wiki folder.
#[tauri::command]
fn read_wiki_page(path: String) -> String {
    wiki::wiki_read(&serde_json::json!({ "path": path }))
}

#[tauri::command]
fn get_wiki_graph(min_score: Option<f32>) -> wiki::WikiGraph {
    wiki::wiki_graph(min_score.unwrap_or(0.6).clamp(0.0, 1.0))
}

#[derive(Deserialize)]
pub struct ConversationIdArgs {
    pub id: String,
}

/// Load a saved conversation: restore its wire history as the active backend
/// context and return the display messages for the frontend to render.
/// What the frontend needs to rebuild a chat: its rendered messages plus the attachment
/// ledger, split by whether each file is still on disk. `missing` is returned rather than
/// dropped so the model can tell the user a file has moved instead of quietly reinventing it.
#[derive(Serialize)]
pub struct LoadedConversation {
    pub display: serde_json::Value,
    pub attachments: Vec<String>,
    pub missing_attachments: Vec<String>,
}

#[tauri::command]
fn load_conversation(
    args: ConversationIdArgs,
    state: State<'_, AppState>,
) -> Result<LoadedConversation, String> {
    let conv = history::load_one(&args.id).ok_or("conversation not found")?;
    let (attachments, missing_attachments) = history::split_attachments(&conv.attachments);
    *state.conversation.lock().unwrap() = conv.wire;
    *state.active_conversation_id.lock().unwrap() = Some(args.id);
    Ok(LoadedConversation { display: conv.display, attachments, missing_attachments })
}

/// Current length of the backend wire history. The frontend stamps this on each user message as its
/// `wireBase` so an edit/regenerate can later truncate the wire back to exactly that point — a
/// robust anchor that a display-side message count is not, since internal nudges append extra
/// `user` messages to the wire that never appear in the display.
#[tauri::command]
fn conversation_len(state: State<'_, AppState>) -> usize {
    state.conversation.lock().unwrap().len()
}

#[derive(Deserialize)]
pub struct TruncateArgs { pub len: usize }

/// Truncate the wire history to `len`, dropping that turn and everything after it. Edit and
/// regenerate call this with a user message's `wireBase`, then re-send — so the model re-runs from
/// exactly the right point with full-fidelity history (real tool calls and results) before it.
#[tauri::command]
fn truncate_conversation(args: TruncateArgs, state: State<'_, AppState>) -> Result<(), String> {
    let mut conv = state.conversation.lock().unwrap();
    if args.len <= conv.len() { conv.truncate(args.len); }
    Ok(())
}

#[tauri::command]
fn delete_conversation(
    args: ConversationIdArgs,
    state: State<'_, AppState>,
) -> Result<(), String> {
    history::delete_one(&args.id).map_err(|e| e.to_string())?;
    // The chat's working files go with it.
    let _ = std::fs::remove_dir_all(dirs_path().join("conversations").join(format!("{}-files", args.id)));
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

#[derive(Deserialize)]
pub struct SetPinnedArgs {
    pub id: String,
    pub pinned: bool,
}

#[tauri::command]
fn set_conversation_pinned(args: SetPinnedArgs) -> Result<(), String> {
    history::set_pinned(&args.id, args.pinned).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
pub struct SearchConversationsArgs {
    pub query: String,
    #[serde(default)]
    pub profile_id: Option<String>,
}

#[tauri::command]
fn search_conversations(args: SearchConversationsArgs) -> Vec<history::SearchHit> {
    history::search(&args.query, args.profile_id.as_deref())
}

#[derive(Deserialize)]
pub struct SetFolderArgs {
    pub id: String,
    /// The folder name, or null / empty to move the chat back to ungrouped.
    #[serde(default)]
    pub folder: Option<String>,
}

#[tauri::command]
fn set_conversation_folder(args: SetFolderArgs) -> Result<(), String> {
    history::set_folder(&args.id, args.folder).map_err(|e| e.to_string())
}

// ── Send message ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SendMessageArgs {
    pub model: String,
    pub message: String,
    pub system_prompt: String,
    pub tools: Vec<ollama::ToolSchema>,
    // Backend routing: which server this model belongs to. Absent → the active default backend.
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub image_paths: Vec<String>,
    /// Optional user-brushed edit mask per attached image, aligned to `image_paths` (a
    /// `data:image/png;base64,…` string, or "" for none). White = the region to edit.
    #[serde(default)]
    pub image_masks: Vec<String>,
    /// Non-image files the user attached. The run_python sandbox is allowed to
    /// read/write these even if they fall outside the configured allowed dirs.
    #[serde(default)]
    pub file_paths: Vec<String>,
    /// Files attached EARLIER in this conversation. They join the sandbox allow-list so a
    /// later turn can still edit them ("make the beard lighter" three messages on), but are
    /// deliberately NOT re-encoded into `images` — the model already sees them through the
    /// retained history, and re-sending every prior photo as base64 each turn would balloon
    /// the context.
    #[serde(default)]
    pub prior_file_paths: Vec<String>,
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
    /// Ollama reasoning toggle. `Some(false)` disables a thinking model's pre-answer reasoning
    /// pass (much faster per turn). `None` leaves the model's default (Qwen3 etc. reason by default).
    #[serde(default)]
    pub think: Option<bool>,
    /// Whether this profile allows code (`run_python`) to call registered tools via `call_tool`
    /// (code-mode). Off unless the profile opts in.
    #[serde(default)]
    pub allow_code_tools: bool,
    /// Capture verbatim context in the debug panel, not just its size. Off unless the user asks.
    #[serde(default)]
    pub debug_full_context: bool,
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
    /// Skill IDs the active profile enables. `None` (field absent) = all loaded skills (backward
    /// compatible); `Some([...])` = exactly those; `Some([])` = none.
    #[serde(default)]
    pub enabled_skill_ids: Option<Vec<String>>,
    /// Cap on tools shown to the model per step (from the profile's maxTools). None/0 → default.
    #[serde(default)]
    pub max_tools: Option<usize>,
    /// Max chars of a tool result fed back to the model (profile's toolResultLimit). None/0 → default.
    #[serde(default)]
    pub tool_result_limit: Option<usize>,
    /// Per-turn cap on web_search + fetch_webpage calls (profile's webToolCap). None/0 → default (10).
    #[serde(default)]
    pub web_tool_cap: Option<usize>,
}

fn default_web_search_results() -> usize { 10 }
fn default_max_steps() -> usize { 12 }

#[tauri::command]
async fn send_message(
    args: SendMessageArgs,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    // Route to the server this model belongs to (sent by the frontend). Fall back to the active
    // default backend when the caller didn't specify one (e.g. legacy callers).
    let backend = match args.base_url.clone() {
        Some(url) if !url.is_empty() => backend_from(args.provider.clone(), url, args.api_key.clone()),
        _ => state.backend.lock().unwrap().clone(),
    };

    // Supersede any previous run: cancel its (distinct) token and install a fresh one for this run,
    // so a stopped run can't be revived by this new one and leak its answer into this chat.
    let cancel = supersede_run(&state.cancel);
    // Reset the per-turn token accumulator (the stream parser sums into it; the frontend reads it
    // back via record_turn_usage when the turn ends).
    *state.turn_tokens.lock().unwrap() = (0, 0);

    // Stash any user-brushed edit masks, keyed by the attached image's basename, so the
    // generate_image dispatch can inpaint exactly the painted region. Aligned to image_paths.
    {
        use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
        let mut masks = Vec::new();
        for (i, data) in args.image_masks.iter().enumerate() {
            let b64 = data.rsplit(',').next().unwrap_or("");
            if b64.is_empty() { continue; }
            if let (Some(path), Ok(bytes)) = (args.image_paths.get(i), B64.decode(b64)) {
                if let Some(name) = std::path::Path::new(path).file_name().and_then(|n| n.to_str()) {
                    masks.push((name.to_string(), bytes));
                }
            }
        }
        *state.attachment_masks.lock().unwrap() = masks;
    }

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

    // Code-mode: record whether this run's profile lets code call tools, and which built-ins are
    // callable from code (enabled built-ins minus run_python, to prevent code re-entering itself).
    *state.code_tools_allowed.lock().unwrap() = args.allow_code_tools;
    *state.debug_full_context.lock().unwrap() = args.debug_full_context;
    *state.run_callable_builtins.lock().unwrap() = builtin_tools.iter()
        .map(|t| t.function.name.clone())
        .filter(|n| n != "run_python" && n != "find_tools")
        .collect();

    // The wiki tools are one workflow, not independent tools: the system prompt requires a
    // wiki_search before every write and a wiki_append to log.md after it. Per-step selection
    // would hand back only wiki_write, so hold them out and always offer the whole group.
    let (always_tools, discoverable): (Vec<_>, Vec<_>) = builtin_tools
        .into_iter()
        .partition(|t| t.function.name.starts_with("wiki_"));

    // `generate_image` is a heavy schema (~200 tokens) wanted only when the user actually asks for
    // a picture, so it is discovered on demand rather than sent on every step of every run.
    // Two tools were tried here and put back, for the same underlying reason — a tool the model can
    // ROUTE AROUND must not be gated, because it will:
    //   * create_artifact — asked for "an HTML page with a bar chart", the model skipped find_tools
    //     entirely and wrote a file to disk with write_file instead. Gating it did not prompt
    //     discovery, it prompted avoidance, and the user silently lost the inline artifact.
    //   * run_python — an offloaded tool result instructs the model to process the file with
    //     run_python, so it must exist the moment that is read, not a discovery round-trip later.
    //   * the file-management tools (create_directory, move_file, delete_file, find_old_files) —
    //     tried and reverted. Asked to create a folder, the model did not call find_tools; it used
    //     run_python, whose sandbox is /work, so it made the directory inside the VIRTUAL
    //     filesystem, saw it succeed there, and reported "the folder has been created and
    //     confirmed to exist" — while nothing appeared on the real disk. run_python is a universal
    //     substitute for anything file-shaped, and a silently WRONG one, so no file tool can be
    //     gated while run_python is core. That is strictly worse than avoidance: it fabricates a
    //     success the user cannot see is false.
    // generate_image survives this rule because nothing else can produce an image.
    const ON_DEMAND: &[&str] = &["generate_image"];
    let (media, core): (Vec<_>, Vec<_>) = discoverable
        .into_iter()
        .partition(|t| ON_DEMAND.contains(&t.function.name.as_str()));

    // Build discoverable tool groups: built-ins, the on-demand groups, then one per OpenAPI spec /
    // SPARQL endpoint / MCP server. The agent loop narrows these per step when they're numerous.
    let mut tool_groups: Vec<ollama::ToolGroup> = Vec::new();
    if !core.is_empty() {
        tool_groups.push(ollama::ToolGroup {
            label: "Built-in tools".into(),
            description: "Read, list and search files; web search; fetch web pages; date and time; \
                email; and run code.".into(),
            tools: core,
        });
    }
    if !media.is_empty() {
        tool_groups.push(ollama::ToolGroup {
            // Keyword-rich: this is what find_tools matches against.
            label: "Image generation".into(),
            description: "Generate, draw, create or edit an image, picture, photo, illustration or \
                artwork from a text description.".into(),
            tools: media,
        });
    }
    for spec in state.openapi_specs.lock().unwrap().iter() {
        let tools: Vec<ollama::ToolSchema> = spec.tools.iter()
            .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok())
            .collect();
        if !tools.is_empty() {
            // Sample a few operation descriptions/names so group selection can match on what
            // the API actually does, not just its title.
            let ops = tools.iter().take(12)
                .map(|t| {
                    let d = t.function.description.trim();
                    if d.is_empty() { t.function.name.clone() }
                    else { d.chars().take(80).collect::<String>() }
                })
                .collect::<Vec<_>>().join("; ");
            tool_groups.push(ollama::ToolGroup {
                label: spec.title.clone(),
                description: format!("{}. Operations include: {ops}", spec.title),
                tools,
            });
        }
    }
    for ep in state.sparql_endpoints.lock().unwrap().iter() {
        let tools: Vec<ollama::ToolSchema> = ep.tools.iter()
            .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok())
            .collect();
        if !tools.is_empty() {
            let desc = if ep.usage_hint.trim().is_empty() { ep.title.clone() } else { ep.usage_hint.clone() };
            tool_groups.push(ollama::ToolGroup {
                label: ep.title.clone(),
                description: format!("SPARQL / linked data — {desc}"),
                tools,
            });
        }
    }
    // MCP: every registered server sits in the pool regardless of profile, so
    // `enabled_mcp_server_ids` is the authority (strict — empty means none).
    {
        let connections = state.mcp_connections.lock().await;
        for (_, conn) in connections.iter().filter(|(id, _)| args.enabled_mcp_server_ids.contains(*id)) {
            let tools: Vec<ollama::ToolSchema> = conn.tools.iter()
                .filter(|t| !args.disabled_mcp_tools.contains(&t.name))
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok())
                .collect();
            if !tools.is_empty() {
                // List the tool names so per-step selection can match on capability (the names
                // are keyword-rich, e.g. static_map_image_tool, search_and_geocode_tool) rather
                // than a generic "connected server" blurb that matches nothing.
                let names = tools.iter().map(|t| t.function.name.as_str())
                    .collect::<Vec<_>>().join(", ");
                tool_groups.push(ollama::ToolGroup {
                    label: format!("{} (MCP)", conn.config.name),
                    description: format!("{} MCP server. Tools: {names}", conn.config.name),
                    tools,
                });
            }
        }
    }

    let specs_snapshot: Vec<openapi::RegisteredSpec> = state.openapi_specs.lock().unwrap().clone();
    let sparql_snapshot: Vec<sparql::RegisteredSparqlEndpoint> = state.sparql_endpoints.lock().unwrap().clone();
    let allowed_dirs_snapshot: Vec<String> = state.allowed_dirs.lock().unwrap().clone();
    // Offer a skill when (a) the active profile enables it (None = all) and (b) every tool it
    // `requires` is available — so run_python skills need the sandbox, but instructions-only skills
    // (citation-format, literature-review, …) show up anywhere.
    let skills_snapshot: Vec<skills::RegisteredSkill> = {
        let tool_names: std::collections::HashSet<&str> =
            args.tools.iter().map(|t| t.function.name.as_str()).collect();
        state.skills.lock().unwrap().iter()
            .filter(|s| match &args.enabled_skill_ids {
                Some(ids) => ids.contains(&s.id),
                None => true,
            })
            .filter(|s| s.requires.iter().all(|r| tool_names.contains(r.as_str())))
            .cloned()
            .collect()
    };

    let options = if args.temperature.is_some() || args.top_p.is_some() || args.top_k.is_some()
        || args.repeat_penalty.is_some() || args.seed.is_some()
        || args.num_ctx.is_some() || args.num_predict.is_some() || args.stop.is_some()
        || args.think.is_some()
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
            think: args.think,
        })
    } else {
        None
    };

    ollama::agent_loop(
        &backend,
        &args.model,
        &args.system_prompt,
        &always_tools,
        &tool_groups,
        args.max_tools.unwrap_or(0),
        options,
        args.keep_alive.clone(),
        &state.conversation,
        specs_snapshot,
        sparql_snapshot,
        &state.mcp_connections,
        allowed_dirs_snapshot,
        {
            // The sandbox may read/write attached files. Attached IMAGES are included too (in
            // addition to being base64'd for vision above) so they stage into /work/uploads and
            // run_python (Pillow) can open/edit the actual pixels — otherwise the model has no path
            // to the photo and loops searching the user's folders for it.
            let mut p = args.file_paths.clone();
            p.extend(args.image_paths.iter().cloned());
            // Earlier attachments stay reachable for the whole conversation.
            p.extend(args.prior_file_paths.iter().cloned());
            p.sort();
            p.dedup();
            p
        },
        args.web_search_results,
        args.tool_result_limit.unwrap_or(0), // 0 → default
        &app,
        false, // silent = false for interactive chat
        false, // allow_code_exec: interactive uses the per-session approval prompt instead
        if args.max_steps == 0 { usize::MAX } else { args.max_steps.max(1) }, // 0 = no limit; else uncapped (loop guards still stop runaways)
        args.web_tool_cap.unwrap_or(0), // 0 → default web-tool cap
        cancel,
        true, // discover_tools: interactive chat uses find_tools discovery for large tool sets
        skills_snapshot,
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

/// Pre-unlock (or re-lock) code execution without a prompt. The frontend calls this at startup
/// when the persisted "Always allow code execution" setting is on, so approved users skip the
/// per-session approval dialog. Persistence lives in the frontend settings (localStorage); this
/// just seeds the in-memory session flag the run_python gate reads.
/// Push the image-generation settings (sd.cpp binary + model paths + defaults) into AppState. The
/// frontend calls this at startup and whenever the user edits them, mirroring set_mcp_servers etc.
#[tauri::command]
fn set_image_gen_config(args: image_gen::ImageGenConfig, state: State<'_, AppState>) -> Result<(), String> {
    *state.image_gen_config.lock().unwrap() = args;
    Ok(())
}

#[derive(serde::Deserialize)]
struct DownloadModelArgs {
    url: String,
    #[serde(default)]
    filename: Option<String>,
}

/// Stream-download an image model into the app's image-gen/models dir, emitting progress on the
/// `image-model-download` event so the UI can show a bar. Auto-detection then picks it up. The
/// AppHandle (not a borrowed State, which can't cross an await) is used so the long download is
/// safe to hold across await points.
#[tauri::command]
async fn download_image_model(args: DownloadModelArgs, app: AppHandle) -> Result<String, String> {
    use std::sync::atomic::Ordering;
    use tauri::Emitter as _;
    // Grab the cancel flag (a cloned Arc) without holding the State guard across the await.
    let cancel = {
        let state = app.state::<AppState>();
        state.image_model_download_cancel.store(false, Ordering::SeqCst);
        state.image_model_download_cancel.clone()
    };
    let app2 = app.clone();
    let mut last: u64 = 0;
    let result = image_gen::download_model(&args.url, args.filename.as_deref(), &cancel, |recv, total| {
        // Throttle progress events: at start, every ~8 MB, and at the end.
        if recv == 0 || recv.saturating_sub(last) >= 8_000_000 || Some(recv) == total {
            last = recv;
            let _ = app2.emit("image-model-download", serde_json::json!({
                "received": recv, "total": total, "done": false
            }));
        }
    }).await;
    match result {
        Ok(p) => {
            let path = p.display().to_string();
            let _ = app.emit("image-model-download", serde_json::json!({ "done": true, "path": path }));
            Ok(path)
        }
        Err(e) => {
            let _ = app.emit("image-model-download", serde_json::json!({ "done": true, "error": e }));
            Err(e)
        }
    }
}

/// Cancel an in-flight model or engine download (the streaming loops check this each chunk).
#[tauri::command]
fn cancel_image_model_download(state: State<'_, AppState>) {
    state.image_model_download_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// Download + install the offline image engine (stable-diffusion.cpp) for this platform on first
/// use, emitting progress on `image-engine-download`. Returns the installed sd path. The app ships
/// without the engine (bundling unsigned binaries breaks macOS notarization), so it's fetched here.
#[tauri::command]
async fn download_image_engine(app: AppHandle) -> Result<String, String> {
    use std::sync::atomic::Ordering;
    use tauri::Emitter as _;
    let cancel = {
        let state = app.state::<AppState>();
        state.image_model_download_cancel.store(false, Ordering::SeqCst);
        state.image_model_download_cancel.clone()
    };
    let app2 = app.clone();
    let mut last: u64 = 0;
    let result = image_gen::download_engine(&cancel, |recv, total| {
        if recv == 0 || recv.saturating_sub(last) >= 2_000_000 || Some(recv) == total {
            last = recv;
            let _ = app2.emit("image-engine-download", serde_json::json!({ "received": recv, "total": total, "done": false }));
        }
    }).await;
    match result {
        Ok(p) => {
            let path = p.display().to_string();
            let _ = app.emit("image-engine-download", serde_json::json!({ "done": true, "path": path }));
            Ok(path)
        }
        Err(e) => {
            let _ = app.emit("image-engine-download", serde_json::json!({ "done": true, "error": e }));
            Err(e)
        }
    }
}

/// Whether this platform has a prebuilt engine + whether it's already installed. Drives the Images
/// tab's "Install engine" affordance.
#[tauri::command]
fn image_engine_status() -> serde_json::Value {
    let installed = image_gen::image_gen_dir_has_engine();
    serde_json::json!({ "supported": image_gen::engine_supported(), "installed": installed })
}

#[derive(serde::Deserialize)]
struct UsageStatsArgs { #[serde(default)] since: Option<i64> }

/// Aggregate local usage records for the Usage panel's History view (all local; `since` = 0 = all).
#[tauri::command]
fn get_usage_stats(args: UsageStatsArgs) -> usage::UsageStats {
    usage::aggregate(args.since.unwrap_or(0))
}

/// Record one completed turn. The frontend supplies everything it knows (model, profile, provider,
/// tools, images, timing, error); the backend merges in the token counts it accumulated from the
/// model stream (reset for the next turn) and appends the record. Best-effort, fully local.
#[tauri::command]
fn record_turn_usage(args: usage::UsageRecord, state: State<'_, AppState>) {
    let mut rec = args;
    let (p, c) = { let mut t = state.turn_tokens.lock().unwrap(); let v = *t; *t = (0, 0); v };
    rec.prompt_tokens = p;
    rec.completion_tokens = c;
    usage::append(&rec);
}

/// Real-time system snapshot for the Usage panel's Live tab: global CPU%, memory, this process's
/// memory, and Ollama's loaded models + VRAM (from /api/ps). All local.
#[tauri::command]
async fn system_stats(app: AppHandle) -> serde_json::Value {
    use sysinfo::ProcessesToUpdate;
    let (cpu, used, total, app_mem, cores) = {
        let state = app.state::<AppState>();
        let mut sys = state.sys.lock().unwrap();
        sys.refresh_cpu_all();
        sys.refresh_memory();
        let pid = sysinfo::get_current_pid().ok();
        if let Some(p) = pid { sys.refresh_processes(ProcessesToUpdate::Some(&[p]), true); }
        let app_mem = pid.and_then(|p| sys.process(p)).map(|p| p.memory()).unwrap_or(0);
        (sys.global_cpu_usage(), sys.used_memory(), sys.total_memory(), app_mem, sys.cpus().len())
    };
    // Ollama loaded models (only meaningful for an Ollama backend; default host otherwise).
    let base = {
        let state = app.state::<AppState>();
        let b = state.backend.lock().unwrap();
        if matches!(b.kind, ollama::ProviderKind::Ollama) { b.base_url.trim_end_matches('/').to_string() }
        else { "http://localhost:11434".to_string() }
    };
    let models = fetch_ollama_ps(&base).await;
    serde_json::json!({
        "cpu": cpu, "mem_used": used, "mem_total": total, "app_mem": app_mem, "cores": cores,
        "models": models,
        "engine": { "supported": image_gen::engine_supported(), "installed": image_gen::image_gen_dir_has_engine() },
    })
}

async fn fetch_ollama_ps(base: &str) -> serde_json::Value {
    let client = reqwest::Client::new();
    match client.get(format!("{base}/api/ps")).timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(r) => match r.json::<serde_json::Value>().await {
            Ok(v) => serde_json::Value::Array(
                v["models"].as_array().map(|a| a.iter().map(|m| serde_json::json!({
                    "name": m["name"].as_str().unwrap_or(""),
                    "size_vram": m["size_vram"].as_u64().unwrap_or(0),
                    "size": m["size"].as_u64().unwrap_or(0),
                    "expires_at": m["expires_at"].as_str().unwrap_or(""),
                })).collect()).unwrap_or_default()),
            Err(_) => serde_json::Value::Array(vec![]),
        },
        Err(_) => serde_json::Value::Array(vec![]),
    }
}

#[tauri::command]
fn set_code_exec_unlocked(unlocked: bool, state: State<'_, AppState>) -> Result<(), String> {
    *state.code_exec_unlocked.lock().unwrap() = unlocked;
    Ok(())
}

#[derive(Deserialize)]
struct PyOutFile { name: String, b64: String }

/// A text payload the code staged to /work/artifacts for a `{{data:name}}` token. Deliberately
/// carries only the NAME and size — the content stays in the frontend and is spliced into the
/// artifact HTML there, so bulk data never enters the model's context (see `substituteData`).
#[derive(Deserialize)]
struct PyDataFile {
    name: String,
    #[serde(default)] chars: usize,
    #[serde(default)] error: Option<String>,
    /// Contents, persisted to `artifact_data_dir` so the file survives into later turns. Written to
    /// disk only — never echoed back to the model, which still sees just the name and size.
    #[serde(default)] text: Option<String>,
}

#[derive(Deserialize)]
struct RespondPythonResultArgs {
    #[serde(default)] request_id: u64,
    #[serde(default)] output: String,
    #[serde(default)] error: Option<String>,
    #[serde(default)] images: Vec<String>,
    #[serde(default)] out_files: Vec<PyOutFile>,
    #[serde(default)] data_files: Vec<PyDataFile>,
}

/// Frontend's response to a `run-python-request`: the Pyodide worker's output, chart images, and
/// any files it wrote to /work/out. Output files are persisted ONLY inside a configured sandbox
/// directory (the first allowed dir), under a sanitised (traversal-safe) basename; the real
/// absolute path is reported to the model. If no sandbox folder is configured we NEVER write
/// outside the sandbox — the files are stashed and the user is asked to add a folder (see
/// `save_pending_outputs`). Resolves the oneshot the agent loop awaits.
#[tauri::command]
fn respond_python_result(args: RespondPythonResultArgs, app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    use tauri::Emitter as _;
    let allowed = state.allowed_dirs.lock().unwrap().clone();
    let sandbox = allowed.first().cloned();
    let mut saved: Vec<String> = Vec::new();
    let mut failed: Vec<String> = Vec::new();
    let mut pending: Vec<(String, Vec<u8>)> = Vec::new();
    for f in &args.out_files {
        // file_name() strips any path components — no traversal outside the sandbox dir.
        let Some(name) = std::path::Path::new(&f.name).file_name().and_then(|n| n.to_str()).map(String::from) else { continue };
        let bytes = match B64.decode(&f.b64) { Ok(b) => b, Err(e) => { failed.push(format!("{name}: {e}")); continue; } };
        match &sandbox {
            Some(dir) => {
                let dest = std::path::Path::new(dir).join(&name);
                match std::fs::write(&dest, &bytes) {
                    Ok(()) => saved.push(dest.display().to_string()),
                    Err(e) => failed.push(format!("{name}: {e}")),
                }
            }
            // No sandbox folder → don't write anywhere; stash and ask the user to add one.
            None => pending.push((name, bytes)),
        }
    }

    let mut output = args.output;
    if let Some(err) = args.error {
        if !err.is_empty() { output.push_str("\n[Python error]\n"); output.push_str(&err); }
    }
    // Reopening an old chat can leave the history telling the model to read a staged file that has
    // since been removed. A bare FileNotFoundError reads as a code bug; name the real cause and the
    // way forward. Checked against the WHOLE result, not just the error field — code that catches
    // the exception itself prints the message to stdout, which is the common case.
    if output.contains("FileNotFoundError")
        && (output.contains("/work/data/") || output.contains("/work/artifacts/"))
    {
        output.push_str(
            "\n[That file is not in the sandbox. Staged files belong to the chat that created them: \
             offloaded tool results and /work/artifacts data are restored for THIS conversation only, \
             and are gone if the conversation was deleted. Do not retry the same path — re-run the \
             tool call to fetch the data again, then continue.]");
    }
    if !saved.is_empty() {
        output.push_str(&format!(
            "\n[SAVED TO DISK — report THESE exact real path(s) to the user; do NOT mention \
             /work/out (an in-memory scratch path the user cannot open):\n{}]",
            saved.iter().map(|p| format!("- {p}")).collect::<Vec<_>>().join("\n")));
    }
    if !pending.is_empty() {
        let names: Vec<String> = pending.iter().map(|(n, _)| n.clone()).collect();
        *state.pending_output_files.lock().unwrap() = pending;
        let _ = app.emit("sandbox-save-request", serde_json::json!({ "files": names }));
        output.push_str(&format!(
            "\n[File(s) generated ({}) but NO sandbox folder is configured, so they were NOT \
             written to disk (files are only ever saved inside the sandbox). The user has been \
             asked to choose a folder to add to the sandbox. Any chart is shown inline and can be \
             saved with its ⤓ download button. Do NOT claim the file is saved at /work/out.]",
            names.join(", ")));
    }
    if !failed.is_empty() {
        output.push_str(&format!("\n[Could not save to disk: {}]", failed.join("; ")));
    }
    // /work/artifacts payloads: tell the model the token is armed, and NOT what's in the file.
    // That's the whole point — it references the data by name instead of retyping it.
    if !args.data_files.is_empty() {
        // Persist them so they are re-staged into /work/artifacts on later turns (the Pyodide
        // workspace itself is wiped whenever a new message starts).
        let dir = conversation_files_dir(&state, "artifacts")
            .unwrap_or_else(ollama::artifact_data_dir);
        for f in &args.data_files {
            let (Some(text), Some(name)) = (f.text.as_deref(), std::path::Path::new(&f.name)
                .file_name().and_then(|n| n.to_str())) else { continue };
            let _ = std::fs::write(dir.join(name), text);
        }
        let (ok, bad): (Vec<_>, Vec<_>) = args.data_files.iter().partition(|f| f.error.is_none());
        if !ok.is_empty() {
            output.push_str(&format!(
                "\n[ARTIFACT DATA READY — {}. In your create_artifact HTML write the token {{{{data:NAME}}}} \
                 where the data should go (e.g. `const DATA = {{{{data:{}}}}};` inside a <script>), and LexiChat \
                 splices the exact file contents in. Do NOT print these contents or copy the values into your \
                 HTML by hand — reference them by name. This file persists: it is re-staged into \
                 /work/artifacts on later turns, so you can read it back with run_python or reference \
                 it with the token in a subsequent message.]",
                ok.iter().map(|f| format!("/work/artifacts/{} ({} chars)", f.name, f.chars))
                    .collect::<Vec<_>>().join(", "),
                ok[0].name));
        }
        for f in bad {
            output.push_str(&format!("\n[Artifact data /work/artifacts/{} unusable: {}]",
                f.name, f.error.as_deref().unwrap_or("unknown error")));
        }
    }

    if let Some(tx) = state.pending_python_result.lock().unwrap().remove(&args.request_id) {
        let _ = tx.send(ollama::PyResult { output, images: args.images });
    }
    Ok(())
}

/// Complete a deferred `run_python` output save: add `dir` to the sandbox, write the stashed
/// files into it, and return their real paths. Called after the user picks a folder in response
/// to a `sandbox-save-request`.
#[tauri::command]
fn save_pending_outputs(dir: String, state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let files = std::mem::take(&mut *state.pending_output_files.lock().unwrap());
    if files.is_empty() { return Ok(Vec::new()); }
    // Add the chosen folder to the sandbox (idempotent) and persist it.
    {
        let mut dirs = state.allowed_dirs.lock().unwrap();
        if !dirs.contains(&dir) { dirs.push(dir.clone()); }
    }
    persist_allowed_dirs(&state);
    let mut saved = Vec::new();
    for (name, bytes) in files {
        let dest = std::path::Path::new(&dir).join(&name);
        std::fs::write(&dest, &bytes).map_err(|e| format!("{name}: {e}"))?;
        saved.push(dest.display().to_string());
    }
    Ok(saved)
}

#[derive(Deserialize)]
struct CallToolFromCodeArgs {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

/// (name, description, parameters-schema) of every enabled external tool (OpenAPI + SPARQL),
/// parsed from their schemas. The parameter schema lets code-mode construct correct arguments
/// without trial-and-error. MCP tools are added separately (async lock) by the caller.
fn external_tool_catalog(
    specs: &[RegisteredSpec],
    sparql: &[RegisteredSparqlEndpoint],
) -> Vec<(String, String, serde_json::Value)> {
    let mut out = Vec::new();
    let parse = |schema: &serde_json::Value| serde_json::from_value::<ollama::ToolSchema>(schema.clone()).ok();
    for s in specs { for t in &s.tools {
        if let Some(ts) = parse(&t.schema) { out.push((ts.function.name, ts.function.description, ts.function.parameters)); }
    }}
    for e in sparql { for t in &e.tools {
        if let Some(ts) = parse(&t.schema) { out.push((ts.function.name, ts.function.description, ts.function.parameters)); }
    }}
    out
}

/// Code-mode bridge: `run_python`'s `call_tool(name, args)` / `list_tools()` route here. Gated by
/// the profile's "allow code to call tools" flag and an allowlist of the run's enabled tools —
/// code can only invoke tools the profile already exposes. Returns the tool's full result as JSON
/// (parsed when possible, else a string). `name == "__list__"` returns the catalog for discovery.
#[tauri::command]
async fn call_tool_from_code(
    call: CallToolFromCodeArgs,
    app: AppHandle,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    // Channel self-test (dev): echo args straight back, bypassing gates — isolates the
    // worker↔runner↔Rust round-trip from model/tool/permission concerns.
    if call.name == "__echo__" {
        return Ok(serde_json::json!({ "echo": call.args }));
    }
    // Gate 1: the profile must allow code to call tools.
    if !*state.code_tools_allowed.lock().unwrap() {
        return Err("This profile does not allow code to call tools. Enable \"Allow code to call \
                    tools\" in the profile to use call_tool().".into());
    }

    let specs = state.openapi_specs.lock().unwrap().clone();
    let sparql = state.sparql_endpoints.lock().unwrap().clone();
    let allowed_dirs = state.allowed_dirs.lock().unwrap().clone();
    let builtins = state.run_callable_builtins.lock().unwrap().clone();
    let mut ext = external_tool_catalog(&specs, &sparql);
    // MCP tools (name + description + input schema) from the active connections.
    let mcp: Vec<(String, String, serde_json::Value)> = state.mcp_connections.lock().await.values()
        .flat_map(|c| c.tools.iter().map(|t| (t.name.clone(), t.description.clone(), t.input_schema.clone())))
        .collect();
    ext.extend(mcp);

    // Discovery: list all callable tools, with parameter schemas so code can build correct args.
    if call.name == "__list__" {
        let mut out: Vec<serde_json::Value> = builtins.iter()
            .map(|n| serde_json::json!({ "name": n, "description": "built-in tool" }))
            .collect();
        for (n, d, p) in &ext { out.push(serde_json::json!({ "name": n, "description": d, "parameters": p })); }
        return Ok(serde_json::Value::Array(out));
    }

    // Gate 2: allowlist — only tools enabled for this run may be called.
    let enabled = builtins.contains(&call.name) || ext.iter().any(|(n, _, _)| n == &call.name);
    if !enabled {
        return Err(format!("Tool '{}' is not available. Call list_tools() to see callable tools.", call.name));
    }

    let result = ollama::call_one_tool(
        &call.name, &call.args, &specs, &sparql, &state.mcp_connections,
        &allowed_dirs, &[], 10, &app,
    ).await;

    // Hard size cap to protect worker memory.
    const MAX: usize = 10 * 1024 * 1024;
    if result.len() > MAX {
        return Err(format!("Tool result too large ({} bytes). Narrow the query or paginate.", result.len()));
    }
    // OpenAPI tool results are prefixed "HTTP <status>\n<body>". Strip that so code-mode's
    // json.loads() gets the clean body directly (the model was burning retries stripping it).
    let cleaned = result.strip_prefix("HTTP ")
        .and_then(|_| result.split_once('\n').map(|(_, b)| b.to_string()))
        .unwrap_or_else(|| result.clone());
    // Return parsed JSON when possible so Python gets a dict/list; else the raw (uncleaned) string.
    Ok(serde_json::from_str::<serde_json::Value>(&cleaned).unwrap_or(serde_json::Value::String(result)))
}

#[tauri::command]
fn get_allowed_dirs(state: State<'_, AppState>) -> Vec<String> {
    state.allowed_dirs.lock().unwrap().clone()
}

/// List the loaded skills (id, name, description, requires, body, builtin).
#[tauri::command]
fn get_skills(state: State<'_, AppState>) -> Vec<skills::RegisteredSkill> {
    state.skills.lock().unwrap().clone()
}

#[derive(Deserialize)]
struct SaveSkillArgs {
    id: String,
    name: String,
    description: String,
    #[serde(default)] category: String,
    #[serde(default)] requires: Vec<String>,
    body: String,
}

/// Create or update a CUSTOM skill (built-in ids are rejected). Returns the reloaded catalogue.
#[tauri::command]
fn save_skill(args: SaveSkillArgs, state: State<'_, AppState>) -> Result<Vec<skills::RegisteredSkill>, String> {
    skills::write_skill(&args.id, &args.name, &args.description, &args.category, &args.requires, &args.body)?;
    let loaded = skills::load_skills();
    *state.skills.lock().unwrap() = loaded.clone();
    Ok(loaded)
}

/// Delete a custom skill (built-ins can't be deleted). Returns the reloaded catalogue.
#[tauri::command]
fn delete_skill(id: String, state: State<'_, AppState>) -> Result<Vec<skills::RegisteredSkill>, String> {
    skills::delete_skill(&id)?;
    let loaded = skills::load_skills();
    *state.skills.lock().unwrap() = loaded.clone();
    Ok(loaded)
}

#[derive(Deserialize, Serialize, Default)]
struct ResourceBlob { name: String, b64: String }

#[derive(Deserialize)]
struct ImportSkillArgs {
    #[serde(default)] id: Option<String>,
    name: String,
    description: String,
    #[serde(default)] category: String,
    #[serde(default)] requires: Vec<String>,
    body: String,
    #[serde(default)] resources: Vec<ResourceBlob>,
}

/// Install a custom skill from an imported profile bundle (with any resource files). Idempotent when
/// the same id+body already exists (reuses it); otherwise picks a free, filesystem-safe id. Returns
/// the final id so the caller can remap the profile's enabledSkillIds. Built-in ids are never written.
#[tauri::command]
fn import_skill(args: ImportSkillArgs, state: State<'_, AppState>) -> Result<String, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let existing = state.skills.lock().unwrap().clone();
    // Re-importing the identical skill (same id + body) is a no-op — reuse it.
    if let Some(want) = args.id.as_deref() {
        if let Some(s) = existing.iter().find(|s| s.id == want && !s.builtin) {
            if s.body.trim() == args.body.trim() { return Ok(s.id.clone()); }
        }
    }
    let ids: Vec<String> = existing.iter().map(|s| s.id.clone()).collect();
    let id = match args.id.as_deref() {
        Some(i) if !i.trim().is_empty() && !skills::is_builtin(i) && !ids.iter().any(|e| e == i) => i.trim().to_string(),
        _ => skills::unique_skill_id(&args.name, &ids),
    };
    skills::write_skill(&id, &args.name, &args.description, &args.category, &args.requires, &args.body)?;
    for r in &args.resources {
        if let Ok(bytes) = B64.decode(r.b64.trim()) {
            let _ = skills::add_resource(&id, &r.name, &bytes); // best-effort; skip a bad resource
        }
    }
    *state.skills.lock().unwrap() = skills::load_skills();
    Ok(id)
}

#[derive(Deserialize)]
struct AddResourceArgs { id: String, name: String, b64: String }

/// Upload a resource file into a custom skill's folder. Returns the reloaded catalogue.
#[tauri::command]
fn add_skill_resource(args: AddResourceArgs, state: State<'_, AppState>) -> Result<Vec<skills::RegisteredSkill>, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let bytes = B64.decode(args.b64.trim()).map_err(|e| format!("bad file data: {e}"))?;
    skills::add_resource(&args.id, &args.name, &bytes)?;
    let loaded = skills::load_skills();
    *state.skills.lock().unwrap() = loaded.clone();
    Ok(loaded)
}

#[derive(Deserialize)]
struct RemoveResourceArgs { id: String, name: String }

/// Delete a resource file from a custom skill. Returns the reloaded catalogue.
#[tauri::command]
fn remove_skill_resource(args: RemoveResourceArgs, state: State<'_, AppState>) -> Result<Vec<skills::RegisteredSkill>, String> {
    skills::remove_resource(&args.id, &args.name)?;
    let loaded = skills::load_skills();
    *state.skills.lock().unwrap() = loaded.clone();
    Ok(loaded)
}

/// Full skill bundle (fields + resource files as base64) for embedding in a profile export.
#[tauri::command]
fn export_skill_bundle(id: String, state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let s = state.skills.lock().unwrap().iter().find(|s| s.id == id).cloned()
        .ok_or_else(|| format!("skill '{id}' not found"))?;
    let resources: Vec<ResourceBlob> = s.resources.iter().filter_map(|r|
        skills::read_resource(&id, r).ok().map(|b| ResourceBlob { name: r.clone(), b64: B64.encode(&b) })).collect();
    Ok(serde_json::json!({
        "id": s.id, "name": s.name, "description": s.description, "category": s.category,
        "requires": s.requires, "body": s.body, "resources": resources,
    }))
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

/// Write a small diagnostics file into the app data dir. Used by the artifact frame probe, whose
/// result can only be observed inside the real webview — the shipped app has no dev-control server,
/// so it leaves its findings on disk instead. Name is reduced to a bare filename so this can't be
/// steered outside the data dir.
#[tauri::command]
fn write_diagnostics(name: String, content: String) -> Result<String, String> {
    let file = std::path::Path::new(&name)
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| "invalid diagnostics filename".to_string())?;
    let path = crate::dirs_path().join(file);
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(path.display().to_string())
}

#[tauri::command]
fn save_document(path: String, content: String) -> Result<(), String> {
    crate::tools::save_document(&path, &content)
}

#[derive(Deserialize)]
struct RenderReportArgs {
    markdown: String,
    #[serde(default)] title: Option<String>,
    #[serde(default)] subtitle: Option<String>,
    /// `data:` image URLs (charts/maps from the chat) to append as a Figures section.
    #[serde(default)] figures: Vec<String>,
}

/// Render a markdown report into the self-contained, styled HTML "artifact" (for inline preview
/// before the user saves it). Saving writes the previewed HTML directly.
#[tauri::command]
fn render_report_html(args: RenderReportArgs) -> String {
    crate::report::render_report_html(&args.markdown, args.title.as_deref(), args.subtitle.as_deref(), &args.figures)
}

/// Write a report's themed HTML to a temp file and open it in the system's default browser, so the
/// user can Print → Save as PDF for a pixel-faithful copy (CSS theme + embedded charts). This is
/// the only offline way to a styled PDF without bundling a rendering engine.
#[tauri::command]
fn open_html_in_browser(html: String, app: AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt as _;
    let path = std::env::temp_dir().join(format!("lexichat-report-{}.html", uuid_v4()));
    std::fs::write(&path, html).map_err(|e| e.to_string())?;
    app.opener()
        .open_url(format!("file://{}", path.to_string_lossy()), None::<String>)
        .map_err(|e| format!("Could not open the report: {e}"))
}

#[derive(Deserialize)]
struct SaveDataUrlArgs { path: String, data_url: String }

/// Save a base64 `data:` URL (e.g. an inline chart the user wants to keep) to `path`.
#[tauri::command]
fn save_data_url(args: SaveDataUrlArgs) -> Result<(), String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let payload = args.data_url.split(',').nth(1).ok_or("not a base64 data URL")?;
    let bytes = B64.decode(payload).map_err(|e| format!("decode failed: {e}"))?;
    std::fs::write(&args.path, bytes).map_err(|e| e.to_string())
}

#[tauri::command]
fn read_file_text(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

/// Read any file's bytes as base64 (for uploading a skill resource). Capped to protect memory.
#[tauri::command]
fn read_file_base64(path: String) -> Result<String, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if bytes.len() > 25 * 1024 * 1024 { return Err("file is too large (max 25 MB)".into()); }
    Ok(B64.encode(&bytes))
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
    // Unblock the agent loop if it's paused waiting for this decision.
    if let Some(tx) = state.pending_app_approval.lock().unwrap().take() { let _ = tx.send(true); }
    Ok(())
}

/// User dismissed an MCP-app approval prompt (don't allow it) — unblock the paused agent loop so it
/// continues without approving the server.
#[tauri::command]
fn skip_mcp_app(state: State<'_, AppState>) -> Result<(), String> {
    if let Some(tx) = state.pending_app_approval.lock().unwrap().take() { let _ = tx.send(false); }
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

/// Draft a scheduled job from a plain-English goal. Returns a job (disabled) for the user to
/// review and save through the normal form — it persists and enables nothing itself.
#[tauri::command]
async fn draft_job_from_goal(
    args: job_designer::DraftJobArgs,
    state: State<'_, AppState>,
) -> Result<job_designer::DraftedJob, String> {
    let backend = state.backend.lock().unwrap().clone();
    job_designer::draft_job(&backend, args).await
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
// ── Dev control server (debug builds only) ────────────────────────────────────
// A localhost HTTP surface so an external driver (e.g. an agent in the terminal) can run chat
// turns end-to-end and read back a structured trace — WITHOUT clicking the GUI. Routes through
// the frontend's real send() path so webview flows (Pyodide run_python, MCP apps, call_tool)
// actually execute. Gated hard: debug build + LEXICHAT_DEV_CONTROL=1, bound to 127.0.0.1. Never
// compiled into release builds.

/// Frontend's report of a completed `/dev/run`: resolves the parked HTTP request with the trace.
/// (Registered in all builds — harmless in release where the server never runs — so the command
/// list doesn't diverge by build type.)
#[derive(Deserialize)]
struct DevControlReportArgs { id: u64, trace: serde_json::Value }

#[tauri::command]
fn dev_control_report(args: DevControlReportArgs, state: State<'_, AppState>) -> Result<(), String> {
    let tx = state.pending_dev_run.lock().unwrap().remove(&args.id);
    match tx {
        Some(tx) => { let _ = tx.send(args.trace); Ok(()) }
        // Silently succeeding here hid the real failure: the run had completed fine but the HTTP
        // caller had already timed out, so its id was gone. Say so instead of swallowing it.
        None => {
            eprintln!("[dev-control] run {} reported after its caller gave up — the turn finished \
                       but the HTTP request had already timed out", args.id);
            Err(format!("no pending dev run with id {} (caller timed out)", args.id))
        }
    }
}

#[cfg(debug_assertions)]
fn start_dev_control_server(app: AppHandle) {
    if std::env::var("LEXICHAT_DEV_CONTROL").ok().as_deref() != Some("1") { return; }
    let port: u16 = std::env::var("LEXICHAT_DEV_CONTROL_PORT").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(8787);
    tauri::async_runtime::spawn(async move {
        use tokio::net::TcpListener;
        let listener = match TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => { eprintln!("[dev-control] bind 127.0.0.1:{port} failed: {e}"); return; }
        };
        eprintln!("[dev-control] listening on http://127.0.0.1:{port}  — POST /dev/run, GET /dev/ping");
        loop {
            match listener.accept().await {
                Ok((stream, _)) => { let app = app.clone(); tauri::async_runtime::spawn(dev_control_conn(stream, app)); }
                Err(e) => eprintln!("[dev-control] accept: {e}"),
            }
        }
    });
}

#[cfg(debug_assertions)]
fn find_subsequence(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

#[cfg(debug_assertions)]
async fn dev_control_conn(mut stream: tokio::net::TcpStream, app: AppHandle) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    // Read headers, then the body per Content-Length.
    let (headers_end, content_length) = loop {
        if let Some(pos) = find_subsequence(&buf, b"\r\n\r\n") {
            let header_str = String::from_utf8_lossy(&buf[..pos]);
            let cl = header_str.lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:")
                    .map(|v| v.trim().parse::<usize>().unwrap_or(0)))
                .unwrap_or(0);
            break (pos + 4, cl);
        }
        match stream.read(&mut tmp).await {
            Ok(0) => break (buf.len(), 0),
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return,
        }
        if buf.len() > 4_000_000 { break (buf.len(), 0); }
    };
    while buf.len() < headers_end + content_length {
        match stream.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return,
        }
    }
    let request_line = String::from_utf8_lossy(&buf).lines().next().unwrap_or("").to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let body: Vec<u8> = buf.get(headers_end..(headers_end + content_length).min(buf.len()))
        .map(|s| s.to_vec()).unwrap_or_default();

    let (status, json) = route_dev_control(&method, &path, &body, &app).await;
    let payload = serde_json::to_string(&json).unwrap_or_else(|_| "{}".into());
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
        payload.len(), payload);
    let _ = stream.write_all(resp.as_bytes()).await;
    let _ = stream.flush().await;
}

/// Park a oneshot, emit `event` to the frontend with the params, and await the frontend's report
/// (via `dev_control_report`). Shared by /dev/run, /dev/state, /dev/config.
#[cfg(debug_assertions)]
async fn dev_await(app: &AppHandle, event: &str, params: serde_json::Value) -> Result<serde_json::Value, ()> {
    use tauri::Emitter as _;
    let state = app.state::<AppState>();
    let id = state.dev_run_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let (tx, rx) = tokio::sync::oneshot::channel();
    state.pending_dev_run.lock().unwrap().insert(id, tx);
    let _ = app.emit(event, serde_json::json!({ "id": id, "params": params }));
    // 900s was too short and produced a misleading failure: real agent turns on a local model run
    // well past it (2930s observed), so the server gave up while the frontend was still working
    // correctly, then reported "frontend did not respond". `send()` awaits the whole agent loop, so
    // this timeout — not anything on the frontend — is what bounds a /dev/run.
    const DEV_RUN_TIMEOUT_SECS: u64 = 3600;
    match tokio::time::timeout(std::time::Duration::from_secs(DEV_RUN_TIMEOUT_SECS), rx).await {
        Ok(Ok(v)) => Ok(v),
        _ => { state.pending_dev_run.lock().unwrap().remove(&id); Err(()) }
    }
}

#[cfg(debug_assertions)]
async fn route_dev_control(method: &str, path: &str, body: &[u8], app: &AppHandle) -> (&'static str, serde_json::Value) {
    // The old wording ("is the app window open?") sent debugging down the wrong path: the real
    // cause was the frontend still being mid-run and dropping the request. The listener now
    // reports `busy` explicitly, so a genuine timeout here means something else.
    let timeout_err = ("504 Gateway Timeout", serde_json::json!({
        "error": "frontend did not report within 900s — it may be mid-run, wedged, or the window \
                  is closed. A busy app now replies with error \"busy\" instead of timing out." }));
    match (method, path) {
        ("GET", "/dev/ping") => ("200 OK", serde_json::json!({ "ok": true, "app": "lexichat" })),
        ("GET", "/dev/state") => match dev_await(app, "dev-control-state", serde_json::json!({})).await {
            Ok(v) => ("200 OK", v), Err(_) => timeout_err,
        },
        ("POST", "/dev/config") => {
            let params: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::json!({}));
            match dev_await(app, "dev-control-config", params).await {
                Ok(v) => ("200 OK", v), Err(_) => timeout_err,
            }
        }
        ("POST", "/dev/run") => {
            let params: serde_json::Value = serde_json::from_slice(body).unwrap_or(serde_json::json!({}));
            if params.get("message").and_then(|m| m.as_str()).unwrap_or("").trim().is_empty() {
                return ("400 Bad Request", serde_json::json!({ "error": "missing 'message'" }));
            }
            match dev_await(app, "dev-control-run", params).await {
                Ok(trace) => ("200 OK", serde_json::json!({ "trace": trace })), Err(_) => timeout_err,
            }
        }
        _ => ("404 Not Found", serde_json::json!({ "error": "unknown endpoint" })),
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        // Remember the window's size, position and maximised state across restarts.
        // VISIBLE is deliberately excluded: the window is hidden (not destroyed) when closed to the
        // tray, so saving visibility would restore the app with no window on next launch.
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_state_flags(
                    tauri_plugin_window_state::StateFlags::SIZE
                        | tauri_plugin_window_state::StateFlags::POSITION
                        | tauri_plugin_window_state::StateFlags::MAXIMIZED,
                )
                .build(),
        )
        .setup(|app| {
            build_menu(app)?;
            // Drop working files whose conversation is gone (deleted outside the app, or a first
            // turn that was never saved).
            prune_orphan_conversation_files();
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
            #[cfg(debug_assertions)]
            start_dev_control_server(app.handle().clone());
            Ok(())
        })
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_models,
            list_managed_models,
            pull_ollama_model,
            delete_ollama_model,
            cancel_ollama_pull,
            set_backend,
            reset_conversation,
            stop_generation,
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
            skip_mcp_app,
            mcp_ui_call_tool,
            set_openapi_specs,
            oauth2_authorize,
            get_allowed_dirs,
            add_allowed_dir,
            remove_allowed_dir,
            set_allowed_dirs,
            respond_code_permission,
            set_code_exec_unlocked,
            set_image_gen_config,
            download_image_model,
            cancel_image_model_download,
            download_image_engine,
            image_engine_status,
            get_usage_stats,
            record_turn_usage,
            system_stats,
            get_skills,
            save_skill,
            delete_skill,
            import_skill,
            add_skill_resource,
            remove_skill_resource,
            export_skill_bundle,
            respond_python_result,
            save_pending_outputs,
            call_tool_from_code,
            dev_control_report,
            render_report_html,
            open_html_in_browser,
            write_file_text,
            write_diagnostics,
            read_file_text,
            read_file_base64,
            read_image_data_url,
            save_document,
            save_data_url,
            get_jobs,
            draft_job_from_goal,
            save_job,
            delete_job,
            run_job_now,
            get_job_runs,
            clear_job_runs,
            list_conversations,
            save_active_conversation,
            load_conversation,
            conversation_len,
            truncate_conversation,
            get_wiki_graph,
            read_wiki_page,
            delete_wiki_page,
            delete_conversation,
            rename_conversation,
            set_conversation_pinned,
            search_conversations,
            set_conversation_folder,
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

#[cfg(test)]
mod tests {
    use super::supersede_run;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};

    /// The invariant behind the stopped-run-leak fix: starting a run cancels the previous run's
    /// token and hands the new run a fresh, independent one — and a stopped run is NEVER revived,
    /// however many runs start afterwards.
    #[test]
    fn superseding_cancels_the_old_run_and_never_revives_it() {
        let slot: Mutex<Arc<AtomicBool>> = Mutex::new(Arc::new(AtomicBool::new(false)));

        let run_a = slot.lock().unwrap().clone(); // run A's token (currently active)
        let run_b = supersede_run(&slot);         // run B starts, superseding A
        assert!(run_a.load(SeqCst), "A is cancelled the moment B starts");
        assert!(!run_b.load(SeqCst), "B starts uncancelled");

        // Run C starts. The old bug reset a shared flag here, reviving A. It must not.
        let run_c = supersede_run(&slot);
        assert!(run_a.load(SeqCst), "A stays cancelled — a later run cannot revive it");
        assert!(run_b.load(SeqCst), "B is now cancelled by C");
        assert!(!run_c.load(SeqCst), "C is the live run");
    }
}
