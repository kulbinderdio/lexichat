mod ollama;
mod tools;
mod openapi;
mod mcp;

use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, State};
use serde::{Deserialize, Serialize};
use openapi::RegisteredSpec;
use mcp::{MCPServerConfig, MCPConnection, AuthConfig};

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    pub ollama_host:     Mutex<String>,
    pub conversation:    Mutex<Vec<ollama::WireMessage>>,
    pub openapi_specs:   Mutex<Vec<RegisteredSpec>>,
    pub mcp_servers:     Mutex<Vec<MCPServerConfig>>,
    pub mcp_connections: tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    /// Allowed directories for file tool access
    pub allowed_dirs:    Mutex<Vec<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        // Load persisted allowed dirs
        let saved = std::fs::read_to_string(allowed_dirs_path())
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_default();
        Self {
            ollama_host:     Mutex::new("http://localhost:11434".into()),
            conversation:    Mutex::new(Vec::new()),
            openapi_specs:   Mutex::new(Vec::new()),
            mcp_servers:     Mutex::new(Vec::new()),
            mcp_connections: tokio::sync::Mutex::new(HashMap::new()),
            allowed_dirs:    Mutex::new(saved),
        }
    }
}

fn allowed_dirs_path() -> std::path::PathBuf {
    dirs_path().join("allowed_dirs.json")
}

fn dirs_path() -> std::path::PathBuf {
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
    Ok(())
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
    #[serde(default)]
    pub disabled_mcp_tools: Vec<String>,
}

fn default_web_search_results() -> usize { 10 }

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

    // Collect all active tool schemas (builtins + openapi + mcp)
    let mut all_tools = args.tools.clone();

    // Add OpenAPI tools (all currently registered — already profile-scoped by set_openapi_specs)
    {
        let extra: Vec<ollama::ToolSchema> = state.openapi_specs.lock().unwrap().iter()
            .flat_map(|spec| spec.tools.iter()
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);
    }

    // Add MCP tools (profile-scoped by set_mcp_servers, filtered by per-tool disable list)
    {
        let extra: Vec<ollama::ToolSchema> = state.mcp_connections.lock().await.values()
            .flat_map(|conn| conn.tools.iter()
                .filter(|t| !args.disabled_mcp_tools.contains(&t.name))
                .filter_map(|t| serde_json::from_value::<ollama::ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);
    }

    let specs_snapshot: Vec<openapi::RegisteredSpec> = state.openapi_specs.lock().unwrap().clone();
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
        &state.mcp_connections,
        allowed_dirs_snapshot,
        args.web_search_results,
        &app,
    )
    .await
    .map_err(|e| e.to_string())
}

// ── Sandbox commands ──────────────────────────────────────────────────────────

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
fn set_allowed_dirs(dirs: Vec<String>, state: State<'_, AppState>) -> Result<(), String> {
    *state.allowed_dirs.lock().unwrap() = dirs;
    Ok(())
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

// ── MCP commands ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AddMCPServerArgs {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub auth: AuthConfig,
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
}

#[derive(Serialize, Clone)]
pub struct MCPToolInfo {
    pub name: String,
    pub description: String,
}

#[tauri::command]
async fn add_mcp_server(
    args: AddMCPServerArgs,
    state: State<'_, AppState>,
) -> Result<MCPServerInfo, String> {
    let id = uuid_v4();
    let config = MCPServerConfig {
        id: id.clone(),
        name: args.name.clone(),
        command: args.command.clone(),
        args: args.args.clone(),
        env: args.env.clone(),
        enabled: true,
        auth: args.auth,
    };

    state.mcp_servers.lock().unwrap().push(config.clone());

    // Try to connect
    match MCPConnection::connect(config.clone()).await {
        Ok(conn) => {
            let tools: Vec<MCPToolInfo> = conn.tools.iter().map(|t| MCPToolInfo {
                name: t.name.clone(),
                description: t.description.clone(),
            }).collect();
            let tool_count = tools.len();
            state.mcp_connections.lock().await.insert(id.clone(), conn);
            Ok(MCPServerInfo { id, name: args.name, command: args.command, args: args.args, connected: true, tool_count, tools, error: None })
        }
        Err(e) => {
            Ok(MCPServerInfo { id, name: args.name, command: args.command, args: args.args, connected: false, tool_count: 0, tools: vec![], error: Some(e) })
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
            let tools: Vec<MCPToolInfo> = conn.tools.iter().map(|t| MCPToolInfo { name: t.name.clone(), description: t.description.clone() }).collect();
            let tool_count = tools.len();
            state.mcp_connections.lock().await.insert(id.clone(), conn);
            Ok(MCPServerInfo { id, name: config.name, command: config.command, args: config.args, connected: true, tool_count, tools, error: None })
        }
        Err(e) => Ok(MCPServerInfo { id, name: config.name, command: config.command, args: config.args, connected: false, tool_count: 0, tools: vec![], error: Some(e) })
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
        };
        state.mcp_servers.lock().unwrap().push(config.clone());
        match MCPConnection::connect(config).await {
            Ok(conn) => {
                let tools: Vec<MCPToolInfo> = conn.tools.iter()
                    .map(|t| MCPToolInfo { name: t.name.clone(), description: t.description.clone() })
                    .collect();
                let tool_count = tools.len();
                state.mcp_connections.lock().await.insert(srv.id.clone(), conn);
                results.push(MCPServerInfo { id: srv.id, name: srv.name, command: srv.command,
                    args: srv.args, connected: true, tool_count, tools, error: None });
            }
            Err(e) => {
                results.push(MCPServerInfo { id: srv.id, name: srv.name, command: srv.command,
                    args: srv.args, connected: false, tool_count: 0, tools: vec![], error: Some(e) });
            }
        }
    }
    Ok(results)
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

fn uuid_v4() -> String {
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

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            build_menu(app)?;
            Ok(())
        })
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_models,
            set_ollama_host,
            reset_conversation,
            send_message,
            get_builtin_schemas,
            register_openapi_spec,
            remove_openapi_spec,
            list_openapi_specs,
            add_mcp_server,
            remove_mcp_server,
            list_mcp_servers,
            reconnect_mcp_server,
            set_mcp_servers,
            set_openapi_specs,
            oauth2_authorize,
            get_allowed_dirs,
            add_allowed_dir,
            remove_allowed_dir,
            set_allowed_dirs,
            write_file_text,
            read_file_text,
            save_document,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
