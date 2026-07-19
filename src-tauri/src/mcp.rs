use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex as AsyncMutex;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 { REQUEST_ID.fetch_add(1, Ordering::SeqCst) }

// ── Auth ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type")]
pub enum AuthConfig {
    #[default]
    #[serde(rename = "none")]
    None,
    #[serde(rename = "bearer")]
    Bearer {
        #[serde(default)]
        bearer_token: String,
    },
    #[serde(rename = "apikey")]
    ApiKey {
        #[serde(default)]
        api_key_header: String,
        #[serde(default)]
        api_key_value: String,
    },
    #[serde(rename = "basic")]
    Basic {
        #[serde(default)]
        basic_username: String,
        #[serde(default)]
        basic_password: String,
    },
    #[serde(rename = "oauth2")]
    OAuth2 {
        #[serde(default)]
        token_url: String,
        #[serde(default)]
        client_id: String,
        #[serde(default)]
        client_secret: String,
        #[serde(default)]
        scope: String,
        /// Authorization URL for the auth-code flow (optional; if absent, falls back to client-credentials)
        #[serde(default)]
        authorization_url: String,
        /// Stored access token after user has authorized via browser
        #[serde(default)]
        access_token: String,
        /// Stored refresh token for silent renewal
        #[serde(default)]
        refresh_token: String,
    },
}

impl AuthConfig {
    /// Apply static auth headers (all types except OAuth2, which requires async token fetch).
    pub fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            AuthConfig::None => req,
            AuthConfig::Bearer { bearer_token } => {
                req.header("Authorization", format!("Bearer {bearer_token}"))
            }
            AuthConfig::ApiKey { api_key_header, api_key_value } => {
                req.header(api_key_header.as_str(), api_key_value.as_str())
            }
            AuthConfig::Basic { basic_username, basic_password } => {
                use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
                let encoded = B64.encode(format!("{basic_username}:{basic_password}"));
                req.header("Authorization", format!("Basic {encoded}"))
            }
            AuthConfig::OAuth2 { .. } => req, // handled by apply_async
        }
    }

    /// Fetch an OAuth2 token. Uses stored access_token first, then client-credentials flow.
    async fn fetch_oauth2_token(&self, client: &reqwest::Client) -> Option<String> {
        let AuthConfig::OAuth2 { token_url, client_id, client_secret, scope, access_token, .. } = self else {
            return None;
        };
        // If we already have a stored token (from browser auth), use it
        if !access_token.is_empty() {
            return Some(access_token.clone());
        }
        // Fall back to client-credentials flow
        if token_url.is_empty() || client_id.is_empty() { return None; }
        let mut params = vec![
            ("grant_type", "client_credentials"),
            ("client_id",  client_id.as_str()),
            ("client_secret", client_secret.as_str()),
        ];
        if !scope.is_empty() {
            params.push(("scope", scope.as_str()));
        }
        let resp = client.post(token_url.as_str()).form(&params).send().await.ok()?;
        let json: Value = resp.json().await.ok()?;
        json["access_token"].as_str().map(String::from)
    }

    /// Use the refresh_token to get a new access_token. Returns the new token on success.
    /// Works with any RFC 6749-compliant provider. Includes scope when provided (required
    /// by some providers, e.g. Microsoft). Only sends client_secret if non-empty (some
    /// providers use public-client / PKCE flows without a secret).
    pub async fn try_refresh(&self, client: &reqwest::Client) -> Option<String> {
        let AuthConfig::OAuth2 { token_url, client_id, client_secret, refresh_token, scope, .. } = self else {
            return None;
        };
        if refresh_token.is_empty() || token_url.is_empty() { return None; }
        let mut params: Vec<(&str, &str)> = vec![
            ("grant_type",    "refresh_token"),
            ("refresh_token", refresh_token.as_str()),
            ("client_id",     client_id.as_str()),
        ];
        if !client_secret.is_empty() { params.push(("client_secret", client_secret.as_str())); }
        if !scope.is_empty()         { params.push(("scope", scope.as_str())); }
        let resp = client.post(token_url.as_str()).form(&params).send().await.ok()?;
        let json: Value = resp.json().await.ok()?;
        json["access_token"].as_str().map(String::from)
    }

    /// Apply auth, fetching an OAuth2 token when needed.
    pub async fn apply_async(
        &self,
        client: &reqwest::Client,
        req: reqwest::RequestBuilder,
    ) -> reqwest::RequestBuilder {
        if matches!(self, AuthConfig::OAuth2 { .. }) {
            if let Some(token) = self.fetch_oauth2_token(client).await {
                return req.header("Authorization", format!("Bearer {token}"));
            }
            return req;
        }
        self.apply(req)
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    pub id: String,
    pub name: String,
    pub command: String,   // either a shell command or an http(s):// URL
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub enabled: bool,
    #[serde(default)]
    pub auth: AuthConfig,
    /// Opt-in: allow this server's tools to render interactive MCP-App UIs
    /// (sandboxed iframe). Off by default.
    #[serde(default)]
    pub enable_apps: bool,
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

// ── Tool returned by an MCP server ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub server_id: String,
    pub server_name: String,
    pub name: String,        // prefixed name used by the model
    pub raw_name: String,    // original name sent to the MCP server
    pub description: String,
    pub input_schema: Value,
    pub schema: Value,
    /// MCP Apps (SEP-1865): `ui://` resource this tool renders, from `_meta.ui.resourceUri`.
    #[serde(default)]
    pub ui_resource_uri: Option<String>,
}

/// A tool-call result that preserves the structured/UI parts the plain-text
/// path discards. Used for MCP Apps rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    pub text: String,
    #[serde(default)]
    pub structured: Option<Value>,
    /// Raw `content` array from the tool result (preserves image/resource blocks
    /// the text extraction drops — needed to forward to MCP-App UIs).
    #[serde(default)]
    pub content: Value,
    /// Raw `_meta` from the tool result.
    #[serde(default)]
    pub meta: Value,
    #[serde(default)]
    pub is_error: bool,
    /// Inline HTML of a UI resource (MCP-UI embedded resource, or a resource we fetched).
    #[serde(default)]
    pub ui_html: Option<String>,
    /// `ui://` URI referenced by the tool/result when the HTML wasn't inlined.
    #[serde(default)]
    pub ui_uri: Option<String>,
}

// ── Transports ────────────────────────────────────────────────────────────────

enum Transport {
    Stdio {
        stdin:  AsyncMutex<ChildStdin>,
        stdout: AsyncMutex<BufReader<ChildStdout>>,
        /// Tail of the process's stderr, filled by a background reader. Surfaced in errors
        /// so a server that dies during the handshake explains why (e.g. "profile not found").
        stderr: std::sync::Arc<std::sync::Mutex<String>>,
        #[allow(dead_code)]
        child:  Child,
    },
    Http {
        url:    String,
        client: reqwest::Client,
        auth:   AuthConfig,
    },
}

// ── Live connection ───────────────────────────────────────────────────────────

/// PATH augmented with the common tool locations a Finder/Dock-launched app misses
/// (Homebrew, Docker Desktop, user bin dirs). Only existing dirs are appended, so it's
/// harmless on Linux where those paths don't exist. On Windows the inherited PATH is used
/// as-is (different separator and no equivalent gap).
fn augmented_path() -> String {
    #[cfg(not(unix))]
    { std::env::var("PATH").unwrap_or_default() }
    #[cfg(unix)]
    {
        let mut dirs: Vec<String> = std::env::var("PATH")
            .map(|p| p.split(':').map(String::from).collect())
            .unwrap_or_default();
        let mut extra = vec![
            "/opt/homebrew/bin", "/opt/homebrew/sbin",
            "/usr/local/bin", "/usr/local/sbin",
            "/usr/bin", "/bin", "/usr/sbin", "/sbin",
            "/Applications/Docker.app/Contents/Resources/bin",
        ].into_iter().map(String::from).collect::<Vec<_>>();
        if let Some(home) = std::env::var_os("HOME") {
            let home = home.to_string_lossy();
            for sub in [".local/bin", ".cargo/bin", ".deno/bin", ".bun/bin", "go/bin", ".docker/bin", ".volta/bin"] {
                extra.push(format!("{home}/{sub}"));
            }
        }
        for d in extra {
            if !dirs.iter().any(|x| x == &d) && std::path::Path::new(&d).is_dir() {
                dirs.push(d);
            }
        }
        dirs.join(":")
    }
}

/// Resolve a bare executable name to its absolute path by searching `path`. Returns None
/// if `exe` already contains a separator (use it verbatim) or isn't found.
fn resolve_in_path(exe: &str, path: &str) -> Option<String> {
    if exe.is_empty() || exe.contains(std::path::MAIN_SEPARATOR) {
        return None;
    }
    let sep = if cfg!(unix) { ':' } else { ';' };
    for dir in path.split(sep) {
        if dir.is_empty() { continue; }
        let cand = std::path::Path::new(dir).join(exe);
        if cand.is_file() {
            return Some(cand.to_string_lossy().into_owned());
        }
    }
    None
}

pub struct MCPConnection {
    pub config: MCPServerConfig,
    pub tools:  Vec<MCPTool>,
    /// `capabilities` object from the server's `initialize` result (empty if none).
    pub server_capabilities: Value,
    transport:  Transport,
}

impl MCPConnection {
    pub async fn connect(config: MCPServerConfig) -> Result<Self, String> {
        let transport = if is_url(&config.command) {
            Transport::Http {
                url:    config.command.clone(),
                client: reqwest::Client::new(),
                auth:   config.auth.clone(),
            }
        } else {
            // Split on whitespace so users can paste a full command string
            // (e.g. "docker mcp gateway run --profile test") into the command field.
            let mut parts = config.command.split_whitespace();
            let exe = parts.next().unwrap_or("");
            let inline_args: Vec<&str> = parts.collect();

            // GUI apps launched from Finder/Dock inherit launchd's minimal PATH, which omits
            // Homebrew, Docker Desktop and user bin dirs — so `docker`/`npx`/`uvx` fail to
            // spawn even though they run fine in a terminal. Resolve against an augmented PATH.
            let path = augmented_path();
            let exe_abs = resolve_in_path(exe, &path);
            let program: &str = exe_abs.as_deref().unwrap_or(exe);

            let mut cmd = tokio::process::Command::new(program);
            cmd.args(&inline_args)
               .args(&config.args)
               .envs(&config.env)
               .env("PATH", &path)   // give the child (and its subprocesses) a full PATH too
               .kill_on_drop(true)
               .stdin(Stdio::piped())
               .stdout(Stdio::piped())
               .stderr(Stdio::piped());

            let mut child = cmd.spawn()
                .map_err(|e| {
                    let hint = if e.kind() == std::io::ErrorKind::NotFound {
                        format!(" — '{exe}' was not found on PATH. Enter its full path (e.g. /usr/local/bin/{exe}), or make sure the tool is installed.")
                    } else { String::new() };
                    format!("Failed to start '{}': {e}{hint}", config.command)
                })?;

            let stdin  = child.stdin.take().ok_or("No stdin")?;
            let stdout = child.stdout.take().ok_or("No stdout")?;

            // Drain stderr in the background into a bounded tail buffer so a server that
            // exits during the handshake can tell us why.
            let stderr_buf = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
            if let Some(errpipe) = child.stderr.take() {
                let sink = stderr_buf.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(errpipe).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        if let Ok(mut s) = sink.lock() {
                            s.push_str(&line);
                            s.push('\n');
                            if s.len() > 4000 { let cut = s.len() - 4000; s.drain(..cut); }
                        }
                    }
                });
            }

            Transport::Stdio {
                stdin:  AsyncMutex::new(stdin),
                stdout: AsyncMutex::new(BufReader::new(stdout)),
                stderr: stderr_buf,
                child,
            }
        };

        let mut conn = MCPConnection { config: config.clone(), tools: Vec::new(), server_capabilities: Value::Null, transport };

        // Initialize handshake. Newer protocolVersion so servers may advertise the
        // MCP Apps / resources capabilities; empty client capabilities are still valid.
        let init = conn.send_request("initialize", serde_json::json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "lexichat", "version": env!("CARGO_PKG_VERSION") }
        })).await.map_err(|e| format!("Initialize failed: {e}"))?;
        conn.server_capabilities = init["capabilities"].clone();

        // Notify initialized
        conn.send_notification("notifications/initialized", serde_json::json!({})).await?;

        // List tools
        let tools_response = conn.send_request("tools/list", serde_json::json!({})).await
            .map_err(|e| format!("tools/list failed: {e}"))?;

        conn.tools = parse_tools(&config, &tools_response);

        Ok(conn)
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = next_id();
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        match &mut self.transport {
            Transport::Http { url, client, auth } => {
                let (val, new_token) = http_rpc(client, url, auth, msg).await?;
                // If a token refresh occurred, update the stored auth in place
                if let Some(tok) = new_token {
                    if let AuthConfig::OAuth2 { ref mut access_token, .. } = auth {
                        *access_token = tok.clone();
                    }
                    // Also update the parent config so future send_request calls use the new token
                    if let AuthConfig::OAuth2 { ref mut access_token, .. } = self.config.auth {
                        *access_token = tok;
                    }
                }
                Ok(val)
            }
            Transport::Stdio { stdin, stdout, stderr, child } => {
                let line = serde_json::to_string(&msg).unwrap() + "\n";
                {
                    let mut s = stdin.lock().await;
                    s.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
                    s.flush().await.map_err(|e| e.to_string())?;
                }
                let mut out = stdout.lock().await;
                loop {
                    let mut buf = String::new();
                    out.read_line(&mut buf).await.map_err(|e| e.to_string())?;
                    if buf.is_empty() {
                        // Give the background reader a moment to flush the process's dying words.
                        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                        let tail = stderr.lock().map(|s| s.trim().to_string()).unwrap_or_default();
                        let exit = match child.try_wait() {
                            Ok(Some(status)) => format!(" (process exited: {status})"),
                            _ => String::new(),
                        };
                        if tail.is_empty() {
                            return Err(format!("MCP server closed connection{exit}"));
                        }
                        let tail = tail.rsplit('\n').take(6).collect::<Vec<_>>()
                            .into_iter().rev().collect::<Vec<_>>().join("\n");
                        return Err(format!("MCP server closed connection{exit}. Its output was:\n{tail}"));
                    }
                    let buf = buf.trim();
                    if buf.is_empty() { continue; }
                    let val: Value = serde_json::from_str(buf)
                        .map_err(|e| format!("JSON error: {e}"))?;
                    if val["id"] == id {
                        if let Some(err) = val.get("error") {
                            return Err(format!("MCP error: {err}"));
                        }
                        return Ok(val["result"].clone());
                    }
                }
            }
        }
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = serde_json::json!({ "jsonrpc": "2.0", "method": method, "params": params });

        match &self.transport {
            Transport::Http { url, client, auth } => {
                // Notifications have no id; server may return 202 or empty body — ignore result
                let req = client.post(url.as_str())
                    .header("Content-Type", "application/json")
                    .json(&msg);
                let _ = auth.apply_async(client, req).await.send().await;
                Ok(())
            }
            Transport::Stdio { stdin, .. } => {
                let line = serde_json::to_string(&msg).unwrap() + "\n";
                let mut s = stdin.lock().await;
                s.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
                s.flush().await.map_err(|e| e.to_string())?;
                Ok(())
            }
        }
    }

    /// Text-only tool call — unchanged behaviour for the plain agent path.
    pub async fn call_tool(&mut self, tool_name: &str, args: &Value) -> String {
        self.call_tool_rich(tool_name, args).await.text
    }

    /// Tool call that preserves structured content and any MCP-App UI resource.
    pub async fn call_tool_rich(&mut self, tool_name: &str, args: &Value) -> ToolCallResult {
        // tool_name is prefixed; look up the raw name + any declared UI resource.
        // Match the model-facing prefixed name OR the server's raw name (UI apps
        // call tools by their raw name over the bridge).
        let (raw, tool_ui_uri) = self.tools.iter()
            .find(|t| t.name == tool_name || t.raw_name == tool_name)
            .map(|t| (t.raw_name.clone(), t.ui_resource_uri.clone()))
            .unwrap_or_else(|| (tool_name.to_string(), None));

        let result = self.send_request("tools/call", serde_json::json!({
            "name": raw,
            "arguments": args
        })).await;

        let resp = match result {
            Ok(r) => r,
            Err(e) => return ToolCallResult {
                text: format!("MCP tool error: {e}"),
                structured: None, content: Value::Null, meta: Value::Null,
                is_error: true, ui_html: None, ui_uri: None,
            },
        };

        // Text: join content[].text, else pretty-print the whole result (unchanged).
        let text = if let Some(content) = resp["content"].as_array() {
            content.iter().filter_map(|c| c["text"].as_str()).collect::<Vec<_>>().join("\n")
        } else {
            serde_json::to_string_pretty(&resp).unwrap_or_default()
        };
        let is_error = resp["isError"].as_bool().unwrap_or(false);
        let structured = resp.get("structuredContent").cloned();

        // Detect a UI resource (MCP-UI embedded resource or ext-apps _meta ref).
        let (mut ui_html, ui_uri) = extract_ui(&resp, &tool_ui_uri);
        // If we have a ui:// uri but no inline HTML, fetch it via resources/read.
        if ui_html.is_none() {
            if let Some(uri) = ui_uri.clone() {
                if let Ok(Some(html)) = self.read_resource(&uri).await {
                    ui_html = Some(html);
                }
            }
        }

        let content = resp["content"].clone();
        let meta = resp["_meta"].clone();
        ToolCallResult { text, structured, content, meta, is_error, ui_html, ui_uri }
    }

    /// Fetch a resource's text/HTML via `resources/read`.
    pub async fn read_resource(&mut self, uri: &str) -> Result<Option<String>, String> {
        let resp = self.send_request("resources/read", serde_json::json!({ "uri": uri })).await?;
        if let Some(contents) = resp["contents"].as_array() {
            for c in contents {
                if let Some(t) = c["text"].as_str() {
                    return Ok(Some(t.to_string()));
                }
                if let Some(html) = c["blob"].as_str().and_then(decode_b64_utf8) {
                    return Ok(Some(html));
                }
            }
        }
        Ok(None)
    }
}

/// Decode a base64 string to a UTF-8 String (lossy), or None on failure.
fn decode_b64_utf8(b: &str) -> Option<String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.decode(b).ok().map(|bytes| String::from_utf8_lossy(&bytes).to_string())
}

/// Extract an MCP-App UI resource from a `tools/call` result.
/// Returns `(inline_html, ui_uri)`. Handles both dialects:
///   (a) MCP-UI — an embedded resource block in `content[]` (`ui://` uri or `text/html`),
///   (b) ext-apps — `_meta.ui.resourceUri` on the result, falling back to the tool's declared uri.
fn extract_ui(resp: &Value, tool_ui_uri: &Option<String>) -> (Option<String>, Option<String>) {
    let mut ui_html: Option<String> = None;
    let mut ui_uri: Option<String> = None;
    if let Some(content) = resp["content"].as_array() {
        for c in content {
            let res = &c["resource"];
            let uri  = res["uri"].as_str().or_else(|| c["uri"].as_str());
            let mime = res["mimeType"].as_str().or_else(|| c["mimeType"].as_str()).unwrap_or("");
            let is_ui = uri.map(|u| u.starts_with("ui://")).unwrap_or(false)
                || mime.starts_with("text/html");
            if is_ui {
                if let Some(h) = res["text"].as_str().or_else(|| c["text"].as_str()) {
                    ui_html = Some(h.to_string());
                } else if let Some(html) = res["blob"].as_str().and_then(decode_b64_utf8) {
                    ui_html = Some(html);
                }
                if ui_uri.is_none() { ui_uri = uri.map(String::from); }
                if ui_html.is_some() { break; }
            }
        }
    }
    if ui_uri.is_none() {
        ui_uri = resp["_meta"]["ui"]["resourceUri"].as_str().map(String::from)
            .or_else(|| tool_ui_uri.clone());
    }
    (ui_html, ui_uri)
}

// ── HTTP JSON-RPC helper ──────────────────────────────────────────────────────

async fn http_rpc(client: &reqwest::Client, url: &str, auth: &AuthConfig, msg: Value) -> Result<(Value, Option<String>), String> {
    let make_req = |token_override: Option<&str>| {
        let mut r = client
            .post(url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .json(&msg);
        if let Some(tok) = token_override {
            r = r.header("Authorization", format!("Bearer {tok}"));
        }
        r
    };

    let req = auth.apply_async(client, make_req(None)).await;
    let resp = req.send().await.map_err(|e| format!("HTTP request failed: {e}"))?;
    let status = resp.status();

    // On 401, attempt a token refresh and retry once
    if status.as_u16() == 401 {
        if let Some(new_token) = auth.try_refresh(client).await {
            let retry_resp = make_req(Some(&new_token))
                .send().await
                .map_err(|e| format!("HTTP request failed after refresh: {e}"))?;
            let retry_status = retry_resp.status();
            if !retry_status.is_success() && retry_status.as_u16() != 202 {
                return Err(format!("HTTP {retry_status}"));
            }
            let body = parse_http_rpc_body(retry_resp).await?;
            return Ok((body, Some(new_token)));
        }
        return Err(format!("HTTP 401 — re-authentication required"));
    }

    if !status.is_success() && status.as_u16() != 202 {
        return Err(format!("HTTP {status}"));
    }

    let body = parse_http_rpc_body(resp).await?;
    Ok((body, None))
}

async fn parse_http_rpc_body(resp: reqwest::Response) -> Result<Value, String> {
    // 202 No Content (notifications) or empty body
    let content_type = resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let body = resp.text().await.map_err(|e| e.to_string())?;
    if body.trim().is_empty() {
        return Ok(Value::Null);
    }

    if content_type.contains("text/event-stream") {
        for line in body.lines() {
            let line = line.trim();
            if let Some(data) = line.strip_prefix("data:") {
                let data = data.trim();
                if data == "[DONE]" { break; }
                if let Ok(val) = serde_json::from_str::<Value>(data) {
                    if val.get("result").is_some() || val.get("error").is_some() {
                        if let Some(err) = val.get("error") {
                            return Err(format!("MCP error: {err}"));
                        }
                        return Ok(val["result"].clone());
                    }
                }
            }
        }
        Err("No result found in SSE response".into())
    } else {
        let val: Value = serde_json::from_str(&body)
            .map_err(|e| format!("JSON parse error: {e}\nBody: {body}"))?;
        if let Some(err) = val.get("error") {
            return Err(format!("MCP error: {err}"));
        }
        Ok(val["result"].clone())
    }
}

// ── Parse tools from tools/list response ─────────────────────────────────────

fn parse_tools(config: &MCPServerConfig, result: &Value) -> Vec<MCPTool> {
    let arr = match result["tools"].as_array() {
        Some(a) => a,
        None    => return Vec::new(),
    };
    let prefix = crate::openapi::tool_prefix(&config.name);
    arr.iter().map(|t| {
        let raw_name = t["name"].as_str().unwrap_or("unknown").to_string();
        let sanitized_raw = crate::openapi::sanitize_tool_name(&raw_name);
        let combined = format!("{prefix}{sanitized_raw}");
        let name = if combined.len() > 64 { combined[..64].trim_end_matches('_').to_string() } else { combined };
        let description = t["description"].as_str().unwrap_or("").to_string();
        let input_schema = t["inputSchema"].clone();

        // MCP Apps: a tool may declare a UI resource via _meta.ui.resourceUri.
        let ui_resource_uri = t["_meta"]["ui"]["resourceUri"].as_str().map(String::from);

        let schema = serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": format!("[{}] {}", config.name, description),
                "parameters": input_schema
            }
        });

        MCPTool { server_id: config.id.clone(), server_name: config.name.clone(),
                  name, raw_name, description, input_schema, schema, ui_resource_uri }
    }).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_url ────────────────────────────────────────────────────────────────

    #[test]
    fn is_url_detects_https() {
        assert!(is_url("https://example.com/mcp"));
    }

    #[test]
    fn is_url_detects_http() {
        assert!(is_url("http://localhost:3000"));
    }

    #[test]
    fn is_url_rejects_shell_command() {
        assert!(!is_url("npx @modelcontextprotocol/server-filesystem"));
        assert!(!is_url("/usr/local/bin/mcp-server"));
        assert!(!is_url("python"));
    }

    // ── AuthConfig serialization ──────────────────────────────────────────────

    #[test]
    fn auth_none_round_trips() {
        let auth = AuthConfig::None;
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AuthConfig::None));
    }

    #[test]
    fn auth_bearer_round_trips() {
        let auth = AuthConfig::Bearer { bearer_token: "tok123".into() };
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AuthConfig::Bearer { bearer_token } if bearer_token == "tok123"));
    }

    #[test]
    fn auth_apikey_round_trips() {
        let auth = AuthConfig::ApiKey {
            api_key_header: "X-API-Key".into(),
            api_key_value: "secret".into(),
        };
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AuthConfig::ApiKey { api_key_header, api_key_value }
            if api_key_header == "X-API-Key" && api_key_value == "secret"));
    }

    #[test]
    fn auth_basic_round_trips() {
        let auth = AuthConfig::Basic {
            basic_username: "user".into(),
            basic_password: "pass".into(),
        };
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AuthConfig::Basic { basic_username, basic_password }
            if basic_username == "user" && basic_password == "pass"));
    }

    #[test]
    fn auth_oauth2_round_trips() {
        let auth = AuthConfig::OAuth2 {
            token_url: "https://oauth2.googleapis.com/token".into(),
            client_id: "client_id".into(),
            client_secret: "secret".into(),
            scope: "https://www.googleapis.com/auth/gmail.readonly".into(),
            authorization_url: "https://accounts.google.com/o/oauth2/auth".into(),
            access_token: "".into(),
            refresh_token: "".into(),
        };
        let json = serde_json::to_string(&auth).unwrap();
        let back: AuthConfig = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, AuthConfig::OAuth2 { token_url, .. } if token_url.contains("googleapis")));
    }

    #[test]
    fn auth_type_tag_is_lowercase() {
        // The serde tag must be the lowercase variant name — Tauri frontend sends lowercase
        let none_json = serde_json::to_string(&AuthConfig::None).unwrap();
        assert!(none_json.contains(r#""type":"none""#));

        let bearer_json = serde_json::to_string(&AuthConfig::Bearer { bearer_token: "t".into() }).unwrap();
        assert!(bearer_json.contains(r#""type":"bearer""#));

        let apikey_json = serde_json::to_string(&AuthConfig::ApiKey {
            api_key_header: "h".into(), api_key_value: "v".into()
        }).unwrap();
        assert!(apikey_json.contains(r#""type":"apikey""#));

        let basic_json = serde_json::to_string(&AuthConfig::Basic {
            basic_username: "u".into(), basic_password: "p".into()
        }).unwrap();
        assert!(basic_json.contains(r#""type":"basic""#));

        let oauth2_json = serde_json::to_string(&AuthConfig::OAuth2 {
            token_url: "u".into(), client_id: "c".into(), client_secret: "s".into(),
            scope: "".into(), authorization_url: "".into(),
            access_token: "".into(), refresh_token: "".into(),
        }).unwrap();
        assert!(oauth2_json.contains(r#""type":"oauth2""#));
    }

    #[test]
    fn auth_deserializes_from_frontend_none() {
        // Frontend sends { "type": "none" }
        let back: AuthConfig = serde_json::from_str(r#"{"type":"none"}"#).unwrap();
        assert!(matches!(back, AuthConfig::None));
    }

    #[test]
    fn auth_deserializes_from_frontend_bearer() {
        let back: AuthConfig = serde_json::from_str(r#"{"type":"bearer","bearer_token":"abc"}"#).unwrap();
        assert!(matches!(back, AuthConfig::Bearer { bearer_token } if bearer_token == "abc"));
    }

    #[test]
    fn auth_deserializes_from_frontend_apikey() {
        let back: AuthConfig = serde_json::from_str(
            r#"{"type":"apikey","api_key_header":"X-Key","api_key_value":"val"}"#
        ).unwrap();
        assert!(matches!(back, AuthConfig::ApiKey { api_key_header, api_key_value }
            if api_key_header == "X-Key" && api_key_value == "val"));
    }

    // ── MCP tool name prefixing ───────────────────────────────────────────────

    fn make_config(name: &str) -> MCPServerConfig {
        MCPServerConfig {
            id: "test".into(),
            name: name.into(),
            command: "echo".into(),
            args: vec![],
            env: HashMap::new(),
            enabled: true,
            auth: AuthConfig::None,
            enable_apps: false,
        }
    }

    #[test]
    fn parse_tools_prefixes_names() {
        let config = make_config("Filesystem");
        let result = serde_json::json!({
            "tools": [{
                "name": "list_files",
                "description": "List files",
                "inputSchema": { "type": "object", "properties": {} }
            }]
        });
        let tools = parse_tools(&config, &result);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "filesystem_list_files");
        assert_eq!(tools[0].raw_name, "list_files");
    }

    #[test]
    fn parse_tools_preserves_raw_name() {
        // tool_prefix("My Server") strips " server" → "my_"
        // sanitize("do.something") → "do_something"
        // combined: "my_do_something"
        let config = make_config("My Server");
        let result = serde_json::json!({
            "tools": [{
                "name": "do.something",
                "description": "Do something",
                "inputSchema": { "type": "object", "properties": {} }
            }]
        });
        let tools = parse_tools(&config, &result);
        assert_eq!(tools[0].raw_name, "do.something");
        assert_eq!(tools[0].name, "my_do_something");
    }

    #[test]
    fn parse_tools_caps_name_at_64_chars() {
        let config = make_config("LongServiceNameThatIsVeryLong API");
        let long_tool = "a".repeat(60);
        let result = serde_json::json!({
            "tools": [{
                "name": long_tool,
                "description": "",
                "inputSchema": { "type": "object", "properties": {} }
            }]
        });
        let tools = parse_tools(&config, &result);
        assert!(tools[0].name.len() <= 64);
    }

    #[test]
    fn parse_tools_empty_list() {
        let config = make_config("Test");
        let result = serde_json::json!({ "tools": [] });
        let tools = parse_tools(&config, &result);
        assert!(tools.is_empty());
    }

    #[test]
    fn parse_tools_missing_tools_key() {
        let config = make_config("Test");
        let result = serde_json::json!({});
        let tools = parse_tools(&config, &result);
        assert!(tools.is_empty());
    }

    // ── MCP Apps: UI resource detection ───────────────────────────────────────

    #[test]
    fn parse_tools_captures_ui_resource_uri() {
        let config = make_config("Maps");
        let result = serde_json::json!({
            "tools": [{
                "name": "show_map",
                "description": "Show a map",
                "inputSchema": { "type": "object", "properties": {} },
                "_meta": { "ui": { "resourceUri": "ui://maps/view" } }
            }]
        });
        let tools = parse_tools(&config, &result);
        assert_eq!(tools[0].ui_resource_uri.as_deref(), Some("ui://maps/view"));
    }

    #[test]
    fn parse_tools_no_ui_by_default() {
        let config = make_config("Plain");
        let result = serde_json::json!({
            "tools": [{ "name": "t", "description": "", "inputSchema": {} }]
        });
        let tools = parse_tools(&config, &result);
        assert!(tools[0].ui_resource_uri.is_none());
    }

    #[test]
    fn extract_ui_mcp_ui_embedded_resource() {
        // MCP-UI: an embedded resource block in content[] with ui:// + inline HTML.
        let resp = serde_json::json!({
            "content": [
                { "type": "text", "text": "here is your chart" },
                { "type": "resource", "resource": {
                    "uri": "ui://chart/1",
                    "mimeType": "text/html;profile=mcp-app",
                    "text": "<html><body>chart</body></html>"
                }}
            ]
        });
        let (html, uri) = extract_ui(&resp, &None);
        assert_eq!(uri.as_deref(), Some("ui://chart/1"));
        assert!(html.unwrap().contains("chart"));
    }

    #[test]
    fn extract_ui_ext_apps_meta_ref() {
        // ext-apps: result carries _meta.ui.resourceUri, no inline HTML.
        let resp = serde_json::json!({
            "content": [{ "type": "text", "text": "ok" }],
            "_meta": { "ui": { "resourceUri": "ui://app/main" } }
        });
        let (html, uri) = extract_ui(&resp, &None);
        assert!(html.is_none());
        assert_eq!(uri.as_deref(), Some("ui://app/main"));
    }

    #[test]
    fn extract_ui_falls_back_to_tool_declared_uri() {
        let resp = serde_json::json!({ "content": [{ "type": "text", "text": "ok" }] });
        let (_, uri) = extract_ui(&resp, &Some("ui://declared/on/tool".to_string()));
        assert_eq!(uri.as_deref(), Some("ui://declared/on/tool"));
    }

    #[test]
    fn extract_ui_none_for_plain_text() {
        let resp = serde_json::json!({ "content": [{ "type": "text", "text": "just text" }] });
        let (html, uri) = extract_ui(&resp, &None);
        assert!(html.is_none());
        assert!(uri.is_none());
    }

    // ── stdio transport integration (spawns a Node JSON-RPC stub) ─────────────

    fn node_available() -> bool {
        std::process::Command::new("node").arg("--version").output().is_ok()
    }

    #[tokio::test]
    async fn stdio_connect_call_and_fetch_ui_resource() {
        if !node_available() {
            eprintln!("node not available — skipping stdio MCP integration test");
            return;
        }
        let stub = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mcp-stub.js");
        let config = MCPServerConfig {
            id: "stub".into(), name: "Stub".into(),
            command: format!("node {stub}"),
            args: vec![], env: HashMap::new(), enabled: true,
            auth: AuthConfig::None, enable_apps: true,
        };

        let mut conn = MCPConnection::connect(config).await.expect("connect to stub");

        // tools/list parsed, and the UI tool carries its resource URI.
        assert!(conn.tools.iter().any(|t| t.raw_name == "echo"));
        let show = conn.tools.iter().find(|t| t.raw_name == "show_ui").expect("show_ui tool");
        assert_eq!(show.ui_resource_uri.as_deref(), Some("ui://stub/app"));

        // Plain text tool call (by prefixed name).
        let echoed = conn.call_tool("stub_echo", &serde_json::json!({ "text": "hi" })).await;
        assert_eq!(echoed, "hi");

        // MCP-App tool → result references a ui:// resource → read_resource fetches its HTML.
        let rich = conn.call_tool_rich("stub_show_ui", &serde_json::json!({})).await;
        assert_eq!(rich.ui_uri.as_deref(), Some("ui://stub/app"));
        assert!(rich.ui_html.as_deref().unwrap_or("").contains("stub app"), "ui_html: {:?}", rich.ui_html);
    }
}
