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
}

// ── Transports ────────────────────────────────────────────────────────────────

enum Transport {
    Stdio {
        stdin:  AsyncMutex<ChildStdin>,
        stdout: AsyncMutex<BufReader<ChildStdout>>,
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

pub struct MCPConnection {
    pub config: MCPServerConfig,
    pub tools:  Vec<MCPTool>,
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

            let mut cmd = tokio::process::Command::new(exe);
            cmd.args(&inline_args)
               .args(&config.args)
               .envs(&config.env)
               .kill_on_drop(true)
               .stdin(Stdio::piped())
               .stdout(Stdio::piped())
               .stderr(Stdio::null());

            let mut child = cmd.spawn()
                .map_err(|e| format!("Failed to start '{}': {e}", config.command))?;

            let stdin  = child.stdin.take().ok_or("No stdin")?;
            let stdout = child.stdout.take().ok_or("No stdout")?;

            Transport::Stdio {
                stdin:  AsyncMutex::new(stdin),
                stdout: AsyncMutex::new(BufReader::new(stdout)),
                child,
            }
        };

        let mut conn = MCPConnection { config: config.clone(), tools: Vec::new(), transport };

        // Initialize handshake
        conn.send_request("initialize", serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "lexichat", "version": "0.1.0" }
        })).await.map_err(|e| format!("Initialize failed: {e}"))?;

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
            Transport::Stdio { stdin, stdout, .. } => {
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
                    if buf.is_empty() { return Err("MCP server closed connection".into()); }
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

    pub async fn call_tool(&mut self, tool_name: &str, args: &Value) -> String {
        // tool_name is prefixed; look up the raw name the MCP server expects
        let raw = self.tools.iter()
            .find(|t| t.name == tool_name)
            .map(|t| t.raw_name.clone())
            .unwrap_or_else(|| tool_name.to_string());
        let result = self.send_request("tools/call", serde_json::json!({
            "name": raw,
            "arguments": args
        })).await;

        match result {
            Ok(resp) => {
                if let Some(content) = resp["content"].as_array() {
                    content.iter()
                        .filter_map(|c| c["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    serde_json::to_string_pretty(&resp).unwrap_or_default()
                }
            }
            Err(e) => format!("MCP tool error: {e}"),
        }
    }
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

        let schema = serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": format!("[{}] {}", config.name, description),
                "parameters": input_schema
            }
        });

        MCPTool { server_id: config.id.clone(), server_name: config.name.clone(),
                  name, raw_name, description, input_schema, schema }
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
}
