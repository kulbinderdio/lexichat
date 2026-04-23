use std::sync::Mutex;
use std::collections::HashMap;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use crate::openapi::RegisteredSpec;
use crate::mcp::MCPConnection;

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Base64-encoded images for vision models (Ollama `images` field)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireToolCall {
    pub function: WireToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireToolFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub r#type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ── Events emitted to the frontend ───────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct TokenEvent {
    pub delta: String,
}

#[derive(Clone, Serialize)]
pub struct ThinkingEvent {
    pub delta: String,
}

#[derive(Clone, Serialize)]
pub struct ToolCallEvent {
    pub name: String,
    pub args: String,
}

#[derive(Clone, Serialize)]
pub struct ToolResultEvent {
    pub name: String,
    pub result: String,
}

#[derive(Clone, Serialize)]
pub struct DoneEvent {
    pub error: Option<String>,
}

// Debug events
#[derive(Clone, Serialize)]
pub struct DebugStepEvent {
    pub step: usize,
    pub schema_names: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct DebugStepDoneEvent {
    pub step: usize,
    pub llm_text: String,
    pub duration_ms: u64,
}

#[derive(Clone, Serialize)]
pub struct DebugRunDoneEvent {
    pub total_ms: u64,
    pub error: Option<String>,
}

// ── Chat parameter options ────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

// ── Ollama REST API ───────────────────────────────────────────────────────────

pub async fn list_models(host: &str) -> anyhow::Result<Vec<String>> {
    #[derive(Deserialize)]
    struct TagsResponse {
        models: Vec<ModelEntry>,
    }
    #[derive(Deserialize)]
    struct ModelEntry {
        name: String,
    }

    let url = format!("{}/api/tags", host);
    let resp: TagsResponse = reqwest::get(&url).await?.json().await?;
    Ok(resp.models.into_iter().map(|m| m.name).collect())
}

// ── Streaming chat ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [WireMessage],
    stream: bool,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [ToolSchema],
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<&'a ChatOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<&'a str>,
}

#[derive(Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    #[allow(dead_code)]
    done: Option<bool>,
}

#[derive(Deserialize)]
struct ChunkMessage {
    content: Option<String>,
    thinking: Option<String>,
    tool_calls: Option<Vec<WireToolCall>>,
}

/// Stream one LLM turn. Returns (full_text, tool_calls).
/// When `silent` is true all `app.emit` calls are skipped (used by background jobs).
async fn stream_chat(
    host: &str,
    model: &str,
    messages: &[WireMessage],
    tools: &[ToolSchema],
    options: Option<&ChatOptions>,
    keep_alive: Option<&str>,
    app: &AppHandle,
    silent: bool,
) -> anyhow::Result<(String, Vec<WireToolCall>)> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;
    let url = format!("{}/api/chat", host);

    let body = ChatRequest { model, messages, stream: true, tools, options, keep_alive };
    let resp = client.post(&url).json(&body).send().await?;

    // Surface HTTP errors immediately — Ollama returns JSON {"error":"..."} on 4xx/5xx
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        // Try to extract Ollama's error message
        let msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["error"].as_str().map(String::from))
            .unwrap_or_else(|| format!("HTTP {status}: {body}"));
        return Err(anyhow::anyhow!(msg));
    }

    let mut full_text = String::new();
    let mut tool_calls: Vec<WireToolCall> = Vec::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        // Ollama sends one JSON object per line
        for line in std::str::from_utf8(&bytes)?.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            // Check for inline error (can appear mid-stream on some Ollama versions)
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(err) = v["error"].as_str() {
                    return Err(anyhow::anyhow!("{err}"));
                }
            }
            let parsed: ChatChunk = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(msg) = parsed.message {
                if let Some(delta) = msg.thinking {
                    if !delta.is_empty() {
                        if !silent { let _ = app.emit("agent-thinking", ThinkingEvent { delta }); }
                    }
                }
                if let Some(delta) = msg.content {
                    if !delta.is_empty() {
                        full_text.push_str(&delta);
                        if !silent { let _ = app.emit("agent-token", TokenEvent { delta }); }
                    }
                }
                if let Some(tcs) = msg.tool_calls {
                    tool_calls.extend(tcs);
                }
            }
        }
    }

    Ok((full_text, tool_calls))
}

// ── Agent loop ────────────────────────────────────────────────────────────────

const MAX_STEPS: usize = 10;

pub async fn agent_loop(
    host: &str,
    model: &str,
    system_prompt: &str,
    tools: &[ToolSchema],
    options: Option<ChatOptions>,
    keep_alive: Option<String>,
    conversation: &Mutex<Vec<WireMessage>>,
    openapi_specs: Vec<RegisteredSpec>,
    mcp_connections: &tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    allowed_dirs: Vec<String>,
    web_search_results: usize,
    app: &AppHandle,
    silent: bool,
) -> anyhow::Result<()> {
    let run_start = std::time::Instant::now();
    let mut nudged = false;
    for step in 0..MAX_STEPS {
        let step_start = std::time::Instant::now();
        let schema_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        if !silent { let _ = app.emit("debug-step-start", DebugStepEvent { step, schema_names }); }

        // Build wire messages: system + history
        let wire = {
            let conv = conversation.lock().unwrap();
            let mut w = vec![WireMessage {
                role: "system".into(),
                content: Some(system_prompt.into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                images: None,
            }];
            w.extend(conv.clone());
            w
        };

        let (full_text, tool_calls) = match stream_chat(host, model, &wire, tools, options.as_ref(), keep_alive.as_deref(), app, silent).await {
            Ok(v) => v,
            Err(e) => {
                if !silent {
                    let _ = app.emit("agent-done", DoneEvent { error: Some(e.to_string()) });
                    let _ = app.emit("debug-run-done", DebugRunDoneEvent {
                        total_ms: run_start.elapsed().as_millis() as u64,
                        error: Some(e.to_string()),
                    });
                }
                return Err(e);
            }
        };

        let step_ms = step_start.elapsed().as_millis() as u64;
        if !silent {
            let _ = app.emit("debug-step-done", DebugStepDoneEvent {
                step,
                llm_text: full_text.clone(),
                duration_ms: step_ms,
            });
        }

        // Append assistant message to history
        {
            let mut conv = conversation.lock().unwrap();
            conv.push(WireMessage {
                role: "assistant".into(),
                content: if full_text.is_empty() { None } else { Some(full_text.clone()) },
                tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls.clone()) },
                tool_call_id: None,
                name: None,
                images: None,
            });
        }

        if tool_calls.is_empty() {
            // Model returned nothing. Nudge once regardless of step so the user always
            // gets a response (step 0 = model gave no reply at all; step > 0 = silent
            // finish after tool results).
            if full_text.is_empty() && !nudged {
                nudged = true;
                let mut conv = conversation.lock().unwrap();
                conv.push(WireMessage {
                    role: "user".into(),
                    content: Some("Please respond to my previous request.".into()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    images: None,
                });
                continue;
            }
            if !silent {
                let _ = app.emit("agent-done", DoneEvent { error: None });
                let _ = app.emit("debug-run-done", DebugRunDoneEvent {
                    total_ms: run_start.elapsed().as_millis() as u64,
                    error: None,
                });
            }
            return Ok(());
        }

        // Dispatch each tool call
        for call in &tool_calls {
            let name = &call.function.name;
            let args = &call.function.arguments;
            let pretty_args = serde_json::to_string_pretty(args).unwrap_or_default();

            if !silent {
                let _ = app.emit("agent-tool-call", ToolCallEvent {
                    name: name.clone(),
                    args: pretty_args,
                });
            }

            // Route: builtin → openapi → mcp
            let result = dispatch_tool(name, args, &openapi_specs, mcp_connections, &allowed_dirs, web_search_results, app).await;

            // Cap large responses — but never truncate passthrough values like compose_email
            // whose full content must be forwarded intact to the next tool call.
            let passthrough = ["compose_email"];
            let result = if !passthrough.contains(&name.as_str()) && result.len() > 6000 {
                format!("{}\n…[truncated: {} chars total]", &result[..6000], result.len())
            } else {
                result
            };

            if !silent {
                let _ = app.emit("agent-tool-result", ToolResultEvent {
                    name: name.clone(),
                    result: result.clone(),
                });
            }

            let mut conv = conversation.lock().unwrap();
            conv.push(WireMessage {
                role: "tool".into(),
                content: Some(result),
                tool_calls: None,
                tool_call_id: None,
                name: Some(name.clone()),
                images: None,
            });
        }
    }

    // Hit max steps
    let msg = "Stopped: reached maximum steps without a final answer.".to_string();
    if !silent {
        let _ = app.emit("agent-done", DoneEvent { error: Some(msg.clone()) });
        let _ = app.emit("debug-run-done", DebugRunDoneEvent {
            total_ms: run_start.elapsed().as_millis() as u64,
            error: Some(msg),
        });
    }
    Ok(())
}

/// Returns true if `s` is NOT already valid base64url — i.e. the model forgot
/// to call compose_email and passed the raw MIME text directly.
fn needs_base64url_encoding(s: &str) -> bool {
    // Valid base64url contains only A-Za-z0-9+/=_- and no whitespace.
    // Plain email text always has spaces, newlines, or colon headers.
    s.contains('\n') || s.contains('\r') || s.contains(": ")
}

/// Route a tool call to the right executor.
async fn dispatch_tool(
    name: &str,
    args: &serde_json::Value,
    openapi_specs: &[RegisteredSpec],
    mcp_connections: &tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    allowed_dirs: &[String],
    web_search_results: usize,
    app: &AppHandle,
) -> String {
    // 1. Try built-in tools first
    let builtin_names = ["read_file","write_file","list_files","search_files",
        "search_in_files","get_file_info","list_directory_tree","create_directory",
        "move_file","delete_file","find_old_files","web_search","fetch_webpage","compose_email",
        "get_current_datetime"];
    if builtin_names.contains(&name) {
        return crate::tools::dispatch_builtin(name, args, allowed_dirs, web_search_results).await;
    }

    // 2. Try OpenAPI tools
    for spec in openapi_specs.iter() {
        if let Some(tool) = spec.tools.iter().find(|t| t.name == name) {
            // Auto-encode the 'raw' field for Gmail sendMessage if the model passes
            // plain text instead of going through compose_email first.
            let patched;
            let effective_args = if let Some(raw) = args["raw"].as_str() {
                if needs_base64url_encoding(raw) {
                    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
                    let mut obj = args.clone();
                    if let Some(map) = obj.as_object_mut() {
                        map.insert("raw".into(), serde_json::Value::String(
                            URL_SAFE_NO_PAD.encode(raw.as_bytes())
                        ));
                    }
                    patched = obj;
                    &patched
                } else { args }
            } else { args };
            return crate::openapi::execute(spec, tool, effective_args, Some(app)).await;
        }
    }

    // 3. Try MCP tools
    {
        let mut connections = mcp_connections.lock().await;
        for conn in connections.values_mut() {
            if conn.tools.iter().any(|t| t.name == name) {
                // Snapshot token before the call to detect if a refresh occurred
                let token_before = if let crate::mcp::AuthConfig::OAuth2 { ref access_token, .. } = conn.config.auth {
                    Some(access_token.clone())
                } else { None };

                let result = conn.call_tool(name, args).await;

                // If the token changed (refresh happened), persist it to the frontend
                if let (Some(before), crate::mcp::AuthConfig::OAuth2 { ref access_token, .. }) =
                    (token_before, &conn.config.auth)
                {
                    if *access_token != before {
                        use tauri::Emitter;
                        let _ = app.emit("openapi-token-refreshed", serde_json::json!({
                            "spec_id": conn.config.id,
                            "access_token": access_token,
                        }));
                        // Also update AppState so the next call within this run uses the new token
                        // (AppState.mcp_connections IS the live connection map — already updated)
                    }
                }

                return result;
            }
        }
    }

    format!("Unknown tool: {name}")
}
