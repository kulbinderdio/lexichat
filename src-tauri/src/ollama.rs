use std::sync::Mutex;
use std::collections::HashMap;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use crate::openapi::RegisteredSpec;
use crate::sparql::RegisteredSparqlEndpoint;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui: Option<ToolUiPayload>,
}

/// MCP Apps (SEP-1865) UI payload attached to a tool result so the frontend can
/// render it in a sandboxed iframe. Only produced for app-enabled MCP servers in
/// interactive (non-silent) chats.
#[derive(Clone, Serialize)]
pub struct ToolUiPayload {
    pub server_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured: Option<serde_json::Value>,
    /// Raw tool-result `content` array — forwarded to the app via ui/notifications/tool-result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    /// Raw tool-result `_meta`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
    /// The arguments the tool was called with — forwarded via ui/notifications/tool-input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

#[derive(Clone, Serialize)]
pub struct DoneEvent {
    pub error: Option<String>,
}

/// A step is being re-sampled after the model emitted an unparseable tool call.
/// The frontend drops any partial text streamed by the failed attempt.
#[derive(Clone, Serialize)]
pub struct RetryEvent {
    pub step: usize,
    pub attempt: usize,
    pub error: String,
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

/// True when an Ollama error is its tool-call parser rejecting the model's own output
/// (models emitting XML-dialect tool calls — e.g. `<function=x><parameter=y>` — sometimes
/// drop a closing tag). The next sample almost always parses, so the step is worth retrying
/// rather than killing the run.
fn is_malformed_tool_call_error(msg: &str) -> bool {
    let m = msg.to_lowercase();
    m.contains("syntax error")
        || m.contains("unexpected eof")
        || m.contains("invalid character")
        || m.contains("closed by")
}

/// Stream one LLM turn. Returns (full_text, tool_calls).
/// When `silent` is true all `app.emit` calls are skipped (used by background jobs).
async fn stream_chat<R: tauri::Runtime>(
    host: &str,
    model: &str,
    messages: &[WireMessage],
    tools: &[ToolSchema],
    options: Option<&ChatOptions>,
    keep_alive: Option<&str>,
    app: &AppHandle<R>,
    silent: bool,
) -> anyhow::Result<(String, Vec<WireToolCall>)> {
    let client = reqwest::Client::builder()
        // Only time out the initial TCP connection — not the streaming duration.
        // Long multi-tool responses can take many minutes; a global timeout kills them.
        .connect_timeout(std::time::Duration::from_secs(30))
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
        // A mid-stream network error (e.g. "error decoding response body") should not
        // discard a response that's already been partially received — break and return
        // what we have rather than propagating the error.
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                if full_text.is_empty() && tool_calls.is_empty() {
                    return Err(anyhow::anyhow!("{e}"));
                }
                break;
            }
        };
        // Ollama sends one JSON object per line; skip chunks with invalid UTF-8
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for line in text.lines() {
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

/// How many times to re-sample a step whose tool call Ollama couldn't parse.
const MALFORMED_TOOL_CALL_RETRIES: usize = 2;

// ── Tool discovery pre-flight ─────────────────────────────────────────────────

/// Minimum number of tools that triggers a pre-flight discovery call.
/// Below this threshold all tools are used directly.
const DISCOVERY_THRESHOLD: usize = 12;

/// Run a silent pre-flight call to let the model pick which tools it actually
/// needs for the given request. Returns the filtered [`ToolSchema`] slice.
/// Falls back to returning all tools on any error or empty selection.
pub async fn discover_tools(
    host: &str,
    model: &str,
    user_message: &str,
    all_tools: &[ToolSchema],
) -> Vec<ToolSchema> {
    if all_tools.len() <= DISCOVERY_THRESHOLD {
        return all_tools.to_vec();
    }

    // Build compact tool list: name + description only (no parameters schema)
    let compact: Vec<serde_json::Value> = all_tools
        .iter()
        .map(|t| serde_json::json!({ "name": t.function.name, "description": t.function.description }))
        .collect();
    let compact_json = serde_json::to_string(&compact).unwrap_or_default();

    let system = "You are a tool selection assistant. \
        Given a user request and a list of available tools (name + description only), \
        select the minimum set of tools needed to fulfil the request. \
        Reply with ONLY a valid JSON array of tool name strings, e.g. [\"tool_a\",\"tool_b\"]. \
        If no tools are needed, reply with []. \
        Do not call any tools. Do not explain. Output only the JSON array.";

    let user_msg = format!("User request: {user_message}\n\nAvailable tools: {compact_json}");

    let messages = vec![
        WireMessage { role: "system".into(), content: Some(system.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None },
        WireMessage { role: "user".into(), content: Some(user_msg),
            tool_calls: None, tool_call_id: None, name: None, images: None },
    ];

    #[derive(Serialize)]
    struct DiscoverReq<'a> {
        model: &'a str,
        messages: &'a [WireMessage],
        stream: bool,
        /// Picking tool names needs no reasoning. Left on, a thinking model spends thousands
        /// of tokens (tens of seconds) here — and this call blocks the whole run, before the
        /// first debug event, so the UI just sits there.
        #[serde(skip_serializing_if = "Option::is_none")]
        think: Option<bool>,
    }

    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(_) => return all_tools.to_vec(),
    };

    let url = format!("{}/api/chat", host);

    let mut resp_text: Option<String> = None;
    // Models without a thinking capability may reject `think`; fall back to a plain request.
    for think in [Some(false), None] {
        let body = DiscoverReq { model, messages: &messages, stream: false, think };
        if let Ok(r) = client.post(&url).json(&body).send().await.and_then(|r| r.error_for_status()) {
            if let Ok(t) = r.text().await {
                resp_text = Some(t);
                break;
            }
        }
    }
    let resp_text = match resp_text {
        Some(t) => t,
        None => return all_tools.to_vec(),
    };

    // Ollama non-streaming: {"message":{"content":"..."}, ...}
    let content = serde_json::from_str::<serde_json::Value>(&resp_text)
        .ok()
        .and_then(|v| v["message"]["content"].as_str().map(String::from))
        .unwrap_or_default();

    let json_str = extract_json_array(&content);
    let selected: Vec<String> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return all_tools.to_vec(),
    };

    if selected.is_empty() {
        return all_tools.to_vec();
    }

    let selected_set: std::collections::HashSet<&str> =
        selected.iter().map(String::as_str).collect();

    let filtered: Vec<ToolSchema> = all_tools
        .iter()
        .filter(|t| selected_set.contains(t.function.name.as_str()))
        .cloned()
        .collect();

    // Safety: if the model picked nothing recognisable, fall back to all tools
    if filtered.is_empty() { all_tools.to_vec() } else { filtered }
}

/// One-shot, non-streaming completion with no tools. Used for meta tasks like drafting a
/// job spec, where we just want text (usually JSON) back. `think: false` keeps thinking
/// models from spending tokens reasoning, with a fallback for models that reject the flag.
pub async fn complete(host: &str, model: &str, system: &str, user: &str) -> anyhow::Result<String> {
    #[derive(Serialize)]
    struct Req<'a> {
        model: &'a str,
        messages: &'a [WireMessage],
        stream: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        think: Option<bool>,
    }

    let messages = vec![
        WireMessage { role: "system".into(), content: Some(system.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None },
        WireMessage { role: "user".into(), content: Some(user.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None },
    ];

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;
    let url = format!("{}/api/chat", host);

    let mut last_err = None;
    for think in [Some(false), None] {
        let body = Req { model, messages: &messages, stream: false, think };
        match client.post(&url).json(&body).send().await.and_then(|r| r.error_for_status()) {
            Ok(r) => {
                let text = r.text().await?;
                let content = serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| v["message"]["content"].as_str().map(String::from))
                    .unwrap_or_default();
                return Ok(content);
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::anyhow!(last_err
        .map(|e| e.to_string())
        .unwrap_or_else(|| "completion failed".into())))
}

fn extract_json_array(s: &str) -> String {
    let s = s.trim();
    // Strip markdown code fences if present
    let body = if let Some(rest) = s.strip_prefix("```") {
        let inner = rest.trim_start_matches("json").trim_start_matches('\n');
        inner.split("```").next().unwrap_or(inner).trim()
    } else {
        s
    };
    // Find outermost [ ... ]
    let start = body.find('[').unwrap_or(0);
    let end = body.rfind(']').map(|i| i + 1).unwrap_or(body.len());
    body[start..end.min(body.len())].to_string()
}

// ── Agent loop ────────────────────────────────────────────────────────────────

pub async fn agent_loop<R: tauri::Runtime>(
    host: &str,
    model: &str,
    system_prompt: &str,
    tools: &[ToolSchema],
    options: Option<ChatOptions>,
    keep_alive: Option<String>,
    conversation: &Mutex<Vec<WireMessage>>,
    openapi_specs: Vec<RegisteredSpec>,
    sparql_endpoints: Vec<RegisteredSparqlEndpoint>,
    mcp_connections: &tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    allowed_dirs: Vec<String>,
    // Extra paths the run_python sandbox may access (e.g. user-attached files),
    // in addition to `allowed_dirs`. Empty for background jobs.
    sandbox_paths: Vec<String>,
    web_search_results: usize,
    app: &AppHandle<R>,
    silent: bool,
    max_steps: usize,
) -> anyhow::Result<()> {
    let run_start = std::time::Instant::now();
    let mut nudged = false;
    let mut continuations = 0usize;
    let mut consecutive_text_without_tools = 0usize; // detect "I'm done" loops
    for step in 0..max_steps {
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

        // Re-sample the step when the model emits a tool call Ollama can't parse; only a
        // persistent failure (or any other error) ends the run.
        let mut attempt = 0usize;
        let (full_text, tool_calls) = loop {
            match stream_chat(host, model, &wire, tools, options.as_ref(), keep_alive.as_deref(), app, silent).await {
                Ok(v) => break v,
                Err(e) if attempt < MALFORMED_TOOL_CALL_RETRIES
                    && is_malformed_tool_call_error(&e.to_string()) =>
                {
                    attempt += 1;
                    if !silent {
                        let _ = app.emit("agent-retry", RetryEvent {
                            step,
                            attempt,
                            error: e.to_string(),
                        });
                    }
                }
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
            // Model returned nothing. Nudge once so the user always gets a response.
            if full_text.is_empty() && !nudged {
                nudged = true;
                let mut conv = conversation.lock().unwrap();
                conv.push(WireMessage {
                    role: "user".into(),
                    content: Some("Please respond to my previous request.".into()),
                    tool_calls: None, tool_call_id: None, name: None, images: None,
                });
                continue;
            }

            // In silent (job) mode the model often outputs a step-completion note
            // and stops, expecting a human to say "continue". There's no human here —
            // push a continuation prompt so the remaining workflow steps execute.
            // BUT: if the model responds to two consecutive continuations with only
            // text (no tool calls), it has genuinely finished — stop the loop.
            if silent && !full_text.is_empty() {
                consecutive_text_without_tools += 1;
                if consecutive_text_without_tools >= 2 || continuations >= 20 {
                    // Model is done — two text-only responses in a row = workflow complete
                    if !silent {
                        let _ = app.emit("agent-done", DoneEvent { error: None });
                    }
                    return Ok(());
                }
                continuations += 1;
                let mut conv = conversation.lock().unwrap();
                conv.push(WireMessage {
                    role: "user".into(),
                    content: Some("Continue executing the workflow. Call the next tool now. Do not output text — call the tool directly.".into()),
                    tool_calls: None, tool_call_id: None, name: None, images: None,
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

        // Tool call received — reset both detectors so the next empty/text response
        // after this tool's result is handled correctly (can nudge again if needed)
        consecutive_text_without_tools = 0;
        nudged = false;

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

            // Route: builtin → openapi → sparql → mcp
            let result = dispatch_tool(name, args, &openapi_specs, &sparql_endpoints, mcp_connections, &allowed_dirs, &sandbox_paths, web_search_results, silent, app).await;

            // Cap large responses — but never truncate passthrough values like compose_email
            // whose full content must be forwarded intact to the next tool call.
            let passthrough = ["compose_email"];
            let result = if !passthrough.contains(&name.as_str()) && result.len() > 6000 {
                format!("{}\n…[truncated: {} chars total]", &result[..6000], result.len())
            } else {
                result
            };

            // An MCP-App UI payload may have been stashed by dispatch_tool; take it.
            let ui = if !silent {
                app.try_state::<crate::AppState>()
                    .and_then(|s| s.pending_tool_ui.lock().unwrap().take())
            } else { None };
            let had_ui = ui.is_some();
            if !silent {
                let _ = app.emit("agent-tool-result", ToolResultEvent {
                    name: name.clone(),
                    result: result.clone(),
                    ui,
                });
            }

            // When an interactive UI was rendered, tell the model so it references
            // the app naturally instead of apologising that it can't display images.
            let conv_text = if had_ui {
                format!("{result}\n\n[Note: an interactive UI for this result is now displayed to the user in the chat. Refer to it naturally (e.g. \"shown above\") and do NOT claim you are unable to display images or this content.]")
            } else {
                result
            };

            let mut conv = conversation.lock().unwrap();
            conv.push(WireMessage {
                role: "tool".into(),
                content: Some(conv_text),
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
async fn dispatch_tool<R: tauri::Runtime>(
    name: &str,
    args: &serde_json::Value,
    openapi_specs: &[RegisteredSpec],
    sparql_endpoints: &[RegisteredSparqlEndpoint],
    mcp_connections: &tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    allowed_dirs: &[String],
    sandbox_paths: &[String],
    web_search_results: usize,
    silent: bool,
    app: &AppHandle<R>,
) -> String {
    // 0. Code-execution sandbox — gated behind a per-session permission prompt.
    if name == "run_python" {
        return dispatch_run_python(args, allowed_dirs, sandbox_paths, silent, app).await;
    }

    // 1. Try built-in tools first
    let builtin_names = ["read_file","write_file","list_files","search_files",
        "search_in_files","get_file_info","list_directory_tree","create_directory",
        "move_file","delete_file","find_old_files","web_search","fetch_webpage","compose_email",
        "get_current_datetime",
        "wiki_list","wiki_search","wiki_read","wiki_write","wiki_patch","wiki_delete",
        "wiki_append","wiki_lint"];
    if builtin_names.contains(&name) {
        let result = crate::tools::dispatch_builtin(name, args, allowed_dirs, web_search_results).await;

        // In silent (job) mode, compose_email returns a large base64 string that
        // overwhelms the context window and causes the model to skip the send step.
        // Store the full result and return a short acknowledgment instead so the
        // model can proceed to the send step without needing to handle the raw value.
        if name == "compose_email" && !result.starts_with("Error") {
            use tauri::Manager;
            if let Some(state) = app.try_state::<crate::AppState>() {
                *state.pending_email_raw.lock().unwrap() = Some(result);
            }
            return "Email composed and ready to send. Call the email send tool now to deliver it — the encoded message will be supplied automatically.".into();
        }

        return result;
    }

    // 2. Try OpenAPI tools
    for spec in openapi_specs.iter() {
        if let Some(tool) = spec.tools.iter().find(|t| t.name == name) {
            use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};

            // For any tool whose name contains "sendmessage", supply the raw field
            // automatically from the stored compose_email result when:
            //   a) raw is absent, OR
            //   b) raw is a placeholder/empty string
            let pending_raw: Option<String> = {
                use tauri::Manager;
                if let Some(state) = app.try_state::<crate::AppState>() {
                    state.pending_email_raw.lock().unwrap().clone()
                } else {
                    None
                }
            };

            let patched_for_send;
            let patched_for_encode;
            let effective_args = if name.to_lowercase().contains("sendmessage") {
                let raw_val = args["raw"].as_str().unwrap_or("");
                let use_pending = raw_val.is_empty()
                    || raw_val.len() < 20          // placeholder / incomplete
                    || raw_val.contains('[')        // model pasted template text
                    || raw_val.contains("PASTE");
                if use_pending {
                    if let Some(ref stored) = pending_raw {
                        let mut obj = args.clone();
                        if let Some(map) = obj.as_object_mut() {
                            map.insert("raw".into(), serde_json::Value::String(stored.clone()));
                        }
                        patched_for_send = obj;
                        // Clear the stored value — it's been consumed
                        use tauri::Manager;
                        if let Some(state) = app.try_state::<crate::AppState>() {
                            *state.pending_email_raw.lock().unwrap() = None;
                        }
                        &patched_for_send
                    } else { args }
                } else if needs_base64url_encoding(raw_val) {
                    let mut obj = args.clone();
                    if let Some(map) = obj.as_object_mut() {
                        map.insert("raw".into(), serde_json::Value::String(
                            URL_SAFE_NO_PAD.encode(raw_val.as_bytes())
                        ));
                    }
                    patched_for_encode = obj;
                    &patched_for_encode
                } else { args }
            } else if let Some(raw) = args["raw"].as_str() {
                if needs_base64url_encoding(raw) {
                    let mut obj = args.clone();
                    if let Some(map) = obj.as_object_mut() {
                        map.insert("raw".into(), serde_json::Value::String(
                            URL_SAFE_NO_PAD.encode(raw.as_bytes())
                        ));
                    }
                    patched_for_encode = obj;
                    &patched_for_encode
                } else { args }
            } else { args };

            return crate::openapi::execute(spec, tool, effective_args, Some(app)).await;
        }
    }

    // 2b. Try SPARQL endpoint tools (query + schema)
    for ep in sparql_endpoints.iter() {
        if let Some(tool) = ep.tools.iter().find(|t| t.name == name) {
            // The query tool has a "query" param; the schema tool has none.
            if tool.parameters.iter().any(|p| p.name == "query") {
                let query = args["query"].as_str().unwrap_or("");
                if query.trim().is_empty() {
                    return "Error: missing 'query' argument — supply the full SPARQL query string.".into();
                }
                return crate::sparql::execute(ep, query, Some(app)).await;
            }
            return crate::sparql::schema_text(ep);
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

                let enable_apps = conn.config.enable_apps;
                let server_id = conn.config.id.clone();
                let rich = conn.call_tool_rich(name, args).await;

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

                // MCP Apps: stash any UI resource for the loop to emit. Interactive
                // chats only (never in background jobs), and only for opted-in servers.
                if !silent && enable_apps && (rich.ui_html.is_some() || rich.ui_uri.is_some()) {
                    if let Some(state) = app.try_state::<crate::AppState>() {
                        *state.pending_tool_ui.lock().unwrap() = Some(ToolUiPayload {
                            server_id,
                            html: rich.ui_html.clone(),
                            uri: rich.ui_uri.clone(),
                            structured: rich.structured.clone(),
                            content: (!rich.content.is_null()).then(|| rich.content.clone()),
                            meta: (!rich.meta.is_null()).then(|| rich.meta.clone()),
                            arguments: Some(args.clone()),
                        });
                    }
                }

                return rich.text;
            }
        }
    }

    format!("Unknown tool: {name}")
}

/// Handle a `run_python` call: enforce the per-session execution permission, then
/// run the code in the Monty sandbox with file access scoped to `allowed_dirs`
/// plus any attached files (`sandbox_paths`).
async fn dispatch_run_python<R: tauri::Runtime>(
    args: &serde_json::Value,
    allowed_dirs: &[String],
    sandbox_paths: &[String],
    silent: bool,
    app: &AppHandle<R>,
) -> String {
    let code = args["code"].as_str().unwrap_or("").to_string();
    if code.trim().is_empty() {
        return "Error: run_python requires a non-empty 'code' string.".into();
    }

    // Permission gate (session toggle): once approved, stays unlocked until the
    // app restarts. Background jobs can never prompt, so they require a prior
    // interactive unlock.
    if let Some(state) = app.try_state::<crate::AppState>() {
        let unlocked = *state.code_exec_unlocked.lock().unwrap();
        if !unlocked {
            if silent {
                return "Error: code execution requires interactive approval and is \
                        disabled in background jobs.".into();
            }
            let (tx, rx) = tokio::sync::oneshot::channel::<bool>();
            *state.pending_code_permission.lock().unwrap() = Some(tx);
            let _ = app.emit("agent-permission-request", serde_json::json!({ "code": code }));

            let approved = matches!(
                tokio::time::timeout(std::time::Duration::from_secs(300), rx).await,
                Ok(Ok(true))
            );
            *state.pending_code_permission.lock().unwrap() = None;

            if !approved {
                return "User denied code execution.".into();
            }
            *state.code_exec_unlocked.lock().unwrap() = true;
        }
    }

    // Sandbox may touch the run's allowed dirs plus any attached files.
    let mut allow = allowed_dirs.to_vec();
    allow.extend(sandbox_paths.iter().cloned());
    crate::sandbox::run_python(code, allow).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_tool_call_errors_are_retryable() {
        // The error Ollama returns when a model drops a closing tag in an XML-dialect tool call.
        assert!(is_malformed_tool_call_error(
            "XML syntax error on line 14: element <parameter> closed by </function>"
        ));
        assert!(is_malformed_tool_call_error("unexpected EOF"));
        assert!(is_malformed_tool_call_error("invalid character '<' looking for beginning of value"));
    }

    #[test]
    fn real_failures_are_not_retryable() {
        assert!(!is_malformed_tool_call_error("model \"qwen3.6\" not found, try pulling it first"));
        assert!(!is_malformed_tool_call_error("connection refused"));
        assert!(!is_malformed_tool_call_error("HTTP 500: internal server error"));
    }

    /// Drive the real agent loop against an Ollama that rejects the first sample's tool call
    /// (the exact error qwen3.6 produced) and answers on the second. The run must survive.
    #[tokio::test]
    async fn agent_loop_retries_a_malformed_tool_call_and_completes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

        /// Sample 1 fails the way qwen3.6 did; every later sample answers cleanly.
        struct FailFirstSample(Arc<AtomicUsize>);
        impl Respond for FailFirstSample {
            fn respond(&self, _: &Request) -> ResponseTemplate {
                let n = self.0.fetch_add(1, Ordering::SeqCst);
                let body = if n == 0 {
                    "{\"error\":\"XML syntax error on line 14: element <parameter> closed by </function>\"}\n"
                        .to_string()
                } else {
                    "{\"message\":{\"role\":\"assistant\",\"content\":\"Saved your DIY list.\"},\"done\":false}\n\
                     {\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n"
                        .to_string()
                };
                ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson")
            }
        }

        let server = MockServer::start().await;
        let samples = Arc::new(AtomicUsize::new(0));
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(FailFirstSample(samples.clone()))
            .mount(&server)
            .await;

        let app = tauri::test::mock_app();
        let conversation = Mutex::new(vec![WireMessage {
            role: "user".into(),
            content: Some("track my House DIY todo list".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            images: None,
        }]);
        let mcp: tokio::sync::Mutex<HashMap<String, MCPConnection>> =
            tokio::sync::Mutex::new(HashMap::new());

        let result = agent_loop(
            &server.uri(),
            "qwen3.6:latest",
            "You are a helpful assistant.",
            &[],
            None,
            None,
            &conversation,
            vec![],
            vec![],
            &mcp,
            vec![],
            vec![],
            10,
            app.handle(),
            false, // interactive chat, as in the failing session
            20,
        )
        .await;

        assert!(result.is_ok(), "run should survive a malformed tool call: {result:?}");
        // Exactly one re-sample: the bad one, then the good one.
        assert_eq!(samples.load(Ordering::SeqCst), 2, "step should be retried once");
        let convo = conversation.lock().unwrap();
        let last = convo.last().expect("assistant reply appended");
        assert_eq!(last.role, "assistant");
        assert_eq!(last.content.as_deref(), Some("Saved your DIY list."));
    }

    fn tool(name: &str) -> ToolSchema {
        ToolSchema {
            r#type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: format!("does {name}"),
                parameters: serde_json::json!({ "type": "object", "properties": {} }),
            },
        }
    }

    /// The pre-flight blocks the run before the first debug event fires, so it must not let a
    /// thinking model reason its way through tool selection (qwen3.6 spent 22s doing exactly that).
    #[tokio::test]
    async fn discovery_disables_thinking() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_partial_json(serde_json::json!({ "think": false })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "[\"read_file\"]" }
            })))
            .expect(1)
            .mount(&server)
            .await;

        // More than DISCOVERY_THRESHOLD tools, or the pre-flight is skipped entirely.
        let all: Vec<ToolSchema> = (0..DISCOVERY_THRESHOLD + 3)
            .map(|i| tool(&format!("tool_{i}")))
            .chain(std::iter::once(tool("read_file")))
            .collect();

        let picked = discover_tools(&server.uri(), "qwen3.6:latest", "read my notes", &all).await;

        let names: Vec<&str> = picked.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["read_file"], "should narrow to the model's selection");
        // The mock only matches requests carrying `think: false`; `.expect(1)` proves it was sent.
    }

    /// A model that never emits a parseable tool call must give up and surface the error,
    /// not retry forever.
    #[tokio::test]
    async fn agent_loop_gives_up_after_persistent_malformed_tool_calls() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"error\":\"XML syntax error on line 14: element <parameter> closed by </function>\"}\n",
                "application/x-ndjson",
            ))
            // One initial sample plus MALFORMED_TOOL_CALL_RETRIES re-samples, then stop.
            .expect(1 + MALFORMED_TOOL_CALL_RETRIES as u64)
            .mount(&server)
            .await;

        let app = tauri::test::mock_app();
        let conversation = Mutex::new(vec![WireMessage {
            role: "user".into(),
            content: Some("track my House DIY todo list".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            images: None,
        }]);
        let mcp: tokio::sync::Mutex<HashMap<String, MCPConnection>> =
            tokio::sync::Mutex::new(HashMap::new());

        let result = agent_loop(
            &server.uri(), "qwen3.6:latest", "You are a helpful assistant.", &[], None, None,
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, app.handle(), false, 20,
        )
        .await;

        assert!(result.is_err(), "a persistent parse failure must surface, not hang");
        // The mock's `.expect(...)` verifies the retry count when `server` drops.
    }
}
