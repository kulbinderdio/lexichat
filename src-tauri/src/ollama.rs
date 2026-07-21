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
    /// OpenAI requires a stable id linking an assistant tool call to its tool result. Ollama
    /// omits it; we synthesise ids when converting stored history to the OpenAI wire format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
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
    /// Base64 `data:` image URLs pulled from the tool result's image content blocks, rendered
    /// inline. Independent of the MCP-App flow — any tool that returns an image shows it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
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
    /// Total candidate tools (always-on + all groups) before per-step narrowing. When larger
    /// than `schema_names.len()`, selection filtered the list for this step.
    pub candidate_total: usize,
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

// ── Backend (inference provider) ───────────────────────────────────────────────

/// Which inference API dialect a backend speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Native Ollama REST — `/api/chat`, `/api/tags`, NDJSON streaming, no auth.
    Ollama,
    /// OpenAI-compatible Chat Completions — `/v1/chat/completions`, `/v1/models`, SSE,
    /// Bearer auth. Covers OpenAI, Anthropic (compat endpoint), Groq, Together, OpenRouter,
    /// Mistral, and local servers (LM Studio, llama.cpp, vLLM).
    OpenAI,
}

impl Default for ProviderKind {
    fn default() -> Self { ProviderKind::Ollama }
}

/// A resolved inference endpoint: dialect + base URL + optional API key.
#[derive(Debug, Clone)]
pub struct Backend {
    pub kind: ProviderKind,
    /// Base URL. Ollama: e.g. `http://localhost:11434`. OpenAI: includes the version
    /// segment, e.g. `https://api.openai.com/v1`.
    pub base_url: String,
    pub api_key: Option<String>,
}

impl Backend {
    /// Convenience for the default local backend and for tests.
    pub fn ollama(base_url: impl Into<String>) -> Self {
        Self { kind: ProviderKind::Ollama, base_url: base_url.into(), api_key: None }
    }

    fn base(&self) -> &str { self.base_url.trim_end_matches('/') }

    fn chat_url(&self) -> String {
        match self.kind {
            ProviderKind::Ollama => format!("{}/api/chat", self.base()),
            ProviderKind::OpenAI => format!("{}/chat/completions", self.base()),
        }
    }

    fn models_url(&self) -> String {
        match self.kind {
            ProviderKind::Ollama => format!("{}/api/tags", self.base()),
            ProviderKind::OpenAI => format!("{}/models", self.base()),
        }
    }

    fn is_openai(&self) -> bool { self.kind == ProviderKind::OpenAI }

    /// Attach auth to a request builder (Bearer for OpenAI when a key is set).
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match (self.kind, self.api_key.as_deref()) {
            (ProviderKind::OpenAI, Some(k)) if !k.is_empty() => rb.bearer_auth(k),
            _ => rb,
        }
    }
}

// ── Model listing ───────────────────────────────────────────────────────────────

pub async fn list_models(backend: &Backend) -> anyhow::Result<Vec<String>> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;
    let resp = backend.auth(client.get(backend.models_url()))
        .send().await?
        .error_for_status()?;
    let v: serde_json::Value = resp.json().await?;
    let mut names: Vec<String> = match backend.kind {
        ProviderKind::Ollama => v["models"].as_array().map(|a| a.iter()
            .filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default(),
        // OpenAI `/v1/models` → { data: [ { id, ... } ] }
        ProviderKind::OpenAI => v["data"].as_array().map(|a| a.iter()
            .filter_map(|m| m["id"].as_str().map(String::from)).collect()).unwrap_or_default(),
    };
    // OpenAI catalogs come unsorted and long — sort for a usable dropdown. Ollama's order
    // (roughly recency) is already sensible, so leave it.
    if backend.is_openai() { names.sort(); }
    Ok(names)
}

// ── Streaming chat ─────────────────────────────────────────────────────────────

/// Build the (non-streaming flag toggled by caller) chat request body for either dialect.
fn build_chat_request(
    backend: &Backend,
    model: &str,
    messages: &[WireMessage],
    tools: &[ToolSchema],
    options: Option<&ChatOptions>,
    keep_alive: Option<&str>,
    stream: bool,
) -> serde_json::Value {
    use serde_json::json;
    match backend.kind {
        ProviderKind::Ollama => {
            let mut b = json!({ "model": model, "messages": messages, "stream": stream });
            if !tools.is_empty() { b["tools"] = json!(tools); }
            if let Some(o) = options { b["options"] = json!(o); }
            if let Some(k) = keep_alive { b["keep_alive"] = json!(k); }
            b
        }
        ProviderKind::OpenAI => {
            let mut b = json!({
                "model": model,
                "messages": to_openai_messages(messages),
                "stream": stream,
            });
            if !tools.is_empty() { b["tools"] = json!(tools); }
            if let Some(o) = options { apply_openai_options(&mut b, o); }
            b
        }
    }
}

/// Map the Ollama-shaped `ChatOptions` onto OpenAI's top-level sampling fields. Ollama-only
/// knobs (top_k, repeat_penalty, num_ctx) have no standard OpenAI equivalent and are dropped.
fn apply_openai_options(b: &mut serde_json::Value, o: &ChatOptions) {
    use serde_json::json;
    if let Some(t) = o.temperature { b["temperature"] = json!(t); }
    if let Some(p) = o.top_p { b["top_p"] = json!(p); }
    if let Some(n) = o.num_predict { b["max_tokens"] = json!(n); }
    if let Some(s) = o.seed { b["seed"] = json!(s); }
    if let Some(stop) = &o.stop { b["stop"] = json!(stop); }
}

/// Convert stored history into OpenAI chat messages, synthesising tool-call ids (Ollama omits
/// them). Assistant tool calls and their following `tool` results are matched FIFO — the agent
/// loop always appends an assistant(tool_calls) message immediately followed by its results in
/// order, so a running queue of ids lines up correctly.
fn to_openai_messages(messages: &[WireMessage]) -> Vec<serde_json::Value> {
    use serde_json::json;
    let mut out = Vec::with_capacity(messages.len());
    let mut pending: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut seq = 0usize;
    for m in messages {
        let has_calls = m.tool_calls.as_ref().map_or(false, |t| !t.is_empty());
        if m.role == "assistant" && has_calls {
            let mut arr = Vec::new();
            for tc in m.tool_calls.as_ref().unwrap() {
                let id = match &tc.id {
                    Some(x) if !x.is_empty() => x.clone(),
                    _ => { seq += 1; format!("call_{seq}") }
                };
                pending.push_back(id.clone());
                arr.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        // OpenAI wants arguments as a JSON *string*.
                        "arguments": serde_json::to_string(&tc.function.arguments)
                            .unwrap_or_else(|_| "{}".into()),
                    }
                }));
            }
            let mut msg = json!({ "role": "assistant", "tool_calls": arr });
            if let Some(c) = &m.content { if !c.is_empty() { msg["content"] = json!(c); } }
            out.push(msg);
        } else if m.role == "tool" {
            let id = pending.pop_front().unwrap_or_else(|| { seq += 1; format!("call_{seq}") });
            out.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": m.content.clone().unwrap_or_default(),
            }));
        } else {
            out.push(openai_plain_message(m));
        }
    }
    out
}

/// A system/user/assistant message with no tool calls. User messages carrying images become an
/// OpenAI content-parts array (`image_url` data URIs); everything else is a plain string.
fn openai_plain_message(m: &WireMessage) -> serde_json::Value {
    use serde_json::json;
    if let Some(imgs) = &m.images {
        if !imgs.is_empty() {
            let mut parts = Vec::new();
            if let Some(c) = &m.content {
                if !c.is_empty() { parts.push(json!({ "type": "text", "text": c })); }
            }
            for b64 in imgs {
                let mime = guess_image_mime(b64);
                parts.push(json!({
                    "type": "image_url",
                    "image_url": { "url": format!("data:{mime};base64,{b64}") }
                }));
            }
            return json!({ "role": m.role, "content": parts });
        }
    }
    json!({ "role": m.role, "content": m.content.clone().unwrap_or_default() })
}

/// Guess an image MIME from the leading bytes of its base64 (magic-number prefixes). Good enough
/// for a data URI — the wrong guess only matters to strict servers, and PNG is a safe default.
fn guess_image_mime(b64: &str) -> &'static str {
    if b64.starts_with("/9j/") { "image/jpeg" }
    else if b64.starts_with("iVBORw0KGgo") { "image/png" }
    else if b64.starts_with("R0lGOD") { "image/gif" }
    else if b64.starts_with("UklGR") { "image/webp" }
    else { "image/png" }
}

/// Accumulator for an OpenAI streamed tool call — name and argument fragments arrive across many
/// SSE deltas keyed by `index` and must be concatenated before the JSON is parseable.
#[derive(Default)]
struct OaiPartialCall {
    id: String,
    name: String,
    args: String,
}

/// Parse one Ollama NDJSON line into the running response. `Err` propagates a stream error.
fn parse_ollama_line<R: tauri::Runtime>(
    line: &str,
    full_text: &mut String,
    tool_calls: &mut Vec<WireToolCall>,
    app: &AppHandle<R>,
    silent: bool,
) -> anyhow::Result<()> {
    let v: serde_json::Value = match serde_json::from_str(line) { Ok(v) => v, Err(_) => return Ok(()) };
    if let Some(err) = v["error"].as_str() { return Err(anyhow::anyhow!("{err}")); }
    let msg = &v["message"];
    if let Some(t) = msg["thinking"].as_str() {
        if !t.is_empty() && !silent { let _ = app.emit("agent-thinking", ThinkingEvent { delta: t.into() }); }
    }
    if let Some(c) = msg["content"].as_str() {
        if !c.is_empty() {
            full_text.push_str(c);
            if !silent { let _ = app.emit("agent-token", TokenEvent { delta: c.into() }); }
        }
    }
    if let Some(tcs) = msg["tool_calls"].as_array() {
        for tc in tcs {
            if let Ok(wtc) = serde_json::from_value::<WireToolCall>(tc.clone()) { tool_calls.push(wtc); }
        }
    }
    Ok(())
}

/// Parse one OpenAI SSE line into the running response + tool-call accumulator. Non-`data:`
/// lines (comments, `event:`) and `[DONE]` are skipped; `Err` propagates a stream error.
fn parse_openai_line<R: tauri::Runtime>(
    line: &str,
    full_text: &mut String,
    calls: &mut Vec<OaiPartialCall>,
    app: &AppHandle<R>,
    silent: bool,
) -> anyhow::Result<()> {
    let data = match line.strip_prefix("data:") { Some(d) => d.trim(), None => return Ok(()) };
    if data == "[DONE]" { return Ok(()); }
    let v: serde_json::Value = match serde_json::from_str(data) { Ok(v) => v, Err(_) => return Ok(()) };
    if let Some(err) = v["error"]["message"].as_str().or_else(|| v["error"].as_str()) {
        return Err(anyhow::anyhow!("{err}"));
    }
    let delta = &v["choices"][0]["delta"];
    // Some OpenAI-compatible servers (e.g. reasoning models via vLLM) stream reasoning here.
    if let Some(r) = delta["reasoning_content"].as_str().or_else(|| delta["reasoning"].as_str()) {
        if !r.is_empty() && !silent { let _ = app.emit("agent-thinking", ThinkingEvent { delta: r.into() }); }
    }
    if let Some(c) = delta["content"].as_str() {
        if !c.is_empty() {
            full_text.push_str(c);
            if !silent { let _ = app.emit("agent-token", TokenEvent { delta: c.into() }); }
        }
    }
    if let Some(tcs) = delta["tool_calls"].as_array() {
        for tc in tcs {
            let idx = tc["index"].as_u64().unwrap_or(0) as usize;
            while calls.len() <= idx { calls.push(OaiPartialCall::default()); }
            let slot = &mut calls[idx];
            if let Some(id) = tc["id"].as_str() { if !id.is_empty() { slot.id = id.to_string(); } }
            let f = &tc["function"];
            if let Some(n) = f["name"].as_str() { if !n.is_empty() { slot.name.push_str(n); } }
            if let Some(a) = f["arguments"].as_str() { slot.args.push_str(a); }
        }
    }
    Ok(())
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

/// True when a server rejects the request because the model/endpoint can't do tool calling at all
/// (e.g. OpenRouter: "No endpoints found that support tool use"). Such a model can still chat, so
/// the run should retry without tools rather than hard-fail.
fn is_tools_unsupported_error(msg: &str) -> bool {
    let m = msg.to_lowercase();
    let unsupported = m.contains("not support") || m.contains("unsupported") || m.contains("no endpoints found");
    unsupported && (m.contains("tool") || m.contains("function call"))
}

/// Stream one LLM turn. Returns (full_text, tool_calls).
/// When `silent` is true all `app.emit` calls are skipped (used by background jobs).
async fn stream_chat<R: tauri::Runtime>(
    backend: &Backend,
    model: &str,
    messages: &[WireMessage],
    tools: &[ToolSchema],
    options: Option<&ChatOptions>,
    keep_alive: Option<&str>,
    app: &AppHandle<R>,
    silent: bool,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> anyhow::Result<(String, Vec<WireToolCall>)> {
    let client = reqwest::Client::builder()
        // Only time out the initial TCP connection — not the streaming duration.
        // Long multi-tool responses can take many minutes; a global timeout kills them.
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    let body = build_chat_request(backend, model, messages, tools, options, keep_alive, true);
    let resp = backend.auth(client.post(backend.chat_url()).json(&body)).send().await?;

    // Surface HTTP errors immediately. Ollama returns {"error":"..."}; OpenAI {"error":{"message":...}}.
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let msg = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().or_else(|| v["error"].as_str()).map(String::from))
            .unwrap_or_else(|| format!("HTTP {status}: {body}"));
        return Err(anyhow::anyhow!(msg));
    }

    let openai = backend.is_openai();
    let mut full_text = String::new();
    let mut tool_calls: Vec<WireToolCall> = Vec::new();
    let mut oai_calls: Vec<OaiPartialCall> = Vec::new();
    // Line buffer: SSE/NDJSON events can be split across network chunks, so hold the trailing
    // partial line until its newline arrives rather than parsing chunk boundaries directly.
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        // Stop pressed mid-generation — abandon the rest of the stream and return what
        // we have. The agent loop's step-boundary check then ends the run.
        if let Some(c) = cancel {
            if c.load(std::sync::atomic::Ordering::SeqCst) { break; }
        }
        // A mid-stream network error (e.g. "error decoding response body") should not
        // discard a response that's already been partially received — break and return
        // what we have rather than propagating the error.
        let bytes = match chunk {
            Ok(b) => b,
            Err(e) => {
                if full_text.is_empty() && tool_calls.is_empty() && oai_calls.is_empty() {
                    return Err(anyhow::anyhow!("{e}"));
                }
                break;
            }
        };
        match std::str::from_utf8(&bytes) {
            Ok(s) => buf.push_str(s),
            Err(_) => continue, // skip chunks that split a multibyte sequence
        }
        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            let line = line.trim();
            if line.is_empty() { continue; }
            if openai {
                parse_openai_line(line, &mut full_text, &mut oai_calls, app, silent)?;
            } else {
                parse_ollama_line(line, &mut full_text, &mut tool_calls, app, silent)?;
            }
        }
    }

    // Assemble OpenAI streamed tool calls (fragments concatenated across deltas → parse once).
    if openai {
        for c in oai_calls {
            if c.name.is_empty() { continue; }
            let args: serde_json::Value = serde_json::from_str(&c.args)
                .unwrap_or_else(|_| serde_json::json!({}));
            tool_calls.push(WireToolCall {
                id: if c.id.is_empty() { None } else { Some(c.id) },
                function: WireToolFunction { name: c.name, arguments: args },
            });
        }
    }

    Ok((full_text, tool_calls))
}

/// How many times to re-sample a step whose tool call Ollama couldn't parse.
const MALFORMED_TOOL_CALL_RETRIES: usize = 2;

/// Default cap on a tool result's size before it's fed back to the model (protects the
/// context). Overridable per profile — raise it for data-heavy APIs that return large JSON.
pub const DEFAULT_TOOL_RESULT_LIMIT: usize = 6000;

const PASSTHROUGH_TOOLS: [&str; 1] = ["compose_email"];

/// Truncate a tool result to `limit` characters (0 → default). Passthrough tools whose full
/// output must reach the next call are left intact. Char-based so it never splits a multibyte
/// sequence (a byte slice at a fixed offset can panic).
fn cap_tool_result(result: String, tool_name: &str, limit: usize) -> String {
    let limit = if limit == 0 { DEFAULT_TOOL_RESULT_LIMIT } else { limit };
    let total = result.chars().count();
    if !PASSTHROUGH_TOOLS.contains(&tool_name) && total > limit {
        let head: String = result.chars().take(limit).collect();
        format!("{head}\n…[truncated: {total} chars total]")
    } else {
        result
    }
}

/// OpenAPI results are prefixed with a "HTTP <status>\n" line by `openapi::execute`. Return the
/// body after it (so an offloaded file is parseable JSON); leave other results untouched.
fn strip_http_status_line(s: &str) -> &str {
    if s.starts_with("HTTP ") {
        if let Some(nl) = s.find('\n') {
            let first = &s["HTTP ".len()..nl];
            if first.len() < 40 && first.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                return &s[nl + 1..];
            }
        }
    }
    s
}

/// Session directory where oversized tool results are dropped so `run_python` can read them.
fn tool_results_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("lexichat-tool-results");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Best-effort removal of offloaded result files older than a few hours, so the dir doesn't grow.
fn clean_tool_results(dir: &std::path::Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(6 * 3600);
        for e in entries.flatten() {
            if let Ok(meta) = e.metadata() {
                if meta.modified().map(|m| m < cutoff).unwrap_or(false) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
}

/// Like `cap_tool_result`, but when a result is too big to fit, the FULL result is written to a
/// file in `dir` and the model is told the path so it can process all of it with `run_python`
/// (accurate counting/aggregation without blowing the context). Interactive chat only —
/// background jobs can't run code, so they fall back to plain truncation.
fn offload_tool_result(result: String, tool_name: &str, limit: usize, dir: &std::path::Path) -> String {
    let limit = if limit == 0 { DEFAULT_TOOL_RESULT_LIMIT } else { limit };
    let total = result.chars().count();
    if PASSTHROUGH_TOOLS.contains(&tool_name) || total <= limit {
        return result;
    }
    let head: String = result.chars().take(limit).collect();
    // OpenAPI tool results are wrapped as "HTTP <status>\n<body>". Strip that status line so the
    // file is the raw body (valid JSON), which is what the model's json.loads expects.
    let body = strip_http_status_line(&result);
    let ext = match body.trim_start().chars().next() { Some('{') | Some('[') => "json", _ => "txt" };
    let safe: String = tool_name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    let path = dir.join(format!("{safe}-{}.{ext}", crate::uuid_v4()));
    match std::fs::write(&path, body) {
        Ok(_) => format!(
            "{head}\n…[truncated — showing the first {limit} of {total} characters. The FULL result is saved to this file:\n{}\nTo use ALL of it (count/aggregate/filter/sort), call the run_python tool. Sandbox rules: read_file() is ALREADY a built-in function — call it directly, do NOT define your own read_file and do NOT use open() (open is DISABLED); parse with json.loads() (json.load(fp) is NOT available); there is no `collections`, so use a plain dict. The JSON may be a bare list OR wrap the list in an object (e.g. {{\"records\":[...]}} or {{\"message\":{{\"items\":[...]}}}}), so find the list first. Example:\nimport json\nraw = json.loads(read_file(r\"{}\"))\ndef first_list(x):\n    if isinstance(x, list): return x\n    if isinstance(x, dict):\n        for v in x.values():\n            r = first_list(v)\n            if r is not None: return r\n    return None\ndata = first_list(raw) or []\ncounts = {{}}\nfor r in data: counts[r.get(\"category\")] = counts.get(r.get(\"category\"), 0) + 1\nprint(len(data), counts)]",
            path.display(), path.display()
        ),
        // File write failed → behave exactly like plain truncation.
        Err(_) => format!("{head}\n…[truncated: {total} chars total]"),
    }
}

// ── Tool selection (per-step discovery) ───────────────────────────────────────

/// At or below this many discoverable tools, send them all — the selection pre-flight
/// isn't worth its latency. Above it, narrow to the ones relevant to the current step.
pub const SELECTION_THRESHOLD: usize = 25;
/// Default upper bound on how many tools reach the model when a caller gives no cap.
pub const DEFAULT_TOOL_CAP: usize = 40;

/// A named collection of tools — one built-in set, or one OpenAPI spec / SPARQL endpoint /
/// MCP server. Selection is two-level: choose relevant groups first (a tiny prompt), then,
/// only if the chosen groups still overflow the cap, choose specific operations within them.
#[derive(Clone)]
pub struct ToolGroup {
    pub label: String,
    pub description: String,
    pub tools: Vec<ToolSchema>,
}

fn first_sentence(s: &str, max: usize) -> String {
    let s = s.trim();
    let cut = s.find(['.', '\n']).map(|i| i + 1).unwrap_or(s.len()).min(max).min(s.len());
    s[..cut].trim().to_string()
}

/// Parse a JSON array of integers, tolerating fences/prose (e.g. "[0, 2]").
fn parse_usize_array(s: &str) -> Vec<usize> {
    serde_json::from_str::<Vec<usize>>(&extract_json_array(s)).unwrap_or_default()
}

/// Pick the tools relevant to `context` from `groups`, capped at `cap`. Two-level with pure
/// fallbacks — a failed or empty selection never stalls the run, it just widens the set.
pub async fn select_tools_for_step(
    backend: &Backend,
    model: &str,
    context: &str,
    groups: &[ToolGroup],
    cap: usize,
) -> Vec<ToolSchema> {
    let all: Vec<&ToolSchema> = groups.iter().flat_map(|g| g.tools.iter()).collect();
    // Small enough: skip the LLM entirely; the cap is a backstop.
    if all.len() <= cap {
        return all.into_iter().cloned().collect();
    }

    // ── Level 1: choose relevant groups ──
    let group_list = groups.iter().enumerate()
        .map(|(i, g)| format!("{}. {} — {}", i, g.label, truncate_str(&g.description, 240)))
        .collect::<Vec<_>>().join("\n");
    let sys1 = "You choose which groups of tools could help with a task. This is a research \
        assistant that often combines several data sources, so favour recall: include EVERY \
        group that might plausibly be relevant, and exclude only groups that are clearly \
        unrelated. When in doubt, include it. Reply with ONLY a JSON array of the group \
        numbers, e.g. [0,2,3]. Output only the JSON array.";
    let user1 = format!("Task/context:\n{context}\n\nTool groups:\n{group_list}");
    let picked = complete(backend, model, sys1, &user1).await.ok()
        .map(|r| parse_usize_array(&r)).unwrap_or_default();
    let mut chosen: Vec<&ToolGroup> = {
        let c: Vec<&ToolGroup> = picked.iter().filter_map(|&i| groups.get(i)).collect();
        if c.is_empty() { groups.iter().collect() } else { c } // nothing picked → consider all
    };
    // Deterministic recall net: also include any group whose name appears in the context, so
    // a task that literally names an API (e.g. "the Bills API", "how MPs voted") always gets
    // that group even if the pre-flight missed it. Over-inclusion is fine — level 2 + the cap
    // bound the final count.
    let ctx_lower = context.to_lowercase();
    let sig = |w: &str| w.len() >= 4 && !matches!(w, "tool" | "tools" | "server" | "search" | "list" | "data");
    for g in groups {
        if chosen.iter().any(|c| std::ptr::eq(*c, g)) { continue; }
        // Match on the group label OR any of its tool names, so "show me a map" reaches a
        // server whose tool is `static_map_image_tool` even though the label is just "Mapbox".
        let named = g.label.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .any(|w| sig(w) && ctx_lower.contains(w))
            || g.tools.iter().any(|t| t.function.name.to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .any(|w| sig(w) && ctx_lower.contains(w)));
        if named { chosen.push(g); }
    }

    let candidate: Vec<&ToolSchema> = chosen.iter().flat_map(|g| g.tools.iter()).collect();
    if candidate.len() <= cap {
        return candidate.into_iter().cloned().collect();
    }

    // ── Level 2: choose specific operations within the chosen groups ──
    let tool_list = candidate.iter()
        .map(|t| format!("{}: {}", t.function.name, first_sentence(&t.function.description, 120)))
        .collect::<Vec<_>>().join("\n");
    let sys2 = "You select the minimum set of tools needed for a task. Reply with ONLY a JSON \
        array of tool name strings, e.g. [\"tool_a\",\"tool_b\"]. Output only the JSON array.";
    let user2 = format!("Task/context:\n{context}\n\nAvailable tools:\n{tool_list}");
    let names: Vec<String> = complete(backend, model, sys2, &user2).await.ok()
        .map(|r| serde_json::from_str(&extract_json_array(&r)).unwrap_or_default())
        .unwrap_or_default();
    let nameset: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
    let selected: Vec<ToolSchema> = candidate.iter()
        .filter(|t| nameset.contains(t.function.name.as_str()))
        .take(cap).map(|t| (*t).clone()).collect();
    // Model gave nothing usable → fall back to the chosen groups' first `cap` tools.
    if selected.is_empty() {
        candidate.into_iter().take(cap).cloned().collect()
    } else {
        selected
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max { s.to_string() }
    else { s.chars().take(max).collect::<String>() + "…" }
}

/// Compact description of the current step for tool selection: the task (first user message)
/// plus a short tail of recent activity, so selection adapts as a multi-tool chain progresses.
fn build_discovery_context(conv: &[WireMessage]) -> String {
    let task = conv.iter().find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref()).unwrap_or_default();
    let mut tail: Vec<String> = Vec::new();
    for m in conv.iter().rev().take(4) {
        match m.role.as_str() {
            "assistant" => if let Some(c) = m.content.as_deref() {
                if !c.trim().is_empty() { tail.push(format!("Assistant: {}", truncate_str(c, 300))); }
            },
            "tool" => if let Some(c) = m.content.as_deref() {
                tail.push(format!("Tool result: {}", truncate_str(c, 200)));
            },
            _ => {}
        }
    }
    tail.reverse();
    let mut out = format!("Task: {}", truncate_str(task, 500));
    if !tail.is_empty() { out.push_str("\n\nRecent activity:\n"); out.push_str(&tail.join("\n")); }
    out
}

/// One non-streaming chat POST, returning the raw body text (or a reqwest error). `think` is an
/// Ollama-only hint (skip reasoning); it's ignored for OpenAI backends.
async fn post_chat(
    client: &reqwest::Client, backend: &Backend, model: &str, messages: &[WireMessage], think: Option<bool>,
) -> reqwest::Result<String> {
    let mut body = build_chat_request(backend, model, messages, &[], None, None, false);
    if backend.kind == ProviderKind::Ollama {
        if let Some(t) = think { body["think"] = serde_json::json!(t); }
    }
    backend.auth(client.post(backend.chat_url()).json(&body)).send().await?
        .error_for_status()?
        .text().await
}

/// reqwest's Display drops the underlying cause ("error sending request for url (...)" alone),
/// so walk the source chain to reveal *why* — e.g. "connection closed before message completed".
fn describe_reqwest(e: &reqwest::Error) -> String {
    use std::error::Error;
    let mut msg = e.to_string();
    let mut src = e.source();
    while let Some(s) = src {
        let s_str = s.to_string();
        if !msg.contains(&s_str) { msg.push_str(": "); msg.push_str(&s_str); }
        src = s.source();
    }
    msg
}

/// One-shot, non-streaming completion with no tools. Used for meta tasks like drafting a
/// job spec, where we just want text (usually JSON) back. `think: false` keeps thinking
/// models from spending tokens reasoning, with a fallback for models that reject the flag.
///
/// Retries transport failures: swapping/loading a large model briefly drops connections, and
/// a one-shot call has no user in the loop to hit "retry", so a single dropped connection must
/// not fail the whole operation. A genuine server *response* error (bad request, model missing)
/// is returned immediately — retrying it is pointless.
pub async fn complete(backend: &Backend, model: &str, system: &str, user: &str) -> anyhow::Result<String> {
    let messages = vec![
        WireMessage { role: "system".into(), content: Some(system.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None },
        WireMessage { role: "user".into(), content: Some(user.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None },
    ];

    let client = reqwest::Client::builder()
        // Only bound the TCP connect — a cold model load can legitimately take a while, and a
        // hard total timeout would cut it off.
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    const ATTEMPTS: usize = 3;
    let mut last = String::new();
    for attempt in 0..ATTEMPTS {
        // think:false first; if the server *responds* with an error (some models reject the
        // flag), retry once without it. A transport error falls through to the backoff below.
        let result = match post_chat(&client, backend, model, &messages, Some(false)).await {
            Ok(t) => Ok(t),
            Err(e) if e.is_status() => post_chat(&client, backend, model, &messages, None).await,
            Err(e) => Err(e),
        };

        match result {
            Ok(text) => {
                let content = serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| match backend.kind {
                        ProviderKind::Ollama => v["message"]["content"].as_str().map(String::from),
                        ProviderKind::OpenAI => v["choices"][0]["message"]["content"].as_str().map(String::from),
                    })
                    .unwrap_or_default();
                return Ok(content);
            }
            Err(e) => {
                last = describe_reqwest(&e);
                // Transport failures and 5xx are transient (model swap/load); a 4xx (bad
                // request, model not found) won't change on retry, so stop.
                let retryable = match e.status() {
                    Some(code) => code.is_server_error(),
                    None => true,
                };
                if !retryable { break; }
                if attempt + 1 < ATTEMPTS {
                    tokio::time::sleep(std::time::Duration::from_millis(500 * (attempt as u64 + 1))).await;
                }
            }
        }
    }
    Err(anyhow::anyhow!(
        "{last} — the model server may be loading or busy; wait a moment and try again."
    ))
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

/// Backstop that guarantees the user always sees a reply. Called when the interactive
/// loop is about to end but nothing was ever streamed to the chat window. Forces one
/// final, tool-free completion and streams it; if the model still returns nothing, emits
/// a plain message so the chat is never left blank.
async fn ensure_final_answer<R: tauri::Runtime>(
    backend: &Backend,
    model: &str,
    system_prompt: &str,
    conversation: &Mutex<Vec<WireMessage>>,
    options: Option<&ChatOptions>,
    keep_alive: Option<&str>,
    app: &AppHandle<R>,
) {
    {
        let mut conv = conversation.lock().unwrap();
        conv.push(WireMessage {
            role: "user".into(),
            content: Some("Now write your complete answer to the user, based on all the work above. Respond in full. Do NOT call any tools.".into()),
            tool_calls: None, tool_call_id: None, name: None, images: None,
        });
    }
    let wire = {
        let conv = conversation.lock().unwrap();
        let mut w = vec![WireMessage {
            role: "system".into(),
            content: Some(system_prompt.into()),
            tool_calls: None, tool_call_id: None, name: None, images: None,
        }];
        w.extend(conv.clone());
        w
    };
    // No tools this turn — we want prose, not another tool call.
    let no_tools: Vec<ToolSchema> = Vec::new();
    let streamed = match stream_chat(backend, model, &wire, &no_tools, options, keep_alive, app, false, None).await {
        Ok((text, _)) => !text.trim().is_empty(),
        Err(_) => false,
    };
    if !streamed {
        let _ = app.emit("agent-token", TokenEvent {
            delta: "I completed the steps but couldn't produce a written summary this time — the model returned no final text. Please try again, or switch to a more capable model (e.g. qwen3).".into(),
        });
    }
}

pub async fn agent_loop<R: tauri::Runtime>(
    backend: &Backend,
    model: &str,
    system_prompt: &str,
    // Tools sent every step, never filtered (e.g. the wiki workflow).
    always_tools: &[ToolSchema],
    // Discoverable tool groups; when numerous, narrowed per step to those relevant now.
    tool_groups: &[ToolGroup],
    // Upper bound on tools shown to the model per step (0 → DEFAULT_TOOL_CAP).
    tool_cap: usize,
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
    // Max chars of a tool result fed back to the model (0 → DEFAULT_TOOL_RESULT_LIMIT).
    tool_result_limit: usize,
    app: &AppHandle<R>,
    silent: bool,
    max_steps: usize,
    // Set by the Stop button; checked between steps and while streaming to abort the run.
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    let run_start = std::time::Instant::now();
    let mut nudged = false;
    let mut continuations = 0usize;
    let mut consecutive_text_without_tools = 0usize; // detect "I'm done" loops
    let cap = if tool_cap == 0 { DEFAULT_TOOL_CAP } else { tool_cap };
    // Oversized tool results are offloaded here for run_python to read (interactive chat only —
    // jobs can't run code). run_python is given read access to this dir via dispatch_paths.
    let results_dir = tool_results_dir();
    let dispatch_paths: Vec<String> = if silent {
        sandbox_paths.clone()
    } else {
        clean_tool_results(&results_dir);
        let mut v = sandbox_paths.clone();
        v.push(results_dir.to_string_lossy().into_owned());
        v
    };
    // Tools the model called last step — kept available next step so a multi-step chain
    // isn't broken by them dropping out of the fresh selection.
    let mut last_used: Vec<String> = Vec::new();
    // Whether any assistant text was ever streamed to the chat. If a run ends with this
    // still false, we force a final answer so the user never sees a blank reply.
    let mut streamed_text = false;
    // Set once a server reports the model can't do tool calling; the rest of the run then sends no
    // tools (and skips per-step selection) so a non-tool model degrades to plain chat.
    let mut disable_tools = false;
    let discoverable_total: usize = tool_groups.iter().map(|g| g.tools.len()).sum();
    for step in 0..max_steps {
        // Stop requested — end the run cleanly before doing any more work.
        if cancel.load(Ordering::SeqCst) {
            if !silent {
                let _ = app.emit("agent-done", DoneEvent { error: None });
                let _ = app.emit("debug-run-done", DebugRunDoneEvent {
                    total_ms: run_start.elapsed().as_millis() as u64,
                    error: None,
                });
            }
            return Ok(());
        }
        let step_start = std::time::Instant::now();

        // Per-step tool selection: with many candidate tools, narrow to those relevant to the
        // current step; small sets are sent whole. `always_tools` are always included. Once the
        // backend has told us tools aren't supported, send none (and skip the selection call).
        let mut tools: Vec<ToolSchema> = if disable_tools { Vec::new() } else {
            let mut v = always_tools.to_vec();
            if discoverable_total <= SELECTION_THRESHOLD {
                v.extend(tool_groups.iter().flat_map(|g| g.tools.iter().cloned()));
            } else {
                let context = { let conv = conversation.lock().unwrap(); build_discovery_context(&conv) };
                let mut selected = select_tools_for_step(backend, model, &context, tool_groups, cap).await;
                for name in &last_used {
                    if !selected.iter().any(|t| &t.function.name == name)
                        && !v.iter().any(|t| &t.function.name == name) {
                        if let Some(t) = tool_groups.iter().flat_map(|g| &g.tools)
                            .find(|t| &t.function.name == name) {
                            selected.push(t.clone());
                        }
                    }
                }
                v.extend(selected);
            }
            v
        };

        let schema_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        if !silent {
            let _ = app.emit("debug-step-start", DebugStepEvent {
                step, schema_names, candidate_total: always_tools.len() + discoverable_total,
            });
        }

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
            match stream_chat(backend, model, &wire, &tools, options.as_ref(), keep_alive.as_deref(), app, silent, Some(cancel.as_ref())).await {
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
                // Model/endpoint can't do tool calling — drop tools and re-sample this step (and
                // every later step). `!tools.is_empty()` guards against re-entry once cleared.
                Err(e) if !tools.is_empty() && is_tools_unsupported_error(&e.to_string()) => {
                    disable_tools = true;
                    tools.clear();
                    if !silent {
                        let _ = app.emit("agent-retry", RetryEvent {
                            step,
                            attempt,
                            error: format!("This model doesn't support tool use — continuing without tools. ({e})"),
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

        if !full_text.trim().is_empty() { streamed_text = true; }

        // Remember which tools were called so they stay available next step (chain continuity).
        last_used = tool_calls.iter().map(|tc| tc.function.name.clone()).collect();

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
                // Never end an interactive run blank — salvage a written answer if the
                // model finished without ever streaming any text.
                if !streamed_text {
                    ensure_final_answer(backend, model, system_prompt, conversation,
                        options.as_ref(), keep_alive.as_deref(), app).await;
                }
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
            let result = dispatch_tool(name, args, &openapi_specs, &sparql_endpoints, mcp_connections, &allowed_dirs, &dispatch_paths, web_search_results, silent, app).await;

            // Cap large responses so they don't blow the context (configurable per profile).
            // In interactive chat, oversized results are also saved to a file run_python can
            // read, so the model can process the whole thing accurately.
            let result = if silent {
                cap_tool_result(result, &name, tool_result_limit)
            } else {
                offload_tool_result(result, &name, tool_result_limit, &results_dir)
            };

            // An MCP-App UI payload and/or inline images may have been stashed by dispatch_tool.
            let (ui, images) = if !silent {
                app.try_state::<crate::AppState>()
                    .map(|s| (
                        s.pending_tool_ui.lock().unwrap().take(),
                        std::mem::take(&mut *s.pending_tool_images.lock().unwrap()),
                    ))
                    .unwrap_or((None, Vec::new()))
            } else { (None, Vec::new()) };
            let had_media = ui.is_some() || !images.is_empty();
            if !silent {
                let _ = app.emit("agent-tool-result", ToolResultEvent {
                    name: name.clone(),
                    result: result.clone(),
                    ui,
                    images,
                });
            }

            // When an interactive UI or image was rendered, tell the model so it references
            // it naturally instead of apologising that it can't display images.
            let conv_text = if had_media {
                format!("{result}\n\n[Note: the image/UI for this result is ALREADY displayed to the user in the chat above. Refer to it naturally (e.g. \"shown above\"). Do NOT output an image URL, a markdown image, or a link to it, and do NOT claim you are unable to display images — it is already visible.]")
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
        // The model ran out of steps still working — force it to write up what it has
        // so the user gets the report rather than a blank window.
        if !streamed_text {
            ensure_final_answer(backend, model, system_prompt, conversation,
                options.as_ref(), keep_alive.as_deref(), app).await;
        }
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

                // Render any base64 image blocks inline — this is not gated by enable_apps or
                // approval, so a tool that returns an image (e.g. a Mapbox static map) always
                // shows it, even when the richer MCP-App panel isn't used.
                if !silent {
                    let imgs = crate::mcp::extract_image_data_urls(&rich.content);
                    if !imgs.is_empty() {
                        if let Some(state) = app.try_state::<crate::AppState>() {
                            *state.pending_tool_images.lock().unwrap() = imgs;
                        }
                    }
                }

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

    /// A transient 5xx (the shape of a model still loading) is retried, not surfaced.
    #[tokio::test]
    async fn complete_retries_transient_server_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // First call: 503 (transient). Later calls: a valid completion.
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(1).expect(1)
            .mount(&server).await;
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "{\"ok\":true}" }
            })))
            .expect(1..)
            .mount(&server).await;

        let out = complete(&Backend::ollama(server.uri()), "m", "sys", "user").await.expect("should recover");
        assert_eq!(out, "{\"ok\":true}");
    }

    /// A 4xx (bad request / model missing) is returned immediately — no wasted retries.
    #[tokio::test]
    async fn complete_does_not_retry_client_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // think:false then the think:None fallback = 2 requests, then it must give up.
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(404))
            .expect(2)
            .mount(&server).await;

        let err = complete(&Backend::ollama(server.uri()), "missing", "s", "u").await.unwrap_err();
        assert!(err.to_string().contains("404"), "got: {err}");
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
            &Backend::ollama(server.uri()),
            "qwen3.6:latest",
            "You are a helpful assistant.",
            &[],
            &[],
            0,
            None,
            None,
            &conversation,
            vec![],
            vec![],
            &mcp,
            vec![],
            vec![],
            10,
            0,
            app.handle(),
            false, // interactive chat, as in the failing session
            20,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
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

    fn group(label: &str, names: &[&str]) -> ToolGroup {
        ToolGroup {
            label: label.into(),
            description: format!("{label} tools"),
            tools: names.iter().map(|n| tool(n)).collect(),
        }
    }

    /// When the candidate set fits under the cap, selection returns everything and makes no
    /// LLM call (the host here is unreachable, so a call would error the test).
    #[tokio::test]
    async fn select_under_cap_skips_the_model() {
        let groups = vec![group("A", &["a1", "a2"]), group("B", &["b1"])];
        let picked = select_tools_for_step(&Backend::ollama("http://127.0.0.1:1"), "m", "task", &groups, 10).await;
        let names: Vec<&str> = picked.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["a1", "a2", "b1"]);
    }

    /// Level 1 narrows to the relevant group; if that group fits the cap, no level-2 call runs.
    #[tokio::test]
    async fn select_narrows_to_the_chosen_group() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Level-1 group pick: choose group 0 only.
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "[0]" }
            })))
            .expect(1) // exactly one call: group 0 fits the cap, so no level-2
            .mount(&server).await;

        let groups = vec![
            group("Members", &["m1", "m2"]),                       // group 0 (3 ≤ cap)
            group("Bills", &["b1", "b2", "b3", "b4", "b5", "b6"]), // group 1
        ];
        let picked = select_tools_for_step(&Backend::ollama(server.uri()), "m", "find an MP", &groups, 5).await;
        let names: Vec<&str> = picked.iter().map(|t| t.function.name.as_str()).collect();
        assert_eq!(names, ["m1", "m2"], "should return only the chosen group's tools");
    }

    /// End-to-end: with many candidate tools, the agent loop runs the selection pre-flight
    /// (non-streaming) before the streaming turn, and completes.
    #[tokio::test]
    async fn agent_loop_runs_per_step_selection_when_many_tools() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Selection pre-flight is non-streaming — pick group 0.
        Mock::given(method("POST")).and(path("/api/chat"))
            .and(body_partial_json(serde_json::json!({ "stream": false })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "[0]" }
            })))
            .expect(1..) // proves per-step selection ran
            .mount(&server).await;
        // The actual turn is streaming — a plain final answer, no tool call.
        Mock::given(method("POST")).and(path("/api/chat"))
            .and(body_partial_json(serde_json::json!({ "stream": true })))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"message\":{\"role\":\"assistant\",\"content\":\"Done.\"},\"done\":false}\n\
                 {\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
                "application/x-ndjson"))
            .mount(&server).await;

        // 2 + 7*4 = 30 tools: over SELECTION_THRESHOLD and over the cap, so selection engages.
        let groups = vec![
            group("A", &["a1", "a2"]),
            group("B", &["b1","b2","b3","b4","b5","b6","b7"]),
            group("C", &["c1","c2","c3","c4","c5","c6","c7"]),
            group("D", &["d1","d2","d3","d4","d5","d6","d7"]),
            group("E", &["e1","e2","e3","e4","e5","e6","e7"]),
        ];

        let app = tauri::test::mock_app();
        let conversation = Mutex::new(vec![WireMessage {
            role: "user".into(), content: Some("do the thing".into()),
            tool_calls: None, tool_call_id: None, name: None, images: None }]);
        let mcp: tokio::sync::Mutex<HashMap<String, MCPConnection>> =
            tokio::sync::Mutex::new(HashMap::new());

        let result = agent_loop(
            &Backend::ollama(server.uri()), "m", "sys", &[], &groups, 5, None, None,
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0, app.handle(), true, 5,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        ).await;
        assert!(result.is_ok(), "run should complete: {result:?}");
    }

    /// A group whose name appears in the task is included even if the level-1 pre-flight
    /// misses it — the fix for "I don't have a Bills API" when Bills is enabled.
    #[tokio::test]
    async fn select_includes_group_named_in_context() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // Level-1 picks only group 0 (Members); the keyword net must still add Bills.
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "[0]" }
            })))
            .expect(1) // Members(4)+Bills(2)=6 ≤ cap 8, so no level-2 call
            .mount(&server).await;

        let groups = vec![
            group("Members", &["m1", "m2", "m3", "m4"]),
            group("Bills", &["b1", "b2"]),
            group("Hansard", &["h1","h2","h3","h4","h5","h6","h7","h8","h9","h10"]),
        ]; // 16 total > cap 8
        let picked = select_tools_for_step(&Backend::ollama(server.uri()), "m",
            "Summarise the Employment Rights Bill using the Bills API", &groups, 8).await;
        let names: Vec<&str> = picked.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"b1") && names.contains(&"b2"), "Bills must be pulled in: {names:?}");
        assert!(names.contains(&"m1"), "Members (level-1 pick) kept: {names:?}");
        assert!(!names.contains(&"h1"), "Hansard neither picked nor named: {names:?}");
    }

    /// A group is pulled in when the task names one of its TOOLS, even if the group's label
    /// doesn't match — the fix for "show me a map" not reaching a server labelled "Mapbox"
    /// whose tool is `static_map_image_tool`. (Uses a >=4 keyword so the recall net fires.)
    #[tokio::test]
    async fn select_includes_group_by_tool_name() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "message": { "role": "assistant", "content": "[0]" }  // level-1 picks only group 0
            })))
            .mount(&server).await;

        let groups = vec![
            group("Postcodes", &["p1", "p2", "p3", "p4"]),
            group("Mapbox", &["search_and_geocode_tool", "static_map_image_tool"]),
            group("Other", &["o1","o2","o3","o4","o5","o6","o7","o8","o9","o10"]),
        ]; // 16 > cap 8
        let picked = select_tools_for_step(&Backend::ollama(server.uri()), "m",
            "please geocode this place for me", &groups, 8).await;
        let names: Vec<&str> = picked.iter().map(|t| t.function.name.as_str()).collect();
        assert!(names.contains(&"search_and_geocode_tool"),
            "Mapbox pulled in via its tool name matching 'geocode': {names:?}");
    }

    #[test]
    fn cap_tool_result_truncates_and_respects_limit_and_passthrough() {
        // Under the limit: untouched.
        assert_eq!(cap_tool_result("small".into(), "read_file", 6000), "small");
        // Over a custom limit: truncated with a marker, at a char boundary.
        let long = "x".repeat(50);
        let out = cap_tool_result(long.clone(), "read_file", 10);
        assert!(out.starts_with(&"x".repeat(10)));
        assert!(out.contains("[truncated: 50 chars total]"));
        // 0 → default limit (so 5000 chars pass untouched under the 6000 default).
        assert_eq!(cap_tool_result("y".repeat(5000), "read_file", 0).len(), 5000);
        // Passthrough tools are never truncated.
        assert_eq!(cap_tool_result("z".repeat(9000), "compose_email", 10).len(), 9000);
    }

    #[test]
    fn offload_writes_full_result_and_points_at_the_file() {
        let dir = std::env::temp_dir().join(format!("lexi-test-{}", crate::uuid_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // Under the limit: untouched, no file written.
        assert_eq!(offload_tool_result("small".into(), "police_streetcrime", 100, &dir), "small");
        // Over the limit: preview + file path + a run_python hint; the file holds the raw JSON
        // BODY (the "HTTP 200" wrapper openapi::execute adds is stripped so json.loads works).
        let body = format!("[{}]", "\"a\",".repeat(2000).trim_end_matches(','));
        let wrapped = format!("HTTP 200\n{body}");
        let out = offload_tool_result(wrapped, "police_streetcrime", 100, &dir);
        assert!(out.contains("run_python"), "must nudge toward run_python");
        assert!(out.contains("saved to this file"));
        let saved = std::fs::read_dir(&dir).unwrap().flatten()
            .map(|e| e.path()).find(|p| p.extension().map(|x| x == "json").unwrap_or(false)).unwrap();
        let content = std::fs::read_to_string(&saved).unwrap();
        assert!(!content.contains("HTTP 200"), "the HTTP wrapper must be stripped from the file");
        assert_eq!(content, body, "file must be the raw JSON body");
        // Passthrough tools are never truncated or offloaded.
        assert_eq!(offload_tool_result("z".repeat(9000), "compose_email", 100, &dir).len(), 9000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn strip_http_status_line_removes_only_the_wrapper() {
        assert_eq!(strip_http_status_line("HTTP 200\n[1,2,3]"), "[1,2,3]");
        assert_eq!(strip_http_status_line("HTTP 404\n{\"e\":1}"), "{\"e\":1}");
        assert_eq!(strip_http_status_line("[1,2,3]"), "[1,2,3]");            // no wrapper
        assert_eq!(strip_http_status_line("HTTPS notes\nx"), "HTTPS notes\nx"); // not a status line
    }

    #[test]
    fn discovery_context_includes_task_and_recent_activity() {
        let conv = vec![
            WireMessage { role: "user".into(), content: Some("Profile the MP for Rotherham".into()),
                tool_calls: None, tool_call_id: None, name: None, images: None },
            WireMessage { role: "assistant".into(), content: Some("Found member 123.".into()),
                tool_calls: None, tool_call_id: None, name: None, images: None },
            WireMessage { role: "tool".into(), content: Some("{\"id\":123}".into()),
                tool_calls: None, tool_call_id: None, name: None, images: None },
        ];
        let ctx = build_discovery_context(&conv);
        assert!(ctx.contains("Rotherham"));
        assert!(ctx.contains("Found member 123"));
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
            &Backend::ollama(server.uri()), "qwen3.6:latest", "You are a helpful assistant.", &[], &[], 0, None, None,
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0, app.handle(), false, 20,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
        .await;

        assert!(result.is_err(), "a persistent parse failure must surface, not hang");
        // The mock's `.expect(...)` verifies the retry count when `server` drops.
    }

    // ── OpenAI adapter ──────────────────────────────────────────────────────────

    #[test]
    fn to_openai_messages_matches_tool_ids_and_stringifies_args() {
        let convo = vec![
            WireMessage { role: "user".into(), content: Some("hi".into()),
                tool_calls: None, tool_call_id: None, name: None, images: None },
            WireMessage { role: "assistant".into(), content: None,
                tool_calls: Some(vec![WireToolCall {
                    id: None, // no id stored → must be synthesised and reused by the result
                    function: WireToolFunction { name: "search".into(),
                        arguments: serde_json::json!({ "q": "cats" }) },
                }]),
                tool_call_id: None, name: None, images: None },
            WireMessage { role: "tool".into(), content: Some("2 results".into()),
                tool_calls: None, tool_call_id: None, name: Some("search".into()), images: None },
        ];
        let out = to_openai_messages(&convo);
        assert_eq!(out[0]["role"], "user");
        // Assistant tool call: arguments serialised as a JSON *string*, id synthesised.
        let call = &out[1]["tool_calls"][0];
        assert_eq!(call["function"]["arguments"], "{\"q\":\"cats\"}");
        let id = call["id"].as_str().unwrap().to_string();
        // Tool result must reference the same id (FIFO match).
        assert_eq!(out[2]["role"], "tool");
        assert_eq!(out[2]["tool_call_id"].as_str().unwrap(), id);
    }

    #[test]
    fn to_openai_messages_encodes_images_as_content_parts() {
        let convo = vec![WireMessage {
            role: "user".into(), content: Some("what's this".into()),
            tool_calls: None, tool_call_id: None, name: None,
            images: Some(vec!["/9j/abc".into()]), // JPEG magic prefix
        }];
        let out = to_openai_messages(&convo);
        let parts = out[0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/jpeg;base64,/9j/abc"));
    }

    /// Drive the real agent loop against an OpenAI-compatible endpoint: step 0 streams a tool
    /// call (arguments fragmented across SSE deltas), step 1 streams the final answer. Verifies
    /// SSE parsing, tool-call accumulation, and that the tool result is fed back with a matching
    /// `tool_call_id` in OpenAI shape.
    #[tokio::test]
    async fn openai_agent_loop_streams_a_tool_call_and_completes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

        struct OaiFlow { n: Arc<AtomicUsize>, bodies: Arc<Mutex<Vec<String>>> }
        impl Respond for OaiFlow {
            fn respond(&self, req: &Request) -> ResponseTemplate {
                self.bodies.lock().unwrap().push(String::from_utf8_lossy(&req.body).into_owned());
                let step = self.n.fetch_add(1, Ordering::SeqCst);
                let sse = if step == 0 {
                    // Tool call with arguments split "{" + "}" across two deltas.
                    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\
                     data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_xyz\",\"type\":\"function\",\"function\":{\"name\":\"get_current_datetime\",\"arguments\":\"{\"}}]}}]}\n\
                     data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]}}]}\n\
                     data: [DONE]\n"
                } else {
                    "data: {\"choices\":[{\"delta\":{\"content\":\"All done.\"}}]}\n\
                     data: [DONE]\n"
                };
                ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream")
            }
        }

        let server = MockServer::start().await;
        let bodies = Arc::new(Mutex::new(Vec::new()));
        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(OaiFlow { n: Arc::new(AtomicUsize::new(0)), bodies: bodies.clone() })
            .mount(&server).await;

        let backend = Backend { kind: ProviderKind::OpenAI, base_url: server.uri(), api_key: None };
        let app = tauri::test::mock_app();
        let conversation = Mutex::new(vec![WireMessage {
            role: "user".into(), content: Some("what time is it".into()),
            tool_calls: None, tool_call_id: None, name: None, images: None,
        }]);
        let mcp: tokio::sync::Mutex<HashMap<String, MCPConnection>> =
            tokio::sync::Mutex::new(HashMap::new());

        let result = agent_loop(
            &backend, "gpt-4o-mini", "You are helpful.",
            &[tool("get_current_datetime")], // always-on tool, no selection LLM call
            &[], 0, None, None,
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0,
            app.handle(), false, 20,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        ).await;

        assert!(result.is_ok(), "OpenAI run should complete: {result:?}");
        let convo = conversation.lock().unwrap();
        let last = convo.last().unwrap();
        assert_eq!(last.role, "assistant");
        assert_eq!(last.content.as_deref(), Some("All done."));
        // The tool must have run (a datetime string was fed back as a tool message).
        assert!(convo.iter().any(|m| m.role == "tool"), "tool result appended");
        // Step-1 request must carry the prior tool result in OpenAI shape with the matching id.
        let second = &bodies.lock().unwrap()[1];
        assert!(second.contains("\"role\":\"tool\""), "history sent in OpenAI tool shape: {second}");
        assert!(second.contains("call_xyz"), "tool_call_id preserved: {second}");
    }

    #[test]
    fn detects_tools_unsupported_errors() {
        assert!(is_tools_unsupported_error("No endpoints found that support tool use."));
        assert!(is_tools_unsupported_error("This model does not support tools"));
        assert!(is_tools_unsupported_error("function calling is not supported by this model"));
        // Not a tools-capability problem — must not trigger the fallback.
        assert!(!is_tools_unsupported_error("rate limit exceeded"));
        assert!(!is_tools_unsupported_error("XML syntax error on line 14"));
    }

    /// A model whose endpoint rejects tool use (OpenRouter-style) should drop tools and still
    /// answer, rather than hard-failing the run.
    #[tokio::test]
    async fn agent_loop_falls_back_when_tools_unsupported() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

        struct ToolGate;
        impl Respond for ToolGate {
            fn respond(&self, req: &Request) -> ResponseTemplate {
                let body = String::from_utf8_lossy(&req.body);
                if body.contains("\"tools\"") {
                    // Reject the tool-bearing request the way OpenRouter does.
                    ResponseTemplate::new(400).set_body_json(serde_json::json!({
                        "error": { "message": "No endpoints found that support tool use." }
                    }))
                } else {
                    ResponseTemplate::new(200).set_body_raw(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"Answered without tools.\"}}]}\n\
                         data: [DONE]\n",
                        "text/event-stream")
                }
            }
        }

        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(ToolGate).mount(&server).await;

        let backend = Backend { kind: ProviderKind::OpenAI, base_url: server.uri(), api_key: None };
        let app = tauri::test::mock_app();
        let conversation = Mutex::new(vec![WireMessage {
            role: "user".into(), content: Some("list my files".into()),
            tool_calls: None, tool_call_id: None, name: None, images: None,
        }]);
        let mcp: tokio::sync::Mutex<HashMap<String, MCPConnection>> =
            tokio::sync::Mutex::new(HashMap::new());

        let result = agent_loop(
            &backend, "some/model", "You are helpful.",
            &[tool("list_files")], // always-on tool → first request carries tools
            &[], 0, None, None,
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0,
            app.handle(), false, 20,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        ).await;

        assert!(result.is_ok(), "run should survive an unsupported-tools endpoint: {result:?}");
        let convo = conversation.lock().unwrap();
        assert_eq!(convo.last().unwrap().content.as_deref(), Some("Answered without tools."));
    }
}
