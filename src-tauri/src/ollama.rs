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
    /// The FULL result (untruncated, capped only at a generous display limit), for the connector
    /// data viewer — so the user can verify the raw data as returned, independent of what the model
    /// saw (`result`, which is truncated to protect context). Empty when it equals `result`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub full_result: String,
    /// True when even `full_result` was clipped at the display cap (multi-MB responses).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub full_truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ui: Option<ToolUiPayload>,
    /// Base64 `data:` image URLs pulled from the tool result's image content blocks, rendered
    /// inline. Independent of the MCP-App flow — any tool that returns an image shows it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// Model-authored HTML artifact (from the `create_artifact` tool), rendered inline in a
    /// sandboxed iframe with a Save button. Distinct from MCP-App `ui` (no server bridge).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ArtifactPayload>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactPayload {
    pub title: String,
    pub html: String,
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

/// One line of the context breakdown: what it is and what it costs. `text` is filled only when the
/// user has turned on full-context capture, since verbatim history on every step of every run is
/// tens of MB in the renderer for a long research turn.
#[derive(Clone, Serialize)]
pub struct ContextItem {
    pub label: String,
    pub tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// What is actually being sent to the model this step, and what each part costs. "Reading 9K
/// tokens" prompts the obvious question — this answers it, and usually the answer is that most of
/// the context is tool schemas the turn never uses.
#[derive(Clone, Serialize)]
pub struct DebugContextEvent {
    pub run_id: u64,
    pub step: usize,
    pub total: usize,
    /// 0 when uncapped. Above it, the prefix cache dies and everything is re-read each step.
    pub num_ctx: usize,
    pub system_tokens: usize,
    pub tools_tokens: usize,
    pub history_tokens: usize,
    /// Biggest schemas first — the usual place a profile is quietly paying for tools it never calls.
    pub schemas: Vec<ContextItem>,
    /// In wire order. The one that grows a turn past num_ctx is nearly always a single big result.
    pub messages: Vec<ContextItem>,
}

// Debug events
#[derive(Clone, Serialize)]
pub struct DebugStepEvent {
    /// Unique per-agent-loop id so the DebugPanel groups steps into the correct run even when two
    /// runs overlap (e.g. a dev-control run during a chat) — without it, interleaved events merge
    /// into one run and steps appear out of order.
    pub run_id: u64,
    pub step: usize,
    pub schema_names: Vec<String>,
    /// Total candidate tools (always-on + all groups) before per-step narrowing. When larger
    /// than `schema_names.len()`, selection filtered the list for this step.
    pub candidate_total: usize,
    /// Rough token cost of the tool schemas sent THIS step. Tool definitions are re-sent every
    /// step, so this is a fixed per-step tax that is otherwise invisible — the reason to know it
    /// is that prompt size, not generation, dominates wall time on a local model.
    pub tools_tokens: usize,
}

#[derive(Clone, Serialize)]
pub struct DebugStepDoneEvent {
    pub run_id: u64,
    pub step: usize,
    pub llm_text: String,
    pub duration_ms: u64,
    /// Prompt (input) and completion (output) tokens for THIS step's model call (Ollama counts).
    pub tokens_in: u64,
    pub tokens_out: u64,
}

#[derive(Clone, Serialize)]
pub struct DebugRunDoneEvent {
    pub run_id: u64,
    pub total_ms: u64,
    pub error: Option<String>,
    /// Total prompt/completion tokens across all of this run's steps.
    pub tokens_in: u64,
    pub tokens_out: u64,
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
    /// Ollama-only reasoning toggle. `Some(false)` makes a thinking model (e.g. Qwen3) skip its
    /// pre-answer reasoning pass — much faster per turn. Set at the TOP level of the request, not
    /// inside `options`, so it's serialized separately in `build_chat_request` (hence `skip`).
    #[serde(skip)]
    pub think: Option<bool>,
}

// ── Backend (inference provider) ───────────────────────────────────────────────

/// Which inference API dialect a backend speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    /// Native Ollama REST — `/api/chat`, `/api/tags`, NDJSON streaming, no auth.
    Ollama,
    /// OpenAI-compatible Chat Completions — `/v1/chat/completions`, `/v1/models`, SSE,
    /// Bearer auth. Covers OpenAI, Groq, Together, OpenRouter, Mistral, Gemini's compat endpoint,
    /// and local servers (LM Studio, llama.cpp, vLLM).
    OpenAI,
    /// Anthropic's OpenAI-compatible endpoint — same request/response shape as OpenAI, but auth is
    /// `x-api-key` + `anthropic-version` headers (NOT Bearer), and `max_tokens` is required.
    Anthropic,
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
            ProviderKind::OpenAI | ProviderKind::Anthropic => format!("{}/chat/completions", self.base()),
        }
    }

    fn models_url(&self) -> String {
        match self.kind {
            ProviderKind::Ollama => format!("{}/api/tags", self.base()),
            ProviderKind::OpenAI | ProviderKind::Anthropic => format!("{}/models", self.base()),
        }
    }

    /// True when the backend speaks the OpenAI request/response dialect (OpenAI proper and
    /// Anthropic's compat endpoint both do — they differ only in auth).
    fn is_openai(&self) -> bool { matches!(self.kind, ProviderKind::OpenAI | ProviderKind::Anthropic) }

    /// Attach auth to a request builder. OpenAI-style uses Bearer; Anthropic uses `x-api-key` +
    /// `anthropic-version` (its `/v1/models` and compat chat endpoint reject Bearer).
    fn auth(&self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match (self.kind, self.api_key.as_deref()) {
            (ProviderKind::OpenAI, Some(k)) if !k.is_empty() => rb.bearer_auth(k),
            (ProviderKind::Anthropic, Some(k)) if !k.is_empty() =>
                rb.header("x-api-key", k).header("anthropic-version", "2023-06-01"),
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
        // OpenAI / Anthropic `/v1/models` → { data: [ { id, ... } ] }
        ProviderKind::OpenAI | ProviderKind::Anthropic => v["data"].as_array().map(|a| a.iter()
            .filter_map(|m| m["id"].as_str().map(String::from)).collect()).unwrap_or_default(),
    };
    // OpenAI catalogs come unsorted and long — sort for a usable dropdown. Ollama's order
    // (roughly recency) is already sensible, so leave it.
    if backend.is_openai() { names.sort(); }
    Ok(names)
}

// ── Model management (Ollama runtime) ─────────────────────────────────────────
// Ollama exposes install/remove; hosted providers (OpenAI/Anthropic) do not — their models are a
// catalog, so the manager only ever *lists* those. These helpers are Ollama-specific and take a
// base_url directly (the provider is always Ollama here).

/// One installed model, with its on-disk size (from `/api/tags`).
#[derive(Debug, Clone, Serialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
}

/// Installed models with sizes.
pub async fn list_ollama_detailed(base_url: &str) -> anyhow::Result<Vec<OllamaModel>> {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    let v: serde_json::Value = reqwest::Client::new().get(url).send().await?
        .error_for_status()?.json().await?;
    Ok(v["models"].as_array().map(|a| a.iter().map(|m| OllamaModel {
        name: m["name"].as_str().unwrap_or("").to_string(),
        size: m["size"].as_u64().unwrap_or(0),
    }).filter(|m| !m.name.is_empty()).collect()).unwrap_or_default())
}

/// Names of models currently loaded in memory (`/api/ps`), so the UI can mark them "running".
pub async fn ollama_loaded(base_url: &str) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/api/ps", base_url.trim_end_matches('/'));
    let v: serde_json::Value = reqwest::Client::new().get(url).send().await?
        .error_for_status()?.json().await?;
    Ok(v["models"].as_array().map(|a| a.iter()
        .filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default())
}

/// Delete an installed model (`DELETE /api/delete`).
pub async fn delete_ollama(base_url: &str, name: &str) -> anyhow::Result<()> {
    let url = format!("{}/api/delete", base_url.trim_end_matches('/'));
    reqwest::Client::new().delete(url)
        .json(&serde_json::json!({ "name": name }))
        .send().await?.error_for_status()?;
    Ok(())
}

/// Pull a model (`POST /api/pull`, streaming). Ollama returns NDJSON status lines with
/// `status`, and for the download phases `completed`/`total` byte counts; `progress` is called for
/// each. A final `{"status":"success"}` ends it. `cancel` is checked per line so a UI Cancel stops
/// the (potentially multi-GB) download promptly.
pub async fn pull_ollama<F: FnMut(&str, u64, u64)>(
    base_url: &str,
    name: &str,
    cancel: &std::sync::atomic::AtomicBool,
    mut progress: F,
) -> anyhow::Result<()> {
    let url = format!("{}/api/pull", base_url.trim_end_matches('/'));
    let resp = reqwest::Client::new().post(url)
        .json(&serde_json::json!({ "name": name, "stream": true }))
        .send().await?.error_for_status()?;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        if cancel.load(std::sync::atomic::Ordering::SeqCst) {
            anyhow::bail!("cancelled");
        }
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        // NDJSON: one JSON object per line; the last fragment may be incomplete, so keep it.
        while let Some(nl) = buf.find('\n') {
            let line = buf[..nl].trim().to_string();
            buf.drain(..=nl);
            if line.is_empty() { continue; }
            let v: serde_json::Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
            if let Some(err) = v["error"].as_str() { anyhow::bail!("{err}"); }
            let status = v["status"].as_str().unwrap_or("");
            progress(status, v["completed"].as_u64().unwrap_or(0), v["total"].as_u64().unwrap_or(0));
        }
    }
    Ok(())
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
            // Default keep_alive to 30m when the user hasn't set one: Ollama otherwise unloads the
            // model after 5m idle, so a large local model cold-reloads (tens of seconds) mid-session.
            b["keep_alive"] = json!(keep_alive.unwrap_or("30m"));
            // `think` is a top-level Ollama field (skipped in the ChatOptions serialization above).
            if let Some(t) = options.and_then(|o| o.think) { b["think"] = json!(t); }
            b
        }
        ProviderKind::OpenAI | ProviderKind::Anthropic => {
            let mut b = json!({
                "model": model,
                "messages": to_openai_messages(messages),
                "stream": stream,
            });
            // Ask for token usage in the stream — OpenAI-compatible servers only include a final
            // `usage` chunk when this is set. Harmless for servers that ignore it.
            if stream { b["stream_options"] = json!({ "include_usage": true }); }
            if !tools.is_empty() { b["tools"] = json!(tools); }
            if let Some(o) = options { apply_openai_options(&mut b, o); }
            // Anthropic's compat endpoint requires max_tokens; default it when the caller didn't set
            // one (Response Length = Auto). OpenAI proper leaves it optional.
            if backend.kind == ProviderKind::Anthropic && b.get("max_tokens").is_none() {
                b["max_tokens"] = json!(4096);
            }
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
    // The final chunk (done:true) carries token counts — accumulate them into the per-turn total
    // (summed across the turn's model calls; the agent loop reads + resets them for the usage record).
    if v["done"].as_bool() == Some(true) {
        let p = v["prompt_eval_count"].as_u64().unwrap_or(0);
        let c = v["eval_count"].as_u64().unwrap_or(0);
        if p > 0 || c > 0 {
            if let Some(s) = app.try_state::<crate::AppState>() {
                let mut t = s.turn_tokens.lock().unwrap();
                t.0 += p; t.1 += c;
            }
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
    // Final usage chunk (present when stream_options.include_usage was set — OpenAI, Anthropic
    // compat, Groq, Together, OpenRouter, …). Accumulate into the per-turn token total, exactly
    // like the Ollama path, so the DebugPanel/usage counts work for cloud models too.
    if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
        let p = u["prompt_tokens"].as_u64().unwrap_or(0);
        let c = u["completion_tokens"].as_u64().unwrap_or(0);
        if p > 0 || c > 0 {
            if let Some(s) = app.try_state::<crate::AppState>() {
                let mut t = s.turn_tokens.lock().unwrap();
                t.0 += p; t.1 += c;
            }
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

/// Shortest and longest repeating unit worth looking for, and how many consecutive
/// repeats count as degeneration rather than legitimate structure. Five identical blocks
/// in a row is well past anything a list or a table produces.
const REPEAT_MIN_PERIOD: usize = 6;
const REPEAT_MAX_PERIOD: usize = 160;
const REPEAT_MIN_TIMES:  usize = 5;

/// Detect a model that has started repeating itself, returning the repeating unit.
///
/// Small local models sometimes lock into a loop mid-response — one observed run emitted
/// "AI SENSI DELL'ART. 1, CO. 668" several hundred times and streamed for over twenty
/// minutes. Every runaway guard in `agent_loop` is checked at the top of a step, so a
/// response that never finishes is never checked at all; this is the only thing watching
/// inside a single generation.
///
/// Works on the tail only, so cost does not grow with the length of the response.
pub(crate) fn runaway_repetition(text: &str) -> Option<String> {
    // Reversed tail: index 0 is the last character, so a repeating suffix becomes a
    // periodic prefix and can be checked with a modulo.
    let tail: Vec<char> = text.chars().rev().take(REPEAT_MAX_PERIOD * REPEAT_MIN_TIMES).collect();
    for period in REPEAT_MIN_PERIOD..=REPEAT_MAX_PERIOD {
        let span = period * REPEAT_MIN_TIMES;
        if span > tail.len() {
            break;
        }
        if (period..span).all(|i| tail[i] == tail[i % period]) {
            let unit: String = tail[..period].iter().rev().collect();
            // Runs of whitespace, dashes or dots are formatting, not degeneration.
            if unit.chars().any(char::is_alphanumeric) {
                return Some(unit);
            }
        }
    }
    None
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
    // Hard stop for this turn. Checked per chunk, because the step-level budget in
    // `agent_loop` cannot fire while a single response is still streaming.
    deadline: Option<std::time::Instant>,
) -> anyhow::Result<(String, Vec<WireToolCall>)> {
    let client = reqwest::Client::builder()
        // Only time out the initial TCP connection — not the streaming duration.
        // Long multi-tool responses can take many minutes; a global timeout kills them.
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    let mut body = build_chat_request(backend, model, messages, tools, options, keep_alive, true);
    let mut resp = backend.auth(client.post(backend.chat_url()).json(&body)).send().await?;

    // Some Ollama versions / non-thinking models reject the top-level `think` flag with an HTTP
    // error. "Reasoning off" must never kill a run, so on any error status strip `think` and retry
    // once — the turn then runs at the model's default reasoning behaviour instead of failing.
    if !resp.status().is_success() && body.get("think").is_some() {
        if let Some(obj) = body.as_object_mut() { obj.remove("think"); }
        resp = backend.auth(client.post(backend.chat_url()).json(&body)).send().await?;
    }

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
    // Degeneration is checked every N characters of growth rather than every token: a loop
    // needs hundreds of characters to establish, so per-token checking buys nothing.
    let repeat_check_every: usize = 256;
    let mut next_repeat_check: usize = 512;
    let mut tool_calls: Vec<WireToolCall> = Vec::new();
    let mut oai_calls: Vec<OaiPartialCall> = Vec::new();
    // Line buffer: SSE/NDJSON events can be split across network chunks, so hold the trailing
    // partial line until its newline arrives rather than parsing chunk boundaries directly.
    let mut buf = String::new();
    let mut stream = resp.bytes_stream();

    loop {
        // Check Stop before each read so it lands even during prompt-eval, when the model is busy
        // and no chunks are arriving — a plain `stream.next().await` would block until the first
        // token. On cancel we break and drop `resp`/`stream`, closing the connection, which tells
        // the backend to abort generation rather than finish a reply we're about to discard.
        if let Some(c) = cancel {
            if c.load(std::sync::atomic::Ordering::SeqCst) { break; }
        }
        // The turn's wall-clock budget, enforced mid-stream. Without this a model that never
        // stops generating is never checked, because the step never ends.
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            if !silent {
                let _ = app.emit("agent-token", TokenEvent {
                    delta: "\n\n[Stopped: this turn ran out of time while the model was still generating.]".into(),
                });
            }
            break;
        }
        // Degeneration check. Only run once per `repeat_check_every` characters of growth —
        // per token it would be wasteful, and a loop takes far longer than that to establish.
        if full_text.len() >= next_repeat_check {
            next_repeat_check = full_text.len() + repeat_check_every;
            if let Some(unit) = runaway_repetition(&full_text) {
                let shown: String = unit.chars().take(40).collect();
                if !silent {
                    let _ = app.emit("agent-token", TokenEvent {
                        delta: format!("\n\n[Stopped: the model began repeating \"{shown}\" and was cut off.]"),
                    });
                }
                break;
            }
        }
        let chunk = match tokio::time::timeout(
            std::time::Duration::from_millis(200), stream.next()).await
        {
            Ok(Some(chunk)) => chunk,
            Ok(None)        => break,    // stream ended normally
            Err(_)          => continue, // read timed out → loop back and re-check the Stop flag
        };
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

/// Where `/work/artifacts` payloads are kept so they survive BETWEEN turns. The Pyodide workspace
/// is wiped on the first run_python of each turn, so a dataset built in one turn used to vanish
/// before it could be aggregated or rendered in the next (observed: a 45-call journey matrix lost
/// to a FileNotFoundError). Like offloaded tool results, these live on real disk and are re-staged
/// into `/work/artifacts` at the start of every turn.
pub fn artifact_data_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("lexichat-artifact-data");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Per-run staging area for skill resource files. `use_skill` copies a loaded skill's resources
/// here; `stage_python_files` then routes them into /work/skills/ (not /work/data/) so run_python
/// can read a template/helper the skill ships with.
fn skill_staging_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("lexichat-skill-resources");
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
/// (accurate counting/aggregation without blowing the context). Used only when `run_python` is
/// actually among this run's tools — otherwise the caller falls back to plain truncation, since
/// pointing at a file the model cannot open is worse than a short result.
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
    let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("result.json").to_string();
    match std::fs::write(&path, body) {
        // The offloaded file is staged into the run_python sandbox at /work/data (see
        // stage_python_files), so the model reads it there with normal Python.
        Ok(_) => format!(
            "{head}\n…[truncated — showing the first {limit} of {total} characters. The FULL result is available ONLY to the run_python tool at /work/data/{fname} — do NOT use list_files or read_file on /work/data (file tools cannot see it). To use ALL of it (count/aggregate/filter/sort/list), call run_python and read it with normal Python:\nimport json\nwith open('/work/data/{fname}') as f:\n    raw = json.load(f)\n# `raw` may be a bare list, or a dict wrapping the list (e.g. {{\"records\": [...]}}); find the list, process it, then print() your result. Pandas also works: pd.read_json('/work/data/{fname}').]"
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
/// Label of the built-in tool group (files/web/code) — always kept directly available in
/// discovery mode; only the *external* groups are gated behind `find_tools`.
pub const BUILTIN_GROUP: &str = "Built-in tools";

/// The `find_tools` discovery meta-tool schema. In discovery mode the model calls this to load
/// specialized (external) tools on demand instead of them all being pushed into every request.
pub fn find_tools_schema() -> ToolSchema {
    serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {
            "name": "find_tools",
            "description": "Discover and load specialized tools (external APIs, linked-data/SPARQL \
                endpoints, MCP servers) for this task. Call it with a short description of what you \
                need — e.g. 'UK street crime statistics', 'draw a diagram', 'company filings'. It \
                returns the matching tools and makes them callable on your NEXT step. You must call \
                find_tools before you can call any non-built-in tool.",
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string", "description": "What capability or data you need." } },
                "required": ["query"]
            }
        }
    })).expect("static find_tools schema is valid")
}

/// The `use_skill` meta-tool schema. When a profile has skills, only their one-line descriptions
/// sit in the prompt; the model calls this to load a skill's full instructions on demand (see
/// docs/skills-framework.md). Registered per step only when ≥1 skill is available.
pub fn use_skill_schema() -> ToolSchema {
    serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Load the full step-by-step instructions for one of the AVAILABLE SKILLS \
                listed in your system prompt. Call it with the skill's name (e.g. use_skill(\"presentation\")) \
                BEFORE attempting that kind of task; it returns the recipe to follow. Only use skill \
                names shown in the AVAILABLE SKILLS list.",
            "parameters": {
                "type": "object",
                "properties": { "name": { "type": "string", "description": "The skill name to load." } },
                "required": ["name"]
            }
        }
    })).expect("static use_skill schema is valid")
}

/// Deterministic keyword retrieval over the *external* tool groups (built-ins are always present,
/// so they're excluded). Scores each tool on term hits in its name (weighted), description, and
/// group label; returns the top `limit` as (name, one-line description). Empty query → a catalog
/// sample so the model still sees what's available.
pub fn search_tools(query: &str, groups: &[ToolGroup], limit: usize) -> Vec<(String, String)> {
    let terms: Vec<String> = query.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(String::from)
        .collect();
    let external: Vec<&ToolGroup> = groups.iter().filter(|g| g.label != BUILTIN_GROUP).collect();
    let describe = |t: &ToolSchema| (t.function.name.clone(), first_sentence(&t.function.description, 120));
    if terms.is_empty() {
        return external.iter().flat_map(|g| &g.tools).take(limit).map(|t| describe(t)).collect();
    }

    let mut out: Vec<(String, String)> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Whole-group match: when a query term hits a group's LABEL (i.e. the user named the server/
    // API, e.g. "excalidraw"), load that group's ENTIRE toolset so the right variant — including
    // an inline MCP-App tool that wouldn't keyword-match "histogram" — is available to pick.
    for g in &external {
        let label = g.label.to_lowercase();
        if terms.iter().any(|term| label.contains(term.as_str())) {
            for t in &g.tools {
                if seen.insert(t.function.name.clone()) { out.push(describe(t)); }
            }
        }
    }

    // Then fill remaining slots with individually-scored tools from the other groups.
    let mut scored: Vec<(i32, &ToolSchema)> = Vec::new();
    for g in &external {
        for t in &g.tools {
            if seen.contains(&t.function.name) { continue; }
            let name = t.function.name.to_lowercase();
            let desc = t.function.description.to_lowercase();
            let mut score = 0;
            for term in &terms {
                if name.contains(term.as_str()) { score += 3; }   // name hit weighted highest
                if desc.contains(term.as_str()) { score += 1; }
            }
            if score > 0 { scored.push((score, t)); }
        }
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, t) in scored {
        if out.len() >= limit { break; }
        if seen.insert(t.function.name.clone()) { out.push(describe(t)); }
    }
    // A named group is loaded in full even if it alone exceeds `limit` (bounded to avoid blowing
    // context); otherwise cap at `limit`.
    out.truncate(limit.max(out.len()).min(25));
    out
}

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
                        ProviderKind::OpenAI | ProviderKind::Anthropic => v["choices"][0]["message"]["content"].as_str().map(String::from),
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
/// Shown when a run ends with the model having produced nothing at all — no text and no tool call.
/// Honest (doesn't pretend a deliverable was made) and actionable.
const EMPTY_RUN_MESSAGE: &str = "The model returned an empty response — it didn't answer or use any \
    tools for this request. This often happens with a complex multi-part follow-up on a local model. \
    Try: rephrasing it as a single, self-contained request; turning Reasoning on; or using a more \
    capable model.";

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
            content: Some("Now write your final answer to the user, based ONLY on the actual tool \
                results and work shown above. Do NOT call any tools. IMPORTANT: state only what the \
                results above actually support. If the user asked you to produce a map, chart, image, \
                file, or other artifact and it is NOT already shown above, do NOT claim you created \
                it — say plainly that you were unable to produce it and why. Never invent data, \
                figures, or a deliverable that isn't really there.".into()),
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
    let streamed = match stream_chat(backend, model, &wire, &no_tools, options, keep_alive, app, false, None, None).await {
        Ok((text, _)) => !text.trim().is_empty(),
        Err(_) => false,
    };
    if !streamed {
        let _ = app.emit("agent-token", TokenEvent {
            delta: "I completed the steps but couldn't produce a written summary this time — the model returned no final text. Please try again, or switch to a more capable model (e.g. qwen3).".into(),
        });
    }
}

/// Rough token count for a string, without a tokenizer.
///
/// The obvious `len / 4` is badly wrong for the payloads that actually blow the context. Prose does
/// sit near 4 chars/token, but digit-dense data does NOT: a route polyline or a coordinate-heavy API
/// response tokenises at ~1 char/token, because tokenizers split long decimal runs into 1–2 digit
/// pieces. Measured on real TfL journey data with qwen3: a 17,249-char coordinate array is 16,088
/// tokens, which `len / 4` estimates at 4,312 — 3.7× low. That under-count is what let an oversized
/// wire sail past the budget check below and get FRONT-TRUNCATED by Ollama instead.
///
/// So bucket the characters instead and weight each class. Weights were fitted against real
/// `prompt_eval_count` values for 13 samples spanning prose, JSON, source code and raw coordinates
/// (1.07–4.12 chars/token); worst-case error is ~9%, against ~73% for `len / 4`. Integer arithmetic,
/// one pass, no allocation — cheap enough to run on every message every step.
pub fn estimate_tokens(s: &str) -> usize {
    let (mut digits, mut alpha, mut other) = (0usize, 0usize, 0usize);
    for c in s.chars() {
        if c.is_ascii_digit() { digits += 1 } else if c.is_alphabetic() { alpha += 1 } else { other += 1 }
    }
    // ×100 to keep this in integers: digits ≈ 1.05 tok, letters ≈ 0.21 tok, everything else ≈ 0.45.
    (digits * 105 + alpha * 21 + other * 45) / 100
}

/// Keep the per-step wire within `num_ctx` so the backend never truncates the FRONT of the prompt
/// — which is the system prompt and the user's task. A long tool-heavy run accumulates large tool
/// results; once the wire exceeds the context window Ollama silently drops the oldest messages and
/// the model loses its instructions mid-run (observed: it loops or asks the user to restate). Rather
/// than delete messages (which would orphan a tool result from its assistant tool-call and make the
/// wire malformed), we shrink the CONTENT of the oldest tool results in place, oldest first, always
/// protecting the system prompt, every user message, and the most recent messages. Returns the
/// number of tool results elided (for logging/tests).
/// Estimated size of what will be sent this step: tool schemas plus every message. This is the
/// work the model must read before it can emit a single token, and it is the bulk of the silence
/// in a slow turn — 27,839 tokens meant 48 seconds of nothing in a measured case.
pub fn wire_tokens(wire: &[WireMessage], tools: &[ToolSchema]) -> usize {
    let msg = |m: &WireMessage| -> usize {
        let mut t = 4; // per-message framing overhead
        if let Some(c) = &m.content { t += estimate_tokens(c); }
        if let Some(tc) = &m.tool_calls {
            t += serde_json::to_string(tc).map(|s| estimate_tokens(&s)).unwrap_or(0);
        }
        t
    };
    serde_json::to_string(tools).map(|s| estimate_tokens(&s)).unwrap_or(0)
        + wire.iter().map(msg).sum::<usize>()
}

fn fit_wire_to_context(
    wire: &mut [WireMessage],
    tools: &[ToolSchema],
    num_ctx: i32,
    num_predict: Option<i32>,
) -> usize {
    if num_ctx <= 0 { return 0; }
    let ctx = num_ctx as usize;
    let tok = estimate_tokens;
    // Reserve room for the model's reply plus slack for the estimate and chat-template
    // framing overhead, so we trim before we're actually at the edge.
    let response_reserve = match num_predict { Some(n) if n > 0 => n as usize, _ => 4096 };
    let budget = ctx.saturating_sub(response_reserve + ctx / 8);
    let msg_tok = |m: &WireMessage| -> usize {
        let mut t = 4; // per-message framing overhead
        if let Some(c) = &m.content { t += tok(c); }
        if let Some(tc) = &m.tool_calls {
            t += serde_json::to_string(tc).map(|s| tok(&s)).unwrap_or(0);
        }
        t
    };
    let tools_tok = serde_json::to_string(tools).map(|s| tok(&s)).unwrap_or(0);
    let mut total: usize = tools_tok + wire.iter().map(msg_tok).sum::<usize>();
    if total <= budget { return 0; }

    const KEEP_RECENT: usize = 6;
    const PLACEHOLDER: &str =
        "[earlier tool result omitted to keep within the context window — call the tool again if you still need this data]";
    let cutoff = wire.len().saturating_sub(KEEP_RECENT);
    let mut elided = 0usize;
    let elide_at = |m: &mut WireMessage, total: &mut usize, text: &'static str| {
        let before = m.content.as_deref().map(tok).unwrap_or(0);
        m.content = Some(text.to_string());
        *total = total.saturating_sub(before.saturating_sub(tok(text)));
    };
    for m in wire.iter_mut().take(cutoff) {
        if total <= budget { break; }
        if m.role != "tool" { continue; } // never touch system / user / assistant messages
        let Some(c) = &m.content else { continue; };
        if c.len() <= PLACEHOLDER.len() + 40 { continue; } // already small — not worth eliding
        elide_at(m, &mut total, PLACEHOLDER);
        elided += 1;
    }
    if total <= budget { return elided; }

    // Still over. One ENORMOUS result inside the protected recent window can hold the wire over
    // budget by itself — the exact shape that broke a route map: a 16k-token coordinate dump sat
    // 3rd from the end, so the oldest-first pass above could never reach it and the wire got
    // front-truncated instead (losing the system prompt and the user's task). Those payloads are
    // precisely the ones `offload_tool_result` already wrote to /work/data, so dropping one from
    // the prompt costs little — the model can re-read it with run_python. Only genuinely huge
    // results qualify, and never the LAST message: that's the result the model just asked for and
    // is about to act on.
    const HUGE_TOOL_RESULT_TOKENS: usize = 4096;
    const HUGE_PLACEHOLDER: &str =
        "[a very large tool result was omitted here to keep within the context window. If it was \
         offloaded to /work/data, read it with run_python; otherwise call the tool again.]";
    let last = wire.len().saturating_sub(1);
    for (i, m) in wire.iter_mut().enumerate() {
        if total <= budget { break; }
        if i < cutoff || i == last { continue; } // older ones already handled; keep the newest
        if m.role != "tool" { continue; }
        let Some(c) = &m.content else { continue; };
        if tok(c) <= HUGE_TOOL_RESULT_TOKENS { continue; }
        elide_at(m, &mut total, HUGE_PLACEHOLDER);
        elided += 1;
    }
    elided
}

/// True when interactive assistant text reads as a "here's what I'll do next" preamble rather than
/// a finished answer — e.g. it ends on a colon, or a short message announces an action ("let me
/// fetch…", "now I need to call…") without actually calling a tool. The agent loop treats plain
/// text (no tool call) as the final answer and ends; this lets it instead nudge the model to take
/// the step (or give its real answer), so a run with more to do isn't cut off mid-narration.
/// Deliberately conservative — the cost of a false positive is one extra step, so we only trip on
/// strong signals to avoid re-prompting genuine final answers.
fn looks_like_continuation(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() { return false; }
    // A finished answer almost never ends on a colon; a "next I'll do X:" preamble does.
    if t.ends_with(':') { return true; }
    let lower = t.to_lowercase();

    const INTENT: &[&str] = &["let me", "i'll", "i will", "let's", "now i", "next i",
        "i'm going to", "i am going to", "i need to", "i should", "now let me", "first, let me"];

    // The FINAL sentence is the tell: a message that ends on an announced-but-unfulfilled action
    // ("…Let me do that now.", "…I'll compile it into a report.") is a narrate-stop even though it
    // contains no colon or specific API verb. Take the last non-empty sentence and check it.
    let last = lower
        .rsplit(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
        .map(str::trim)
        .find(|s| !s.is_empty())
        .unwrap_or(lower.trim());
    // "let me know if…" is a closing pleasantry, i.e. a COMPLETE answer — never a continuation.
    if !last.starts_with("let me know") && INTENT.iter().any(|p| last.starts_with(p)) {
        // A bare intent sentence that ends the message: either it trails off on "…now" or it's short
        // enough to be pure announcement (not "let me explain: <the actual explanation>").
        if last.ends_with("now") || last.chars().count() <= 40 { return true; }
    }

    // Fallback: a SHORT message that both announces intent AND names an action but takes none.
    if t.chars().count() <= 320 {
        let has_intent = INTENT.iter().any(|p| lower.contains(p));
        let has_action = ["do that", "do this", "compile", "put together", "assemble", "create",
            "write ", "generate", "build ", "prepare", "produce", "draft", "finalis", "finaliz",
            "complete the", "summar", "the report", "fetch", "call the", "look up", "retrieve",
            "query the", "run_python", "use the", "get the", "search the", "pull the"]
            .iter().any(|a| lower.contains(a));
        return has_intent && has_action;
    }
    false
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
    // A scheduled job that has opted into running code. False for interactive runs, which use
    // the normal per-session approval prompt instead.
    allow_code_exec: bool,
    max_steps: usize,
    // Per-turn cap on web_search + fetch_webpage calls (0 → default). A runaway guard; raise it for
    // research/scraping profiles that legitimately fetch many pages.
    web_tool_cap: usize,
    // Set by the Stop button; checked between steps and while streaming to abort the run.
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // Discovery mode: when there are many tools, expose only built-ins + `find_tools` and let the
    // model load specialized tools on demand, instead of the LLM pre-flight. Jobs pass false.
    discover_tools: bool,
    // On-demand capability recipes. Their descriptions go in the prompt; the model loads a body via
    // `use_skill`. Empty for background jobs.
    skills: Vec<crate::skills::RegisteredSkill>,
) -> anyhow::Result<()> {
    use std::sync::atomic::Ordering;
    let run_start = std::time::Instant::now();
    // Unique id for this agent-loop invocation — stamped on every debug event so the trace groups
    // steps into the right run and closes the right run, even if two runs overlap.
    static DEBUG_RUN_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let run_id = DEBUG_RUN_SEQ.fetch_add(1, Ordering::Relaxed);
    // Read the turn's cumulative (prompt, completion) token counts (accumulated by the stream parser
    // into AppState). Used for per-step deltas and the per-run total shown in the DebugPanel.
    let read_tokens = || -> (u64, u64) {
        app.try_state::<crate::AppState>()
            .map(|s| *s.turn_tokens.lock().unwrap())
            .unwrap_or((0, 0))
    };
    // Emit debug-run-done with the run id + running token totals (no-op for silent job runs).
    let emit_run_done = |error: Option<String>| {
        if silent { return; }
        let (tokens_in, tokens_out) = read_tokens();
        let _ = app.emit("debug-run-done", DebugRunDoneEvent {
            run_id,
            total_ms: run_start.elapsed().as_millis() as u64,
            error,
            tokens_in,
            tokens_out,
        });
    };
    let mut nudged = false;
    let mut continuations = 0usize;
    let mut consecutive_text_without_tools = 0usize; // detect "I'm done" loops
    // Interactive narrate-stop guard: consecutive times the model ended a step with a "next I'll…"
    // preamble (no tool call). Nudged up to this many times to actually continue, then accepted as
    // final so a stubborn narrator can't loop forever.
    let mut narrate_nudges = 0usize;
    const MAX_NARRATE_NUDGES: usize = 2;
    let cap = if tool_cap == 0 { DEFAULT_TOOL_CAP } else { tool_cap };
    // Whether `run_python` is actually reachable this run. Three things key off it: whether an
    // oversized result is offloaded to a file at all, whether the offload directory is staged into
    // the sandbox, and what the truncation notice tells the model to do. All three used to be
    // keyed off `silent` ("is this a job?"), which was only ever a proxy for "can this run execute
    // code" — true because jobs had no run_python. Now that a job can be granted it, ask the real
    // question: a job with code gets the same treatment as chat, and an interactive run with code
    // tools switched off stops being pointed at a file it cannot open.
    let code_available = always_tools.iter().chain(tool_groups.iter().flat_map(|g| g.tools.iter()))
        .any(|t| t.function.name == "run_python");
    // Oversized tool results are offloaded here for run_python to read. run_python is given read
    // access to this dir via dispatch_paths — including in a job, when that job may run code.
    // Interactive chats keep working files under the conversation so they survive for as long as
    // the chat does; background jobs (silent) have no conversation and use the swept temp dir.
    let (results_dir, artifacts_dir, ephemeral) = {
        let conv = if silent { None } else {
            app.try_state::<crate::AppState>().and_then(|st| {
                let d = crate::conversation_files_dir(&st, "data")?;
                let a = crate::conversation_files_dir(&st, "artifacts")?;
                Some((d, a))
            })
        };
        match conv {
            Some((d, a)) => (d, a, false),
            None => (tool_results_dir(), artifact_data_dir(), true),
        }
    };
    // Skill resources loaded via use_skill are copied here and staged into /work/skills for run_python.
    let skill_staging = skill_staging_dir();
    let dispatch_paths: Vec<String> = if silent {
        // A job with run_python still needs the offload directory staged into /work/data, or the
        // file the truncation notice names is not there when it looks. Deliberately without the
        // interactive housekeeping below: the temp dir is shared, and jobs can run concurrently,
        // so one job must not sweep another's working files out from under it.
        let mut v = sandbox_paths.clone();
        if code_available {
            let _ = std::fs::create_dir_all(&results_dir);
            v.push(results_dir.to_string_lossy().into_owned());
        }
        v
    } else {
        if ephemeral { clean_tool_results(&results_dir); }
        // Fresh turn: reset the per-turn image counter so generated_image_N.png restarts at 1 and
        // lines up with {{figure:N}}. We deliberately do NOT delete previous turns' generated images
        // — they persist (cleaned by the 6h TTL above) so a follow-up turn ("now build the deck")
        // can still read /work/data/generated_image_N.png to embed them into a .pptx.
        if let Some(s) = app.try_state::<crate::AppState>() { *s.turn_image_count.lock().unwrap() = 0; }
        let _ = std::fs::remove_dir_all(&skill_staging); // clear any resources from a previous run
        let _ = std::fs::create_dir_all(&skill_staging);
        let mut v = sandbox_paths.clone();
        v.push(results_dir.to_string_lossy().into_owned());
        // Artifact data from earlier turns, re-staged into /work/artifacts.
        if ephemeral { clean_tool_results(&artifacts_dir); }
        v.push(artifacts_dir.to_string_lossy().into_owned());
        v.push(skill_staging.to_string_lossy().into_owned());
        v
    };
    // Tools the model called last step — kept available next step so a multi-step chain
    // isn't broken by them dropping out of the fresh selection.
    let mut last_used: Vec<String> = Vec::new();
    /// Move `name` to the most-recent end of an LRU list, inserting it if absent.
    fn touch(lru: &mut Vec<String>, name: &str) {
        lru.retain(|n| n != name);
        lru.push(name.to_string());
    }
    // Hashes of run_python code already executed this turn. A stubborn local model sometimes
    // re-issues the identical script (re-rendering the same chart and doubling the wall time)
    // even after being told the charts are already shown; we short-circuit the repeat.
    let mut ran_python_code: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // Whether run_python has executed yet this turn — drives the one-time /work workspace reset so
    // files persist across calls within a turn (but not across turns).
    let mut python_started = false;
    // Loop breaker: signature of the previous step's tool-call set + how many times in a row it's
    // repeated. A model stuck re-issuing the same calls (common with reasoning off) is nudged then
    // force-answered instead of spinning to max_steps.
    let mut last_tool_sig: u64 = 0;
    let mut tool_sig_repeats: usize = 0;
    // Total times each tool-call signature has occurred this run — catches a model that re-issues
    // the same call repeatedly but NON-consecutively (interspersed with others), which the
    // consecutive counter misses (e.g. geocode, details, geocode, details, geocode…).
    let mut sig_counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    // Times each tool NAME has been dispatched this run. Catches flailing on ONE tool with
    // different args each time (e.g. 18× SPARQL query variations) — the signature guard misses
    // that because the args differ. Past the cap, further calls are refused, not executed.
    let mut tool_name_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    const TOOL_NAME_CALL_CAP: usize = 8;
    // Web tools flail more readily (endless search/scrape variations), so they get their own
    // per-name cap. Default 10; a profile can raise it (web_tool_cap) for heavy fetching.
    const DEFAULT_WEB_TOOL_CALL_CAP: usize = 10;
    let web_tool_cap = if web_tool_cap > 0 { web_tool_cap } else { DEFAULT_WEB_TOOL_CALL_CAP };
    // Turn-level runaway guards. The per-tool caps above miss a model that *rotates* among tools
    // (web_search→fetch→query→run_python…), keeping each under its cap while the turn total and
    // wall-clock climb unbounded (observed: 30 calls / 13 min, and a 65-min map spiral). These
    // bound the WHOLE turn regardless of which tools are used.
    const GLOBAL_TOOL_CALL_CAP: usize = 15; // total tool dispatches across ALL tools this turn
    // ~30 min. The global tool cap (15) is the primary runaway guard; this wall is a backstop. It's
    // generous because a legit image deck can spend minutes generating several photoreal images
    // (each ~90s) before building the slides — a tighter wall would cut that off mid-workflow.
    // Raised from 600s: a large-result analysis (e.g. judging ~45 TED tenders from their
    // descriptions on a local 27B model) legitimately needs 10-20 min of generation AFTER its last
    // tool call. At 600s such runs were killed mid-sentence, having already decided what to publish
    // but never emitting the calls to do it — the wall cannot distinguish a spiral from slow work,
    // so it must sit above the slowest legitimate job, leaving the tool cap to catch real runaways.
    const TURN_WALL_BUDGET_SECS: u64 = 1800;
    // Once a deliverable artifact (a map/chart/HTML) has been produced, only a little cleanup is
    // allowed before we force the final answer — otherwise the model "refines" it for many minutes
    // (observed: create_artifact, then 3 more run_python calls, still looping at 65 min).
    const POST_ARTIFACT_TOOL_BUDGET: usize = 2;
    let mut total_tool_calls: usize = 0;
    let mut artifact_emitted = false;
    // Titles of artifacts already rendered this turn. `create_artifact`'s own result says "It is
    // DONE — do NOT call create_artifact again", but that is only an instruction and models ignore
    // it: observed twice in one turn, same title, near-identical narration, so the user got two
    // copies of the same document stacked in the chat. run_python has an enforced idempotency
    // guard for exactly this; artifacts had none.
    let mut emitted_artifact_titles: Vec<String> = Vec::new();
    let mut post_artifact_tool_calls: usize = 0;
    // Discovery mode: names of external tools the model has loaded this run via `find_tools`.
    // Discovery mode: tools the model has loaded via `find_tools`, oldest first. A Vec rather than
    // a set because the order IS the eviction policy: the set only ever grew, so six find_tools
    // calls could leave ~70 schemas in every later step and `maxTools` never applied here at all.
    let mut loaded_tools: Vec<String> = Vec::new();
    // Whether the model ever streamed a genuine FINAL answer — text on a step with NO tool call.
    // Narration that accompanies a tool call ("Let me look that up…") does NOT count: a real answer
    // ends the run immediately, so if we reach a salvage site with only narration behind us, the
    // user still has no answer and we must force one. (Counting narration here caused blank replies.)
    let mut streamed_text = false;
    // Whether the model called ANY tool this run. If a run ends with no final text AND no tool ever
    // ran, the model produced nothing — forcing a prose "summary" then makes it FABRICATE a
    // deliverable it never made (e.g. "here's your map"). In that case we give an honest message
    // instead of a salvaged (and invented) answer.
    let mut any_tool_ran = false;
    // Set once a server reports the model can't do tool calling; the rest of the run then sends no
    // tools (and skips per-step selection) so a non-tool model degrades to plain chat.
    let mut disable_tools = false;
    let discoverable_total: usize = tool_groups.iter().map(|g| g.tools.len()).sum();
    // Discovery mode is only meaningful once there are more tools than we'd send wholesale.
    let discover_active = discover_tools && discoverable_total > SELECTION_THRESHOLD;
    // In discovery mode the model must be told specialized tools are loaded on demand.
    let sys_prompt_effective = if discover_active {
        format!("{system_prompt}\n\nTOOL DISCOVERY: You have built-in tools (files, web search, \
            fetch, date/time, email, code) plus a `find_tools` tool. Specialized tools (external \
            APIs, SPARQL/linked-data endpoints, MCP servers) are NOT loaded yet — nor is image \
            generation, so call find_tools(\"generate an image\") before trying to create a picture. When the task \
            needs one, FIRST call find_tools with a short description of what you need; the matching \
            tools become callable on your next step. Never guess a specialized tool's name before \
            loading it with find_tools. IMPORTANT: for any question about specific data, facts, \
            figures, statistics, records, or a domain your connected tools might cover, call \
            find_tools FIRST — do NOT jump straight to web_search. web_search is a fallback only for \
            general open-web information the connected tools don't cover; prefer authoritative \
            connected tools whenever the topic could match one.")
    } else {
        system_prompt.to_string()
    };
    // Skills preamble: only the one-line descriptions (cheap); the model pulls a full recipe with
    // use_skill on demand. Appended after any discovery preamble.
    let sys_prompt_effective = if skills.is_empty() {
        sys_prompt_effective
    } else {
        let list = skills.iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>().join("\n");
        format!("{sys_prompt_effective}\n\nAVAILABLE SKILLS — when the user's request is one of \
            these, FIRST call use_skill(\"<name>\") to load its full instructions, then follow \
            them:\n{list}")
    };
    for step in 0..max_steps {
        // Stop requested — end the run cleanly before doing any more work.
        if cancel.load(Ordering::SeqCst) {
            if !silent {
                let _ = app.emit("agent-done", DoneEvent { error: None });
                emit_run_done(None);
            }
            return Ok(());
        }
        // Turn-level runaway guard: before sampling another (expensive) step, stop if this turn has
        // spent its global tool-call budget, exceeded the wall-clock budget, or kept tooling after a
        // deliverable artifact was already produced. Force a written final answer and end, so a stuck
        // local model can't burn 10-60 minutes. Mirrors the severity-based force-answer path below.
        let over_budget = total_tool_calls >= GLOBAL_TOOL_CALL_CAP
            || run_start.elapsed().as_secs() >= TURN_WALL_BUDGET_SECS
            || (artifact_emitted && post_artifact_tool_calls > POST_ARTIFACT_TOOL_BUDGET);
        if over_budget {
            if !silent {
                if !streamed_text {
                    ensure_final_answer(backend, model, system_prompt, conversation,
                        options.as_ref(), keep_alive.as_deref(), app).await;
                }
                let _ = app.emit("agent-done", DoneEvent { error: None });
                emit_run_done(None);
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
            } else if discover_active {
                // Discovery mode: built-ins stay directly available; specialized tools appear only
                // after the model loads them via find_tools. No LLM pre-flight.
                for g in tool_groups.iter().filter(|g| g.label == BUILTIN_GROUP) {
                    v.extend(g.tools.iter().cloned());
                }
                v.push(find_tools_schema());
                // Most-recent first, so the cap evicts the tools loaded longest ago rather than
                // whichever the group iteration happened to reach last. Tools called on the previous
                // step outrank everything: dropping one mid-chain would strand the model.
                let mut wanted: Vec<&str> = Vec::new();
                for n in last_used.iter() { if !wanted.contains(&n.as_str()) { wanted.push(n); } }
                for n in loaded_tools.iter().rev() { if !wanted.contains(&n.as_str()) { wanted.push(n); } }
                for name in wanted {
                    if v.len() >= cap { break; }
                    if v.iter().any(|x| x.function.name == name) { continue; }
                    if let Some(t) = tool_groups.iter().filter(|g| g.label != BUILTIN_GROUP)
                        .flat_map(|g| &g.tools).find(|t| t.function.name == name) {
                        v.push(t.clone());
                    }
                }
            } else {
                // Legacy LLM pre-flight (jobs): 1–2 non-streaming calls narrow the tool set.
                if !silent { let _ = app.emit("agent-status", serde_json::json!({ "phase": "Selecting tools…" })); }
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
        // Skills are always loadable when present (independent of tool discovery/selection).
        if !disable_tools && !skills.is_empty() {
            tools.push(use_skill_schema());
        }

        let schema_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();
        if !silent {
            // ~4 chars per token is close enough for a cost indicator, and costs nothing to compute.
            let tools_tokens = tools.iter()
                .filter_map(|t| serde_json::to_string(t).ok().map(|j| j.len()))
                .sum::<usize>() / 4;
            let _ = app.emit("debug-step-start", DebugStepEvent {
                run_id, step, schema_names, candidate_total: always_tools.len() + discoverable_total,
                tools_tokens,
            });
        }
        // Token counts at the start of this step, so the step's own usage is the delta at step-done.
        let tok_before = read_tokens();

        // Build wire messages: system + history
        let mut wire = {
            let conv = conversation.lock().unwrap();
            let mut w = vec![WireMessage {
                role: "system".into(),
                content: Some(sys_prompt_effective.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
                images: None,
            }];
            w.extend(conv.clone());
            w
        };

        // Keep the wire under num_ctx so the backend never truncates the system prompt / user task
        // out from under a long tool-heavy run. Non-destructive: only shrinks this step's copy, the
        // stored conversation keeps the full history.
        if let Some(o) = options.as_ref() {
            if let Some(ctx) = o.num_ctx {
                let elided = fit_wire_to_context(&mut wire, &tools, ctx, o.num_predict);
                if elided > 0 && !silent {
                    let _ = app.emit("debug-context-trim",
                        serde_json::json!({ "step": step, "elided": elided }));
                }
            }
        }

        // The model must eval the whole prompt (history + tool schemas) before the first token
        // streams — silent time that grows with the conversation. Flag it as "Thinking…".
        if !silent {
            // "Thinking…" gave no clue why a turn sat silent for minutes. Name the actual work, and
            // call out the context overflow — past num_ctx the prefix cache dies and the WHOLE
            // prompt is re-read every step, which is the difference between a 300s and a 1400s turn.
            let est = wire_tokens(&wire, &tools);
            let ctx = options.as_ref().and_then(|o| o.num_ctx).unwrap_or(0) as usize;
            let k = |n: usize| if n >= 1000 { format!("{}K", n / 1000) } else { n.to_string() };
            let phase = if ctx > 0 && est > ctx {
                format!("Reading {} tokens — over the {} context limit, so all of it is re-read each step", k(est), k(ctx))
            } else {
                format!("Reading {} tokens of context…", k(est))
            };
            let _ = app.emit("agent-status", serde_json::json!({ "phase": phase }));

            // Break that number down. Cheap (one estimate pass over data already in memory) and it
            // turns "why is my context 9K" into an answer rather than a scrolling exercise.
            let full = app.try_state::<crate::AppState>()
                .map(|st| *st.debug_full_context.lock().unwrap()).unwrap_or(false);
            let cut = |t: &str| -> Option<String> {
                if !full { return None; }
                const MAX: usize = 200_000;   // one absurd result must not wedge the panel
                Some(if t.chars().count() > MAX { t.chars().take(MAX).collect::<String>() + "\n…[truncated]" } else { t.to_string() })
            };
            let mut schemas: Vec<ContextItem> = tools.iter().map(|t| {
                let json = serde_json::to_string(t).unwrap_or_default();
                ContextItem { label: t.function.name.clone(), tokens: estimate_tokens(&json), text: cut(&json) }
            }).collect();
            schemas.sort_by(|a, b| b.tokens.cmp(&a.tokens));
            let messages: Vec<ContextItem> = wire.iter().enumerate().map(|(i, m)| {
                let body = m.content.clone().unwrap_or_default();
                let calls = m.tool_calls.as_ref()
                    .and_then(|tc| serde_json::to_string(tc).ok()).unwrap_or_default();
                let label = match (m.role.as_str(), m.name.as_deref()) {
                    ("tool", Some(n)) => format!("{i}. tool result · {n}"),
                    (r, _) if !calls.is_empty() => format!("{i}. {r} · tool call"),
                    (r, _) => format!("{i}. {r}"),
                };
                ContextItem {
                    label,
                    tokens: estimate_tokens(&body) + estimate_tokens(&calls) + 4,
                    text: cut(&if calls.is_empty() { body } else { format!("{body}\n{calls}") }),
                }
            }).collect();
            let system_tokens = wire.first().filter(|m| m.role == "system")
                .and_then(|m| m.content.as_deref()).map(estimate_tokens).unwrap_or(0);
            let tools_tokens = schemas.iter().map(|c| c.tokens).sum();
            let _ = app.emit("debug-step-context", DebugContextEvent {
                run_id, step, total: est, num_ctx: ctx,
                system_tokens, tools_tokens,
                history_tokens: est.saturating_sub(tools_tokens + system_tokens),
                schemas, messages,
            });
        }

        // Re-sample the step when the model emits a tool call Ollama can't parse; only a
        // persistent failure (or any other error) ends the run.
        let mut attempt = 0usize;
        let (full_text, tool_calls) = loop {
            match stream_chat(backend, model, &wire, &tools, options.as_ref(), keep_alive.as_deref(), app, silent, Some(cancel.as_ref()),
                              Some(run_start + std::time::Duration::from_secs(TURN_WALL_BUDGET_SECS))).await {
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
                        emit_run_done(Some(e.to_string()));
                    }
                    return Err(e);
                }
            }
        };

        let step_ms = step_start.elapsed().as_millis() as u64;
        if !silent {
            let (tp, tc) = read_tokens();
            let _ = app.emit("debug-step-done", DebugStepDoneEvent {
                run_id,
                step,
                llm_text: full_text.clone(),
                duration_ms: step_ms,
                tokens_in: tp.saturating_sub(tok_before.0),
                tokens_out: tc.saturating_sub(tok_before.1),
            });
        }

        if !full_text.trim().is_empty() && tool_calls.is_empty() { streamed_text = true; }

        // Remember which tools were called so they stay available next step (chain continuity).
        last_used = tool_calls.iter().map(|tc| tc.function.name.clone()).collect();
        // Calling a discovered tool makes it most-recent, so an actively-used tool is never the one
        // evicted when the cap bites.
        for n in &last_used { if loaded_tools.iter().any(|l| l == n) { touch(&mut loaded_tools, n); } }

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
            // Interactive narrate-stop: the model ended a step describing a next action ("Now I'll
            // fetch…: ") without calling a tool. Plain text normally ends the run, so an otherwise
            // productive run gets cut off mid-narration. Nudge it (a bounded number of times) to
            // either take the step or give its real final answer, instead of ending here.
            if !silent && narrate_nudges < MAX_NARRATE_NUDGES && looks_like_continuation(&full_text) {
                narrate_nudges += 1;
                let mut conv = conversation.lock().unwrap();
                conv.push(WireMessage {
                    role: "user".into(),
                    content: Some("You described a next step but didn't take it. If you still need \
                        more data, call the tool now. If you already have everything, write your \
                        complete final answer — don't just describe what you would do.".into()),
                    tool_calls: None, tool_call_id: None, name: None, images: None,
                });
                continue;
            }

            if !silent {
                // Never end an interactive run blank. If the model did real work but didn't
                // summarise, salvage a written answer (honesty-guarded). If it did NOTHING (no text,
                // no tool), don't force a prose "answer" — that just makes it fabricate a deliverable
                // it never produced; give an honest message instead.
                if !streamed_text {
                    if any_tool_ran {
                        ensure_final_answer(backend, model, system_prompt, conversation,
                            options.as_ref(), keep_alive.as_deref(), app).await;
                    } else {
                        let _ = app.emit("agent-token", TokenEvent { delta: EMPTY_RUN_MESSAGE.into() });
                    }
                }
                let _ = app.emit("agent-done", DoneEvent { error: None });
                emit_run_done(None);
            }
            return Ok(());
        }

        // Tool call received — reset the detectors so the next empty/text response after this
        // tool's result is handled correctly (can nudge again if needed). narrate_nudges resets too:
        // progress was made, so a later narrate-stop gets a fresh budget (still bounded per episode).
        consecutive_text_without_tools = 0;
        narrate_nudges = 0;
        nudged = false;
        any_tool_ran = true;

        // Loop breaker: has this exact tool-call set been issued before — consecutively OR just
        // repeatedly this run? Either way the model is stuck. `severity` unifies both: 2 = first
        // intervention (nudge), >2 = force an answer and end.
        let sig = tool_call_signature(&tool_calls);
        if sig == last_tool_sig { tool_sig_repeats += 1; } else { last_tool_sig = sig; tool_sig_repeats = 0; }
        let total = { let c = sig_counts.entry(sig).or_insert(0); *c += 1; *c };
        let severity = tool_sig_repeats.max(total.saturating_sub(1));
        if severity >= 2 {
            // Drop the un-executed repeat we just appended so history isn't left with a dangling
            // assistant(tool_calls) that has no matching results.
            { let mut conv = conversation.lock().unwrap(); conv.pop(); }
            if severity == 2 {
                // First intervention: tell it to stop and answer, then give it one more chance.
                conversation.lock().unwrap().push(WireMessage {
                    role: "user".into(),
                    content: Some("You have called the same tool with the same arguments several \
                        times without making progress. STOP calling tools now and give your best \
                        final answer using the information you already have.".into()),
                    tool_calls: None, tool_call_id: None, name: None, images: None,
                });
                continue;
            }
            // Ignored the nudge and repeated again — force a written answer and end the run.
            if !silent {
                if !streamed_text {
                    ensure_final_answer(backend, model, system_prompt, conversation,
                        options.as_ref(), keep_alive.as_deref(), app).await;
                }
                let _ = app.emit("agent-done", DoneEvent { error: None });
                emit_run_done(None);
            }
            return Ok(());
        }

        // Dispatch each tool call
        let mut answered = 0usize;
        for call in &tool_calls {
            // Stop may have been pressed while the step was still streaming, before this loop
            // began. Don't start another outbound call — remaining ones are closed out below.
            if cancel.load(Ordering::SeqCst) { break; }
            let name = &call.function.name;
            let args = &call.function.arguments;
            let pretty_args = serde_json::to_string_pretty(args).unwrap_or_default();
            // Count every dispatched call toward the turn-level budgets (checked at the next step's
            // top). Calls made after a deliverable artifact also count against the tighter post-
            // artifact budget so the model doesn't endlessly "refine" a map/chart it already produced.
            total_tool_calls += 1;
            if artifact_emitted { post_artifact_tool_calls += 1; }

            // Discovery meta-tool: load specialized tools matching the query so they become
            // callable next step. Handled here (not dispatch_tool) because it mutates the per-run
            // loaded set. No-op if discovery mode isn't active.
            if name == "find_tools" {
                let query = args.get("query").and_then(|q| q.as_str()).unwrap_or("");
                let found = search_tools(query, tool_groups, 12);
                for (nm, _) in &found { touch(&mut loaded_tools, nm); }
                let result = if found.is_empty() {
                    format!("No tools matched \"{query}\". Try broader or different keywords, or \
                        proceed with your built-in tools.")
                } else {
                    format!("Loaded {} tool(s) — now available to call directly on your next step:\n{}",
                        found.len(),
                        found.iter().map(|(n, d)| format!("- {n}: {d}")).collect::<Vec<_>>().join("\n"))
                };
                if !silent {
                    let _ = app.emit("agent-tool-call", ToolCallEvent { name: name.clone(), args: pretty_args.clone() });
                    let _ = app.emit("agent-tool-result", ToolResultEvent {
                        name: name.clone(), result: result.clone(), full_result: String::new(), full_truncated: false, ui: None, images: Vec::new(), artifact: None });
                }
                conversation.lock().unwrap().push(WireMessage {
                    role: "tool".into(), content: Some(result),
                    tool_calls: None, tool_call_id: None, name: Some(name.clone()), images: None,
                });
                continue;
            }

            // Skill loader: return the named skill's full instructions so the model can follow them
            // (progressive disclosure — only descriptions were in the prompt). Mirrors find_tools.
            if name == "use_skill" {
                let want = args.get("name").and_then(|q| q.as_str()).unwrap_or("").trim();
                let result = match skills.iter()
                    .find(|s| s.name.eq_ignore_ascii_case(want) || s.id.eq_ignore_ascii_case(want))
                {
                    Some(s) => {
                        // Stage the skill's bundled resources into /work/skills. Interactive only,
                        // because jobs are passed no skills at all. They land in the staging dir
                        // already in dispatch_paths.
                        let mut note = String::new();
                        if !silent && !s.resources.is_empty() {
                            let dir = skill_staging_dir();
                            let mut staged = Vec::new();
                            for src in crate::skills::skill_resource_paths(s) {
                                if let Some(fname) = src.file_name() {
                                    if std::fs::copy(&src, dir.join(fname)).is_ok() {
                                        staged.push(format!("/work/skills/{}", fname.to_string_lossy()));
                                    }
                                }
                            }
                            if !staged.is_empty() {
                                note = format!("\n\nBundled resources are available to run_python at: {}.", staged.join(", "));
                            }
                        }
                        format!("SKILL: {}\n\n{}{}\n\n(Now follow these instructions to complete the task.)", s.name, s.body, note)
                    }
                    None => format!("No skill named \"{want}\". Available skills: {}.",
                        skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")),
                };
                if !silent {
                    let _ = app.emit("agent-tool-call", ToolCallEvent { name: name.clone(), args: pretty_args.clone() });
                    let _ = app.emit("agent-tool-result", ToolResultEvent {
                        name: name.clone(), result: result.clone(), full_result: String::new(), full_truncated: false, ui: None, images: Vec::new(), artifact: None });
                }
                conversation.lock().unwrap().push(WireMessage {
                    role: "tool".into(), content: Some(result),
                    tool_calls: None, tool_call_id: None, name: Some(name.clone()), images: None,
                });
                continue;
            }

            // Duplicate-artifact guard. A second artifact under a title already rendered this turn
            // is a re-run or a "refinement", not a new deliverable — refuse it and push the model
            // to its final answer. Matching on TITLE rather than content is deliberate: the repeat
            // is rarely byte-identical (the model tweaks a heading or a field as it "improves" it),
            // so a content hash would miss it, while a genuinely different deliverable in the same
            // turn gets a different title and still goes through.
            if name == "create_artifact" {
                let title = args.get("title").and_then(|t| t.as_str()).unwrap_or("").trim().to_lowercase();
                if !title.is_empty() && emitted_artifact_titles.iter().any(|t| *t == title) {
                    // The wording has to stop the model DESCRIBING a second artifact as well as
                    // creating one. With a softer note it accepted the refusal, then wrote "Here's
                    // what was created: 1. First … 2. Second (with signature field) …" — narrating
                    // a deliverable the user could not see, which is worse than the duplicate.
                    let note = "[REFUSED: an artifact with this title was already rendered earlier \
                        in this turn and is still displayed. Nothing was created by this call — \
                        there is exactly ONE artifact in this conversation, not two. Do NOT call \
                        create_artifact again, and do NOT describe, list or refer to a second \
                        artifact in your answer: it does not exist and the user cannot see one. \
                        Write your final answer about the single artifact already shown above.]"
                        .to_string();
                    if !silent {
                        let _ = app.emit("agent-tool-call", ToolCallEvent { name: name.clone(), args: pretty_args.clone() });
                        let _ = app.emit("agent-tool-result", ToolResultEvent {
                            name: name.clone(), result: note.clone(), full_result: String::new(),
                            full_truncated: false, ui: None, images: Vec::new(), artifact: None });
                    }
                    conversation.lock().unwrap().push(WireMessage {
                        role: "tool".into(), content: Some(note),
                        tool_calls: None, tool_call_id: None, name: Some(name.clone()), images: None,
                    });
                    continue;
                }
                if !title.is_empty() { emitted_artifact_titles.push(title); }
            }

            // Idempotency guard: if the model re-issues run_python with code it already ran this
            // turn, don't execute it again (it would re-render the same chart and waste time).
            // Feed back an "already ran" note so the model stops and writes its final answer.
            if name == "run_python" {
                if let Some(code) = args.get("code").and_then(|c| c.as_str()) {
                    let key = code.trim();
                    if !key.is_empty() {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        std::hash::Hash::hash(key, &mut h);
                        if !ran_python_code.insert(std::hash::Hasher::finish(&h)) {
                            let note = "[This exact code was already run earlier in this turn; its \
                                output and any chart(s) are already displayed to the user above. It \
                                was NOT run again. Do NOT call run_python with this code again — write \
                                your final answer now, describing the chart in words.]".to_string();
                            conversation.lock().unwrap().push(WireMessage {
                                role: "tool".into(), content: Some(note),
                                tool_calls: None, tool_call_id: None,
                                name: Some(name.clone()), images: None,
                            });
                            continue;
                        }
                    }
                }
            }

            // Per-tool-name cap: a tool hammered too many times this run (with any args) is
            // flailing (e.g. 18 SPARQL query variations). Refuse further calls and push the model
            // to answer. run_python is exempt (iterative coding is legitimate; it has its own guard).
            if name != "run_python" {
                let n = { let c = tool_name_counts.entry(name.clone()).or_insert(0); *c += 1; *c };
                let cap = match name.as_str() {
                    "web_search" | "fetch_webpage" => web_tool_cap,
                    _ => TOOL_NAME_CALL_CAP,
                };
                if n > cap {
                    let note = format!("[You have called '{name}' {n} times this turn without \
                        resolving the request. STOP calling it — answer using the results you \
                        already have, or take a clearly different approach.]");
                    if !silent {
                        let _ = app.emit("agent-tool-result", ToolResultEvent {
                            name: name.clone(), result: note.clone(), full_result: String::new(), full_truncated: false, ui: None, images: Vec::new(), artifact: None });
                    }
                    conversation.lock().unwrap().push(WireMessage {
                        role: "tool".into(), content: Some(note),
                        tool_calls: None, tool_call_id: None, name: Some(name.clone()), images: None });
                    continue;
                }
            }

            if !silent {
                let _ = app.emit("agent-tool-call", ToolCallEvent {
                    name: name.clone(),
                    args: pretty_args,
                });
            }

            // The Pyodide /work workspace is wiped only on the FIRST run_python of this turn, so
            // files an earlier call wrote (chart PNGs, etc.) persist into later calls in the turn.
            let python_reset = name == "run_python" && !python_started;
            if name == "run_python" { python_started = true; }
            // Route: builtin → openapi → sparql → mcp
            let result = dispatch_tool(name, args, &openapi_specs, &sparql_endpoints, mcp_connections, &allowed_dirs, &dispatch_paths, web_search_results, silent, python_reset, allow_code_exec, app).await;

            // Stop pressed during the tool call (e.g. a long-running run_python) — bail before
            // rendering its result, which could otherwise land in a now-different chat. The
            // step-boundary cancel check below ends the run cleanly.
            if cancel.load(Ordering::SeqCst) { break; }

            // Cap large responses so they don't blow the context (configurable per profile). In
            // interactive chat, oversized *structured-data* results are also saved to a file the
            // model can process with run_python. But document/text reads (read_file, fetch_webpage,
            // wiki_read) are meant to be read and summarised — plain-truncate them so the model
            // works from the text it has instead of being nudged toward run_python (which loops
            // chasing a file it can't process, e.g. a PDF).
            let read_for_content = matches!(name.as_str(), "read_file" | "fetch_webpage" | "wiki_read");
            // Keep the FULL result for the connector data viewer (interactive only) before it's
            // truncated for the model. Capped at a generous display limit so a multi-MB response
            // can't bloat the webview / saved chat. Document reads and the URL-list/file views don't
            // need it, so skip them.
            const DISPLAY_LIMIT: usize = 512 * 1024;
            let (full_result, full_truncated) = if !silent && !read_for_content {
                if result.len() > DISPLAY_LIMIT {
                    (result.chars().take(DISPLAY_LIMIT).collect::<String>(), true)
                } else {
                    (result.clone(), false)
                }
            } else { (String::new(), false) };
            let result = if !code_available || read_for_content {
                cap_tool_result(result, &name, tool_result_limit)
            } else {
                offload_tool_result(result, &name, tool_result_limit, &results_dir)
            };
            // Only send full_result when it actually adds detail over the (truncated) model result.
            let full_result = if full_result == result { String::new() } else { full_result };

            // An MCP-App UI payload, inline images, and/or a model-authored artifact may have been
            // stashed by dispatch_tool.
            let (ui, images, artifact) = if !silent {
                app.try_state::<crate::AppState>()
                    .map(|s| (
                        s.pending_tool_ui.lock().unwrap().take(),
                        std::mem::take(&mut *s.pending_tool_images.lock().unwrap()),
                        s.pending_artifact.lock().unwrap().take(),
                    ))
                    .unwrap_or((None, Vec::new(), None))
            } else { (None, Vec::new(), None) };
            let had_media = ui.is_some() || !images.is_empty() || artifact.is_some();
            // A deliverable artifact was produced — arm the post-artifact budget so the model wraps
            // up soon instead of looping on further "refinements" (see POST_ARTIFACT_TOOL_BUDGET).
            if artifact.is_some() { artifact_emitted = true; }
            // Server id of any MCP-app UI this call produced — used to pause below for approval.
            let ui_server = ui.as_ref().map(|u| u.server_id.clone());
            if !silent {
                let _ = app.emit("agent-tool-result", ToolResultEvent {
                    name: name.clone(),
                    result: result.clone(),
                    full_result,
                    full_truncated,
                    ui,
                    images,
                    artifact,
                });
            }

            // Pause on an UNAPPROVED MCP-app: wait for the user to Allow/Skip before the loop
            // continues, so the model can't stack more approval prompts (or keep flailing) while a
            // deliverable awaits your decision. Once a server is approved, later apps don't pause.
            if let Some(server_id) = ui_server {
                if let Some(state) = app.try_state::<crate::AppState>() {
                    let needs = !state.apps_allowed.lock().unwrap().contains(&server_id);
                    if needs {
                        let (tx, mut rx) = tokio::sync::oneshot::channel::<bool>();
                        *state.pending_app_approval.lock().unwrap() = Some(tx);
                        let _ = app.emit("agent-status", serde_json::json!({ "phase": "Waiting for app approval…" }));
                        // Wait for Allow/Skip, the Stop button, or a 5-minute safety timeout.
                        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
                        while std::time::Instant::now() < deadline {
                            match tokio::time::timeout(std::time::Duration::from_millis(200), &mut rx).await {
                                Ok(_) => break,                                   // Allow or Skip
                                Err(_) => if cancel.load(Ordering::SeqCst) { break; }, // Stop pressed
                            }
                        }
                        *state.pending_app_approval.lock().unwrap() = None;
                    }
                }
                if cancel.load(Ordering::SeqCst) { break; }
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
            answered += 1;
        }

        // Any call left unanswered by a Stop gets an explicit cancelled result, so the next
        // user message doesn't make the model re-run it thinking it never returned.
        if answered < tool_calls.len() {
            let mut conv = conversation.lock().unwrap();
            close_unanswered_tool_calls(&mut conv, &tool_calls, answered);
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
        emit_run_done(Some(msg));
    }
    Ok(())
}

/// Execute a single registered tool by name and return its full (untruncated) result string.
/// Used by code-mode (`call_tool` from run_python), which routes here through `call_tool_from_code`.
/// Same routing as the agent loop's dispatch (built-in → OpenAPI → SPARQL → MCP); `silent` so it
/// emits no chat events of its own (the caller surfaces a code-tool trace).
#[allow(clippy::too_many_arguments)]
pub async fn call_one_tool<R: tauri::Runtime>(
    name: &str,
    args: &serde_json::Value,
    openapi_specs: &[RegisteredSpec],
    sparql_endpoints: &[RegisteredSparqlEndpoint],
    mcp_connections: &tokio::sync::Mutex<HashMap<String, MCPConnection>>,
    allowed_dirs: &[String],
    sandbox_paths: &[String],
    web_search_results: usize,
    app: &AppHandle<R>,
) -> String {
    dispatch_tool(name, args, openapi_specs, sparql_endpoints, mcp_connections,
        allowed_dirs, sandbox_paths, web_search_results, /*silent*/ true, /*python_reset*/ true,
        /*allow_code_exec*/ false, app).await
}

/// Order-independent hash of a step's tool calls (name + arguments), used to detect a model that
/// keeps re-issuing the identical set of calls step after step (a no-progress loop).
fn tool_call_signature(calls: &[WireToolCall]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut parts: Vec<String> = calls.iter()
        .map(|c| format!("{}::{}", c.function.name,
            serde_json::to_string(&c.function.arguments).unwrap_or_default()))
        .collect();
    parts.sort();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in &parts { p.hash(&mut h); }
    h.finish()
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
    // Only meaningful for run_python: wipe /work first (true = first call of the turn).
    python_reset: bool,
    // Only meaningful for run_python in a job: this job may execute code (see its field docs).
    allow_code_exec: bool,
    app: &AppHandle<R>,
) -> String {
    // 0. Code-execution sandbox — gated behind a per-session permission prompt.
    if name == "run_python" {
        return dispatch_run_python(args, allowed_dirs, sandbox_paths, silent, allow_code_exec, python_reset, app).await;
    }

    // 0b. Model-authored HTML artifact — stashed for the agent loop to render inline (sandboxed
    // iframe + Save). No file/tool access; display only. Skipped in silent (job) runs.
    if name == "create_artifact" {
        let title = args.get("title").and_then(|t| t.as_str()).unwrap_or("Artifact").to_string();
        let html = args.get("html").and_then(|h| h.as_str()).unwrap_or("").to_string();
        if html.trim().is_empty() {
            return "Error: create_artifact needs a non-empty 'html' argument.".to_string();
        }
        if !silent {
            if let Some(s) = app.try_state::<crate::AppState>() {
                *s.pending_artifact.lock().unwrap() = Some(ArtifactPayload { title: title.clone(), html });
            }
        }
        return format!("[Artifact \"{title}\" is rendered inline to the user, with a Save button. \
            It is DONE and displayed — do NOT call create_artifact again for this content or to \
            'refine' it. Write your final answer now, referring to it as \"shown above\"; do NOT \
            paste the HTML into your reply.]");
    }

    // 0c. Local image generation — shell out to stable-diffusion.cpp (offline, no cloud). The PNG
    // is pushed as a data: URL onto pending_tool_images so the agent loop renders it inline, exactly
    // like a Mapbox static map. Skipped-display in silent (job) runs, which have no UI.
    if name == "generate_image" {
        use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
        let prompt = args.get("prompt").and_then(|p| p.as_str()).unwrap_or("").trim().to_string();
        if prompt.is_empty() {
            return "Error: generate_image needs a non-empty 'prompt' describing the image.".to_string();
        }
        let negative = args.get("negative_prompt").and_then(|p| p.as_str());
        let size = args.get("size").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let steps = args.get("steps").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let seed = args.get("seed").and_then(|v| v.as_i64());
        let strength = args.get("strength").and_then(|v| v.as_f64()).map(|s| s as f32);
        // Optional img2img: the model passes source_image as the /work/uploads/<name> path (or bare
        // filename) of an ATTACHED image. That virtual path doesn't exist on the real disk, so we
        // resolve it by basename against the staged attachment paths to get the actual file.
        let source_ref = args.get("source_image").and_then(|p| p.as_str())
            .map(str::trim).filter(|s| !s.is_empty());
        // Restored conversations carry paths, not bytes, so a match in `sandbox_paths` is not
        // proof the file is still there — require it to exist, or the read fails later with a
        // far less obvious error.
        let source_real: Option<String> = source_ref.and_then(|r| {
            let want = std::path::Path::new(r).file_name().and_then(|n| n.to_str())?.to_string();
            sandbox_paths.iter().find(|p|
                std::path::Path::new(p.as_str()).file_name().and_then(|n| n.to_str()) == Some(want.as_str())
                    && std::path::Path::new(p.as_str()).is_file()
            ).cloned()
        });
        if source_ref.is_some() && source_real.is_none() {
            return format!("Error: generate_image could not find an attached image matching \
                source_image=\"{}\". Images the user attached anywhere in this conversation can be \
                edited — pass the exact /work/uploads/<filename> path shown for the attachment (or \
                omit source_image to generate a brand-new image from the prompt).", source_ref.unwrap());
        }
        let editing = source_real.is_some();
        // Optional inpaint mask: prefer a mask the USER brushed onto this attachment (stored in
        // AppState by basename); else rasterize the model's `mask_regions` DSL against the source's
        // real dimensions. Either way only the masked region changes and the rest is preserved.
        let mask_png: Option<Vec<u8>> = source_real.as_ref().and_then(|src| {
            let base = std::path::Path::new(src).file_name().and_then(|n| n.to_str())?.to_string();
            // Tier 2 — user-brushed mask.
            if let Some(s) = app.try_state::<crate::AppState>() {
                if let Some((_, bytes)) = s.attachment_masks.lock().unwrap().iter().find(|(n, _)| n == &base) {
                    return Some(bytes.clone());
                }
            }
            // Tier 1 — model-drawn region DSL.
            let dsl = args.get("mask_regions").and_then(|v| v.as_str()).map(str::trim).filter(|s| !s.is_empty())?;
            let dim = imagesize::size(src).ok()?;
            crate::image_gen::rasterize_mask(dsl, dim.width as u32, dim.height as u32)
        });
        let masked = mask_png.is_some();
        let cfg = app.try_state::<crate::AppState>()
            .map(|s| s.image_gen_config.lock().unwrap().clone())
            .unwrap_or_default();
        return match crate::image_gen::generate(&cfg, &prompt, negative, size, steps, seed, source_real.as_deref(), strength, mask_png.as_deref()).await {
            Ok(png) => {
                let mut file_note = String::new();
                if !silent {
                    if let Some(s) = app.try_state::<crate::AppState>() {
                        // Show inline.
                        let b64 = B64.encode(&png);
                        s.pending_tool_images.lock().unwrap().push(format!("data:image/png;base64,{b64}"));
                        // Stable per-turn number so the model can reference the SAME image again from a
                        // file (results dir is staged into /work/data) without re-generating it.
                        let n = { let mut c = s.turn_image_count.lock().unwrap(); *c += 1; *c };
                        let dir = tool_results_dir();
                        let _ = std::fs::create_dir_all(&dir);
                        let fname = format!("generated_image_{n}.png");
                        if std::fs::write(dir.join(&fname), &png).is_ok() {
                            file_note = format!(" This is generated image #{n} this turn — to reuse it \
                                WITHOUT re-generating: in run_python read /work/data/{fname} (e.g. \
                                python-pptx add_picture), or embed it in a create_artifact HTML slide \
                                as <img src=\"{{{{figure:{n}}}}}\"> (that figure number is correct if \
                                you have generated only images this turn; if you also made charts, \
                                {{{{figure:K}}}} counts all inline charts+images in creation order).");
                        }
                    }
                }
                let verb = if masked { "Edited the selected region of the attached image (the rest is unchanged)" }
                    else if editing { "Edited the attached image" }
                    else { "Generated an image" };
                format!("[{verb} for the prompt \"{prompt}\". It is displayed inline to \
                    the user above.{file_note} Refer to it naturally as \"shown above\"; do NOT output \
                    an image URL or markdown image, and do NOT claim you cannot show images.]")
            }
            Err(e) => format!("Image generation error: {e}"),
        };
    }

    // wiki_search needs the Ollama backend to embed, and dispatch_builtin has no handle on it.
    // Intercept here (same shape as generate_image above) rather than threading a backend
    // through every dispatch_builtin call site.
    if name == "wiki_search" {
        let backend = app
            .try_state::<crate::AppState>()
            .map(|s| s.backend.lock().unwrap().clone());
        return match backend {
            Some(b) => crate::wiki::wiki_search_hybrid(args, &b).await,
            None => crate::wiki::wiki_search(args),
        };
    }

    // 1. Try built-in tools first
    let builtin_names = ["read_file","write_file","list_files","search_files",
        "search_in_files","get_file_info","list_directory_tree","create_directory",
        "move_file","delete_file","find_old_files","web_search","fetch_webpage","compose_email",
        "get_current_datetime",
        "wiki_list","wiki_search","wiki_read","wiki_write","wiki_patch","wiki_delete",
        "wiki_append","wiki_lint"];
    if builtin_names.contains(&name) {
        // File tools may also touch the user's attached files (`sandbox_paths`), not just the
        // configured sandbox dirs — so "read/summarise this attached PDF" works even when the
        // file lives outside a sandbox folder. `check_path` still rejects anything else.
        let mut allow = allowed_dirs.to_vec();
        allow.extend(sandbox_paths.iter().cloned());
        let result = crate::tools::dispatch_builtin(name, args, &allow, web_search_results).await;

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

/// Handle a `run_python` call: enforce the per-session execution permission, then run the code in
/// the Pyodide worker (webview), with the attached files (`sandbox_paths`) staged into the
/// sandbox workspace (/work/uploads and /work/data).
/// Result of a Pyodide `run_python` run, handed back from the webview worker via
/// `respond_python_result`. `output` is stdout (+ any error/traceback appended); `images`
/// are base64 PNGs of matplotlib figures to render inline.
#[derive(Debug, Clone)]
pub struct PyResult {
    pub output: String,
    pub images: Vec<String>,
}

/// Give every tool call that never produced a result a matching `tool` message.
///
/// Pressing Stop breaks out of the dispatch loop before the result is recorded, which leaves an
/// assistant turn whose `tool_calls` have no answer. Both the wire format and the model read that
/// as "the call never came back": on the user's next message the model re-issues it, announcing
/// that the results didn't come through. Closing the calls out says cancelled instead of silent.
pub(crate) fn close_unanswered_tool_calls(
    conv: &mut Vec<WireMessage>,
    calls: &[WireToolCall],
    answered: usize,
) {
    for call in calls.iter().skip(answered) {
        conv.push(WireMessage {
            role: "tool".into(),
            content: Some(
                "[Cancelled: the user pressed Stop before this call returned, so its result is \
                 unknown. Do NOT re-run it or report it as failed — only run it again if the user \
                 asks for it again.]".into(),
            ),
            tool_calls: None,
            tool_call_id: call.id.clone(),
            name: Some(call.function.name.clone()),
            images: None,
        });
    }
}

/// Stage the run's attached files into the Pyodide workspace payload: each attachment becomes
/// `uploads/<filename>`, base64-encoded. Directories (e.g. the offloaded-results dir) are skipped
/// in this pass — offloaded-result staging into `/work/data` is a follow-up.
fn stage_python_files(sandbox_paths: &[String]) -> Vec<serde_json::Value> {
    use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
    let mut files = Vec::new();
    for p in sandbox_paths {
        let path = std::path::Path::new(p);
        if path.is_file() {
            // Attached files → /work/uploads. Documents (PDF/Word) are staged as their EXTRACTED
            // TEXT so open() returns readable content (Pyodide can't parse PDF/Word); data files
            // (CSV/Excel/JSON, images) are staged as raw bytes for pandas/etc.
            let name = match path.file_name().and_then(|n| n.to_str()) { Some(n) => n, None => continue };
            if let Some(text) = crate::tools::extract_document_text(p) {
                files.push(serde_json::json!({ "path": format!("uploads/{name}"), "b64": B64.encode(text.as_bytes()) }));
            } else if let Ok(bytes) = std::fs::read(path) {
                files.push(serde_json::json!({ "path": format!("uploads/{name}"), "b64": B64.encode(&bytes) }));
            }
        } else if path.is_dir() {
            // A staged dir → /work/data, EXCEPT the skill-resources dir which goes to /work/skills so
            // a skill's bundled template/helper is where its instructions say. The offloaded-results
            // dir → /work/data, so run_python can read a large tool result that didn't fit in context.
            let named = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let dest_prefix = if path == skill_staging_dir() { "skills" }
                else if named == "artifacts" || path == artifact_data_dir() { "artifacts" }
                else { "data" };
            if let Ok(entries) = std::fs::read_dir(path) {
                for e in entries.flatten() {
                    let ep = e.path();
                    if !ep.is_file() { continue; }
                    let name = match ep.file_name().and_then(|n| n.to_str()) { Some(n) => n, None => continue };
                    if let Ok(bytes) = std::fs::read(&ep) {
                        files.push(serde_json::json!({ "path": format!("{dest_prefix}/{name}"), "b64": B64.encode(&bytes) }));
                    }
                }
            }
        }
    }
    files
}

async fn dispatch_run_python<R: tauri::Runtime>(
    args: &serde_json::Value,
    _allowed_dirs: &[String],
    sandbox_paths: &[String],
    silent: bool,
    // Set only by a scheduled job whose own config opts into code execution. Threaded rather
    // than held in shared state because jobs can run concurrently, and a shared flag would let
    // one job's permission apply to another's run.
    allow_code_exec: bool,
    // Wipe /work before running? True only on the first run_python of a turn, so files written by
    // an earlier call this turn (e.g. chart PNGs) persist into later calls.
    reset: bool,
    app: &AppHandle<R>,
) -> String {
    let code = args["code"].as_str().unwrap_or("").to_string();
    if code.trim().is_empty() {
        return "Error: run_python requires a non-empty 'code' string.".into();
    }

    // Permission gate (session toggle): once approved, stays unlocked until the app restarts.
    // A background job can never answer a prompt, so it has exactly two outcomes: its own
    // `allow_code_exec` opt-in stands in for the approval, or it is refused. It must never reach
    // the prompt below — nobody is there to answer, so the wait would always end in a denial.
    if let Some(state) = app.try_state::<crate::AppState>() {
        let unlocked = *state.code_exec_unlocked.lock().unwrap();
        if silent {
            if !allow_code_exec {
                return "Error: code execution requires interactive approval and is disabled in \
                        background jobs. To let THIS job run code, switch on its \
                        'allow code execution' setting.".into();
            }
            // Opted in: proceed without prompting, and WITHOUT setting the session-wide unlock —
            // one job's permission must not silently grant code execution to interactive chat or
            // to another job that was never opted in.
        } else if !unlocked {
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

    // Single runtime: hand the code + staged files to the Pyodide worker in the webview and await
    // its result (stdout + chart images + files written to /work/out). Both interactive chat and
    // background jobs run here — the window is only hidden (never destroyed, see lib.rs), so the
    // webview hosting the worker is alive whenever the in-process scheduler runs a job.
    let Some(state) = app.try_state::<crate::AppState>() else {
        return "Error: run_python is unavailable (no app state).".into();
    };
    let files = stage_python_files(sandbox_paths);
    let request_id = state.python_request_seq.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let (tx, rx) = tokio::sync::oneshot::channel::<PyResult>();
    state.pending_python_result.lock().unwrap().insert(request_id, tx);
    let _ = app.emit("run-python-request",
        serde_json::json!({ "request_id": request_id, "code": code, "files": files, "reset": reset }));

    // Interactive runs abort promptly on Stop (so a slow run can't resume into a new chat).
    // Background jobs manage their own lifecycle and must NOT obey the interactive stop flag
    // (which may be stale from a previous chat) — they just wait with a hard timeout.
    let mut rx = rx;
    let py: Option<PyResult> = if silent {
        tokio::time::timeout(std::time::Duration::from_secs(300), &mut rx).await.ok().and_then(|r| r.ok())
    } else {
        // This run's cancel token (a clone of the current slot). If a newer run supersedes this
        // one, it sets THIS token true, so a slow run_python aborts rather than outliving its chat.
        let cancel = state.cancel.lock().unwrap().clone();
        let deadline = tokio::time::sleep(std::time::Duration::from_secs(300));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                r = &mut rx => break r.ok(),
                _ = &mut deadline => break None,
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                    if cancel.load(std::sync::atomic::Ordering::SeqCst) { break None; }
                }
            }
        }
    };
    state.pending_python_result.lock().unwrap().remove(&request_id);

    match py {
        Some(py) => {
            // Chart/image outputs (already full `data:` URLs) ride the existing inline-image
            // path — the agent loop takes them for the agent-tool-result event.
            let n_images = py.images.len();
            if n_images > 0 {
                state.pending_tool_images.lock().unwrap().extend(py.images);
            }
            // A run that only plots has empty stdout — confirm success (with the inline-render
            // note) so the model doesn't mistake it for a failure and re-run it.
            let mut out = py.output;
            if n_images > 0 {
                if !out.trim().is_empty() { out.push('\n'); }
                out.push_str(&format!("[Success: {n_images} image(s)/chart(s) rendered inline to the user. \
                    Do not re-run; in your reply refer to the chart in words — do NOT embed a markdown image, link, or file path.]"));
            } else if out.trim().is_empty() {
                out = "[Ran successfully with no text output.]".into();
            }
            out
        }
        None => "Error: run_python was cancelled or did not respond (the runtime may still be loading — try again).".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: Option<&str>, name: Option<&str>) -> WireMessage {
        WireMessage {
            role: role.into(),
            content: content.map(str::to_string),
            tool_calls: None,
            tool_call_id: None,
            name: name.map(str::to_string),
            images: None,
        }
    }

    /// A coordinate polyline is the payload that broke this: `len / 4` called a 17k-char route
    /// array 4.3k tokens when it was really 16k, so an oversized wire slipped past the budget and
    /// Ollama front-truncated the system prompt. Both real counts below are measured
    /// `prompt_eval_count` values from qwen3 on the actual TfL journey that failed.
    /// The status line is derived from this, so it must count the tool schemas too — they are a
    /// large fixed cost per step and were a third of the prompt in the measured TED case.
    #[test]
    fn wire_tokens_counts_messages_and_tool_schemas() {
        let wire = vec![msg("system", Some(&"a".repeat(4000)), None),
                        msg("user", Some(&"b".repeat(400)), None)];
        let bare = wire_tokens(&wire, &[]);
        assert!(bare > 800 && bare < 1200, "message estimate was {bare}");
        let with_tools = wire_tokens(&wire, &[tool("some_tool_with_a_schema")]);
        assert!(with_tools > bare, "tool schemas must be counted: {with_tools} vs {bare}");
    }

    #[test]
    fn estimate_tokens_tracks_real_counts_for_prose_and_coordinate_data() {
        let coords: String = std::iter::repeat("[51.44134339536, 0.36633931267], ")
            .take(523).collect::<String>();   // ≈17,259 chars, measured at ~16,088 tokens
        let est = estimate_tokens(&coords);
        assert!((14_500..=17_500).contains(&est), "coordinate estimate {est} should be near 16k");
        // The old chars/4 rule is the bug this replaces — it lands nowhere near.
        assert!(coords.len() / 4 < est / 2, "chars/4 should be the gross under-count we fixed");

        // Prose still lands near the familiar ~4 chars/token, so ordinary chat isn't over-trimmed.
        let prose = "The quick brown fox jumps over the lazy dog and then writes a short report about \
                     everything it saw along the way, in plain English prose. ".repeat(40);
        let ratio = prose.len() as f64 / estimate_tokens(&prose) as f64;
        assert!((3.0..=5.0).contains(&ratio), "prose ratio {ratio:.2} should stay near 4 chars/token");
    }

    /// The failing route map in miniature: a huge tool result sits inside the protected recent
    /// window, so the oldest-first pass can't reach it and the wire stays over budget. It must be
    /// elided as a last resort — but never the newest result, which the model is about to use.
    #[test]
    fn fit_wire_to_context_elides_a_huge_result_inside_the_recent_window_but_not_the_newest() {
        let huge = "1".repeat(60_000);   // digit-dense: ~63k tokens, way over any budget
        let newest = "2".repeat(20_000);
        let mut wire = vec![
            msg("system", Some("SYSTEM"), None),
            msg("user", Some("map the route"), None),
            msg("assistant", Some("working"), None),
            msg("tool", Some(&huge), Some("run_python")),      // 3rd from the end
            msg("assistant", Some("now the map"), None),
            msg("tool", Some(&newest), Some("run_python")),     // last — must survive
        ];
        let elided = fit_wire_to_context(&mut wire, &[], 32768, None);
        assert_eq!(elided, 1, "the huge in-window result should be elided");
        assert!(wire[3].content.as_deref().unwrap().contains("very large tool result"));
        assert_eq!(wire[5].content.as_deref(), Some(newest.as_str()), "newest result must survive");
        assert_eq!(wire[0].content.as_deref(), Some("SYSTEM"));
        assert_eq!(wire[1].content.as_deref(), Some("map the route"));
    }

    /// The escape hatch must stay shut for ordinary turns: nothing in the recent window is touched
    /// when the wire already fits, so a normal conversation keeps every tool result intact.
    #[test]
    fn fit_wire_to_context_leaves_a_recent_large_result_alone_when_within_budget() {
        let big = "x".repeat(40_000); // letters: ~8.4k tokens, comfortably inside a 128k window
        let mut wire = vec![
            msg("system", Some("SYSTEM"), None),
            msg("user", Some("q"), None),
            msg("tool", Some(&big), Some("t")),
        ];
        assert_eq!(fit_wire_to_context(&mut wire, &[], 131072, None), 0);
        assert_eq!(wire[2].content.as_deref(), Some(big.as_str()));
    }

    #[test]
    fn fit_wire_to_context_elides_old_tool_results_protecting_system_user_and_recent() {
        let big = "x".repeat(24_000); // ~6k tokens each
        let mut wire = vec![
            msg("system", Some("THE SYSTEM PROMPT WITH ALL THE RULES"), None),
            msg("user", Some("Give me the census religion breakdown for Pelham ward"), None),
        ];
        // 8 tool-call / tool-result pairs — well over an 8k-ish budget.
        for i in 0..8 {
            let mut a = msg("assistant", None, None);
            a.tool_calls = Some(vec![WireToolCall {
                id: None,
                function: WireToolFunction { name: format!("t{i}"), arguments: serde_json::json!({}) },
            }]);
            wire.push(a);
            wire.push(msg("tool", Some(&big), Some(&format!("t{i}"))));
        }

        let tools: Vec<ToolSchema> = Vec::new();
        let elided = fit_wire_to_context(&mut wire, &tools, 32768, None);
        assert!(elided > 0, "expected some old tool results to be elided");

        // System prompt and the original task are never touched.
        assert_eq!(wire[0].content.as_deref(), Some("THE SYSTEM PROMPT WITH ALL THE RULES"));
        assert_eq!(wire[1].content.as_deref(), Some("Give me the census religion breakdown for Pelham ward"));
        // The most recent tool result (last message) is preserved verbatim.
        assert_eq!(wire.last().unwrap().content.as_deref(), Some(big.as_str()));
        // A second pass finds it already within budget — proving the first brought it under.
        assert_eq!(fit_wire_to_context(&mut wire, &tools, 32768, None), 0);
    }

    #[test]
    fn looks_like_continuation_trips_on_preambles_not_final_answers() {
        // The exact test-#4 case: had the data, narrated the next step, ended on a colon.
        assert!(looks_like_continuation(
            "I have the settlement and coordinates. Now I need to fetch the building footprint for the TOID using OS Features. Let me get that:"));
        // The reported real case: gathered everything, announced compiling, then stopped (ends "now").
        assert!(looks_like_continuation(
            "You're absolutely right — I gathered all the data but never compiled it into a report. Let me do that now."));
        // Bare "…now" and short intent endings, and intent+action announcements.
        assert!(looks_like_continuation("Great, I have everything I need. Let me compile the report now."));
        assert!(looks_like_continuation("I'll put that together into a summary."));
        assert!(looks_like_continuation("Let me look up the USRN for this property."));
        assert!(looks_like_continuation("Next, I'll call the crime API to get the figures."));
        // Genuine final answers must NOT trip it.
        assert!(!looks_like_continuation(
            "The property is on USRN 8400071 (Bristol). Its building polygon is TOID osgb1000005572568. That completes the lookup."));
        assert!(!looks_like_continuation("The answer is 42."));
        assert!(!looks_like_continuation("Let me know if you'd like anything else."));
        assert!(!looks_like_continuation("Here is the full report. Let me highlight the key point: Bristol is the largest."));
        assert!(!looks_like_continuation(""));
    }

    #[test]
    fn fit_wire_to_context_noop_when_within_budget_or_uncapped() {
        let mut wire = vec![
            msg("system", Some("short system"), None),
            msg("user", Some("hi"), None),
            msg("tool", Some("small result"), Some("t")),
        ];
        let tools: Vec<ToolSchema> = Vec::new();
        assert_eq!(fit_wire_to_context(&mut wire, &tools, 32768, None), 0);
        // num_ctx <= 0 means "no cap" — never trims regardless of size.
        let mut big_wire = vec![msg("tool", Some(&"x".repeat(200_000)), Some("t"))];
        assert_eq!(fit_wire_to_context(&mut big_wire, &tools, 0, None), 0);
    }

    #[test]
    fn anthropic_backend_uses_openai_shape_and_requires_max_tokens() {
        let be = Backend { kind: ProviderKind::Anthropic,
            base_url: "https://api.anthropic.com/v1".into(), api_key: Some("sk-ant-x".into()) };
        // OpenAI-shaped URLs + request/response dialect.
        assert!(be.is_openai());
        assert_eq!(be.chat_url(), "https://api.anthropic.com/v1/chat/completions");
        assert_eq!(be.models_url(), "https://api.anthropic.com/v1/models");
        // max_tokens defaulted when the caller sets none (Anthropic's compat endpoint requires it).
        let body = build_chat_request(&be, "claude-opus-4-8", &[], &[], None, None, true);
        assert_eq!(body["max_tokens"], serde_json::json!(4096));
        assert_eq!(body["model"], "claude-opus-4-8");
        // A caller-provided value wins.
        let opts = ChatOptions { num_predict: Some(1000), ..Default::default() };
        let body2 = build_chat_request(&be, "claude-opus-4-8", &[], &[], Some(&opts), None, true);
        assert_eq!(body2["max_tokens"], serde_json::json!(1000));
    }

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
            false, // allow_code_exec: not a job
            20,
            0, // web_tool_cap: default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false, // discover_tools: exercise the legacy pre-flight path
            vec![], // skills
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

    #[test]
    fn search_tools_finds_by_name_and_excludes_builtins() {
        let groups = vec![
            group(BUILTIN_GROUP, &["read_file", "web_search"]),
            group("Police API", &["police_street_crimes", "police_forces"]),
            group("Maps", &["static_map_image"]),
        ];
        let hits = search_tools("street crimes", &groups, 5);
        assert!(hits.iter().any(|(n, _)| n == "police_street_crimes"), "{hits:?}");
        // Built-ins are always available, so discovery search never returns them.
        assert!(!hits.iter().any(|(n, _)| n == "read_file"));
        // Empty query → a catalog sample of external tools only (no built-ins).
        let sample = search_tools("", &groups, 2);
        assert_eq!(sample.len(), 2);
        assert!(sample.iter().all(|(n, _)| n != "read_file" && n != "web_search"));
    }

    #[test]
    fn search_tools_loads_whole_group_when_label_matches() {
        // Naming a server ("excalidraw") loads its ENTIRE toolset — so a variant that wouldn't
        // keyword-match the rest of the query (e.g. the inline-app renderer) is still available.
        let groups = vec![
            group("Excalidraw (MCP)", &["excalidraw_render_app", "excalidraw_create_link"]),
            group("Police API", &["police_street_crimes"]),
        ];
        let hits = search_tools("draw a histogram with excalidraw", &groups, 12);
        assert!(hits.iter().any(|(n, _)| n == "excalidraw_render_app"), "{hits:?}");
        assert!(hits.iter().any(|(n, _)| n == "excalidraw_create_link"), "{hits:?}");
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
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0, app.handle(), true, false, 5,
            0, // web_tool_cap: default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false, // discover_tools: exercise the legacy pre-flight path
            vec![], // skills
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
        // Over the limit: preview + a run_python hint pointing at /work/data; the file holds the
        // raw JSON BODY (the "HTTP 200" wrapper openapi::execute adds is stripped so json.load works).
        let body = format!("[{}]", "\"a\",".repeat(2000).trim_end_matches(','));
        let wrapped = format!("HTTP 200\n{body}");
        let out = offload_tool_result(wrapped, "police_streetcrime", 100, &dir);
        assert!(out.contains("run_python"), "must nudge toward run_python");
        assert!(out.contains("/work/data/"), "must point at the /work/data sandbox path");
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

    /// Stop pressed mid-dispatch used to leave an assistant turn whose tool_calls had no
    /// matching `tool` message. The model read the gap as "no result yet" and re-ran the call
    /// on the user's next message — a stopped TED query firing again unasked.
    #[test]
    fn cancelled_tool_calls_get_an_explicit_result() {
        let calls = vec![
            WireToolCall { id: Some("a".into()),
                function: WireToolFunction { name: "ted_query".into(), arguments: serde_json::json!({}) } },
            WireToolCall { id: None,
                function: WireToolFunction { name: "web_search".into(), arguments: serde_json::json!({}) } },
        ];
        let mut conv: Vec<WireMessage> = Vec::new();
        close_unanswered_tool_calls(&mut conv, &calls, 0);

        assert_eq!(conv.len(), 2, "every dangling call needs an answer");
        assert!(conv.iter().all(|m| m.role == "tool"));
        assert_eq!(conv[0].name.as_deref(), Some("ted_query"));
        assert_eq!(conv[0].tool_call_id.as_deref(), Some("a"), "id must link back to the call");
        assert_eq!(conv[1].name.as_deref(), Some("web_search"));
        let text = conv[0].content.as_deref().unwrap();
        assert!(text.contains("Cancelled"), "must say cancelled, not look like an empty result");
        assert!(text.contains("Do NOT re-run"), "must not invite a retry");
    }

    /// Calls that DID return keep their real results — only the tail is backfilled.
    #[test]
    fn already_answered_tool_calls_are_left_alone() {
        let calls = vec![
            WireToolCall { id: None,
                function: WireToolFunction { name: "first".into(), arguments: serde_json::json!({}) } },
            WireToolCall { id: None,
                function: WireToolFunction { name: "second".into(), arguments: serde_json::json!({}) } },
        ];
        let mut conv: Vec<WireMessage> = Vec::new();
        close_unanswered_tool_calls(&mut conv, &calls, 1);
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0].name.as_deref(), Some("second"));

        // A fully answered step adds nothing at all.
        let mut none: Vec<WireMessage> = Vec::new();
        close_unanswered_tool_calls(&mut none, &calls, 2);
        assert!(none.is_empty());
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
            &conversation, vec![], vec![], &mcp, vec![], vec![], 10, 0, app.handle(), false, false, 20,
            0, // web_tool_cap: default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false, // discover_tools: exercise the legacy pre-flight path
            vec![], // skills
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
            app.handle(), false, false, 20,
            0, // web_tool_cap: default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false, // discover_tools: exercise the legacy pre-flight path
            vec![], // skills
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

    // ── Duplicate artifacts ───────────────────────────────────────────────────

    /// The matching rule from `agent_loop`'s duplicate-artifact guard, isolated so the policy is
    /// testable without standing up a run: a title already rendered this turn is a repeat.
    fn artifact_is_duplicate(seen: &[String], title: &str) -> bool {
        let t = title.trim().to_lowercase();
        !t.is_empty() && seen.iter().any(|s| *s == t)
    }

    #[test]
    fn second_artifact_with_the_same_title_is_refused() {
        let seen = vec!["daily task list".to_string()];
        // The observed failure: same title, model "refining" it into a second copy.
        assert!(artifact_is_duplicate(&seen, "Daily Task List"));
        assert!(artifact_is_duplicate(&seen, "  daily task list  "), "case and padding must not evade it");
    }

    #[test]
    fn a_genuinely_different_deliverable_still_renders() {
        let seen = vec!["daily task list".to_string()];
        assert!(!artifact_is_duplicate(&seen, "Closing Checklist"),
            "a second, different artifact in one turn is legitimate");
    }

    #[test]
    fn an_untitled_artifact_is_never_treated_as_a_duplicate() {
        // Empty titles would otherwise all collide with each other.
        let seen = vec!["".to_string()];
        assert!(!artifact_is_duplicate(&seen, ""));
        assert!(!artifact_is_duplicate(&seen, "   "));
    }

    // ── Discovery-mode tool cap ───────────────────────────────────────────────

    /// Mirrors the selection in `agent_loop`'s discovery branch: most-recently-used first, then
    /// most-recently-loaded, capped. Kept as a free function so the policy can be tested without
    /// standing up a whole agent run.
    fn discovery_pick(cap: usize, base: usize, last_used: &[&str], loaded: &[&str]) -> Vec<String> {
        let mut wanted: Vec<&str> = Vec::new();
        for n in last_used { if !wanted.contains(n) { wanted.push(n); } }
        for n in loaded.iter().rev() { if !wanted.contains(n) { wanted.push(n); } }
        let mut out = Vec::new();
        for name in wanted {
            if base + out.len() >= cap { break; }
            out.push(name.to_string());
        }
        out
    }

    #[test]
    fn discovery_cap_evicts_the_oldest_loaded_tools() {
        // 20 built-ins already in the request, cap 25 → room for 5 more.
        let loaded = ["a", "b", "c", "d", "e", "f", "g", "h"]; // h loaded most recently
        let got = discovery_pick(25, 20, &[], &loaded);
        assert_eq!(got, vec!["h", "g", "f", "e", "d"], "keeps newest, drops the oldest loaded");
    }

    #[test]
    fn discovery_cap_never_drops_a_tool_used_last_step() {
        // "a" is the oldest load but was just called — dropping it would strand a tool chain.
        let loaded = ["a", "b", "c", "d", "e", "f"];
        let got = discovery_pick(23, 20, &["a"], &loaded);
        assert_eq!(got[0], "a", "the tool called last step outranks recency of loading");
        assert_eq!(got.len(), 3, "and the cap still binds");
    }

    #[test]
    fn discovery_cap_with_no_room_adds_nothing() {
        let loaded = ["a", "b"];
        assert!(discovery_pick(20, 20, &[], &loaded).is_empty());
    }

    /// The bug this replaced: the loaded set only grew, so repeated find_tools calls piled every
    /// discovered tool into every later step.
    #[test]
    fn discovery_cap_bounds_a_long_run_of_discoveries() {
        let loaded: Vec<&str> = (0..70).map(|i| Box::leak(format!("t{i}").into_boxed_str()) as &str).collect();
        let got = discovery_pick(40, 20, &[], &loaded);
        assert_eq!(got.len(), 20, "70 discovered tools must not all reach the model");
    }

    // ── Runaway repetition ────────────────────────────────────────────────────

    /// The observed failure: one local model emitted this fragment several hundred times
    /// and streamed for over twenty minutes, because no guard runs inside a single response.
    #[test]
    fn catches_the_repetition_that_hung_a_real_run() {
        let text = "Matching notices: ".to_string()
            + &"\"AI SENSI DELL'ART. 1, CO. 668\", ".repeat(12);
        let unit = runaway_repetition(&text).expect("should have caught the loop");
        assert!(unit.contains("AI SENSI"), "unit should be the repeating fragment: {unit:?}");
    }

    #[test]
    fn ignores_text_that_merely_repeats_a_few_times() {
        // Four repeats is under the threshold — a table or a list can legitimately do this.
        let text = "row one | row two | ".repeat(4);
        assert_eq!(runaway_repetition(&text), None);
    }

    #[test]
    fn ignores_formatting_runs() {
        // Rules, ellipses and indentation repeat constantly in Markdown and are not a loop.
        assert_eq!(runaway_repetition(&"-".repeat(400)), None);
        assert_eq!(runaway_repetition(&"          ".repeat(60)), None);
        assert_eq!(runaway_repetition(&". . . . ".repeat(60)), None);
    }

    #[test]
    fn ignores_ordinary_prose_and_short_text() {
        assert_eq!(runaway_repetition(""), None);
        assert_eq!(runaway_repetition("A short answer about tenders."), None);
        let prose = "The notice concerns an artificial intelligence platform for the \
                     public sector, covering retrieval, agents and audit. ".repeat(3);
        assert_eq!(runaway_repetition(&prose), None);
    }

    /// Only the tail matters — a long, healthy answer that degenerates at the very end must
    /// still be caught, and the preceding text must not mask it.
    #[test]
    fn catches_a_loop_that_starts_late_in_a_long_answer() {
        let good = "Genuine analysis of the tender landscape. ".repeat(40);
        let text = good + &"stuck stuck stuck ".repeat(20);
        assert!(runaway_repetition(&text).is_some());
    }

    /// Cost must not grow with the response, or the check would slow down exactly the runs
    /// it exists to protect.
    #[test]
    fn stays_cheap_on_a_very_long_response() {
        let text = "Ordinary varied sentence number one. ".repeat(20_000);
        let start = std::time::Instant::now();
        let _ = runaway_repetition(&text);
        assert!(start.elapsed().as_millis() < 50, "tail-only check should be fast");
    }

    #[test]
    fn handles_multibyte_text_without_panicking() {
        // Char-based, not byte-based: a naive slice here would panic mid-codepoint.
        let text = "Ø£ØªÙ Ø¯Ø±Ø³Ø© ".repeat(30);
        let _ = runaway_repetition(&text);
        let mixed = "tender — Ünïcödé ".repeat(30);
        assert!(runaway_repetition(&mixed).is_some());
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
            app.handle(), false, false, 20,
            0, // web_tool_cap: default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            false, // discover_tools: exercise the legacy pre-flight path
            vec![], // skills
        ).await;

        assert!(result.is_ok(), "run should survive an unsupported-tools endpoint: {result:?}");
        let convo = conversation.lock().unwrap();
        assert_eq!(convo.last().unwrap().content.as_deref(), Some("Answered without tools."));
    }
}
