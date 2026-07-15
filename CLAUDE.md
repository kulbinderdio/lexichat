# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**LexiChat** — a Tauri 2.x desktop app (React/TypeScript frontend, Rust backend) that provides a local AI chat interface powered by Ollama, with support for built-in file tools, OpenAPI integrations, and MCP servers.

## Build & Run

### Prerequisites
- Node.js 18+ and npm 9+
- Rust stable (via rustup)
- Ollama running locally (`ollama serve`)

### Development
```bash
npm install
npm run tauri dev       # Full Tauri dev with HMR (Vite on port 1420)
```

### Production builds
```bash
# macOS universal binary
rustup target add aarch64-apple-darwin x86_64-apple-darwin
npm run tauri build -- --target universal-apple-darwin

# Windows / Linux
npm run tauri build
```

See `BUILDING.md` for platform-specific dependencies (Linux gtk/webkit), code signing, and CI/CD.

## Architecture

```
src/                        React/TypeScript frontend
  App.tsx                   Main chat UI, message state, model/profile selection, Tauri event listeners
  AdminPanel.tsx            All settings: profiles, tools, OpenAPI specs, MCP servers, auth, sandbox
  DebugPanel.tsx            Real-time agentic trace viewer (steps, tool calls, durations)
  App.css                   All styling

src-tauri/src/
  lib.rs                    Tauri command handlers, AppState, OAuth2 auth-code flow, allowed dirs persistence
  ollama.rs                 Ollama REST API, streaming chat, agent loop (up to 10 steps), tool dispatch routing
  tools.rs                  12 built-in tools (file ops + web search), their JSON schemas, sandbox enforcement
  openapi.rs                OpenAPI 3.0 spec parsing → tool generation, HTTP execution with auth
  mcp.rs                    MCP 2024-11-05 protocol — stdio and HTTP transports, JSON-RPC, tool listing/invocation
```

## Key data flows

### Chat → Agent loop
1. `App.tsx send()` collects message + attachments + enabled tools → `invoke("send_message", { args: {...} })`
2. `lib.rs send_message` base64-encodes images, gathers all active tool schemas → calls `ollama::agent_loop()`
3. `ollama::agent_loop` (0..MAX_STEPS=10): streams LLM response via `agent-token` events, detects tool calls, dispatches them, appends results, loops
4. `dispatch_tool()` routes by name prefix: built-ins first, then OpenAPI tools, then MCP tools
5. Frontend listeners in `App.tsx` handle `agent-token`, `agent-tool-call`, `agent-tool-result`, `agent-done`

### Tool name sanitization & prefixing
All tool names must be `[a-z0-9_]`, no leading/trailing underscores, max 64 chars (Ollama limit).
- **Built-in tools**: no prefix (`read_file`, `list_files`, `web_search`, etc.)
- **OpenAPI tools**: `{service_prefix}_{operation_id}` — prefix is lowercased service name with " API"/" server" stripped and spaces → underscores. e.g. "Google Drive API" → `google_drive_`
- **MCP tools**: `{server_name_prefix}_{raw_tool_name}` — same sanitization applied to server name

`openapi::tool_prefix()` and `openapi::sanitize_tool_name()` are shared by both `openapi.rs` and `mcp.rs`.

### OpenAPI spec registration
1. User pastes spec JSON + base URL + auth in AdminPanel → `invoke("register_openapi_spec")`. Frontend also adds it to the global `toolRegistry.openapiSpecs` (see Profile scoping); the spec is not owned by a profile.
2. `openapi::parse_spec()` iterates `paths`, generates one tool per HTTP operation with prefixed/sanitized name and JSON schema from parameters + request body. **`base_url` is applied to every operation** (`execute` does `format!("{base}{path}")`); per-operation `servers` in the spec are ignored — so one spec = one host.
3. `AppState.openapi_specs` holds `RegisteredSpec { id, title, base_url, auth, tools }` — but only the **active profile's enabled subset**, (re)pushed by `syncServers()` → `set_openapi_specs()` on each profile switch.
4. At execution, `openapi::execute()` substitutes path params, appends query params, applies auth, returns JSON

### MCP server connection
1. User enters name + command (shell path) or HTTP URL + env vars + auth → `invoke("add_mcp_server")`
2. `MCPConnection::connect()`: spawns process (stdio) or connects HTTP; sends `initialize` → `notifications/initialized` → `tools/list`
3. Tools are stored with both `raw_name` (sent to server) and `name` (prefixed, used by model)
4. `call_tool()` looks up `raw_name` from the prefixed `name` before sending JSON-RPC `tools/call`

### Profile scoping (registry model)
Tools are **not** copied per profile. They live once in a global registry —
`settings.toolRegistry` in `App.tsx` with `{ openapiSpecs, mcpServers, sparqlEndpoints }` —
and each profile only holds **enabled-ID references**: `enabledOpenapiSpecIds`,
`enabledMcpServerIds`, `enabledSparqlEndpointIds`, plus `enabledTools` (a built-in-tool
on/off map). Registering a spec/server (the OpenAPI/MCP/SPARQL tabs edit the global registry)
auto-enables it for the *active* profile; the per-profile enable checkboxes live in the
**profile editor** (Profiles tab), not the tool tabs. Legacy per-profile `openapiSpecs`/
`mcpServers` arrays are migrated to `enabled*Ids` on load (`App.tsx` ~line 378).

On profile switch, `syncServers()`:
1. Filters the registry down to the active profile's enabled (and not-`enabled:false`) IDs, then pushes that subset to the Rust backend via `set_mcp_servers()` (drops old connections, connects new), `set_openapi_specs()`, and `set_sparql_endpoints()`. So `AppState` holds only the active profile's enabled tools.
2. MCP is *additionally* filtered at call time: `send_message` receives `enabled_mcp_server_ids` and narrows again inside `dispatch`.
3. Model is set to the profile's model only if that model exists in the fetched Ollama models list.
4. Conversation history is reset.

## Auth system (`mcp.rs`)

`AuthConfig` is an internally-tagged serde enum (`#[serde(tag = "type")]`) with variants: `none`, `bearer`, `apikey`, `basic`, `oauth2`. All variant names are lowercase (explicit `#[serde(rename)]` on each variant).

- `apply()` — sync, handles all types except OAuth2
- `apply_async()` — async, fetches OAuth2 token then applies
- OAuth2 supports: stored access_token (from browser auth-code flow) with client-credentials fallback
- OAuth2 browser flow: `oauth2_authorize` Tauri command binds loopback TCP listener on random port, opens browser, waits for callback, exchanges code for tokens

Tauri IPC convention: all commands take a single `args` struct argument: `invoke("command_name", { args: { snake_case_fields } })`. This matches serde's default snake_case deserialization.

## State persistence

- All settings stored in `localStorage` via `saveSettings()` / `loadSettings()` in `App.tsx`. Tool definitions (OpenAPI specs, MCP servers, SPARQL endpoints) live once in the global `toolRegistry`; profiles reference them by ID (see Profile scoping), not by copy
- Allowed sandbox directories persisted to `~/.local/share/lexichat/allowed_dirs.json` via Rust (`dirs` crate)
- OAuth2 tokens stored in the `AuthConfig` within the profile in localStorage

## Built-in tools (`tools.rs`)

File tools: `read_file`, `write_file`, `list_files`, `search_files`, `search_in_files`, `get_file_info`, `list_directory_tree`, `create_directory`, `delete_file`, `move_file`, `copy_file`
Web: `web_search` (DuckDuckGo free API, no key required)

All file tools enforce the sandbox: `check_path()` canonicalizes the path and verifies it starts with an allowed directory. Images are blocked from `read_file` — users must attach them via the UI.

Tool results are truncated to 6000 chars to avoid overwhelming the LLM context.

## Frontend ↔ Backend events

Backend emits via `app.emit()`, frontend listens via Tauri `listen()` in `useEffect`:

| Event | Payload | Purpose |
|-------|---------|---------|
| `agent-token` | `{ delta }` | Streamed text chunk |
| `agent-tool-call` | `{ name, args_pretty }` | Tool invocation started |
| `agent-tool-result` | `{ name, result }` | Tool result (truncated) |
| `debug-step-start` | `{ step, tools }` | DebugPanel: step began |
| `debug-step-done` | `{ step, text, duration_ms }` | DebugPanel: step complete |
| `debug-run-done` | `{ total_ms, error? }` | DebugPanel: run complete |
| `agent-retry` | `{ step, attempt, error }` | Step is being re-sampled after an unparseable tool call; frontend drops the failed attempt's partial text |
| `agent-done` | `{ error? }` | Agent loop finished |

## Non-obvious behaviours

- **Empty response nudging**: if the model returns empty text with no tool calls at step 0, the agent sends a follow-up nudge message once to prompt a response
- **Malformed tool-call retry**: models that emit XML-dialect tool calls (`<function=x><parameter=y>`, e.g. Qwen) sometimes drop a closing tag, and Ollama rejects the whole response with a Go parser error (`XML syntax error … element <parameter> closed by </function>`). This is a sampling slip, not a real failure, so `agent_loop` re-samples the step up to `MALFORMED_TOOL_CALL_RETRIES` (2) times before surfacing the error — otherwise one bad sample kills a run that had already done its work
- **HTTP error surfacing**: `stream_chat` checks `resp.status().is_success()` and scans stream lines for `{"error":...}` — errors are returned as `Err(String)` so they surface in chat rather than silently dropping
- **Model not found**: profile's stored model is only applied if it exists in the current Ollama models list (prevents stale model names overriding the UI selection)
- **MCP HTTP transport**: supports both plain JSON and SSE (`text/event-stream`) responses for JSON-RPC
- **OpenAPI `operationId` missing**: if an operation has no `operationId`, one is synthesised from HTTP method + path
- **Tool count limit**: `maxTools` setting (default 30) caps how many tool schemas are sent to Ollama per request to stay within context limits
