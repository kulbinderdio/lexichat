# TESTING.md

LexiChat has four automated test layers, all headless and runnable with **no external
services** (Ollama/MCP/SPARQL are mocked or stubbed). CI runs them on every push/PR
(`.github/workflows/test.yml`); `build.yml` stays release-only.

## Running the tests

```bash
npm run test          # Frontend unit + component tests (Vitest + jsdom + RTL)
npm run test:rust     # Rust unit + HTTP-integration tests (cargo test)
npm run test:all      # Both of the above
npm run test:e2e      # Playwright UI e2e (headless Chromium, mocked Tauri IPC)

npm run test:watch    # Frontend in watch mode
npm run test:coverage # Frontend coverage report
```

`test:e2e` starts the Vite dev server automatically (see `playwright.config.ts`).
First run needs the browser: `npx playwright install chromium`.

## The four layers

### 1. Rust unit tests — pure logic
`#[cfg(test)] mod tests` at the bottom of each source file. Cover parsing, serde
round-trips, sanitization, the SPARQL read-only guard, MCP-App UI detection
(`extract_ui`), sandbox path-gating (`tools::check_path`), the Python sandbox
(`sandbox.rs`), wiki FS ops, and the job scheduler's cron logic (`jobs::is_due`).

### 2. Rust HTTP-integration tests — mocked servers (`#[tokio::test]`)
Also in-file, using the `wiremock` dev-dependency to spin up an in-process HTTP
server, or a spawned stub process:
- `openapi.rs` — `execute()` against a mock (query substitution, bearer auth header, response formatting).
- `sparql.rs` — `execute()` (SELECT→table, read-only rejection) and `probe()` (liveness + prefix derivation).
- `mcp.rs` — **stdio transport** driven end-to-end against `src-tauri/tests/fixtures/mcp-stub.js` (a Node JSON-RPC stub): `connect` handshake, `call_tool`, and the MCP-App path (`call_tool_rich` → `read_resource` fetching a `ui://` resource).

### 3. Frontend component tests — Vitest + React Testing Library
`src/**/*.test.tsx`, using the Tauri IPC mocks in `src/test/setup.ts`
(`invoke`/`listen`/`emit`/`openUrl` are `vi.fn()`s). E.g. `McpAppFrame.test.tsx`
covers the MCP-App **consent gate** (Allow → `invoke("approve_mcp_app")` → sandboxed
iframe mounts) and `ToolResultRow` render branches. Pure-logic helpers are tested in
`src/test/*.test.ts` (`extractSpecMeta`, `loadSettings`/`saveSettings`).

### 4. Playwright e2e — real browser, mocked backend
`e2e/*.spec.ts` run the React app in headless Chromium with the Tauri bridge faked by
`e2e/mock-tauri.ts` (injected via `addInitScript`): it stubs `invoke` with canned
responses and implements the event system, exposing `window.__mockEmit(event, payload)`
so a test can push backend events. This exercises the **real** iframe + `postMessage`
bridge that jsdom can't — e.g. emitting an `agent-tool-result` with `ui.html`, clicking
**Allow**, and asserting the content renders inside the sandboxed `srcdoc` iframe.
Cross-platform (runs on macOS/Linux/Windows); the real Rust backend is not exercised
here (that's layer 2).

## Mock seams (how each layer injects test doubles)

| Dependency | Seam |
|---|---|
| Ollama | `host` is a parameter (`ollama.rs`) → point at a mock base URL |
| OpenAPI / SPARQL | `base_url` / `endpoint_url` struct fields → mock server; `execute` takes `app: Option<&AppHandle>` → pass `None` |
| MCP (HTTP) | `config.command` = mock URL |
| MCP (stdio) | `config.command` = `node tests/fixtures/mcp-stub.js` |
| OAuth2 token | `token_url` → mock server |
| Frontend IPC | `src/test/setup.ts` `vi.mock`s (unit) / `e2e/mock-tauri.ts` init-script (e2e) |

## Adding tests
- **Rust pure logic** → add to the file's `#[cfg(test)] mod tests`.
- **Rust HTTP path** → `#[tokio::test]` with a `wiremock::MockServer` (see `sparql.rs`/`openapi.rs`).
- **React component** → `src/**/*.test.tsx`; drive `vi.mocked(invoke)` return values.
- **User flow** → `e2e/*.spec.ts`; use `window.__mockEmit` to simulate backend events.

## Known gaps / future work
- **`agent_loop` behavioural tests** — the core orchestration in `ollama.rs` is coupled
  to `&AppHandle` (it `emit`s directly). Testing it needs Tauri's `MockRuntime` to
  capture events, or an event-sink abstraction. Deferred.
- **`web_search` / `fetch_webpage`** hardcode their endpoints (`tools.rs`) — thread an
  injectable base URL to make them mockable.
- **Real-app E2E** via `tauri-driver` (Linux/Windows CI only — not macOS) for true
  full-stack coverage.
