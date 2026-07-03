# LexiChat


<img width="1284" height="852" alt="image" src="https://github.com/user-attachments/assets/712aee36-6659-4fb5-b5cd-066c775e7dae" />



**Your private AI data assistant.**

LexiChat connects a local LLM to **your data** — REST APIs, SPARQL / linked-data endpoints, MCP servers, and local files — and runs entirely on your machine via [Ollama](https://ollama.com). No cloud, no subscriptions, nothing leaves your device.

Built with Tauri 2.x (Rust backend) and a React/TypeScript frontend.

## Connect your data

LexiChat is built around bringing your own data to a private model. Each source becomes a tool the AI can call, all managed in Settings and scoped per profile:

| Source | What it does |
|--------|--------------|
| **Local files** | Sandboxed read / write / search over folders you explicitly allow |
| **REST APIs (OpenAPI)** | Paste an OpenAPI 3.x spec → one callable tool per operation |
| **SPARQL endpoints** | Query linked-data / RDF services (e.g. UK gov open data); auto-generated query + schema tools, primed with your prefixes and example queries |
| **MCP servers** | Connect Model Context Protocol servers (stdio or HTTP) |
| **Web search** | Built-in DuckDuckGo search, no API key |

Ships with ready-to-use SPARQL endpoints (HM Land Registry, OpenDataCommunities) and a built-in Wikipedia API. See the [Integrations guide](https://github.com/kulbinderdio/lexichat/blob/main/website/docs/integrations.html) for setup details.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| [Node.js](https://nodejs.org) | 18+ | npm 9+ included |
| [Rust](https://rustup.rs) | stable | via `rustup` |
| [Ollama](https://ollama.com) | any | must be running locally |

### macOS-specific
Xcode Command Line Tools are required:
```bash
xcode-select --install
```

### Linux-specific
```bash
# Ubuntu/Debian
sudo apt install libgtk-3-dev libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf
```

### Windows-specific
[Microsoft Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (Desktop development with C++ workload).

---

## Development

```bash
# 1. Install JS dependencies
npm install

# 2. Make sure Ollama is running with a model pulled
ollama pull gemma3  # or any model of your choice

# 3. Start the app in dev mode (hot-reload for both frontend and backend)
npm run tauri dev
```

The Vite dev server runs on port 1420. Changes to `src/` hot-reload instantly; changes to `src-tauri/src/` trigger a Rust recompile.

---

## GitHub Actions — Building a Release

The CI/CD workflow builds installers for all platforms automatically. To trigger it:

```bash
# 1. Tag the commit you want to release
git tag v1.9.0

# 2. Push the tag — this kicks off the build
git push origin v1.9.0
```

GitHub Actions will run 4 parallel jobs (macOS, Windows, Linux x86-64, Linux ARM64). When all finish (typically 15–25 minutes), a **draft release** appears at:

**https://github.com/kulbinderdio/lexichat/releases**

Review the draft, edit the release notes if needed, then click **Publish release** to make it public.

You can also trigger a build manually without a tag from the **Actions** tab → **Build & Release** → **Run workflow**.

### Build outputs per platform

| Platform | Files attached to release |
|----------|--------------------------|
| macOS | `LexiChat_*_universal.dmg` (Intel + Apple Silicon) |
| Windows | `LexiChat_*_x64-setup.exe`, `LexiChat_*_x64_en-US.msi` |
| Linux x86-64 | `lexichat_*_amd64.deb`, `LexiChat_*_amd64.AppImage` |
| Linux ARM64 | `lexichat_*_arm64.deb`, `LexiChat_*_aarch64.AppImage` |

---

## Building locally

### macOS (universal binary — runs on both Intel and Apple Silicon)
```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin
npm run tauri build -- --target universal-apple-darwin
```

Output: `src-tauri/target/universal-apple-darwin/release/bundle/dmg/LexiChat_x.x.x_universal.dmg`

### macOS (current architecture only)
```bash
npm run tauri build
```

### Windows
```bash
npm run tauri build
```
Output: `src-tauri/target/release/bundle/msi/` (.msi installer) and `nsis/` (.exe installer)

### Linux
```bash
npm run tauri build
```
Output: `src-tauri/target/release/bundle/deb/` (.deb) and `appimage/` (.AppImage)

---

## Running Tests

All suites are headless and need **no external services** — Ollama, MCP servers, and
HTTP endpoints are mocked or stubbed.

```bash
npm run test          # Frontend: unit + React component tests (Vitest + jsdom)
npm run test:rust     # Rust: unit + HTTP-integration tests (cargo test)
npm run test:all      # Both of the above
npm run test:e2e      # End-to-end UI tests (Playwright, headless Chromium)

npm run test:coverage # Frontend coverage report
```

First e2e run needs the browser: `npx playwright install chromium`.

### What's covered

- **Rust unit** — parsing, serde, tool-name sanitization, SPARQL read-only guard, MCP-App UI detection, sandbox path-gating, wiki FS ops, job cron logic.
- **Rust integration** — real HTTP calls against an in-process mock (`wiremock`) for OpenAPI / SPARQL / MCP-HTTP, plus the MCP **stdio** transport driven end-to-end against a Node JSON-RPC stub (`src-tauri/tests/fixtures/mcp-stub.js`).
- **Frontend components** — React Testing Library with mocked Tauri IPC (`src/test/setup.ts`); e.g. the MCP-App consent gate.
- **End-to-end** — Playwright runs the real UI in a browser with the Tauri bridge mocked (`e2e/mock-tauri.ts`), exercising the real sandboxed-iframe / postMessage MCP-App flow.

**CI:** `.github/workflows/test.yml` runs the entire suite on every push and pull request. See **[TESTING.md](TESTING.md)** for the full breakdown, mock seams, and how to add tests.

---

## App Data

User settings are stored in `localStorage` within the app's WebView. The sandbox allowed-directories list is persisted at:

- **macOS**: `~/Library/Application Support/lexichat/allowed_dirs.json`
- **Linux**: `~/.local/share/lexichat/allowed_dirs.json`
- **Windows**: `%APPDATA%\lexichat\allowed_dirs.json`

> If upgrading from an older build named `ai-agent-cross`, the allowed directories list is migrated automatically on first launch.

---

## First-time Setup

1. Launch the app (`npm run tauri dev` or the built executable)
2. Ollama models are fetched automatically — select one from the dropdown
3. Open **Settings** (gear icon) to:
   - Create chat profiles with different models and system prompts
   - Enable/disable built-in file and web search tools
   - Register OpenAPI specs for external API access
   - Add SPARQL endpoints for linked-data queries
   - Connect MCP servers
   - Configure sandbox directories for file tool access

---

## Built-in Tools (local files & web)

These ship with the app and cover local-file access and web search. They sit alongside the external data connectors above (OpenAPI, SPARQL, MCP).

| Tool | Description |
|------|-------------|
| `read_file` | Read a file's contents |
| `write_file` | Write content to a file |
| `list_files` | List files in a directory |
| `search_files` | Find files by name pattern |
| `search_in_files` | Search file contents by regex |
| `get_file_info` | File metadata (size, dates) |
| `list_directory_tree` | Recursive directory tree |
| `create_directory` | Create a new directory |
| `move_file` | Move or rename a file |
| `delete_file` | Delete a file |
| `find_old_files` | Find files older than N days |
| `web_search` | DuckDuckGo web search |
| `compose_email` | Build a base64url-encoded RFC 2822 email (for Gmail API) |

All file tools are sandboxed — they only operate within directories you explicitly allow in Settings.

---

## Architecture Overview

See [CLAUDE.md](CLAUDE.md) for a detailed technical architecture guide.

---

## License

LexiChat is dual-licensed:

- **Personal / open-source use**: [GNU AGPLv3](LICENSE) — free to use, modify, and distribute provided any derivative work is also released under AGPLv3.
- **Commercial use**: A separate commercial license is required if you use LexiChat in a proprietary product or service, or do not wish to comply with AGPLv3's copyleft terms.

For commercial licensing enquiries contact: info@lexi-chat.com

See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for details.
