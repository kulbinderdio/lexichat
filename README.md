# LexiChat

A local AI chat desktop application built with Tauri 2.x (Rust backend) and React/TypeScript frontend. Powered by [Ollama](https://ollama.com), with support for file tools, OpenAPI integrations, and MCP servers.

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
git tag v0.2.0

# 2. Push the tag — this kicks off the build
git push origin v0.2.0
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

```bash
# Frontend tests (Vitest)
npm test

# Rust tests
npm run test:rust

# Both together
npm run test:all

# Frontend with coverage
npm run test:coverage
```

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
   - Connect MCP servers
   - Configure sandbox directories for file tool access

---

## Built-in Tools

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
