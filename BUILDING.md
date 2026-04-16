# LexiChat — Build & Distribution Guide

LexiChat is a [Tauri 2](https://tauri.app) application with a React/TypeScript frontend and a Rust backend. Each platform produces a native installer from a single `npm run tauri build` command, but the toolchain requirements differ per platform. **You must build on the target platform** — cross-compilation is not officially supported by Tauri 2.

---

## Prerequisites (all platforms)

| Tool | Version | Notes |
|------|---------|-------|
| Node.js | 18 LTS or 20 LTS | `node --version` |
| npm | 9+ | Comes with Node |
| Rust | stable (≥ 1.77) | Install via [rustup](https://rustup.rs) |
| Cargo | same as Rust | Installed by rustup |

Install Rust/Cargo:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Install Node dependencies once (from the project root):
```bash
npm install
```

---

## macOS

### System requirements
- macOS 10.15 Catalina or later (for development)
- Xcode Command Line Tools

### 1 — Install Xcode CLT
```bash
xcode-select --install
```

### 2 — Add the required Rust targets

For a universal binary (Apple Silicon + Intel):
```bash
rustup target add aarch64-apple-darwin
rustup target add x86_64-apple-darwin
```

For Apple Silicon only:
```bash
rustup target add aarch64-apple-darwin
```

### 3 — Build

**Universal binary (recommended for distribution):**
```bash
npm run tauri build -- --target universal-apple-darwin
```

**Apple Silicon only:**
```bash
npm run tauri build -- --target aarch64-apple-darwin
```

**Intel only:**
```bash
npm run tauri build -- --target x86_64-apple-darwin
```

### 4 — Output

```
src-tauri/target/<target>/release/bundle/
  dmg/   LexiChat_0.1.0_universal.dmg       ← disk image for distribution
  macos/ LexiChat.app                        ← app bundle (drag to Applications)
```

### 5 — Code signing & notarisation (required for Gatekeeper)

Without signing, macOS will block the app with "unidentified developer" for end users.

**Sign the app:**
```bash
# List your signing identities
security find-identity -v -p codesigning

# Set in environment before building
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID)"
npm run tauri build -- --target universal-apple-darwin
```

**Notarise (required for distribution outside the App Store):**
```bash
# Store credentials once
xcrun notarytool store-credentials "lexichat-profile" \
  --apple-id your@email.com \
  --team-id YOURTEAMID \
  --password "app-specific-password"

# Submit the DMG
xcrun notarytool submit \
  "src-tauri/target/universal-apple-darwin/release/bundle/dmg/LexiChat_0.1.0_universal.dmg" \
  --keychain-profile "lexichat-profile" \
  --wait

# Staple the ticket
xcrun stapler staple \
  "src-tauri/target/universal-apple-darwin/release/bundle/dmg/LexiChat_0.1.0_universal.dmg"
```

---

## Windows

### System requirements
- Windows 10 or 11 (64-bit)
- Visual Studio Build Tools 2022 with the **"Desktop development with C++"** workload, **or** a full Visual Studio 2022 installation

### 1 — Install Visual Studio Build Tools

Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Select workload: **Desktop development with C++**

Required individual components (auto-selected with the workload):
- MSVC v143 C++ x64/x86 build tools
- Windows 11 (or 10) SDK
- CMake tools for Windows

### 2 — Install Rust

Download and run `rustup-init.exe` from https://rustup.rs

In the Rust installer, choose the default install (MSVC toolchain).

Verify:
```powershell
rustup toolchain list    # should show stable-x86_64-pc-windows-msvc
```

### 3 — Install Node.js

Download the LTS installer from https://nodejs.org and run it. Ensure "Add to PATH" is checked.

### 4 — Install dependencies

Open **Developer PowerShell** or a regular terminal with Rust and Node in PATH:
```powershell
npm install
```

### 5 — Build

```powershell
npm run tauri build
```

### 6 — Output

```
src-tauri\target\release\bundle\
  msi\    LexiChat_0.1.0_x64_en-US.msi    ← Windows Installer package
  nsis\   LexiChat_0.1.0_x64-setup.exe    ← NSIS installer (smaller, recommended)
```

Both installers are produced by default. The NSIS `.exe` is the better choice for most end-user distribution.

### 7 — Code signing (optional but recommended)

Without signing, Windows SmartScreen will warn users. You need a code signing certificate (EV or OV) from a CA such as DigiCert, Sectigo, or Certum.

```powershell
# Set environment variables before building
$env:TAURI_SIGNING_PRIVATE_KEY = "path\to\private.key"
$env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = "your-password"

npm run tauri build
```

Or use `signtool.exe` post-build:
```powershell
signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 `
  /f certificate.pfx /p "pfx-password" `
  "src-tauri\target\release\bundle\nsis\LexiChat_0.1.0_x64-setup.exe"
```

---

## Linux

### System requirements
- Ubuntu 22.04 / Debian 12 (recommended), or any modern distro with GTK 3+
- The build produces a `.deb` (Debian/Ubuntu), `.rpm` (Fedora/RHEL), and `.AppImage` (portable, runs on any distro)

### 1 — Install system dependencies

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install -y \
  build-essential \
  curl \
  wget \
  file \
  libssl-dev \
  libgtk-3-dev \
  libwebkit2gtk-4.1-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  patchelf
```

**Fedora / RHEL:**
```bash
sudo dnf install -y \
  gcc \
  openssl-devel \
  gtk3-devel \
  webkit2gtk4.1-devel \
  libappindicator-gtk3-devel \
  librsvg2-devel \
  patchelf
```

**Arch Linux:**
```bash
sudo pacman -S --needed \
  base-devel \
  openssl \
  gtk3 \
  webkit2gtk-4.1 \
  libappindicator-gtk3 \
  librsvg \
  patchelf
```

### 2 — Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3 — Install Node.js

Use a version manager (recommended) or the system package manager:
```bash
# Using nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20
```

### 4 — Install dependencies

```bash
npm install
```

### 5 — Build

```bash
npm run tauri build
```

### 6 — Output

```
src-tauri/target/release/bundle/
  deb/       lexichat_0.1.0_amd64.deb        ← Debian/Ubuntu package
  rpm/       lexichat-0.1.0-1.x86_64.rpm     ← Fedora/RHEL package
  appimage/  LexiChat_0.1.0_amd64.AppImage   ← portable (no install needed)
```

The AppImage is the most portable option — users on any distro can just `chmod +x` and run it.

### 7 — Make AppImage executable

```bash
chmod +x "src-tauri/target/release/bundle/appimage/LexiChat_0.1.0_amd64.AppImage"
./LexiChat_0.1.0_amd64.AppImage
```

### 8 — Optional: build specific targets only

```bash
# Only .deb
npm run tauri build -- --bundles deb

# Only AppImage
npm run tauri build -- --bundles appimage

# Only .rpm
npm run tauri build -- --bundles rpm
```

---

## Changing the version number

Edit both files before building:

**`src-tauri/tauri.conf.json`:**
```json
"version": "1.0.0"
```

**`src-tauri/Cargo.toml`:**
```toml
[package]
version = "1.0.0"
```

---

## Replacing the app icon

Tauri reads icons from `src-tauri/icons/`. To update:

1. Create a 1024×1024 PNG source image
2. Install the Tauri icon generator:
   ```bash
   npm run tauri icon path/to/your-icon.png
   ```
   This auto-generates all required sizes for all platforms.

---

## CI/CD with GitHub Actions

To build for all three platforms automatically on every release, create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: macos-latest
            target: universal-apple-darwin
            args: --target universal-apple-darwin
          - os: windows-latest
            target: x86_64-pc-windows-msvc
            args: ''
          - os: ubuntu-22.04
            target: x86_64-unknown-linux-gnu
            args: ''

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Install Node
        uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - name: Install Linux dependencies
        if: matrix.os == 'ubuntu-22.04'
        run: |
          sudo apt update
          sudo apt install -y libgtk-3-dev libwebkit2gtk-4.1-dev \
            libayatana-appindicator3-dev librsvg2-dev patchelf

      - name: Install frontend dependencies
        run: npm install

      - name: Build
        run: npm run tauri build -- ${{ matrix.args }}

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: lexichat-${{ matrix.os }}
          path: |
            src-tauri/target/**/release/bundle/dmg/*.dmg
            src-tauri/target/**/release/bundle/macos/*.app
            src-tauri/target/**/release/bundle/msi/*.msi
            src-tauri/target/**/release/bundle/nsis/*.exe
            src-tauri/target/**/release/bundle/deb/*.deb
            src-tauri/target/**/release/bundle/appimage/*.AppImage
```

Push a tag (`git tag v1.0.0 && git push --tags`) to trigger the workflow. Artifacts are uploaded to GitHub for download.

---

## Runtime requirement — Ollama

LexiChat requires [Ollama](https://ollama.com) running locally. This is **not bundled** in the app. Include this in your release notes:

> **Before launching LexiChat**, install Ollama from https://ollama.com and pull at least one model:
> ```bash
> ollama pull gemma3      # recommended default
> ```
> Ollama must be running in the background while using LexiChat.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `error: linker 'cc' not found` (Linux) | `sudo apt install build-essential` |
| `error: failed to run custom build command for openssl-sys` | `sudo apt install libssl-dev pkg-config` |
| `WebKit2GTK not found` (Linux) | `sudo apt install libwebkit2gtk-4.1-dev` |
| `xcrun: error: invalid active developer path` (macOS) | `xcode-select --install` |
| `LINK : fatal error LNK1181` (Windows) | Ensure MSVC Build Tools are installed with the C++ workload |
| `Could not find Visual Studio` (Windows) | Run from a Developer PowerShell, or ensure `MSVC` is in PATH |
| Build hangs on first run | Normal — Rust is compiling ~200 crates. Subsequent builds are incremental and much faster. |
