# TESTING.md

## Running the tests

```bash
npm test              # Frontend tests only (Vitest, ~0.5s)
npm run test:rust     # Rust tests only (cargo test, ~0.2s after first compile)
npm run test:all      # Both — run this after every build
npm run test:watch    # Frontend in watch mode during development
npm run test:coverage # Frontend with coverage report
```

## What's tested

**84 tests total across 5 suites.**

### Rust — `src-tauri/src/openapi.rs` (23 tests)

`sanitize_tool_name`
- Lowercase alphanumeric preserved unchanged
- Uppercase letters lowercased
- Dots, spaces, hyphens converted to underscores
- Leading/trailing underscores stripped
- Consecutive separators collapsed to single underscore
- Numbers preserved
- Empty string handled

`tool_prefix`
- Strips ` API`, ` Service`, ` Server` suffixes
- Produces trailing underscore (e.g. `"Gmail API"` → `"gmail_"`)
- Empty/whitespace-only input falls back to `"svc_"`

`parse_spec`
- Tool names are prefixed with the service name
- HTTP methods are uppercased
- Path parameters detected as `location = "path"` and marked required
- Query parameters detected as optional by default
- Request body properties extracted as `location = "body"`
- Tool names capped at 64 characters
- Non-HTTP methods (e.g. `x-custom`) ignored
- Missing `operationId` synthesised from method + path
- Invalid JSON returns `Err`
- Missing `paths` key returns `Err`

### Rust — `src-tauri/src/mcp.rs` (17 tests)

`is_url`
- Detects `http://` and `https://` URLs
- Rejects shell commands and file paths

`AuthConfig` serialization
- All 5 variants round-trip through `serde_json` (`none`, `bearer`, `apikey`, `basic`, `oauth2`)
- All type tags serialize as lowercase strings (critical: Tauri frontend sends lowercase)
- Deserializes correctly from the JSON format the TypeScript frontend sends

`parse_tools`
- Tool names are prefixed with sanitized server name
- Original `raw_name` is preserved (used when calling the MCP server)
- Names capped at 64 characters
- Empty tool list handled
- Missing `tools` key handled

### Rust — `src-tauri/src/tools.rs` (17 tests)

`check_path`
- Allowed when no restrictions configured
- Allowed when path is within an allowed directory
- Denied when path is outside all allowed directories
- Denied on `../` traversal attempts
- Non-existent files resolved via parent directory
- Note: tests canonicalize temp dir paths to handle macOS `/var` → `/private/var` symlinks

`glob_matches`
- `*` matches any sequence including empty
- `?` matches exactly one character
- Exact match with no wildcards
- Non-matching patterns return false
- Empty pattern matches empty string only

`all_builtin_schemas`
- Returns exactly 12 schemas
- Every schema has `name`, `description`, and `parameters` fields
- All tool names are valid: `[a-z0-9_]`, no leading/trailing `_`, max 64 chars

### Frontend — `src/test/extractSpecMeta.test.ts` (17 tests)

OpenAPI 3.x extraction:
- Title from `info.title`
- Base URL from `servers[0].url` (trailing slash stripped)
- Token URL and Authorization URL from `authorizationCode` flow
- Scopes as space-separated string
- Prefers `authorizationCode` over `clientCredentials` when both present
- Returns `undefined` fields when no security schemes present
- Ignores non-OAuth2 schemes (e.g. apiKey)

Swagger 2.0 extraction:
- Base URL reconstructed from `host` + `basePath` + `schemes`
- Defaults to `https` when `schemes` is absent
- Token URL and Authorization URL from `accessCode` flow
- Authorization URL from `implicit` flow (no token URL)
- Scopes extracted from Swagger 2.0 definitions

Edge cases:
- Invalid JSON returns `{}`
- Empty string returns `{}`
- Empty object `{}` returns `{}`
- Missing title returns `undefined` while other fields still extracted

### Frontend — `src/test/settings.test.ts` (10 tests)

`loadSettings`
- Returns all defaults when localStorage is empty
- Merges saved values over defaults (missing keys filled from defaults)
- Returns defaults when localStorage contains corrupted JSON
- Preserves saved profiles array
- Preserves saved `enabledTools` overrides

`saveSettings` + `loadSettings` round-trip:
- Host change persists
- Model list persists
- Active profile ID persists
- `maxTools` persists
- Second save overwrites first

## Test infrastructure

**Frontend** — Vitest + jsdom + React Testing Library

- Config: `vite.config.ts` → `test` block
- Setup file: `src/test/setup.ts`
  - Provides a `localStorage` mock (jsdom's built-in can be incomplete)
  - Mocks all Tauri APIs (`@tauri-apps/api/core`, `@tauri-apps/api/event`, plugins)

**Rust** — built-in `cargo test`

- Tests live in `#[cfg(test)] mod tests` at the bottom of each source file
- Dev dependency: `tempfile = "3"` (temp directories for sandbox path tests)

## Keeping tests up to date

When changing the code, update the corresponding tests:

| Changed area | Test file |
|---|---|
| Tool name sanitization or prefixing | `src-tauri/src/openapi.rs` tests |
| `AuthConfig` variants or serde rename tags | `src-tauri/src/mcp.rs` tests |
| File sandbox (`check_path`) or built-in schemas | `src-tauri/src/tools.rs` tests |
| `extractSpecMeta` parsing logic | `src/test/extractSpecMeta.test.ts` |
| Settings structure, keys, or defaults | `src/test/settings.test.ts` |

When adding a new Rust module with pure functions, add a `#[cfg(test)] mod tests` block at the bottom of that file following the same pattern.

When adding a new pure TypeScript utility function, add a `.test.ts` file under `src/test/`.
