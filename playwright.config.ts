import { defineConfig, devices } from "@playwright/test";

// E2E tests run the React frontend in a real headless browser (Chromium) with the
// Tauri IPC bridge mocked (see e2e/mock-tauri.ts). Cross-platform — runs on macOS,
// Linux, and Windows. The real Rust backend is NOT exercised here (that's covered
// by the Rust integration tests); this layer covers UI flows and the real iframe /
// postMessage bridge that jsdom can't.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:1420",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:1420",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
