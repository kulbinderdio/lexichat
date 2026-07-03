import { test, expect } from "@playwright/test";
import { tauriMockInit } from "./mock-tauri";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(tauriMockInit);
});

test("app boots and shows the composer", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByPlaceholder("Message…")).toBeVisible();
});

test("an MCP-App tool result renders a consented, sandboxed iframe", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("Message…").waitFor(); // listeners registered

  // Simulate the backend emitting a tool result that carries an MCP-App UI.
  await page.evaluate(() =>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (window as any).__mockEmit("agent-tool-result", {
      name: "qr", result: "done",
      ui: { server_id: "e2e", html: "<h1>QR-CODE-HERE</h1>" },
    })
  );

  // Consent gate first — no iframe until Allow.
  const allow = page.getByRole("button", { name: /Allow app/i });
  await expect(allow).toBeVisible();
  await allow.click();

  // Real browser iframe with srcdoc — assert content renders inside it.
  const frame = page.frameLocator('iframe[title="mcp-app-qr"]');
  await expect(frame.locator("h1")).toHaveText("QR-CODE-HERE");
});
