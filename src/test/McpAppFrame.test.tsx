import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { invoke } from "@tauri-apps/api/core";
import { McpAppFrame, ToolResultRow } from "../App";

// Consent gating is the security-critical, jsdom-testable part of the MCP-App
// bridge. (Full postMessage traffic is exercised by the Playwright e2e suite,
// which runs in a real browser with a real iframe.)
describe("McpAppFrame consent gating", () => {
  beforeEach(() => { vi.mocked(invoke).mockReset(); });

  it("shows a consent prompt and no iframe before approval", () => {
    render(<McpAppFrame ui={{ server_id: "srv-a", html: "<p>hi</p>" }} toolName="qr" onSend={() => {}} />);
    expect(screen.getByRole("button", { name: /Allow app/i })).toBeInTheDocument();
    expect(screen.queryByTitle("mcp-app-qr")).not.toBeInTheDocument();
  });

  it("mounts a sandboxed iframe after approval and records consent", async () => {
    vi.mocked(invoke).mockResolvedValue(undefined);
    render(<McpAppFrame ui={{ server_id: "srv-b", html: "<p>hi</p>" }} toolName="qr" onSend={() => {}} />);

    await userEvent.click(screen.getByRole("button", { name: /Allow app/i }));

    await waitFor(() => expect(screen.getByTitle("mcp-app-qr")).toBeInTheDocument());
    expect(invoke).toHaveBeenCalledWith("approve_mcp_app", { args: { server_id: "srv-b" } });

    const iframe = screen.getByTitle("mcp-app-qr");
    const sandbox = iframe.getAttribute("sandbox") ?? "";
    expect(sandbox).toContain("allow-scripts");
    expect(sandbox).not.toContain("allow-same-origin"); // must not be able to reach the host origin
  });

  it("renders nothing when the payload has no html", () => {
    const { container } = render(<McpAppFrame ui={{ server_id: "srv-c" }} toolName="qr" onSend={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ToolResultRow", () => {
  it("renders an MCP app frame when ui.html is present", () => {
    render(<ToolResultRow name="qr" result="done" ui={{ server_id: "srv-d", html: "<p>hi</p>" }}
      onSend={() => {}} onAttach={() => {}} />);
    expect(screen.getByRole("button", { name: /Allow app/i })).toBeInTheDocument();
  });

  it("renders a text preview for a plain result", () => {
    render(<ToolResultRow name="custom_tool" result="just some text" onSend={() => {}} onAttach={() => {}} />);
    expect(screen.getByText("custom_tool")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Allow app/i })).not.toBeInTheDocument();
  });
});
