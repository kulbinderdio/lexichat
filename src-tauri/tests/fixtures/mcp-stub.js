#!/usr/bin/env node
// Minimal MCP server stub for integration tests. Speaks JSON-RPC (newline-delimited)
// over stdin/stdout: initialize → tools/list → tools/call → resources/read.
// Includes an MCP-App tool (show_ui) that declares a ui:// resource.

let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let idx;
  while ((idx = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, idx).trim();
    buf = buf.slice(idx + 1);
    if (line) { try { handle(JSON.parse(line)); } catch { /* ignore */ } }
  }
});

function send(obj) { process.stdout.write(JSON.stringify(obj) + "\n"); }

function handle(msg) {
  const { id, method, params } = msg;
  if (id === undefined || id === null) return; // notification (e.g. notifications/initialized)

  if (method === "initialize") {
    return send({ jsonrpc: "2.0", id, result: {
      protocolVersion: "2025-06-18",
      capabilities: { resources: {} },
      serverInfo: { name: "stub", version: "0" },
    }});
  }
  if (method === "tools/list") {
    return send({ jsonrpc: "2.0", id, result: { tools: [
      { name: "echo", description: "Echo text", inputSchema: { type: "object", properties: { text: { type: "string" } } } },
      { name: "show_ui", description: "Show a UI", inputSchema: { type: "object" }, _meta: { ui: { resourceUri: "ui://stub/app" } } },
    ] }});
  }
  if (method === "tools/call") {
    const name = params && params.name;
    if (name === "echo") {
      const text = (params.arguments && params.arguments.text) || "echoed";
      return send({ jsonrpc: "2.0", id, result: { content: [{ type: "text", text }] } });
    }
    if (name === "show_ui") {
      return send({ jsonrpc: "2.0", id, result: {
        content: [{ type: "text", text: "opened" }],
        _meta: { ui: { resourceUri: "ui://stub/app" } },
      }});
    }
    return send({ jsonrpc: "2.0", id, error: { code: -32601, message: "unknown tool" } });
  }
  if (method === "resources/read") {
    return send({ jsonrpc: "2.0", id, result: { contents: [
      { uri: "ui://stub/app", mimeType: "text/html", text: "<h1>stub app</h1>" },
    ] }});
  }
  return send({ jsonrpc: "2.0", id, result: {} });
}
