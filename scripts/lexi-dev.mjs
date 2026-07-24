#!/usr/bin/env node
// Client for LexiChat's dev control server (Phase 1).
// Requires the app running as a DEBUG build with the server enabled:
//     LEXICHAT_DEV_CONTROL=1 npm run tauri dev
// (optionally LEXICHAT_DEV_CONTROL_PORT=8787). Bound to 127.0.0.1 only; never in release builds.
//
// Usage:
//   node scripts/lexi-dev.mjs ping
//   node scripts/lexi-dev.mjs run "draw a histogram of 1..100 with matplotlib"
//   node scripts/lexi-dev.mjs run "..." --raw     # full JSON instead of the summary
//
// Runs against the CURRENTLY active profile/settings in the app; auto-approves the code-exec
// prompt. Prints a compact trace: each tool call, tool result (truncated), inline-image counts,
// MCP-app UI presence, and the final answer.

import http from "node:http";

const PORT = Number(process.env.LEXICHAT_DEV_CONTROL_PORT || 8787);
const argv = process.argv.slice(2);
const [cmd] = argv;
const raw = argv.includes("--raw");
// Flags: --reasoning=on|off|auto  --numctx=16384  --model="server / name"
const flag = (name) => {
  const p = argv.find((a) => a.startsWith(`--${name}=`));
  return p ? p.slice(name.length + 3) : undefined;
};
const message = argv.slice(1).filter((a) => !a.startsWith("--")).join(" ");
const overrides = {};
if (flag("reasoning")) overrides.reasoning = flag("reasoning");
if (flag("numctx")) overrides.numCtx = Number(flag("numctx"));
if (flag("model")) overrides.model = flag("model");
if (flag("profile")) overrides.profile = flag("profile");
if (argv.includes("--code-tools")) overrides.allowCodeTools = true;
if (argv.includes("--no-code-tools")) overrides.allowCodeTools = false;
if (argv.includes("--continue")) overrides.fresh = false; // keep conversation history

// Raw node:http request — no headers/body timeout, so a slow model (minutes) won't drop the
// connection the way global fetch()'s undici defaults do.
function request(method, path, bodyObj, timeoutMs = 0) {
  return new Promise((resolve, reject) => {
    const body = bodyObj ? JSON.stringify(bodyObj) : null;
    const req = http.request(
      { host: "127.0.0.1", port: PORT, path, method,
        headers: body ? { "content-type": "application/json", "content-length": Buffer.byteLength(body) } : {} },
      (res) => {
        let data = "";
        res.on("data", (c) => (data += c));
        res.on("end", () => { try { resolve(JSON.parse(data)); } catch { resolve(data); } });
      });
    req.on("error", reject);
    if (timeoutMs > 0) req.setTimeout(timeoutMs, () => req.destroy(new Error(`timeout after ${timeoutMs}ms`)));
    else req.setTimeout(0);
    if (body) req.write(body);
    req.end();
  });
}

function trunc(s, n = 220) {
  if (typeof s !== "string") return s;
  return s.length > n ? s.slice(0, n) + `… (${s.length} chars)` : s;
}

async function main() {
  if (cmd === "ping") {
    console.log(JSON.stringify(await request("GET", "/dev/ping")));
    return;
  }
  if (cmd === "state") {
    console.log(JSON.stringify(await request("GET", "/dev/state", null, 45000), null, 2));
    return;
  }
  if (cmd === "config") {
    // e.g. config --profile="Maps" --code-tools --reasoning=off
    console.log(JSON.stringify(await request("POST", "/dev/config", overrides, 45000), null, 2));
    return;
  }
  if (cmd === "run") {
    if (!message) { console.error('usage: lexi-dev run "your message" [--reasoning=off] [--numctx=16384] [--model="..."]'); process.exit(1); }
    const j = await request("POST", "/dev/run", { message, ...overrides });
    if (raw) { console.log(JSON.stringify(j, null, 2)); return; }
    const trace = j.trace ?? j;
    if (j.error) { console.error("server error:", j.error); process.exit(1); }
    if (trace.error) { console.error("run error:", trace.error, trace.elapsedMs ? `(${(trace.elapsedMs/1000).toFixed(1)}s)` : ""); process.exit(1); }
    for (const m of trace.messages ?? []) {
      if (m.role === "user") console.log(`\n👤 ${trunc(m.text)}`);
      else if (m.toolCalls) for (const tc of m.toolCalls) console.log(`  🔧 ${tc.name}(${trunc(tc.args, 160)})`);
      else if (m.role === "tool-result") {
        const extra = [m.images ? `${m.images} image(s)` : null, m.ui ? `MCP-app ui(html=${m.ui.hasHtml})` : null].filter(Boolean).join(", ");
        console.log(`  ⇒ [${m.toolName}] ${trunc(m.toolResult)}${extra ? "  {" + extra + "}" : ""}`);
      } else if (m.role === "error") console.log(`  ⛔ ${trunc(m.text)}`);
    }
    if (trace.codeToolCalls?.length) {
      console.log(`  🐍→🔧 code-mode calls: ${trace.codeToolCalls.map((c) => `${c.name}${c.ok ? "" : "✗"}`).join(", ")}`);
    }
    console.log(`\n🤖 ${trace.finalAnswer ?? "(no final answer)"}`);
    console.log(`⏱  ${trace.elapsedMs != null ? (trace.elapsedMs / 1000).toFixed(1) + "s" : "?"}\n`);
    return;
  }
  console.error('usage: node scripts/lexi-dev.mjs <ping | run "message" [--raw]>');
  process.exit(1);
}
main().catch((e) => { console.error("request failed:", String(e), `\nis the app running with LEXICHAT_DEV_CONTROL=1?`); process.exit(1); });
