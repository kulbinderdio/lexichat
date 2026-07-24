// Main-thread manager for the Pyodide worker. Loads the worker once (kept warm for the
// life of the app), serialises run requests, and resolves each with the worker's result.
// The heavy WASM runtime lives entirely in the worker; this file is a thin RPC shim.
//
// Robustness: any failure to create/load the worker or initialise Pyodide is turned into an
// error result (never a hang) so the model/user sees a diagnostic. A load watchdog bounds
// cold-start.

import { invoke } from "@tauri-apps/api/core";

export interface PyFile { path: string; b64: string }
export interface PyResult {
  output: string;
  images: string[];              // base64 PNGs (matplotlib figures)
  outFiles: { name: string; b64: string }[]; // files written to /work/out
  error: string | null;
}

const LOAD_TIMEOUT_MS = 90_000;

let worker: Worker | null = null;
let ready: Promise<void> | null = null;
let loadError: string | null = null;
let seq = 0;
const pending = new Map<number, (r: PyResult) => void>();

// Code-mode: bound how many tools one run_python call may invoke, so a runaway loop can't hammer
// an API. Reset at the start of each run.
const MAX_TOOL_CALLS_PER_RUN = 200;
let toolCallsRemaining = 0;

// Observability (dev control): a log of code-mode tool calls, drained by the harness after a run.
let codeToolLog: { name: string; ok: boolean }[] = [];
export function drainCodeToolCalls(): { name: string; ok: boolean }[] {
  const c = codeToolLog; codeToolLog = []; return c;
}

// The worker asked to call a registered tool (code-mode `call_tool`). Route it to Rust
// (`call_tool_from_code`, which enforces the profile opt-in + allowlist) and post the reply back.
async function handleCallTool(msg: { callId: number; name: string; args: string }): Promise<void> {
  if (!worker) return;
  const { callId, name } = msg;
  if (toolCallsRemaining <= 0) {
    worker.postMessage({ type: "call-tool-result", callId,
      error: "call_tool budget exceeded for this run (200 calls)." });
    return;
  }
  toolCallsRemaining--;
  try {
    let parsedArgs: unknown = {};
    try { parsedArgs = msg.args ? JSON.parse(msg.args) : {}; } catch { parsedArgs = {}; }
    const result = await invoke("call_tool_from_code", { call: { name, args: parsedArgs } });
    codeToolLog.push({ name, ok: true });
    worker.postMessage({ type: "call-tool-result", callId, result: JSON.stringify(result) });
  } catch (e) {
    codeToolLog.push({ name, ok: false });
    worker.postMessage({ type: "call-tool-result", callId, error: String(e) });
  }
}

function fail(msg: string): void {
  loadError = msg;
  console.error("[pyodide]", msg);
}

function ensureWorker(): Promise<void> {
  if (ready) return ready;
  ready = new Promise<void>((resolve) => {
    let settled = false;
    const done = () => { if (!settled) { settled = true; clearTimeout(watchdog); resolve(); } };
    const watchdog = setTimeout(() => {
      fail(`Pyodide worker did not finish loading within ${LOAD_TIMEOUT_MS / 1000}s.`);
      done();
    }, LOAD_TIMEOUT_MS);

    try {
      console.log("[pyodide] creating worker…");
      // Module worker (WKWebView doesn't support classic workers). Same-origin, native ESM.
      worker = new Worker("/pyodide-worker.js", { type: "module" });
    } catch (e) {
      fail("Failed to create the Pyodide worker: " + String(e));
      done();
      return;
    }

    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg?.type === "ready") { console.log("[pyodide] runtime ready"); done(); return; }
      if (msg?.type === "init-error") { fail("Pyodide failed to initialise: " + msg.error); done(); return; }
      if (msg?.type === "call-tool") { void handleCallTool(msg); return; }
      if (msg?.type === "result") {
        const cb = pending.get(msg.id);
        if (cb) { pending.delete(msg.id); cb(msg as PyResult); }
      }
    };
    worker.onerror = (e: ErrorEvent) => {
      fail("Pyodide worker error: " + (e.message || "failed to load /pyodide-worker.js"));
      done();
    };
  });
  return ready;
}

/** Pre-load Pyodide so the first real call (or a scheduled job) doesn't pay cold-start. */
export function warmPyodide(): void { void ensureWorker(); }

/** Execute Python in the sandbox with a staged workspace. Never rejects — errors come back on `.error`. */
export async function runPython(code: string, files: PyFile[]): Promise<PyResult> {
  await ensureWorker();
  if (loadError || !worker) {
    return { output: "", images: [], outFiles: [], error: loadError ?? "Pyodide worker unavailable." };
  }
  const id = ++seq;
  toolCallsRemaining = MAX_TOOL_CALLS_PER_RUN; // reset code-mode call budget per run
  return new Promise<PyResult>((resolve) => {
    pending.set(id, resolve);
    worker!.postMessage({ type: "run", id, code, files });
  });
}
