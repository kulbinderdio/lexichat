/* Classic Web Worker: hosts the Pyodide (WASM CPython) runtime for `run_python`.
 *
 * Loaded once and kept warm. Per run it sets up a workspace in Pyodide's in-memory FS,
 * writes the staged files (user uploads + offloaded tool results) into it, executes the
 * model's code with normal Python I/O, then returns stdout, any matplotlib figures (as
 * PNGs), and files the model wrote to /work/out. WASM has no host filesystem, so the code
 * can only see what we staged in — the sandbox is structural.
 *
 * Same-origin static ESM asset (served from public/). Runs as a MODULE worker because
 * WKWebView (macOS) doesn't support classic workers, so Pyodide is loaded via native ESM
 * import rather than importScripts. CSP 'self' + 'wasm-unsafe-eval' permit this.
 */
import { loadPyodide } from "/pyodide/pyodide.mjs";

let pyodide = null;
let ready = false;

// ── base64 <-> bytes (binary-safe) ──────────────────────────────────────────────
function b64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function bytesToB64(bytes) {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function mkdirp(path) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur += "/" + p;
    try { pyodide.FS.mkdir(cur); } catch { /* exists */ }
  }
}

// ── Code-mode: call registered tools from Python (call_tool / list_tools) ──────
// Python awaits a JS promise that round-trips through the main thread to Rust and back.
let callSeq = 0;
const pendingCalls = new Map();
globalThis.__lexiCallTool = (name, argsJson) => new Promise((resolve, reject) => {
  const callId = ++callSeq;
  pendingCalls.set(callId, { resolve, reject });
  postMessage({ type: "call-tool", callId, name, args: argsJson });
});

async function init() {
  try {
    console.log("[pyodide-worker] loadPyodide indexURL=/pyodide/ …");
    pyodide = await loadPyodide({ indexURL: "/pyodide/" });
    // Define the code-mode tool API. call_tool/list_tools are async — user code MUST await them.
    pyodide.runPython(`
import json as _json
from js import __lexiCallTool as _lexi_call
async def call_tool(name, args=None):
    """Call a registered LexiChat tool. Returns a dict/list (parsed JSON) or a string.
    Must be awaited:  data = await call_tool("tool_name", {"k": "v"})"""
    _raw = await _lexi_call(name, _json.dumps(args or {}))
    try: return _json.loads(str(_raw))
    except Exception: return str(_raw)
async def list_tools():
    """List the tools callable from code:  [{'name','description'}, ...]. Must be awaited."""
    _raw = await _lexi_call("__list__", "{}")
    try: return _json.loads(str(_raw))
    except Exception: return []
`);
    ready = true;
    console.log("[pyodide-worker] ready");
    postMessage({ type: "ready" });
    // Preload the common science stack in the background so the FIRST chart doesn't pay a
    // multi-second package-load stall mid-run. `ready` is already posted, so this doesn't
    // delay startup; it warms while the app idles. All wheels are local (offline).
    pyodide.loadPackage(["numpy", "pandas", "matplotlib"])
      .then(() => console.log("[pyodide-worker] science stack preloaded"))
      .catch((e) => console.warn("[pyodide-worker] preload skipped:", String(e)));
  } catch (e) {
    const msg = String(e && e.message ? e.message : e);
    console.error("[pyodide-worker] init failed:", msg);
    postMessage({ type: "init-error", error: msg });
    throw e;
  }
}

// Reset the workspace to a clean state for each run.
function resetWorkspace() {
  // Best-effort wipe of /work between runs so state never leaks across calls.
  try { pyodide.runPython("import shutil, os\nshutil.rmtree('/work', ignore_errors=True)"); } catch { /* first run */ }
  for (const d of ["/work", "/work/uploads", "/work/data", "/work/out"]) mkdirp(d);
}

// Capture any open matplotlib figures as inline PNG data URLs (no-op if matplotlib absent).
function captureFigures() {
  try {
    const b64s = pyodide.runPython(`
def __lexi_capture():
    import sys
    if 'matplotlib' not in sys.modules:
        return []
    import base64, io
    import matplotlib.pyplot as plt
    out = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format='png', bbox_inches='tight')
        out.append(base64.b64encode(buf.getvalue()).decode())
    plt.close('all')
    return out
__lexi_capture()
`);
    const arr = b64s.toJs ? b64s.toJs() : b64s;
    if (b64s.destroy) b64s.destroy();
    return Array.from(arr || []).map((b64) => "data:image/png;base64," + b64);
  } catch { return []; }
}

// data: URL for an output file if it's a displayable image/SVG, else null.
const IMG_MIME = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", gif: "image/gif",
  webp: "image/webp", svg: "image/svg+xml", bmp: "image/bmp" };
function imageDataUrl(name, b64) {
  const ext = (name.split(".").pop() || "").toLowerCase();
  const mime = IMG_MIME[ext];
  return mime ? `data:${mime};base64,${b64}` : null;
}

// Collect files the model wrote under /work/out.
function collectOutputs() {
  const out = [];
  const walk = (dir) => {
    let entries = [];
    try { entries = pyodide.FS.readdir(dir).filter((e) => e !== "." && e !== ".."); } catch { return; }
    for (const e of entries) {
      const full = dir + "/" + e;
      const st = pyodide.FS.stat(full);
      if (pyodide.FS.isDir(st.mode)) walk(full);
      else out.push({ name: full.slice("/work/out/".length), b64: bytesToB64(pyodide.FS.readFile(full)) });
    }
  };
  walk("/work/out");
  return out;
}

async function run(code, files) {
  resetWorkspace();
  for (const f of files || []) {
    const path = "/work/" + f.path.replace(/^\/+/, "");
    mkdirp(path.slice(0, path.lastIndexOf("/")));
    pyodide.FS.writeFile(path, b64ToBytes(f.b64));
  }

  let stdout = "";
  pyodide.setStdout({ batched: (s) => { stdout += s + "\n"; } });
  pyodide.setStderr({ batched: (s) => { stdout += s + "\n"; } });

  // Force a headless matplotlib backend before user code imports pyplot.
  pyodide.runPython("import os; os.chdir('/work'); os.environ['MPLBACKEND'] = 'AGG'");

  let error = null;
  try {
    // Auto-load any bundled packages the code imports (numpy/pandas/matplotlib), then run.
    await pyodide.loadPackagesFromImports(code);
    await pyodide.runPythonAsync(code);
  } catch (e) {
    error = String(e && e.message ? e.message : e);
  }

  const images = captureFigures();
  let outFiles = collectOutputs();
  // Charts are shown INLINE (with a download button) — they must NOT also count as output files,
  // or a model that savefig()s a displayed chart triggers an intrusive "save to a folder" prompt
  // for something already on screen. So: any image in /work/out is rendered inline and dropped
  // from outFiles; only genuine non-image outputs (xlsx/csv/pdf the user asked for) remain to save.
  if (images.length === 0) {
    // No open figure captured — promote saved image files to the inline view, then drop them.
    outFiles = outFiles.filter((f) => {
      const url = imageDataUrl(f.name, f.b64);
      if (url) { images.push(url); return false; }
      return true;
    });
  } else {
    // A figure was captured inline already — drop redundant image files (the same chart).
    outFiles = outFiles.filter((f) => !imageDataUrl(f.name, f.b64));
  }
  return { output: stdout, images, outFiles, error };
}

const initPromise = init();
initPromise.catch(() => {}); // init-error is already posted; suppress unhandled-rejection noise

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type === "call-tool-result") {
    const p = pendingCalls.get(msg.callId);
    if (!p) return;
    pendingCalls.delete(msg.callId);
    if (msg.error) p.reject(new Error(msg.error));
    else p.resolve(msg.result);   // JSON string; Python call_tool json.loads() it
    return;
  }
  if (msg.type !== "run") return;
  try {
    if (!ready) await initPromise;
    const res = await run(msg.code, msg.files);
    postMessage({ type: "result", id: msg.id, ...res });
  } catch (e) {
    postMessage({ type: "result", id: msg.id, output: "", images: [], outFiles: [],
      error: "python runtime error: " + String(e && e.message ? e.message : e) });
  }
};
