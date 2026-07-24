// Stages the Pyodide runtime + the numpy/pandas/matplotlib wheel closure into public/pyodide/
// so Vite serves them same-origin (CSP 'self'). Wheels are downloaded ONCE at build time from
// the Pyodide CDN (or reused from node_modules / a previous run); the app itself never fetches
// from the network — the runtime stays fully offline/private. Idempotent; safe to re-run.
import { mkdirSync, copyFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "pyodide");
const dest = join(root, "public", "pyodide");

if (!existsSync(src)) {
  console.error("[prepare-pyodide] node_modules/pyodide not found — run `npm install pyodide`.");
  process.exit(1);
}
mkdirSync(dest, { recursive: true });

// Core runtime (ESM entry + asm + stdlib + package index).
const core = ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"];
for (const f of core) copyFileSync(join(src, f), join(dest, f));

// Resolve the dependency closure of the science stack from the lock file.
const lock = JSON.parse(readFileSync(join(dest, "pyodide-lock.json"), "utf8"));
const version = JSON.parse(readFileSync(join(src, "package.json"), "utf8")).version;
const cdn = `https://cdn.jsdelivr.net/pyodide/v${version}/full/`;
const pkgs = lock.packages || {};
const roots = ["numpy", "pandas", "matplotlib", "scipy", "sympy", "beautifulsoup4", "geopandas"];
const closure = new Set();
const stack = [...roots];
while (stack.length) {
  const name = stack.pop();
  if (closure.has(name)) continue;
  const p = pkgs[name];
  if (!p) { console.warn(`[prepare-pyodide] '${name}' not in lock — skipping`); continue; }
  closure.add(name);
  for (const d of p.depends || []) stack.push(d);
}

let have = 0, fetched = 0;
for (const name of closure) {
  const file = pkgs[name].file_name;
  const out = join(dest, file);
  if (existsSync(out)) { have++; continue; }                         // already staged
  const cached = join(src, file);
  if (existsSync(cached)) { copyFileSync(cached, out); have++; continue; } // cached in node_modules
  try {
    const res = await fetch(cdn + file);
    if (!res.ok) { console.warn(`[prepare-pyodide] fetch ${file} -> HTTP ${res.status}`); continue; }
    writeFileSync(out, Buffer.from(await res.arrayBuffer()));
    fetched++;
  } catch (e) {
    console.warn(`[prepare-pyodide] fetch ${file} failed: ${e.message}`);
  }
}
// Pure-Python packages not in the Pyodide distribution: fetch their py3-none-any wheels from
// PyPI and register them in the local lock so `loadPackagesFromImports` auto-loads them offline
// at runtime (no CDN, no micropip network calls). Order matters — deps before dependents.
const pypi = [
  // Pyodide canonicalises package names to lowercase-hyphenated; the Python import name keeps
  // its underscore. Register/depend on the canonical "et-xmlfile", import "et_xmlfile".
  { name: "et-xmlfile", imports: ["et_xmlfile"], depends: [] },
  { name: "openpyxl",   imports: ["openpyxl"],   depends: ["et-xmlfile"] },
];
let pypiCount = 0;
for (const p of pypi) {
  if (pkgs[p.name]) { pypiCount++; continue; }
  try {
    const meta = await (await fetch(`https://pypi.org/pypi/${p.name}/json`)).json();
    const version = meta.info.version;
    const rel = (meta.releases[version] || []).find(f =>
      f.filename.endsWith("-py3-none-any.whl") || f.filename.endsWith("-py2.py3-none-any.whl"));
    if (!rel) { console.warn(`[prepare-pyodide] no pure wheel for ${p.name}`); continue; }
    const out = join(dest, rel.filename);
    if (!existsSync(out)) { writeFileSync(out, Buffer.from(await (await fetch(rel.url)).arrayBuffer())); fetched++; }
    const sha256 = createHash("sha256").update(readFileSync(out)).digest("hex");
    pkgs[p.name] = { name: p.name, version, file_name: rel.filename, install_dir: "site",
      sha256, package_type: "package", imports: p.imports, depends: p.depends, unvendored_tests: false };
    pypiCount++;
  } catch (e) { console.warn(`[prepare-pyodide] PyPI ${p.name} failed: ${e.message}`); }
}
// Persist the lock with the injected PyPI entries so the runtime resolver knows about them.
writeFileSync(join(dest, "pyodide-lock.json"), JSON.stringify(lock));

console.log(`[prepare-pyodide] core (${core.length}) + wheels: ${have} present, ${fetched} downloaded (${closure.size} in closure) + ${pypiCount} PyPI (openpyxl) → public/pyodide/`);
