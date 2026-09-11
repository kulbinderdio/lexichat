# AutomationBench vs LexiChat — tool-handling review

Read-and-report only; no code changed. Assesses LexiChat against
[AutomationBench](https://arxiv.org/abs/2604.18934) (Zapier), which benchmarks agents on
cross-app REST workflows. Because LexiChat runs local models via Ollama, the paper's
**small-model** findings carry more weight here than its frontier numbers.

## Headline

**The paper's two primary prescriptions are already implemented in LexiChat.**

- "Give the agent `search` + `execute`, not every schema" → LexiChat has **`find_tools`**
  (model-driven discovery) + **`call_tool`** (code-mode execution). This is the same two-tool
  pattern.
- "Narrow the tool surface per task" → LexiChat scopes tools **per profile** (`enabled*Ids`,
  `maxTools`) and **per step** (discovery mode + LRU cap).

So the real opportunities are the paper's *other* findings, in order of value:
**completion verification (#4)**, **schema simplification (#3)**, and an **eval harness (#5)**.
Items #1 and #2 need refinement at most, not building.

---

## 1. Tool schema injection

**Current behaviour.** Tools are **not** all injected every turn.
- `src-tauri/src/ollama.rs:953` `SELECTION_THRESHOLD = 25`: at/under 25 discoverable tools they
  are all sent; above it, discovery engages.
- `ollama.rs:962` `find_tools_schema()` — the discovery meta-tool (the paper's `search`).
- `ollama.rs:1006` `search_tools()` — keyword scoring over tool name (+3) and description (+1),
  plus a whole-group match on the server label (the paper's BM25-over-schemas, cruder).
- `ollama.rs:1723` `discover_active`: in discovery mode the model gets built-ins + `find_tools`
  only; specialised tools load on demand and are held in an **LRU set capped at `maxTools`**
  (added this session — `loaded_tools`, `ollama.rs:~1618`).
- `call_tool` / code-mode (`ollama.rs:2432`, `list_tools`) — the paper's `execute`.

**Approx token cost** (measured this session; ~4 chars/token):
- Built-ins, always sent: **~3,528 tokens/step** (after this session's description trims).
- Wiki tools when memory on: **~625**.
- Per external group in discovery mode: near-zero until `find_tools` loads it; a whole-server
  load can add a lot (Blender's 28 tools ≈ 7,200 tokens; Playwright's 24 ≈ similar).
- With everything enabled at once, the external registry measured **~40,054 tokens** — which is
  exactly what discovery mode exists to avoid: those never all reach the model at once.
- So "2 / 5 / 10 servers attached" does **not** scale linearly in context — above the threshold,
  cost is driven by what `find_tools` has loaded this turn, not by how many servers are attached.

**Gap vs finding.** Small. The architecture matches the paper. Two refinements worth noting:
- `search_tools` scores by substring keyword, not BM25. For large registries a real BM25 (or an
  embedding match — the wiki index already embeds via Ollama `/api/embed`) would rank better.
- A **whole-group match loads the entire server** (e.g. all 28 Blender tools). The paper argues
  for top-k *individual* tools. A refinement: strip the terms that matched the group label, score
  the residual, and cap — falling back to the whole group only when there's no residual signal.

**Proposed change.** Refine `search_tools` ranking (BM25 or reuse the embed index) and add a
top-k cap to whole-group loads. **Size: M.**

**Risks.** The whole-group load exists deliberately (a variant that wouldn't keyword-match, e.g.
an inline MCP-App tool, must still be reachable) — see the comment at `search_tools`. Capping it
risks hiding the right tool; keep the whole-group fallback.

---

## 2. Per-profile tool scoping

**Current behaviour.** Fully supported — this is the "registry model" in CLAUDE.md.
- Profiles hold **enabled-ID references**: `enabledOpenapiSpecIds`, `enabledMcpServerIds`,
  `enabledSparqlEndpointIds`, plus `enabledTools` (built-in on/off map) — `src/App.tsx:~450`,
  `App.tsx:475`.
- `maxTools` per profile (`App.tsx:376`) is the hard per-step cap (default 40).
- **Per-tool MCP disabling** exists (`disabledMcpTools`, `AdminPanel` per-server tool checkboxes),
  narrowed again at call time in `dispatch`.
- `syncServers()` pushes only the active profile's enabled subset to the Rust backend.

**Gap vs finding.** None material. LexiChat already does allowlist-per-profile at server, spec,
endpoint, and individual-tool granularity. This is *more* granular than the paper assumes.

**Proposed change.** None required. (Optional S: a per-profile **denylist** for built-ins is
effectively already there via `enabledTools[name] = false`.)

**Risks.** One pre-existing quirk (noted this session): built-in tools default **on** when absent
from a profile's map, so a newly-added built-in silently appears in old profiles. Unrelated to
the paper but worth fixing alongside any scoping work. **Size: S.**

---

## 3. Schema simplification

**Current behaviour.** Partial, and only on the *discovery menu*, not the *loaded schema*.
- `find_tools`/`search_tools` present each discovered tool as `first_sentence(desc, 120)`
  (`ollama.rs:1013`, `1137`); group descriptions truncate to 240 (`ollama.rs:1097`).
- Built-in tool descriptions were hand-trimmed this session (`run_python` 931→329 tokens, etc.).
- **But** once a tool is loaded/called, its **full raw schema** goes to the model — no pass trims
  long parameter descriptions, drops rarely-used optional params, or flattens nested objects.
  For verbose OpenAPI-generated or MCP schemas this is the bulk of the cost.

**Gap vs finding.** Real. The paper found curated/simplified schemas outscore raw ones, and
smaller models benefit most. LexiChat compacts the *menu* but ships *raw* loaded schemas.

**Proposed change.** A compaction pass applied where a `ToolSchema` is assembled for the wire —
`openapi::parse_spec` (generation time) and the MCP tool-list ingestion (`mcp.rs`), or a single
normalise step in `agent_loop` before send. Trim descriptions to N chars, drop `default`/`example`
noise, optionally elide optional params below a relevance bar, flatten one level of nesting.
**Size: M.**

**Risks.** Over-trimming a parameter description can make a model mis-call a tool (the opposite
of the goal). Must be conservative and reversible (keep required params + their descriptions
intact). OpenAPI path/query param semantics must survive — `openapi::execute` still needs the
real names. Best done behind a per-profile toggle so it can be measured, not assumed.

---

## 4. Completion verification  ← highest value

**Current behaviour.** Nothing verifies a success claim.
- `ensure_final_answer` (`ollama.rs:1306`) forces the model to *write* an answer when a run ends
  without one — it does not check the answer is true.
- Honesty is handled by **prompting only** — the "BE HONEST" / "never invent a field" sections in
  task prompts, and the offload-truncation notices. Effective but not enforced.
- The existing guards (`GLOBAL_TOOL_CALL_CAP`, wall-clock, `runaway_repetition`) bound *runaway*
  behaviour, not *false completion*.

**Gap vs finding.** This is the paper's **dominant failure mode** (false completion: 72–91% of
frontier failures; and the "processed part of a list, summarised as if whole" pattern). It hits
small local models hardest — exactly LexiChat's target. LexiChat has no defence beyond prompting.

**Proposed change.** An **optional per-profile "verify before done" step**: when the model emits
its final answer, if the profile opts in, run one extra bounded turn that (a) re-reads the state
the task claims to have changed (re-call the read tools / re-fetch), and (b) asks the model to
confirm its claim against what it just read, or correct it. Cheapest hook: in `agent_loop`, at the
point a final answer is detected, before `agent-done` — gated by a profile flag so it costs
nothing when off. **Size: M.**

**Risks / conflicts.**
- The agent loop **streams**; a post-hoc verify turn either streams after the answer (a visible
  "checking…" phase) or must buffer the answer until verified. The former is simpler and honest.
- Interacts with the wall-clock budget — a verify turn needs its own small budget or it can be
  cut (this session's `ensure_final_answer` already loses tool access on timeout; verify would
  need tools, so budget it explicitly).
- Only meaningful when the task *mutated* state or *claims a complete list*; for a plain chat
  answer it's wasted. The profile flag makes that the user's call. A lighter first version:
  detect "list-shaped" claims and re-read only then, rather than always.

---

## 5. Eval harness

**Current behaviour.** Strong foundation, missing the assertion layer.
- **dev-control** (`lib.rs:114`, `/dev/run`, `scripts/lexi-dev.mjs`) already drives LexiChat
  headlessly and returns a full trace — used all session to A/B models and prompts.
- `usage.jsonl` records per-turn prompt/completion tokens, steps, tools, duration, error — enough
  to compare models quantitatively (already mined this session).
- An "Eval Tester" profile exists.
- **Missing:** a mock MCP server seeded with known state, and deterministic positive/negative
  assertions over *final* state after a task.

**Gap vs finding.** The paper's grading (end-state assertions + negative "must NOT have happened"
assertions to block reward-hacking) has no local equivalent. Without it, "which Ollama model is
usable with LexiChat's tools" is judged by eye, not by assertion.

**Proposed change.** A local harness: (1) a scriptable **mock MCP stdio server** (a small Node/
Python script LexiChat connects to like any MCP server) exposing a few CRUD tools over an
in-memory store seeded per scenario; (2) a runner that seeds state, fires a task via `/dev/run`,
then checks positive + negative assertions against the mock's final store; (3) a matrix over
installed Ollama models (now easy to enumerate via this session's model-manager commands).
**Size: L** (but incremental — the runner and model matrix are S/M on top of existing dev-control;
the mock server + scenarios are the bulk).

**Risks.** Mock fidelity — a mock that's easier than real APIs over-reports model capability.
Keep scenarios adversarial (irrelevant/misleading tools present, as the paper does). Negative
assertions are the important half and the easy half to forget.

---

## Recommended ordering — (impact on small local models) / (effort)

1. **#4 Completion verification (M).** Highest impact: it targets the paper's dominant failure
   mode, which is worst on small models — LexiChat's exact audience — and there is currently no
   defence but prompting. Ship it per-profile so it's opt-in and measurable.
2. **#3 Schema simplification (M).** Directly trims per-step context, which this session showed is
   the dominant local-model latency factor, and the paper ties simpler schemas to higher small-
   model scores. Low user-facing risk behind a toggle.
3. **#5 Eval harness (L, incremental).** Not user-facing, but it's the only way to *know* #4/#3
   helped and which Ollama models are actually usable. Build the runner + model matrix first
   (cheap, reuses dev-control), then grow scenarios.
4. **#1 retrieval refinement (M) / #2 scoping (S).** Mostly done. Do only the cheap wins: better
   `search_tools` ranking, a top-k cap on whole-group loads, and the built-in default-on fix.

## Where the paper does not transfer

- **#1 and #2 are already built.** The task brief asked "where would a retrieval layer slot in?"
  and "identify the config changes for an allowlist" — both already exist (`find_tools`/`call_tool`,
  `enabled*Ids`/`maxTools`). Treat these as validation, not backlog.
- **Frontier numbers (9.6%→14.3% etc.) don't map to Ollama-class models** and shouldn't be quoted
  as targets. The *direction* (narrowing helps small models disproportionately) is what transfers.
- **Multi-app state mutation is the benchmark's core**; much LexiChat use is read/analyse (crime
  stats, tenders, SPARQL). Completion verification (#4) matters most where a task *writes* state or
  *claims a full list* — the profile flag scopes it to those uses rather than taxing every chat.
