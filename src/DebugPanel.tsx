import { useState, useEffect, useRef } from "react";
import { listen, UnlistenFn } from "@tauri-apps/api/event";

// ── Types ─────────────────────────────────────────────────────────────────────

interface DebugStep {
  index: number;
  schemaNames: string[];
  llmText?: string;
  durationMs?: number;
  toolCalls: { name: string; args: string }[];
  toolResults: { name: string; result: string }[];
  tokens: string;
  thinking: string;
}

interface DebugRun {
  id: number;
  steps: DebugStep[];
  totalMs?: number;
  error?: string;
  done: boolean;
}

// ── Step row ──────────────────────────────────────────────────────────────────

function StepRow({ step, isLast }: { step: DebugStep; isLast: boolean }) {
  const [open, setOpen] = useState(isLast);
  const [schemasOpen, setSchemasOpen] = useState(false);

  return (
    <div style={{ marginBottom: 4 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", textAlign: "left", background: "var(--dbg-step-bg)",
          border: "1px solid var(--dbg-border)", borderRadius: 6,
          padding: "5px 10px", cursor: "pointer", display: "flex",
          alignItems: "center", gap: 8, fontSize: 12,
        }}
      >
        <span style={{ fontSize: 10, opacity: 0.5 }}>{open ? "▼" : "▶"}</span>
        <span style={{ fontWeight: 600 }}>Step {step.index + 1}</span>
        {step.toolCalls.length > 0 && (
          <span style={{ fontSize: 10, background: "var(--purple)", color: "#fff",
            padding: "1px 6px", borderRadius: 10 }}>
            {step.toolCalls.length} tool{step.toolCalls.length > 1 ? "s" : ""}
          </span>
        )}
        {step.durationMs !== undefined && (
          <span style={{ fontSize: 10, opacity: 0.45, marginLeft: "auto" }}>{step.durationMs}ms</span>
        )}
      </button>

      {open && (
        <div style={{ paddingLeft: 8, paddingTop: 4, display: "flex", flexDirection: "column", gap: 4 }}>
          {/* Schemas */}
          {step.schemaNames.length > 0 && (
            <div>
              <button
                onClick={() => setSchemasOpen(o => !o)}
                style={{ background: "none", border: "none", cursor: "pointer", padding: "2px 0",
                  fontSize: 11, color: "var(--dbg-schemas-color)", display: "flex", alignItems: "center", gap: 4 }}
              >
                <span style={{ fontSize: 9 }}>{schemasOpen ? "▼" : "▶"}</span>
                Schemas ({step.schemaNames.length} tools sent)
              </button>
              {schemasOpen && (
                <div style={{ paddingLeft: 14, paddingBottom: 4 }}>
                  {step.schemaNames.map(n => (
                    <div key={n} style={{ fontSize: 11, fontFamily: "monospace", opacity: 0.7 }}>{n}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Thinking */}
          {step.thinking && (
            <details style={{ fontSize: 11 }}>
              <summary style={{
                cursor: "pointer", userSelect: "none", opacity: 0.55,
                padding: "2px 0", listStyle: "none", display: "flex", alignItems: "center", gap: 4,
              }}>
                <span style={{ fontSize: 9 }}>▶</span>
                <span>💭 Thinking ({step.thinking.length} chars)</span>
              </summary>
              <div style={{
                background: "var(--dbg-text-bg)", borderRadius: 4, padding: "6px 8px",
                fontSize: 11, fontFamily: "monospace", whiteSpace: "pre-wrap",
                wordBreak: "break-word", maxHeight: 200, overflowY: "auto",
                opacity: 0.6, fontStyle: "italic", marginTop: 4,
                borderLeft: "2px solid var(--dbg-border)",
              }}>
                {step.thinking}
              </div>
            </details>
          )}

          {/* LLM output */}
          {(step.llmText || step.tokens) && (
            <div style={{
              background: "var(--dbg-text-bg)", borderRadius: 4, padding: "6px 8px",
              fontSize: 11, fontFamily: "monospace", whiteSpace: "pre-wrap",
              wordBreak: "break-word", maxHeight: 160, overflowY: "auto",
              opacity: step.llmText ? 1 : 0.6,
            }}>
              {step.llmText || step.tokens || <span style={{ opacity: 0.4 }}>(no text output)</span>}
            </div>
          )}

          {/* Tool calls + results */}
          {step.toolCalls.map((tc, i) => (
            <div key={i}>
              <div style={{
                background: "var(--dbg-tool-bg)", border: "1px solid var(--purple)33",
                borderRadius: 4, padding: "5px 8px",
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--purple)", marginBottom: 2 }}>
                  ⚡ {tc.name}
                </div>
                <pre style={{ margin: 0, fontSize: 10, fontFamily: "monospace",
                  whiteSpace: "pre-wrap", wordBreak: "break-all", opacity: 0.8 }}>
                  {tc.args}
                </pre>
              </div>
              {step.toolResults[i] && (
                <div style={{
                  background: "var(--dbg-result-bg)", borderRadius: 4, padding: "5px 8px",
                  marginTop: 2, fontSize: 10, fontFamily: "monospace",
                  whiteSpace: "pre-wrap", wordBreak: "break-all",
                  maxHeight: 100, overflowY: "auto", opacity: 0.7,
                }}>
                  {step.toolResults[i].result}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Run row ───────────────────────────────────────────────────────────────────

function RunRow({ run }: { run: DebugRun }) {
  const [open, setOpen] = useState(true);

  return (
    <div style={{ marginBottom: 8 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", textAlign: "left", background: "var(--dbg-run-bg)",
          border: "1px solid var(--dbg-border)", borderRadius: 6,
          padding: "6px 10px", cursor: "pointer", display: "flex",
          alignItems: "center", gap: 8, fontSize: 12,
        }}
      >
        <span style={{ fontSize: 10, opacity: 0.5 }}>{open ? "▼" : "▶"}</span>
        <span style={{ fontWeight: 700, fontSize: 11, opacity: 0.6 }}>RUN #{run.id}</span>
        {!run.done && (
          <span style={{ fontSize: 10, color: "#60a5fa" }}>running…</span>
        )}
        {run.done && run.error && (
          <span style={{ fontSize: 10, color: "#f87171" }}>✕ error</span>
        )}
        {run.done && !run.error && (
          <span style={{ fontSize: 10, color: "#4ade80" }}>✓ done</span>
        )}
        {run.totalMs !== undefined && (
          <span style={{ fontSize: 10, opacity: 0.4, marginLeft: "auto" }}>{run.totalMs}ms</span>
        )}
      </button>

      {open && (
        <div style={{ paddingLeft: 8, paddingTop: 4 }}>
          {run.steps.map((step, i) => (
            <StepRow key={step.index} step={step} isLast={i === run.steps.length - 1} />
          ))}
          {run.error && (
            <div style={{ fontSize: 11, color: "#f87171", padding: "4px 8px" }}>
              {run.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Debug Panel ───────────────────────────────────────────────────────────────

interface Props {
  visible: boolean;
  clearKey?: number;
}

export function DebugPanel({ visible, clearKey }: Props) {
  const [runs, setRuns] = useState<DebugRun[]>([]);
  const runCounter = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (clearKey === undefined) return;
    setRuns([]);
    runCounter.current = 0;
  }, [clearKey]);

  useEffect(() => {
    if (!visible) return;
    const unsubs: UnlistenFn[] = [];

    const setup = async () => {
      // New step starting
      unsubs.push(await listen<{ step: number; schema_names: string[] }>("debug-step-start", ({ payload }) => {
        if (payload.step === 0) {
          // New run
          runCounter.current += 1;
          const runId = runCounter.current;
          setRuns(prev => [...prev, {
            id: runId, steps: [{
              index: 0,
              schemaNames: payload.schema_names,
              toolCalls: [], toolResults: [], tokens: "", thinking: "",
            }], done: false,
          }]);
        } else {
          setRuns(prev => {
            if (prev.length === 0) return prev;
            const runs = [...prev];
            const run = { ...runs[runs.length - 1] };
            run.steps = [...run.steps, {
              index: payload.step,
              schemaNames: payload.schema_names,
              toolCalls: [], toolResults: [], tokens: "", thinking: "",
            }];
            runs[runs.length - 1] = run;
            return runs;
          });
        }
      }));

      // Thinking tokens
      unsubs.push(await listen<{ delta: string }>("agent-thinking", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          if (run.steps.length === 0) return prev;
          const steps = [...run.steps];
          const last = { ...steps[steps.length - 1] };
          last.thinking = (last.thinking || "") + payload.delta;
          steps[steps.length - 1] = last;
          run.steps = steps;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));

      // Streaming tokens
      unsubs.push(await listen<{ delta: string }>("agent-token", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          if (run.steps.length === 0) return prev;
          const steps = [...run.steps];
          const last = { ...steps[steps.length - 1] };
          last.tokens = (last.tokens || "") + payload.delta;
          steps[steps.length - 1] = last;
          run.steps = steps;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));

      // Step done
      unsubs.push(await listen<{ step: number; llm_text: string; duration_ms: number }>("debug-step-done", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          const steps = [...run.steps];
          const idx = steps.findIndex(s => s.index === payload.step);
          if (idx >= 0) {
            steps[idx] = { ...steps[idx], llmText: payload.llm_text, durationMs: payload.duration_ms, tokens: "" };
          }
          run.steps = steps;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));

      // Tool call
      unsubs.push(await listen<{ name: string; args: string }>("agent-tool-call", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          const steps = [...run.steps];
          const last = { ...steps[steps.length - 1] };
          last.toolCalls = [...last.toolCalls, { name: payload.name, args: payload.args }];
          steps[steps.length - 1] = last;
          run.steps = steps;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));

      // Tool result
      unsubs.push(await listen<{ name: string; result: string }>("agent-tool-result", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          const steps = [...run.steps];
          const last = { ...steps[steps.length - 1] };
          last.toolResults = [...last.toolResults, { name: payload.name, result: payload.result }];
          steps[steps.length - 1] = last;
          run.steps = steps;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));

      // Run done
      unsubs.push(await listen<{ total_ms: number; error?: string }>("debug-run-done", ({ payload }) => {
        setRuns(prev => {
          if (prev.length === 0) return prev;
          const runs = [...prev];
          const run = { ...runs[runs.length - 1] };
          run.done = true;
          run.totalMs = payload.total_ms;
          run.error = payload.error;
          runs[runs.length - 1] = run;
          return runs;
        });
      }));
    };

    setup();
    return () => { unsubs.forEach(u => u()); };
  }, [visible]);

  // Scroll to bottom on new data
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [runs]);

  if (!visible) return null;

  return (
    <div style={{
      width: 320, minWidth: 260, maxWidth: 400, height: "100%",
      borderLeft: "1px solid var(--dbg-border)",
      background: "var(--dbg-bg)",
      display: "flex", flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 14px", borderBottom: "1px solid var(--dbg-border)",
        display: "flex", alignItems: "center", gap: 8,
        fontSize: 12, fontWeight: 600, opacity: 0.7,
        flexShrink: 0,
      }}>
        <span>🔍</span>
        <span>Agent Trace</span>
        {runs.length > 0 && (
          <button
            onClick={() => setRuns([])}
            style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer",
              fontSize: 10, opacity: 0.4, padding: 0 }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
        {runs.length === 0 && (
          <div style={{ fontSize: 12, opacity: 0.35, textAlign: "center", marginTop: 40 }}>
            Send a message to see the agent trace.
          </div>
        )}
        {runs.map(run => <RunRow key={run.id} run={run} />)}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
