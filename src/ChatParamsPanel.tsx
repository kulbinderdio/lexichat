import { useState, useRef, useEffect } from "react";
import ReactDOM from "react-dom";
import { SlidersHorizontal } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ChatParams {
  style: "precise" | "balanced" | "creative";
  responseLength: "short" | "medium" | "long" | "auto";
  contextSize: "short" | "long";
  // Advanced overrides (undefined = use preset value)
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  seed?: number | null;
  numCtx?: number;
  numPredict?: number;
  stopSequences?: string;
  systemPromptOverride?: string;
  keepAlive?: string;
  // Reasoning toggle for thinking models (Qwen3, etc.). undefined = model default.
  // "off" skips the pre-answer reasoning pass — much faster per turn on local models.
  reasoning?: "on" | "off";
}

export const DEFAULT_CHAT_PARAMS: ChatParams = {
  style: "balanced",
  responseLength: "auto",
  contextSize: "short",
};

export function resolveParams(p: ChatParams): {
  temperature?: number;
  numCtx?: number;
  numPredict?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  seed?: number | null;
  stop?: string[];
  keepAlive?: string;
  systemPromptOverride?: string;
  think?: boolean;
} {
  // Only non-default presets override Ollama — "balanced", "auto", "short" let
  // the model use its own defaults so we never accidentally truncate tool schemas.
  const styleTemp: Record<ChatParams["style"], number | undefined> = { precise: 0.2, balanced: undefined, creative: 1.0 };
  const lengthTokens: Record<ChatParams["responseLength"], number | undefined> = { short: 256, medium: 1024, long: 4096, auto: undefined };
  const ctxTokens: Record<ChatParams["contextSize"], number | undefined> = { short: undefined, long: 8192 };
  return {
    temperature: p.temperature ?? styleTemp[p.style],
    numCtx: p.numCtx ?? ctxTokens[p.contextSize],
    numPredict: p.numPredict ?? lengthTokens[p.responseLength],
    topP: p.topP,
    topK: p.topK,
    repeatPenalty: p.repeatPenalty,
    seed: p.seed,
    stop: p.stopSequences ? p.stopSequences.split(",").map(s => s.trim()).filter(Boolean) : undefined,
    keepAlive: p.keepAlive,
    systemPromptOverride: p.systemPromptOverride || undefined,
    think: p.reasoning === "off" ? false : p.reasoning === "on" ? true : undefined,
  };
}

export function hasAdvancedOverrides(p: ChatParams): boolean {
  return p.temperature !== undefined || p.topP !== undefined || p.topK !== undefined ||
    p.repeatPenalty !== undefined || (p.seed !== undefined && p.seed !== null) ||
    p.numCtx !== undefined || p.numPredict !== undefined ||
    !!p.stopSequences || !!p.systemPromptOverride || !!p.keepAlive ||
    p.reasoning !== undefined;
}

export function hasCustomParams(p: ChatParams): boolean {
  const d = DEFAULT_CHAT_PARAMS;
  return p.style !== d.style || p.responseLength !== d.responseLength ||
    p.contextSize !== d.contextSize || hasAdvancedOverrides(p);
}

// ── Shared primitives ─────────────────────────────────────────────────────────

function ChipRow<T extends string>({
  label, desc, value, options, onChange,
}: {
  label: string; desc?: string; value: T;
  options: { value: T; label: string; desc?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
      {desc && <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>{desc}</div>}
      <div style={{ display: "flex", gap: 5 }}>
        {options.map(opt => (
          <button key={opt.value} title={opt.desc} onClick={() => onChange(opt.value)} style={{
            flex: 1, padding: "5px 0", fontSize: 12,
            fontWeight: value === opt.value ? 600 : 400,
            background: value === opt.value ? "var(--accent)" : "var(--surface3)",
            color: value === opt.value ? "#fff" : "var(--text-secondary)",
            border: value === opt.value ? "1px solid var(--accent)" : "1px solid var(--border)",
            borderRadius: 7, cursor: "pointer", transition: "all 0.12s",
          }}>{opt.label}</button>
        ))}
      </div>
    </div>
  );
}

// ── Advanced params form (shared between dialog and Settings tab) ──────────────

export function AdvancedParamsContent({
  draft, onChange,
}: { draft: ChatParams; onChange: (p: ChatParams) => void }) {
  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => onChange({ ...draft, [k]: v });
  const styleTemp: Record<ChatParams["style"], number | undefined> = { precise: 0.2, balanced: undefined, creative: 1.0 };
  const lengthTokens: Record<ChatParams["responseLength"], number | undefined> = { short: 256, medium: 1024, long: 4096, auto: undefined };
  const ctxTokens: Record<ChatParams["contextSize"], number | undefined> = { short: undefined, long: 8192 };

  const SliderRow = ({ label, tooltip, value, min, max, step, presetValue, onChg }: {
    label: string; tooltip?: string; value: number | undefined;
    min: number; max: number; step: number; presetValue: number | undefined;
    onChg: (v: number | undefined) => void;
  }) => {
    const display = value ?? presetValue ?? min;
    const usingDefault = value === undefined;
    return (
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
          <span title={tooltip} style={{ fontSize: 12, fontWeight: 500, cursor: tooltip ? "help" : undefined }}>
            {label}{tooltip ? " ?" : ""}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 600 }}>
              {usingDefault ? "model default" : display.toFixed(step < 1 ? 2 : 0)}
            </span>
            {!usingDefault && (
              <button onClick={() => onChg(undefined)} title="Reset to model default"
                style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer", padding: 0 }}>↺</button>
            )}
          </div>
        </div>
        <input type="range" min={min} max={max} step={step} value={display}
          onChange={e => onChg(parseFloat(e.target.value))}
          style={{ width: "100%", accentColor: "var(--accent)", opacity: usingDefault ? 0.45 : 1 }}
        />
        {usingDefault && <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 2 }}>Not set — model uses its own default</div>}
      </div>
    );
  };

  const numInput = (label: string, tooltip: string, value: number | undefined | null, placeholder: string, min: number, max: number, step: number, onChg: (v: number | undefined) => void, note?: string) => (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span title={tooltip} style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>{label} ?</span>
        {value !== undefined && value !== null && (
          <button onClick={() => onChg(undefined)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ reset</button>
        )}
      </div>
      <input type="number" min={min} max={max} step={step}
        value={value ?? ""}
        placeholder={placeholder}
        onChange={e => onChg(e.target.value ? parseInt(e.target.value) : undefined)}
        style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
      />
      {note && <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 3 }}>{note}</div>}
    </div>
  );

  return (
    <>
      <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 14, padding: "6px 8px", background: "var(--surface2)", borderRadius: 6 }}>
        Preset: <b>{draft.style}</b> · These override the preset values for this context.
      </div>

      <ChipRow<"default" | "on" | "off">
        label="Reasoning"
        desc="Thinking models (Qwen3, etc.) reason before answering. Off is much faster per turn; On is better for hard multi-step tasks. (Ollama only.)"
        value={draft.reasoning ?? "default"}
        options={[
          { value: "default", label: "Model default" },
          { value: "off", label: "Off (faster)" },
          { value: "on", label: "On" },
        ]}
        onChange={v => set("reasoning", v === "default" ? undefined : v)}
      />

      <SliderRow label="Temperature"
        tooltip="Controls randomness. Lower = more focused, Higher = more creative."
        value={draft.temperature} min={0} max={2} step={0.05}
        presetValue={styleTemp[draft.style]} onChg={v => set("temperature", v)} />

      <SliderRow label="Top-p (nucleus sampling)"
        tooltip="Controls vocabulary diversity. 0.9 = considers top 90% of probability mass."
        value={draft.topP} min={0} max={1} step={0.05}
        presetValue={0.9} onChg={v => set("topP", v)} />

      <SliderRow label="Repeat Penalty"
        tooltip="Discourages repeating the same phrases. 1.0 = no penalty."
        value={draft.repeatPenalty} min={0.5} max={2} step={0.05}
        presetValue={1.1} onChg={v => set("repeatPenalty", v)} />

      {numInput("Top-k", "Limits the token choices at each step. 0 = disabled.",
        draft.topK, "default (40)", 0, 200, 1, v => set("topK", v))}

      {/* Seed */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <span title="Set for reproducible outputs. Leave blank for random." style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>Seed ?</span>
          {draft.seed !== undefined && draft.seed !== null && (
            <button onClick={() => set("seed", null)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ random</button>
          )}
        </div>
        <input type="number" min={0} step={1}
          value={draft.seed ?? ""}
          placeholder="blank = random"
          onChange={e => set("seed", e.target.value ? parseInt(e.target.value) : null)}
          style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
        />
      </div>

      {numInput("Context Window (tokens)", "How many tokens of conversation history the model sees.",
        draft.numCtx,
        ctxTokens[draft.contextSize] ? `preset: ${ctxTokens[draft.contextSize]}` : "model default",
        512, 131072, 512,
        v => set("numCtx", v), "Larger values use more RAM — leave blank to use the Memory preset (or model default for Standard).")}

      {numInput("Max Output Tokens", "Maximum tokens in the model's response. -1 = unlimited.",
        draft.numPredict,
        lengthTokens[draft.responseLength] !== undefined ? `preset: ${lengthTokens[draft.responseLength]}` : "model default",
        -1, 32768, 256, v => set("numPredict", v), "Leave blank to use the Response Length preset (or model default for Auto).")}

      {/* Stop sequences */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 4 }}>Stop Sequences</div>
        <input type="text"
          value={draft.stopSequences ?? ""}
          placeholder="comma-separated, e.g.  ###, END"
          onChange={e => set("stopSequences", e.target.value || undefined)}
          style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
        />
      </div>

      {/* System prompt override */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <span style={{ fontSize: 12, fontWeight: 500 }}>System Prompt Override</span>
          {draft.systemPromptOverride && (
            <button onClick={() => set("systemPromptOverride", undefined)}
              style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ clear</button>
          )}
        </div>
        <textarea
          value={draft.systemPromptOverride ?? ""}
          placeholder="Leave blank to use the profile system prompt"
          onChange={e => set("systemPromptOverride", e.target.value || undefined)}
          rows={4}
          style={{ width: "100%", padding: "6px 8px", fontSize: 11, fontFamily: "monospace", background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", resize: "vertical" }}
        />
      </div>

      {/* Keep-alive */}
      <div style={{ marginBottom: 4 }}>
        <span title="How long to keep the model loaded in RAM. e.g. 5m, 1h, -1 for forever." style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>Keep-alive ?</span>
        <input type="text"
          value={draft.keepAlive ?? ""}
          placeholder="e.g. 5m, 1h, -1 for forever (default: 5m)"
          onChange={e => set("keepAlive", e.target.value || undefined)}
          style={{ width: "100%", marginTop: 4, padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
        />
      </div>
    </>
  );
}

// ── Advanced dialog (used from the chat input popover) ────────────────────────

function AdvancedParamsDialog({ params, onClose, onChange }: {
  params: ChatParams; onClose: () => void; onChange: (p: ChatParams) => void;
}) {
  const [draft, setDraft] = useState<ChatParams>({ ...params });

  const resetAdvanced = () => setDraft(prev => ({
    ...prev,
    temperature: undefined, topP: undefined, topK: undefined,
    repeatPenalty: undefined, seed: undefined, numCtx: undefined,
    numPredict: undefined, stopSequences: undefined,
    systemPromptOverride: undefined, keepAlive: undefined,
  }));

  return ReactDOM.createPortal(
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-box" style={{ maxWidth: 480, maxHeight: "85vh", overflowY: "auto" }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
          <div style={{ fontSize: 15, fontWeight: 700 }}>Advanced Chat Settings</div>
          <button className="btn icon-only" onClick={onClose}>✕</button>
        </div>
        <AdvancedParamsContent draft={draft} onChange={setDraft} />
        <div style={{ display: "flex", gap: 8, justifyContent: "space-between", marginTop: 20 }}>
          <button className="btn" onClick={resetAdvanced} style={{ fontSize: 11 }}>Reset to preset</button>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" onClick={onClose}>Cancel</button>
            <button className="btn primary" onClick={() => { onChange(draft); onClose(); }}>Apply</button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

// ── Simple popover (tier 1) — rendered via portal so it escapes overflow:hidden ─

function SimpleParamsPopover({ params, onChange, onClose, onAdvanced, anchor }: {
  params: ChatParams;
  onChange: (p: ChatParams) => void;
  onClose: () => void;
  onAdvanced: () => void;
  anchor: { bottom: number; left: number };
}) {
  const ref = useRef<HTMLDivElement>(null);
  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => onChange({ ...params, [k]: v });

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return ReactDOM.createPortal(
    <div ref={ref} style={{
      position: "fixed",
      bottom: anchor.bottom,
      left: anchor.left,
      zIndex: 500,
      background: "var(--surface)",
      border: "1px solid var(--border)",
      borderRadius: 14,
      padding: 16,
      width: 300,
      boxShadow: "0 8px 32px rgba(0,0,0,0.35), 0 2px 8px rgba(0,0,0,0.2)",
    }} onClick={e => e.stopPropagation()}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 14 }}>Chat Settings</div>

      <ChipRow label="Response Style" value={params.style}
        options={[
          { value: "precise",  label: "Precise",  desc: "Best for facts, code, analysis (temp 0.2)" },
          { value: "balanced", label: "Balanced", desc: "Best for most tasks (temp 0.7)" },
          { value: "creative", label: "Creative", desc: "Best for writing, brainstorming (temp 1.0)" },
        ]}
        onChange={v => set("style", v)}
      />

      <ChipRow label="Response Length" value={params.responseLength}
        options={[
          { value: "short",  label: "Short",  desc: "~256 tokens" },
          { value: "medium", label: "Medium", desc: "~1 024 tokens" },
          { value: "long",   label: "Long",   desc: "~4 096 tokens" },
          { value: "auto",   label: "Auto",   desc: "Let the model decide" },
        ]}
        onChange={v => set("responseLength", v)}
      />

      <ChipRow
        label="Memory"
        desc="How much conversation Lexi remembers — longer uses more RAM"
        value={params.contextSize}
        options={[
          { value: "short", label: "Standard", desc: "2 048 tokens" },
          { value: "long",  label: "Extended", desc: "8 192 tokens" },
        ]}
        onChange={v => set("contextSize", v)}
      />

      <ChipRow<"default" | "on" | "off">
        label="Reasoning"
        desc="Thinking models (e.g. Qwen3) reason before answering. Off is much faster per reply; On helps on hard multi-step tasks."
        value={params.reasoning ?? "default"}
        options={[
          { value: "default", label: "Auto",  desc: "Model default" },
          { value: "off",     label: "Off",   desc: "Faster replies" },
          { value: "on",      label: "On",    desc: "Deeper thinking" },
        ]}
        onChange={v => set("reasoning", v === "default" ? undefined : v)}
      />

      <div style={{ borderTop: "1px solid var(--border-light)", margin: "2px 0 12px" }} />

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button className="link-btn" onClick={onAdvanced}
          style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}>
          {hasAdvancedOverrides(params) && <span style={{ color: "#f59e0b", fontSize: 8 }}>●</span>}
          Advanced settings…
        </button>
      </div>
    </div>,
    document.body,
  );
}

// ── Public button component ───────────────────────────────────────────────────

export function ChatParamsButton({ params, onChange, disabled }: {
  params: ChatParams; onChange: (p: ChatParams) => void; disabled?: boolean;
}) {
  const [showPopover, setShowPopover] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [anchor, setAnchor] = useState<{ bottom: number; left: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const open = () => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setAnchor({ bottom: window.innerHeight - rect.top + 8, left: rect.left });
    }
    setShowPopover(true);
  };

  const isCustom = hasCustomParams(params);
  const advOverrides = hasAdvancedOverrides(params);

  return (
    <>
      <button ref={buttonRef} className="attach-btn" disabled={disabled}
        title="Chat settings" onClick={open} style={{ position: "relative" }}>
        <SlidersHorizontal size={14} />
        {isCustom && (
          <span style={{
            position: "absolute", top: 2, right: 2,
            width: 6, height: 6, borderRadius: "50%",
            background: advOverrides ? "#f59e0b" : "var(--accent)",
            border: "1px solid var(--bg)",
          }} />
        )}
      </button>

      {showPopover && !showAdvanced && anchor && (
        <SimpleParamsPopover
          params={params} onChange={onChange}
          onClose={() => setShowPopover(false)}
          onAdvanced={() => setShowAdvanced(true)}
          anchor={anchor}
        />
      )}

      {showAdvanced && (
        <AdvancedParamsDialog
          params={params}
          onClose={() => { setShowAdvanced(false); setShowPopover(false); }}
          onChange={p => { onChange(p); setShowAdvanced(false); setShowPopover(false); }}
        />
      )}
    </>
  );
}

// ── Inline defaults editor (reused in AdminPanel) ─────────────────────────────

export function ChatParamsDefaults({ params, onChange }: {
  params: ChatParams; onChange: (p: ChatParams) => void;
}) {
  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => onChange({ ...params, [k]: v });
  return (
    <div>
      <ChipRow label="Default Response Style" value={params.style}
        options={[
          { value: "precise",  label: "Precise" },
          { value: "balanced", label: "Balanced" },
          { value: "creative", label: "Creative" },
        ]}
        onChange={v => set("style", v)}
      />
      <ChipRow label="Default Response Length" value={params.responseLength}
        options={[
          { value: "short",  label: "Short" },
          { value: "medium", label: "Medium" },
          { value: "long",   label: "Long" },
          { value: "auto",   label: "Auto" },
        ]}
        onChange={v => set("responseLength", v)}
      />
      <ChipRow label="Default Memory" value={params.contextSize}
        options={[
          { value: "short", label: "Standard" },
          { value: "long",  label: "Extended" },
        ]}
        onChange={v => set("contextSize", v)}
      />
    </div>
  );
}
