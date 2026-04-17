import { useState, useRef, useEffect } from "react";
import { SlidersHorizontal } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ChatParams {
  // Tier 1 — plain-English presets
  style: "precise" | "balanced" | "creative";
  responseLength: "short" | "medium" | "long" | "auto";
  contextSize: "short" | "long";
  webSearch: boolean;
  fileAccess: boolean;
  // Tier 2 — advanced overrides (undefined = use preset value)
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
}

export const DEFAULT_CHAT_PARAMS: ChatParams = {
  style: "balanced",
  responseLength: "auto",
  contextSize: "short",
  webSearch: true,
  fileAccess: true,
};

/** Map presets + overrides → concrete Ollama options. */
export function resolveParams(p: ChatParams): {
  temperature: number;
  numCtx: number;
  numPredict: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  seed?: number | null;
  stop?: string[];
  keepAlive?: string;
  systemPromptOverride?: string;
} {
  const styleTemp: Record<ChatParams["style"], number> = { precise: 0.2, balanced: 0.7, creative: 1.0 };
  const lengthTokens: Record<ChatParams["responseLength"], number> = { short: 256, medium: 1024, long: 4096, auto: -1 };
  const ctxTokens: Record<ChatParams["contextSize"], number> = { short: 2048, long: 8192 };

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
  };
}

/** True if any advanced field has been set (tier 2 override active). */
function hasAdvancedOverrides(p: ChatParams): boolean {
  return p.temperature !== undefined ||
    p.topP !== undefined ||
    p.topK !== undefined ||
    p.repeatPenalty !== undefined ||
    (p.seed !== undefined && p.seed !== null) ||
    p.numCtx !== undefined ||
    p.numPredict !== undefined ||
    !!p.stopSequences ||
    !!p.systemPromptOverride ||
    !!p.keepAlive;
}

/** True if anything differs from the defaults. */
export function hasCustomParams(p: ChatParams): boolean {
  const d = DEFAULT_CHAT_PARAMS;
  return p.style !== d.style ||
    p.responseLength !== d.responseLength ||
    p.contextSize !== d.contextSize ||
    !p.webSearch || !p.fileAccess ||
    hasAdvancedOverrides(p);
}

// ── Shared chip-button row ────────────────────────────────────────────────────

function ChipRow<T extends string>({
  label, desc, value, options, onChange,
}: {
  label: string;
  desc?: string;
  value: T;
  options: { value: T; label: string; desc?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </div>
      {desc && <div style={{ fontSize: 11, color: "var(--text-tertiary)", marginBottom: 6 }}>{desc}</div>}
      <div style={{ display: "flex", gap: 5 }}>
        {options.map(opt => (
          <button
            key={opt.value}
            title={opt.desc}
            onClick={() => onChange(opt.value)}
            style={{
              flex: 1,
              padding: "5px 0",
              fontSize: 12,
              fontWeight: value === opt.value ? 600 : 400,
              background: value === opt.value ? "var(--accent)" : "var(--surface3)",
              color: value === opt.value ? "#fff" : "var(--text-secondary)",
              border: value === opt.value ? "1px solid var(--accent)" : "1px solid var(--border)",
              borderRadius: 7,
              cursor: "pointer",
              transition: "all 0.12s",
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
      <span style={{ fontSize: 12, color: "var(--text)" }}>{label}</span>
      <button
        onClick={() => onChange(!checked)}
        style={{
          width: 38, height: 22, borderRadius: 11,
          background: checked ? "var(--accent)" : "var(--surface3)",
          border: checked ? "1px solid var(--accent)" : "1px solid var(--border)",
          position: "relative", cursor: "pointer", transition: "all 0.2s", flexShrink: 0,
        }}
      >
        <span style={{
          position: "absolute", top: 2, left: checked ? 17 : 2,
          width: 16, height: 16, borderRadius: "50%",
          background: "#fff", transition: "left 0.2s",
          boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
        }} />
      </button>
    </div>
  );
}

// ── Advanced dialog ───────────────────────────────────────────────────────────

function SliderRow({
  label, tooltip, value, min, max, step, presetValue, onChange,
}: {
  label: string; tooltip?: string; value: number | undefined;
  min: number; max: number; step: number;
  presetValue: number; onChange: (v: number | undefined) => void;
}) {
  const display = value ?? presetValue;
  const isCustom = value !== undefined;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span title={tooltip} style={{ fontSize: 12, fontWeight: 500, cursor: tooltip ? "help" : undefined }}>
          {label}{tooltip ? " ?" : ""}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, fontFamily: "monospace", fontWeight: 600 }}>{display.toFixed(step < 1 ? 2 : 0)}</span>
          {isCustom && (
            <button onClick={() => onChange(undefined)} style={{
              fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer", padding: 0,
            }} title="Reset to preset">↺</button>
          )}
        </div>
      </div>
      <input type="range" min={min} max={max} step={step} value={display}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", accentColor: "var(--accent)" }}
      />
      {!isCustom && (
        <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 2 }}>Using preset value</div>
      )}
    </div>
  );
}

function AdvancedParamsDialog({
  params, onClose, onChange,
}: { params: ChatParams; onClose: () => void; onChange: (p: ChatParams) => void }) {
  const [draft, setDraft] = useState<ChatParams>({ ...params });
  const styleTemp: Record<ChatParams["style"], number> = { precise: 0.2, balanced: 0.7, creative: 1.0 };
  const lengthTokens: Record<ChatParams["responseLength"], number> = { short: 256, medium: 1024, long: 4096, auto: -1 };
  const ctxTokens: Record<ChatParams["contextSize"], number> = { short: 2048, long: 8192 };

  const resetAdvanced = () => setDraft(prev => ({
    ...prev,
    temperature: undefined, topP: undefined, topK: undefined,
    repeatPenalty: undefined, seed: undefined, numCtx: undefined,
    numPredict: undefined, stopSequences: undefined,
    systemPromptOverride: undefined, keepAlive: undefined,
  }));

  const apply = () => { onChange(draft); onClose(); };

  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => setDraft(prev => ({ ...prev, [k]: v }));

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-box" style={{ maxWidth: 500, maxHeight: "85vh", overflowY: "auto" }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700 }}>Advanced Settings</div>
            <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 2 }}>
              Preset: <b>{draft.style}</b> — overrides apply to this chat only
            </div>
          </div>
          <button className="btn icon-only" onClick={onClose}>✕</button>
        </div>

        <SliderRow
          label="Temperature"
          tooltip="Controls randomness. Lower = more focused, Higher = more creative."
          value={draft.temperature}
          min={0} max={2} step={0.05}
          presetValue={styleTemp[draft.style]}
          onChange={v => set("temperature", v)}
        />

        <SliderRow
          label="Top-p (nucleus sampling)"
          tooltip="Controls vocabulary diversity. 0.9 = considers top 90% of probability mass."
          value={draft.topP}
          min={0} max={1} step={0.05}
          presetValue={0.9}
          onChange={v => set("topP", v)}
        />

        <SliderRow
          label="Repeat Penalty"
          tooltip="Discourages repeating the same phrases. 1.0 = no penalty."
          value={draft.repeatPenalty}
          min={0.5} max={2} step={0.05}
          presetValue={1.1}
          onChange={v => set("repeatPenalty", v)}
        />

        {/* Top-k */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span title="Limits the token choices at each step. 0 = disabled." style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>Top-k ?</span>
            {draft.topK !== undefined && (
              <button onClick={() => set("topK", undefined)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ reset</button>
            )}
          </div>
          <input type="number" min={0} max={200} step={1}
            value={draft.topK ?? ""}
            placeholder="default (40)"
            onChange={e => set("topK", e.target.value ? parseInt(e.target.value) : undefined)}
            style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
          />
        </div>

        {/* Seed */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span title="Set to get reproducible outputs. Leave blank for random." style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>Seed ?</span>
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

        {/* Context window */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 500 }}>Context Window (tokens)</span>
            {draft.numCtx !== undefined && (
              <button onClick={() => set("numCtx", undefined)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ reset</button>
            )}
          </div>
          <input type="number" min={512} max={131072} step={512}
            value={draft.numCtx ?? ""}
            placeholder={`preset: ${ctxTokens[draft.contextSize]}`}
            onChange={e => set("numCtx", e.target.value ? parseInt(e.target.value) : undefined)}
            style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
          />
          <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 3 }}>Larger values use more RAM — leave blank to use the Memory preset.</div>
        </div>

        {/* Max tokens */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 500 }}>Max Output Tokens</span>
            {draft.numPredict !== undefined && (
              <button onClick={() => set("numPredict", undefined)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ reset</button>
            )}
          </div>
          <input type="number" min={-1} max={32768} step={256}
            value={draft.numPredict ?? ""}
            placeholder={`preset: ${lengthTokens[draft.responseLength] === -1 ? "unlimited" : lengthTokens[draft.responseLength]}`}
            onChange={e => set("numPredict", e.target.value ? parseInt(e.target.value) : undefined)}
            style={{ width: "100%", padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
          />
          <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 3 }}>-1 = unlimited. Leave blank to use the Response Length preset.</div>
        </div>

        {/* Stop sequences */}
        <div style={{ marginBottom: 14 }}>
          <span style={{ fontSize: 12, fontWeight: 500 }}>Stop Sequences</span>
          <input type="text"
            value={draft.stopSequences ?? ""}
            placeholder="comma-separated, e.g.  ###, END"
            onChange={e => set("stopSequences", e.target.value || undefined)}
            style={{ width: "100%", marginTop: 4, padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
          />
        </div>

        {/* System prompt override */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 500 }}>System Prompt Override</span>
            {draft.systemPromptOverride && (
              <button onClick={() => set("systemPromptOverride", undefined)} style={{ fontSize: 10, color: "var(--text-tertiary)", background: "none", border: "none", cursor: "pointer" }}>↺ clear</button>
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
        <div style={{ marginBottom: 20 }}>
          <span title="How long to keep the model loaded in RAM after the last request. e.g. 5m, 1h, -1 (forever)" style={{ fontSize: 12, fontWeight: 500, cursor: "help" }}>Keep-alive ?</span>
          <input type="text"
            value={draft.keepAlive ?? ""}
            placeholder="e.g. 5m, 1h, -1 for forever (default: 5m)"
            onChange={e => set("keepAlive", e.target.value || undefined)}
            style={{ width: "100%", marginTop: 4, padding: "5px 8px", fontSize: 12, background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", fontFamily: "monospace" }}
          />
        </div>

        <div style={{ display: "flex", gap: 8, justifyContent: "space-between" }}>
          <button className="btn" onClick={resetAdvanced} style={{ fontSize: 11 }}>Reset to preset</button>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn" onClick={onClose}>Cancel</button>
            <button className="btn primary" onClick={apply}>Apply</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Simple popover (tier 1) ───────────────────────────────────────────────────

function SimpleParamsPopover({
  params, onChange, onClose, onAdvanced,
}: {
  params: ChatParams;
  onChange: (p: ChatParams) => void;
  onClose: () => void;
  onAdvanced: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => onChange({ ...params, [k]: v });
  const advOverrides = hasAdvancedOverrides(params);

  return (
    <div ref={ref} className="chat-params-popover" onClick={e => e.stopPropagation()}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 14, color: "var(--text)" }}>Chat Settings</div>

      <ChipRow
        label="Response Style"
        value={params.style}
        options={[
          { value: "precise",  label: "Precise",  desc: "Best for facts, code, analysis (temp 0.2)" },
          { value: "balanced", label: "Balanced", desc: "Best for most tasks (temp 0.7)" },
          { value: "creative", label: "Creative", desc: "Best for writing, brainstorming (temp 1.0)" },
        ]}
        onChange={v => set("style", v)}
      />

      <ChipRow
        label="Response Length"
        value={params.responseLength}
        options={[
          { value: "short",  label: "Short",     desc: "Up to ~256 tokens" },
          { value: "medium", label: "Medium",    desc: "Up to ~1 024 tokens" },
          { value: "long",   label: "Long",      desc: "Up to ~4 096 tokens" },
          { value: "auto",   label: "Auto",      desc: "Let the model decide (unlimited)" },
        ]}
        onChange={v => set("responseLength", v)}
      />

      <ChipRow
        label="Memory"
        desc="How much conversation context Lexi remembers — longer memory uses more RAM"
        value={params.contextSize}
        options={[
          { value: "short", label: "Standard", desc: "2 048 tokens (~10–15 turns)" },
          { value: "long",  label: "Extended", desc: "8 192 tokens (~50+ turns)" },
        ]}
        onChange={v => set("contextSize", v)}
      />

      <div style={{ borderTop: "1px solid var(--border-light)", marginBottom: 12, marginTop: 2 }} />

      <Toggle label="Web Search" checked={params.webSearch} onChange={v => set("webSearch", v)} />
      <Toggle label="File Access" checked={params.fileAccess} onChange={v => set("fileAccess", v)} />

      <div style={{ borderTop: "1px solid var(--border-light)", marginTop: 6, paddingTop: 10, display: "flex", justifyContent: "flex-end" }}>
        <button
          className="link-btn"
          onClick={onAdvanced}
          style={{ fontSize: 11, display: "flex", alignItems: "center", gap: 4 }}
        >
          {advOverrides && <span style={{ color: "var(--accent)", fontSize: 8 }}>●</span>}
          Advanced settings…
        </button>
      </div>
    </div>
  );
}

// ── Public button component ───────────────────────────────────────────────────

export function ChatParamsButton({
  params, onChange, disabled,
}: {
  params: ChatParams;
  onChange: (p: ChatParams) => void;
  disabled?: boolean;
}) {
  const [showPopover, setShowPopover] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const isCustom = hasCustomParams(params);
  const advOverrides = hasAdvancedOverrides(params);

  return (
    <div style={{ position: "relative" }}>
      <button
        className="attach-btn"
        disabled={disabled}
        title="Chat settings"
        onClick={() => setShowPopover(v => !v)}
        style={{ position: "relative" }}
      >
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

      {showPopover && !showAdvanced && (
        <SimpleParamsPopover
          params={params}
          onChange={onChange}
          onClose={() => setShowPopover(false)}
          onAdvanced={() => setShowAdvanced(true)}
        />
      )}

      {showAdvanced && (
        <AdvancedParamsDialog
          params={params}
          onClose={() => { setShowAdvanced(false); setShowPopover(false); }}
          onChange={p => { onChange(p); setShowAdvanced(false); setShowPopover(false); }}
        />
      )}
    </div>
  );
}

// ── Inline defaults editor (used in AdminPanel profile edit) ──────────────────

export function ChatParamsDefaults({
  params, onChange,
}: { params: ChatParams; onChange: (p: ChatParams) => void }) {
  const set = <K extends keyof ChatParams>(k: K, v: ChatParams[K]) => onChange({ ...params, [k]: v });

  return (
    <div>
      <ChipRow
        label="Default Response Style"
        value={params.style}
        options={[
          { value: "precise",  label: "Precise" },
          { value: "balanced", label: "Balanced" },
          { value: "creative", label: "Creative" },
        ]}
        onChange={v => set("style", v)}
      />
      <ChipRow
        label="Default Response Length"
        value={params.responseLength}
        options={[
          { value: "short",  label: "Short" },
          { value: "medium", label: "Medium" },
          { value: "long",   label: "Long" },
          { value: "auto",   label: "Auto" },
        ]}
        onChange={v => set("responseLength", v)}
      />
      <ChipRow
        label="Default Memory"
        value={params.contextSize}
        options={[
          { value: "short", label: "Standard" },
          { value: "long",  label: "Extended" },
        ]}
        onChange={v => set("contextSize", v)}
      />
      <Toggle label="Web Search enabled by default" checked={params.webSearch} onChange={v => set("webSearch", v)} />
      <Toggle label="File Access enabled by default" checked={params.fileAccess} onChange={v => set("fileAccess", v)} />
    </div>
  );
}
