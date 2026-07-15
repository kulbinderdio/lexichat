import React, { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { Profile, StoredMCPServer } from "./AdminPanel";
import { save } from "@tauri-apps/plugin-dialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ScheduledJob, JobRun, JobSchedule, TraceStep, JobStep } from "./jobTypes";

// Backend response shapes for tool discovery
interface SpecInfoLocal {
  id: string; title: string; tool_count: number;
  tools: { name: string; description: string; method: string; path: string }[];
}
interface MCPServerInfoLocal {
  id: string; name: string; connected: boolean;
  tools: { name: string; description: string }[];
}
interface ToolOption {
  name: string; label: string; description: string;
  category: "builtin" | "openapi" | "mcp"; service?: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const BUILTIN_TOOLS = [
  { name: "list_files",          label: "List Files",         icon: "📁" },
  { name: "read_file",           label: "Read File",          icon: "📄" },
  { name: "write_file",          label: "Write File",         icon: "✏️" },
  { name: "search_files",        label: "Search Files",       icon: "🔎" },
  { name: "search_in_files",     label: "Search In Files",    icon: "🔍" },
  { name: "get_file_info",       label: "Get File Info",      icon: "ℹ️" },
  { name: "list_directory_tree", label: "Directory Tree",     icon: "🌳" },
  { name: "create_directory",    label: "Create Directory",   icon: "📂" },
  { name: "move_file",           label: "Move / Rename",      icon: "↕️" },
  { name: "delete_file",         label: "Delete File",        icon: "🗑️" },
  { name: "find_old_files",      label: "Find Old Files",     icon: "🗂️" },
  { name: "web_search",          label: "Web Search",         icon: "🌐" },
  { name: "fetch_webpage",       label: "Fetch Webpage",      icon: "🔗" },
  { name: "get_current_datetime", label: "Get Date / Time",   icon: "🕐" },
  { name: "compose_email",        label: "Compose Email",      icon: "✉️" },
];

const WEEKDAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];

function buildToolOptions(
  _enabledBuiltins: string[],  // kept for API compat but unused — step builder shows all tools
  specs: SpecInfoLocal[],
  mcpServers: MCPServerInfoLocal[],
): ToolOption[] {
  const opts: ToolOption[] = [];
  // Show every built-in tool — the step builder is explicit about which tools to call,
  // so the profile's "enabled" filter doesn't apply here.
  for (const bt of BUILTIN_TOOLS)
    opts.push({ name: bt.name, label: bt.label, description: bt.label, category: "builtin" });
  for (const sp of specs)
    for (const t of sp.tools)
      opts.push({ name: t.name, label: t.name, description: t.description, category: "openapi", service: sp.title });
  for (const srv of mcpServers)
    for (const t of srv.tools)
      opts.push({ name: t.name, label: t.name, description: t.description, category: "mcp", service: srv.name });
  return opts;
}

function compileStepsPreview(steps: JobStep[]): string {
  if (!steps.length) return "";
  const n = steps.length;
  const lines = [
    `Execute the following ${n} step${n === 1 ? "" : "s"} IN ORDER. Do not skip any step.`,
    `After each step output: ✓ Step N complete: [one line summary]`,
    "",
  ];
  steps.forEach((step, i) => {
    const instr = (step.instruction ?? "").trim();
    lines.push(`STEP ${i + 1} — ${instr}`);
    if (step.step_type === "tool") {
      if (step.tool_name) lines.push(`Call tool: ${step.tool_name}`);
      const hint = (step.tool_hint ?? "").trim();
      if (hint) lines.push(`Arguments hint: ${hint}`);
    }
    lines.push("");
  });
  return lines.join("\n").trimEnd();
}

const pad = (n: number) => String(n).padStart(2, "0");

function uid() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function scheduleLabel(s: JobSchedule): string {
  if (s.type === "Daily")    return `Daily at ${pad(s.hour!)}:${pad(s.minute!)}`;
  if (s.type === "Interval") return `Every ${s.hours}h`;
  if (s.type === "Weekly")   return `${WEEKDAYS[s.weekday!]?.slice(0,3)} ${pad(s.hour!)}:${pad(s.minute!)}`;
  return "Manual";
}

function relativeTime(iso: string): string {
  const d = new Date(iso);
  const diff = Date.now() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 2)   return "just now";
  if (mins < 60)  return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)   return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7)   return `${days}d ago`;
  return d.toLocaleDateString();
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function resolveProfileContext(
  profile: Profile,
  selectedSpecIds: string[],
  selectedMcpIds: string[],
  disabledMcpTools: string[],
  globalAllowedDirs: string[] = [],
  registrySpecs: import("./AdminPanel").StoredOpenAPISpec[] = [],
  registryMcp: import("./AdminPanel").StoredMCPServer[] = [],
): import("./jobTypes").JobProfileContext {
  const allowed_dirs = (profile.allowedDirs?.length ?? 0) > 0
    ? profile.allowedDirs!
    : globalAllowedDirs;
  return {
    ollama_host: profile.host ?? "http://localhost:11434",
    allowed_dirs,
    openapi_specs: registrySpecs
      .filter(s => s.enabled !== false && selectedSpecIds.includes(s.id))
      .map(s => ({ id: s.id, title: s.title, base_url: s.base_url, spec_json: s.spec_json, auth: s.auth })),
    mcp_servers: registryMcp
      .filter(s => selectedMcpIds.includes(s.id))
      .map(s => ({ id: s.id, name: s.name, command: s.command, args: s.args ?? [], env: s.env ?? {}, auth: s.auth })),
    disabled_mcp_tools: disabledMcpTools,
    snapshot_at: new Date().toISOString(),
  };
}

function blankJob(profile: Profile | null = null): ScheduledJob {
  const enabledBuiltins = profile
    ? Object.entries(profile.enabledTools ?? {}).filter(([, v]) => v !== false).map(([k]) => k)
    : ["web_search"];
  return {
    id: uid(),
    name: "",
    enabled: true,
    schedule: { type: "Daily", hour: 9, minute: 0 },
    prompt: "",
    model: profile?.model ?? "",
    system_prompt: profile?.systemPrompt ?? null,
    enabled_builtin_tools: enabledBuiltins,
    output_file: null,
    created_at: new Date().toISOString(),
    last_run_at: null,
    profile_id: profile?.id ?? null,
    profile_name: profile?.name ?? null,
    profile_context: null,
    steps: [],
  };
}

function nextRunLabel(job: ScheduledJob): string {
  if (!job.enabled) return "Disabled";
  const s = job.schedule;
  const now = new Date();
  if (s.type === "Manual") return "Manual only";
  if (s.type === "Interval") {
    if (!job.last_run_at) return "Within 30 seconds";
    const next = new Date(new Date(job.last_run_at).getTime() + (s.hours ?? 1) * 3600_000);
    return next <= now ? "Due now" : `Next: ${next.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  }
  if (s.type === "Daily" || s.type === "Weekly") {
    const h = pad(s.hour ?? 0), m = pad(s.minute ?? 0);
    if (s.type === "Weekly") {
      const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];
      return `Next: ${days[s.weekday ?? 0]} ${h}:${m}`;
    }
    const target = new Date(); target.setHours(s.hour ?? 0, s.minute ?? 0, 0, 0);
    if (job.last_run_at && new Date(job.last_run_at).toDateString() === now.toDateString()) {
      target.setDate(target.getDate() + 1);
    } else if (target < now) {
      // Due today but past — will run at next tick if within the minute window, else tomorrow
      const diff = now.getTime() - target.getTime();
      if (diff < 60_000) return "Running soon…";
      target.setDate(target.getDate() + 1);
    }
    return `Next: ${target.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} ${h}:${m}`;
  }
  return "";
}

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

// ── Props ─────────────────────────────────────────────────────────────────────

interface Props {
  models: string[];
  profiles: Profile[];
  activeProfileId: string | null;
  globalOpenapiSpecs: import("./AdminPanel").StoredOpenAPISpec[];
  globalMcpServers: import("./AdminPanel").StoredMCPServer[];
  globalEnabledTools: Record<string, boolean>;
  globalAllowedDirs: string[];
  onClose: () => void;
}

// ── Main component ────────────────────────────────────────────────────────────

export function JobsPanel({ models, profiles, activeProfileId, globalOpenapiSpecs, globalMcpServers, globalEnabledTools, globalAllowedDirs, onClose }: Props) {
  const [tab, setTab]               = useState<"jobs" | "history">("jobs");
  const [jobs, setJobs]             = useState<ScheduledJob[]>([]);
  const [runs, setRuns]             = useState<JobRun[]>([]);
  const [editingJob, setEditingJob] = useState<ScheduledJob | null>(null);
  const [runningIds, setRunningIds] = useState<Set<string>>(new Set());
  const [filterJobId, setFilterJobId] = useState<string | null>(null);
  const [error, setError]           = useState("");
  const [designWarnings, setDesignWarnings] = useState<string[]>([]);

  useEffect(() => {
    invoke<ScheduledJob[]>("get_jobs").then(setJobs).catch(() => {});
    invoke<JobRun[]>("get_job_runs", { jobId: null }).then(setRuns).catch(() => {});

    // Listen for job completions — updates history and clears running state
    // regardless of whether this panel is open or was closed mid-run.
    const unlisten = listen<JobRun>("job-run-done", e => {
      const run = e.payload;
      setRuns(prev => prev.some(r => r.id === run.id) ? prev : [run, ...prev]);
      setRunningIds(prev => { const s = new Set(prev); s.delete(run.job_id); return s; });
      // Refresh job list to pick up updated last_run_at
      invoke<ScheduledJob[]>("get_jobs").then(setJobs).catch(() => {});
    });
    return () => { unlisten.then(f => f()); };
  }, []);

  const saveJob = async (job: ScheduledJob) => {
    try {
      await invoke("save_job", { job });
      setJobs(prev => {
        const idx = prev.findIndex(j => j.id === job.id);
        return idx >= 0 ? prev.map(j => j.id === job.id ? job : j) : [...prev, job];
      });
      setEditingJob(null);
    } catch (e) {
      setError(String(e));
    }
  };

  const deleteJob = async (id: string) => {
    await invoke("delete_job", { id });
    setJobs(prev => prev.filter(j => j.id !== id));
  };

  // Draft a job from a plain-English goal. When `dryRun` is false, open it in the normal
  // form for review. When true, save it (disabled) and run it once so the user can see the
  // real artifact before enabling the schedule. The backend never enables anything itself.
  const designJob = async (goal: string, model: string, dryRun: boolean) => {
    const tool_catalog = BUILTIN_TOOLS.map(t => ({ name: t.name, description: t.label }));
    const ap = profiles.find(p => p.id === activeProfileId) ?? null;
    const res = await invoke<{ job: ScheduledJob; warnings: string[] }>("draft_job_from_goal", {
      args: {
        goal, model,
        allowed_dirs: globalAllowedDirs,
        tool_catalog,
        profile_id: ap?.id ?? null,
        profile_name: ap?.name ?? null,
      },
    });
    setDesignWarnings(res.warnings);

    if (!dryRun) {
      setEditingJob(res.job);
      return;
    }

    // Persist disabled, then dry-run and show the result in Run History.
    const job = res.job;
    await invoke("save_job", { job });
    setJobs(prev => [...prev, job]);
    setFilterJobId(job.id);
    setTab("history");
    runNow(job.id);
  };

  // Fire-and-forget — the job runs in Rust until completion.
  // Closing the panel NEVER interrupts it; result arrives via job-run-done event.
  const runNow = (id: string) => {
    setRunningIds(prev => new Set([...prev, id]));
    setError("");
    invoke("run_job_now", { id }).catch((e: unknown) => {
      setError(String(e));
      setRunningIds(prev => { const s = new Set(prev); s.delete(id); return s; });
    });
  };

  const clearRuns = async () => {
    await invoke("clear_job_runs", { jobId: filterJobId ?? null });
    setRuns(filterJobId ? runs.filter(r => r.job_id !== filterJobId) : []);
  };

  const filteredRuns = filterJobId ? runs.filter(r => r.job_id === filterJobId) : runs;

  return (
    <div className="jobs-page">

      {/* Page header — back button + title + running indicator */}
      <div className="jobs-page-header">
        <button className="btn icon-only" onClick={onClose} title="Back to chat"
          style={{ marginRight: 4 }}>
          ← Chat
        </button>
        <span className="jobs-page-title">
          Scheduled Jobs
          {runningIds.size > 0 && (
            <span style={{ fontSize: 11, color: "var(--accent)", marginLeft: 8, fontWeight: 400 }}>
              · {runningIds.size} running…
            </span>
          )}
        </span>
        {/* Tabs sit in the header row on the right */}
        <div className="jobs-page-tabs">
          <button className={`admin-tab ${tab === "jobs" ? "active" : ""}`} onClick={() => setTab("jobs")}>
            Scheduled Jobs
            {jobs.length > 0 && <span className="tab-count">{jobs.length}</span>}
          </button>
          <button className={`admin-tab ${tab === "history" ? "active" : ""}`} onClick={() => setTab("history")}>
            Run History
            {runs.length > 0 && <span className="tab-count">{runs.length}</span>}
          </button>
        </div>
      </div>
      <div className="admin-divider" />

      {/* AI-designer review notes — shown over the pre-filled form (or the dry-run result) */}
      {designWarnings.length > 0 && (
        <div style={{ margin: "8px 16px", padding: "8px 12px", borderRadius: 8,
          background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.35)", fontSize: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>⚠ Review before saving</div>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {designWarnings.map((w, i) => <li key={i} style={{ marginBottom: 2 }}>{w}</li>)}
          </ul>
        </div>
      )}

      {/* Content — fills remaining height */}
      <div className="jobs-page-content">
        {tab === "jobs" && (
          <JobsTab
            jobs={jobs}
            models={models}
            profiles={profiles}
            globalOpenapiSpecs={globalOpenapiSpecs}
            globalMcpServers={globalMcpServers}
            globalEnabledTools={globalEnabledTools}
            globalAllowedDirs={globalAllowedDirs}
            activeProfileId={activeProfileId}
            editingJob={editingJob}
            runningIds={runningIds}
            error={error}
            onEdit={j => { setDesignWarnings([]); setEditingJob(j); }}
            onSave={saveJob}
            onCancel={() => { setDesignWarnings([]); setEditingJob(null); }}
            onDelete={deleteJob}
            onRunNow={runNow}
            onDesign={designJob}
            onNew={() => {
              setDesignWarnings([]);
              const ap = profiles.find(p => p.id === activeProfileId) ?? null;
              setEditingJob(blankJob(ap ?? null));
            }}
            onToggle={j => saveJob({ ...j, enabled: !j.enabled })}
          />
        )}
        {tab === "history" && (
          <HistoryTab
            runs={filteredRuns}
            jobs={jobs}
            filterJobId={filterJobId}
            onFilterChange={setFilterJobId}
            onClear={clearRuns}
          />
        )}
      </div>

    </div>
  );
}

// ── Jobs tab ──────────────────────────────────────────────────────────────────

function JobsTab({ jobs, models, profiles, globalOpenapiSpecs, globalMcpServers, globalEnabledTools, globalAllowedDirs, activeProfileId, editingJob, runningIds, error, onEdit, onSave, onCancel, onDelete, onRunNow, onDesign, onNew, onToggle }: {
  jobs: ScheduledJob[];
  models: string[];
  profiles: Profile[];
  globalOpenapiSpecs: import("./AdminPanel").StoredOpenAPISpec[];
  globalMcpServers: import("./AdminPanel").StoredMCPServer[];
  globalEnabledTools: Record<string, boolean>;
  globalAllowedDirs: string[];
  activeProfileId: string | null;
  editingJob: ScheduledJob | null;
  runningIds: Set<string>;
  error: string;
  onEdit: (j: ScheduledJob) => void;
  onSave: (j: ScheduledJob) => void;
  onCancel: () => void;
  onDelete: (id: string) => void;
  onRunNow: (id: string) => void;
  onDesign: (goal: string, model: string, dryRun: boolean) => Promise<void>;
  onNew: () => void;
  onToggle: (j: ScheduledJob) => void;
}) {
  const [designOpen, setDesignOpen] = useState(false);
  const [goal, setGoal] = useState("");
  const activeModel = profiles.find(p => p.id === activeProfileId)?.model || models[0] || "";
  const [designModel, setDesignModel] = useState(activeModel);
  const [designBusy, setDesignBusy] = useState(false);
  const [designErr, setDesignErr] = useState("");

  const runDesign = async (dryRun: boolean) => {
    if (!goal.trim() || !designModel) return;
    setDesignBusy(true);
    setDesignErr("");
    try {
      // On success: opens the pre-filled form, or (dryRun) saves disabled + runs + shows history.
      await onDesign(goal.trim(), designModel, dryRun);
    } catch (e) {
      setDesignErr(String(e));
    } finally {
      setDesignBusy(false);
    }
  };

  if (editingJob) {
    return (
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0, overflow: "hidden" }}>
        <JobForm job={editingJob} models={models} profiles={profiles}
          globalOpenapiSpecs={globalOpenapiSpecs} globalMcpServers={globalMcpServers}
          globalEnabledTools={globalEnabledTools} globalAllowedDirs={globalAllowedDirs}
          onSave={onSave} onCancel={onCancel} />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {jobs.length === 0 && (
          <div className="admin-empty">No scheduled jobs yet. Create one to run prompts automatically.</div>
        )}
        {jobs.map(job => (
          <div key={job.id} className="job-row">
            {/* Enable toggle */}
            <input
              type="checkbox"
              checked={job.enabled}
              onChange={() => onToggle(job)}
              style={{ accentColor: "var(--accent)", flexShrink: 0 }}
            />
            {/* Info */}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="job-row-name">{job.name}</div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
                <span className="job-schedule-pill">{scheduleLabel(job.schedule)}</span>
                {job.profile_name && (
                  <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 8, background: "var(--surface2)", color: "var(--text-secondary)" }}>
                    {job.profile_name}
                  </span>
                )}
                <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>
                  {job.last_run_at ? `Last run ${relativeTime(job.last_run_at)}` : "Never run"}
                </span>
                <span style={{ fontSize: 10, color: "var(--accent)", fontWeight: 500 }}>
                  {nextRunLabel(job)}
                </span>
                {job.profile_context && (() => {
                  const stale = Date.now() - new Date(job.profile_context!.snapshot_at).getTime() > 7 * 86400_000;
                  return stale ? (
                    <span title="Profile may have changed. Edit job to refresh snapshot." style={{ fontSize: 10, color: "#f59e0b", cursor: "help" }}>⚠ stale</span>
                  ) : null;
                })()}
              </div>
            </div>
            {/* Actions */}
            <button className="btn" style={{ fontSize: 11, padding: "3px 8px" }} onClick={() => onEdit(job)}>Edit</button>
            <button
              className="btn"
              style={{ fontSize: 11, padding: "3px 8px", minWidth: 60 }}
              onClick={() => onRunNow(job.id)}
              disabled={runningIds.has(job.id)}
            >
              {runningIds.has(job.id) ? "…" : "▶ Run"}
            </button>
            <button className="icon-btn danger" onClick={() => onDelete(job.id)}>✕</button>
          </div>
        ))}
        {error && <div style={{ padding: "8px 16px", color: "#f87171", fontSize: 12 }}>{error}</div>}
      </div>
      {/* Footer */}
      <div style={{ padding: 12, borderTop: "1px solid var(--border)" }}>
        {designOpen && (
          <div style={{ marginBottom: 8, padding: 10, borderRadius: 10, background: "var(--surface2)", border: "1px solid var(--border)" }}>
            <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 6 }}>Describe what you want, and LexiChat will draft the job for you.</div>
            <textarea
              className="admin-input"
              rows={3}
              value={goal}
              onChange={e => setGoal(e.target.value)}
              placeholder="e.g. Every weekday at 7am, brief me on UK energy policy from gov.uk and Hansard, skip anything you already covered, and save it as a PDF in ~/Briefings."
              style={{ width: "100%", resize: "vertical", fontSize: 12 }}
            />
            <div style={{ marginTop: 6 }}>
              <select className="admin-input" value={designModel} onChange={e => setDesignModel(e.target.value)}
                style={{ width: "100%", fontSize: 11 }}>
                {models.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 6 }}>
              <button className="btn" style={{ flex: 1, fontSize: 11 }} disabled={designBusy || !goal.trim() || !designModel}
                onClick={() => runDesign(false)}>
                {designBusy ? "Drafting…" : "✨ Draft it"}
              </button>
              <button className="btn primary" style={{ flex: 1, fontSize: 11 }} disabled={designBusy || !goal.trim() || !designModel}
                onClick={() => runDesign(true)}>
                {designBusy ? "Working…" : "Draft & dry-run"}
              </button>
            </div>
            {designErr && <div style={{ color: "#f87171", fontSize: 11, marginTop: 6 }}>{designErr}</div>}
            <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 6 }}>
              Both leave the job <strong>disabled</strong> — it won't run on a schedule until you enable it.
              "Draft &amp; dry-run" saves it and runs it once now so you can see the result.
            </div>
          </div>
        )}
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn" style={{ flex: 1, fontSize: 12 }}
            onClick={() => { setDesignErr(""); setDesignOpen(o => !o); }}>
            {designOpen ? "Close" : "✨ Design with AI"}
          </button>
          <button className="btn primary" style={{ flex: 1, fontSize: 12 }} onClick={onNew}>
            + New Job
          </button>
        </div>
        <p style={{ fontSize: 10, color: "var(--text-tertiary)", textAlign: "center", margin: "6px 0 0" }}>
          Jobs run while this app is open.
        </p>
      </div>
    </div>
  );
}

// ── Step Builder ──────────────────────────────────────────────────────────────

function StepBuilder({ steps, onChange, toolOptions, toolsLoading }: {
  steps: JobStep[];
  onChange: (steps: JobStep[]) => void;
  toolOptions: ToolOption[];
  toolsLoading: boolean;
}) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [searchState, setSearchState] = useState<Record<string, string>>({});
  const [dropdownOpen, setDropdownOpen] = useState<string | null>(null);

  const addStep = () => onChange([...steps, { id: uid(), step_type: "text", instruction: "", tool_name: undefined, tool_hint: undefined }]);
  const deleteStep = (id: string) => onChange(steps.filter(s => s.id !== id));
  const updateStep = (id: string, patch: Partial<JobStep>) => onChange(steps.map(s => s.id === id ? { ...s, ...patch } : s));
  const moveStep = (i: number, dir: -1 | 1) => {
    const j = i + dir;
    if (j < 0 || j >= steps.length) return;
    const c = [...steps]; [c[i], c[j]] = [c[j], c[i]]; onChange(c);
  };

  // Group filtered options for display
  const getGroups = (id: string) => {
    const q = (searchState[id] ?? "").toLowerCase();
    const filtered = q
      ? toolOptions.filter(t => t.name.toLowerCase().includes(q) || (t.service ?? "").toLowerCase().includes(q) || t.description.toLowerCase().includes(q))
      : toolOptions;
    const groups: Record<string, ToolOption[]> = {};
    for (const t of filtered) {
      const key = t.category === "builtin" ? "Built-in" : (t.service ?? t.category);
      (groups[key] ??= []).push(t);
    }
    return groups;
  };

  return (
    <div className="step-builder">
      {steps.length === 0 && (
        <div className="step-builder-empty">No steps yet — add one below to build your job.</div>
      )}

      {steps.map((step, i) => {
        const groups = getGroups(step.id);
        const isOpen = dropdownOpen === step.id;

        return (
          <div key={step.id} className="step-row">
            <div className="step-row-left">
              <span className="step-num-badge">{i + 1}</span>
              <div className="step-move-buttons">
                <button className="step-move-btn" onClick={() => moveStep(i, -1)} disabled={i === 0}>▲</button>
                <button className="step-move-btn" onClick={() => moveStep(i, 1)} disabled={i === steps.length - 1}>▼</button>
              </div>
            </div>

            <div className="step-row-body">
              {/* Type + delete row */}
              <div className="step-type-row">
                <label className={`step-type-chip${step.step_type === "text" ? " active" : ""}`}>
                  <input type="radio" name={`t-${step.id}`} checked={step.step_type === "text"}
                    onChange={() => updateStep(step.id, { step_type: "text", tool_name: undefined, tool_hint: undefined })} />
                  Text instruction
                </label>
                <label className={`step-type-chip${step.step_type === "tool" ? " active" : ""}`}>
                  <input type="radio" name={`t-${step.id}`} checked={step.step_type === "tool"}
                    onChange={() => updateStep(step.id, { step_type: "tool" })} />
                  Tool call
                </label>
                <button className="step-delete-btn" onClick={() => deleteStep(step.id)}>✕</button>
              </div>

              {/* Tool selector */}
              {step.step_type === "tool" && (
                <div className="step-tool-select" style={{ position: "relative" }}>
                  <div className="step-tool-search-wrap">
                    <input
                      className="admin-input step-tool-search"
                      placeholder={toolsLoading ? "Loading tools…" : "Search tools…"}
                      value={searchState[step.id] ?? ""}
                      onFocus={() => setDropdownOpen(step.id)}
                      // Close on click-away/tab-out. Options use onMouseDown+preventDefault, so
                      // picking one doesn't blur first — selection still registers.
                      onBlur={() => setDropdownOpen(null)}
                      onKeyDown={e => { if (e.key === "Escape") { setDropdownOpen(null); e.currentTarget.blur(); } }}
                      onChange={e => setSearchState(prev => ({ ...prev, [step.id]: e.target.value }))}
                    />
                    {step.tool_name && (
                      <span className="step-tool-selected-name">{step.tool_name}</span>
                    )}
                  </div>
                  {isOpen && (
                    <div className="step-tool-dropdown">
                      {Object.entries(groups).map(([group, opts]) => (
                        <div key={group}>
                          <div className="step-tool-group-label">{group}</div>
                          {opts.map(opt => (
                            <button key={opt.name} className="step-tool-option"
                              onMouseDown={e => { e.preventDefault();
                                updateStep(step.id, { tool_name: opt.name });
                                setSearchState(prev => ({ ...prev, [step.id]: "" }));
                                setDropdownOpen(null);
                              }}>
                              <span className="step-tool-option-name">{opt.name}</span>
                              {opt.description && <span className="step-tool-option-desc">{opt.description.slice(0, 70)}{opt.description.length > 70 ? "…" : ""}</span>}
                            </button>
                          ))}
                        </div>
                      ))}
                      {Object.keys(groups).length === 0 && (
                        <div className="step-tool-empty">No matching tools</div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Arg hints */}
              {step.step_type === "tool" && (
                <textarea className="admin-input step-hint-area"
                  value={step.tool_hint ?? ""}
                  onChange={e => updateStep(step.id, { tool_hint: e.target.value || undefined })}
                  placeholder={"Argument hints, one per line:\nsearch_query: cat:cs.AI AND all:transformer\nmax_results: 5"}
                  rows={4}
                  style={{ resize: "vertical" }}
                />
              )}

              {/* Instruction */}
              <textarea className="admin-input step-instruction-area"
                value={step.instruction ?? ""}
                onChange={e => updateStep(step.id, { instruction: e.target.value || undefined })}
                placeholder={step.step_type === "tool"
                  ? "Context for this step (e.g. extract the headline and article URL)"
                  : "Instruction for the LLM (e.g. Write a report using the information gathered above)"}
                rows={2}
                onClick={() => setDropdownOpen(null)}
              />
            </div>
          </div>
        );
      })}

      <button className="btn step-add-btn" onClick={addStep}>+ Add step</button>

      {steps.length > 0 && (
        <div className="step-preview-section">
          <button className="job-trace-toggle" onClick={() => setPreviewOpen(o => !o)}>
            <span style={{ fontSize: 11, fontWeight: 600 }}>Preview compiled prompt</span>
            <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{previewOpen ? "▲" : "▼"}</span>
          </button>
          {previewOpen && <pre className="step-preview-pre">{compileStepsPreview(steps)}</pre>}
        </div>
      )}
    </div>
  );
}

// ── Job form ──────────────────────────────────────────────────────────────────

function ToolSection({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ borderTop: "1px solid var(--border-light)", marginTop: 6 }}>
      <button className="job-trace-toggle" onClick={() => setOpen(o => !o)}>
        <span style={{ fontSize: 11, fontWeight: 600 }}>{title}</span>
        <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && <div style={{ paddingLeft: 4, paddingTop: 4 }}>{children}</div>}
    </div>
  );
}

function MCPServerToolToggle({ srv, checked, disabledTools, onToggleServer, onToggleTool }: {
  srv: StoredMCPServer; checked: boolean; disabledTools: string[];
  onToggleServer: (c: boolean) => void; onToggleTool: (name: string, enabled: boolean) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const toolNames = Object.keys(srv.enabledTools ?? {});
  return (
    <div style={{ marginBottom: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <input type="checkbox" checked={checked} onChange={e => onToggleServer(e.target.checked)} style={{ accentColor: "var(--purple)" }} />
        <span style={{ fontSize: 12, fontWeight: 600, flex: 1 }}>🖧 {srv.name}</span>
        {checked && toolNames.length > 0 && (
          <button style={{ fontSize: 10, background: "none", border: "none", color: "var(--text-tertiary)", cursor: "pointer" }}
            onClick={() => setExpanded(o => !o)}>
            {toolNames.length} tools {expanded ? "▲" : "▼"}
          </button>
        )}
      </div>
      {checked && expanded && toolNames.map(name => (
        <label key={name} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, paddingLeft: 20, cursor: "pointer", marginTop: 2 }}>
          <input type="checkbox" checked={!disabledTools.includes(name)}
            onChange={e => onToggleTool(name, e.target.checked)} style={{ accentColor: "var(--purple)" }} />
          <span style={{ fontFamily: "monospace" }}>{name}</span>
        </label>
      ))}
    </div>
  );
}

function JobForm({ job: initial, models, profiles, globalOpenapiSpecs, globalMcpServers, globalEnabledTools, globalAllowedDirs, onSave, onCancel }: {
  job: ScheduledJob; models: string[]; profiles: Profile[];
  globalOpenapiSpecs: import("./AdminPanel").StoredOpenAPISpec[];
  globalMcpServers: import("./AdminPanel").StoredMCPServer[];
  globalEnabledTools: Record<string, boolean>;
  globalAllowedDirs: string[];
  onSave: (j: ScheduledJob) => void; onCancel: () => void;
}) {
  const [job, setJob] = useState<ScheduledJob>({ ...initial, model: initial.model || models[0] || "", steps: initial.steps ?? [] });
  const set = (patch: Partial<ScheduledJob>) => setJob(j => ({ ...j, ...patch }));
  const setSched = (patch: Partial<JobSchedule>) => set({ schedule: { ...job.schedule, ...patch } });

  // Step builder mode — auto-detect from initial data
  const [promptMode, setPromptMode] = useState<"freeform" | "steps">(
    // Default to steps for new jobs; existing jobs with a prompt but no steps stay freeform
    initial.steps?.length ? "steps" : (initial.prompt ? "freeform" : "steps")
  );

  // Tool discovery for step builder
  const [toolOptions, setToolOptions] = useState<ToolOption[]>([]);
  const [toolsLoading, setToolsLoading] = useState(false);

  const selectedProfile = profiles.find(p => p.id === job.profile_id) ?? null;

  // When no profile is selected, show all registry tools; when a profile is selected, show its enabled subset
  const effectiveOpenAPI  = selectedProfile
    ? globalOpenapiSpecs.filter(s => s.enabled !== false && selectedProfile.enabledOpenapiSpecIds.includes(s.id))
    : globalOpenapiSpecs.filter(s => s.enabled !== false);
  const effectiveMcp      = selectedProfile
    ? globalMcpServers.filter(s => selectedProfile.enabledMcpServerIds.includes(s.id))
    : globalMcpServers;
  const effectiveBuiltins = selectedProfile ? (selectedProfile.enabledTools ?? {}) : globalEnabledTools;
  void !selectedProfile; // isGlobal — no longer directly used in JSX

  const loadToolOptions = async () => {
    setToolsLoading(true);
    try {
      // Registry is the single source of truth — no need to aggregate across profiles
      const allSpecs = globalOpenapiSpecs.filter(s => s.enabled !== false);
      const allMcp   = globalMcpServers;

      // Parse OpenAPI tools via the stateless command (no AppState mutation)
      const specs: SpecInfoLocal[] = allSpecs.length > 0
        ? await invoke<SpecInfoLocal[]>("get_spec_tools", { specs: allSpecs })
        : [];

      // MCP tools: merge stored tool names (from enabledTools map) with any
      // live tool lists from currently connected servers
      const connectedMcp = await invoke<MCPServerInfoLocal[]>("list_mcp_servers").catch(() => [] as MCPServerInfoLocal[]);
      const connectedById = new Map(connectedMcp.map(s => [s.id, s]));

      const mcpTools: MCPServerInfoLocal[] = allMcp.map(srv => {
        const live = connectedById.get(srv.id);
        // Prefer live tool list (has descriptions); fall back to stored enabledTools keys
        if (live && live.tools.length > 0) return live;
        return {
          id: srv.id, name: srv.name, connected: false,
          tools: Object.keys(srv.enabledTools ?? {}).map(name => ({ name, description: "" })),
        };
      });

      setToolOptions(buildToolOptions([], specs, mcpTools));
    } catch { /* ignore */ }
    setToolsLoading(false);
  };

  useEffect(() => { loadToolOptions(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Seed tool selections from the current profile's effective tools.
  // Always start from the full current profile list so newly-added specs/servers
  // are included by default — the old snapshot IDs might be a stale subset.
  const seedSpecIds  = () => effectiveOpenAPI.map(s => s.id);
  const seedMcpIds   = () => effectiveMcp.map(s => s.id);
  const seedDisabled = () => job.profile_context?.disabled_mcp_tools ?? effectiveMcp.flatMap(srv => Object.entries(srv.enabledTools ?? {}).filter(([,en]) => !en).map(([n]) => n));

  const [selSpecIds,  setSelSpecIds]  = useState<string[]>(seedSpecIds);
  const [selMcpIds,   setSelMcpIds]   = useState<string[]>(seedMcpIds);
  const [disabledMcp, setDisabledMcp] = useState<string[]>(seedDisabled);

  const toggleTool = (name: string) => {
    const has = job.enabled_builtin_tools.includes(name);
    set({ enabled_builtin_tools: has ? job.enabled_builtin_tools.filter(t => t !== name) : [...job.enabled_builtin_tools, name] });
  };

  const refreshSnapshot = (p: Profile | null) => {
    const specs = p
      ? globalOpenapiSpecs.filter(s => s.enabled !== false && p.enabledOpenapiSpecIds.includes(s.id))
      : globalOpenapiSpecs.filter(s => s.enabled !== false);
    const mcps = p
      ? globalMcpServers.filter(s => p.enabledMcpServerIds.includes(s.id))
      : globalMcpServers;
    setSelSpecIds(specs.map(s => s.id));
    setSelMcpIds(mcps.map(s => s.id));
    setDisabledMcp(mcps.flatMap(srv => Object.entries(srv.enabledTools ?? {}).filter(([,en]) => !en).map(([n]) => n)));
  };

  const pickOutputFile = async () => {
    const ext = job.output_file?.split(".").pop() ?? "txt";
    const filters = ext === "docx" ? [{ name: "Word", extensions: ["docx"] }] : ext === "pdf" ? [{ name: "PDF", extensions: ["pdf"] }] : [{ name: "Text / Markdown", extensions: ["txt", "md"] }];
    const path = await save({ title: "Job output file", filters });
    if (path) set({ output_file: path });
  };

  const canSave = job.name.trim() && job.model && (
    promptMode === "freeform" ? job.prompt.trim().length > 0 : (job.steps ?? []).length > 0
  );

  const handleSave = () => {
    let profile_context: import("./jobTypes").JobProfileContext | null = null;
    if (selectedProfile) {
      profile_context = resolveProfileContext(selectedProfile, selSpecIds, selMcpIds, disabledMcp, globalAllowedDirs, globalOpenapiSpecs, globalMcpServers);
    } else if (effectiveOpenAPI.length > 0 || effectiveMcp.length > 0) {
      // Build context from global settings so the job carries its tool snapshot
      profile_context = {
        ollama_host: globalAllowedDirs.length > 0 ? "" : "",  // uses AppState host at run time
        allowed_dirs: globalAllowedDirs,
        openapi_specs: effectiveOpenAPI.filter(s => selSpecIds.includes(s.id)).map(s => ({
          id: s.id, title: s.title, base_url: s.base_url, spec_json: s.spec_json, auth: s.auth,
        })),
        mcp_servers: effectiveMcp.filter(s => selMcpIds.includes(s.id)).map(s => ({
          id: s.id, name: s.name, command: s.command, args: s.args ?? [], env: s.env ?? {}, auth: s.auth,
        })),
        disabled_mcp_tools: disabledMcp,
        snapshot_at: new Date().toISOString(),
      };
    }
    onSave({
      ...job,
      profile_context,
      // Exclusive: only persist the active mode's data so Rust execution is deterministic
      prompt: promptMode === "freeform" ? job.prompt : "",
      steps:  promptMode === "steps"    ? (job.steps ?? []) : [],
    });
  };

  const [advancedOpen, setAdvancedOpen] = useState(!!job.output_file);

  // In steps mode, derive the tool list from the steps themselves
  const stepsToolNames = promptMode === "steps"
    ? [...new Set((job.steps ?? []).filter(s => s.step_type === "tool" && s.tool_name).map(s => s.tool_name!))]
    : null;

  // Summary shown in the Advanced accordion header
  const toolSummary = (() => {
    if (stepsToolNames !== null) {
      return stepsToolNames.length > 0 ? stepsToolNames.join(", ") : "derived from steps";
    }
    const nb = job.enabled_builtin_tools.length;
    const ns = selSpecIds.length;
    const nm = selMcpIds.length;
    const parts = [];
    if (nb > 0) parts.push(`${nb} built-in`);
    if (ns > 0) parts.push(`${ns} API`);
    if (nm > 0) parts.push(`${nm} MCP`);
    return parts.length ? parts.join(" · ") : "none";
  })();

  return (
    <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
    <div style={{ padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12 }}>

      {/* ── Row 1: Name + Enabled ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <input className="admin-input" style={{ flex: 1, fontWeight: 600 }}
          value={job.name} onChange={e => set({ name: e.target.value })} placeholder="Job name…" />
        <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, flexShrink: 0, cursor: "pointer" }}>
          <input type="checkbox" checked={job.enabled} onChange={e => set({ enabled: e.target.checked })} style={{ accentColor: "var(--accent)" }} />
          Enabled
        </label>
      </div>

      {/* ── Row 2: Profile + Model side by side ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, alignItems: "start" }}>
        <div className="field" style={{ margin: 0 }}>
          <label>
            Tool connections
            <span style={{ fontWeight: 400, opacity: 0.5, marginLeft: 5, fontSize: 10 }}>
              (provides Gmail, MCP, etc.)
            </span>
          </label>
          <select className="admin-input" value={job.profile_id ?? ""}
            onChange={e => {
              const p = profiles.find(x => x.id === e.target.value) ?? null;
              set({
                profile_id: p?.id ?? null,
                profile_name: p?.name ?? null,
                model: p?.model ?? job.model,
                system_prompt: p?.systemPrompt ?? null,
                enabled_builtin_tools: p
                  ? Object.entries(p.enabledTools ?? {}).filter(([,v]) => v !== false).map(([k]) => k)
                  : Object.entries(globalEnabledTools).filter(([,v]) => v !== false).map(([k]) => k),
              });
              refreshSnapshot(p);
              loadToolOptions();
            }}>
            <option value="">Global</option>
            {profiles.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
          {job.profile_context && (
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
              <span style={{ fontSize: 9, color: "var(--text-tertiary)" }}>
                Snapshot {new Date(job.profile_context.snapshot_at).toLocaleDateString([], { month: "short", day: "numeric" })}
              </span>
              <button className="btn" style={{ fontSize: 9, padding: "1px 5px" }} onClick={() => refreshSnapshot(selectedProfile)}>
                Refresh
              </button>
            </div>
          )}
        </div>
        <div className="field" style={{ margin: 0 }}>
          <label>Model</label>
          <select className="admin-input" value={job.model} onChange={e => set({ model: e.target.value })}>
            {models.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
      </div>

      {/* ── Row 3: Schedule ── */}
      <div className="field" style={{ margin: 0 }}>
        <label>Schedule</label>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <select className="admin-input" style={{ width: "auto" }} value={job.schedule.type}
            onChange={e => setSched({ type: e.target.value as JobSchedule["type"] })}>
            <option value="Daily">Daily</option>
            <option value="Interval">Every N hours</option>
            <option value="Weekly">Weekly</option>
            <option value="Manual">Manual only</option>
          </select>
          {(job.schedule.type === "Daily" || job.schedule.type === "Weekly") && <>
            {job.schedule.type === "Weekly" && (
              <select className="admin-input" style={{ width: "auto" }} value={job.schedule.weekday ?? 0}
                onChange={e => setSched({ weekday: Number(e.target.value) })}>
                {WEEKDAYS.map((d, i) => <option key={i} value={i}>{d}</option>)}
              </select>
            )}
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>at</span>
            <input type="number" className="admin-input" style={{ width: 52 }} min={0} max={23}
              value={job.schedule.hour ?? 9} onChange={e => setSched({ hour: Number(e.target.value) })} />
            <span style={{ fontSize: 12 }}>:</span>
            <input type="number" className="admin-input" style={{ width: 52 }} min={0} max={59}
              value={job.schedule.minute ?? 0} onChange={e => setSched({ minute: Number(e.target.value) })} />
          </>}
          {job.schedule.type === "Interval" && <>
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>every</span>
            <input type="number" className="admin-input" style={{ width: 60 }} min={1}
              value={job.schedule.hours ?? 4} onChange={e => setSched({ hours: Number(e.target.value) })} />
            <span style={{ fontSize: 12, color: "var(--text-secondary)" }}>hours</span>
          </>}
        </div>
      </div>

      {/* ── Divider ── */}
      <div style={{ borderTop: "1px solid var(--border-light)", margin: "2px 0" }} />

      {/* ── Prompt / Step builder (main content) ── */}
      <div className="field" style={{ margin: 0 }}>
        <div className="step-mode-toggle">
          <button className={`step-mode-btn${promptMode === "steps" ? " active" : ""}`}
            onClick={() => setPromptMode("steps")}>Step builder</button>
        </div>
        {promptMode === "freeform" ? (
          <textarea className="admin-input" value={job.prompt} onChange={e => set({ prompt: e.target.value })}
            placeholder="What should Lexi do each time this runs?" style={{ minHeight: 90, resize: "vertical" }} />
        ) : (
          <>
            {job.profile_id && (
              <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginBottom: 4 }}>
                Tool dropdown reflects the currently active profile's connections. Ensure the target profile is active when creating this job.
              </div>
            )}
            <StepBuilder steps={job.steps ?? []} onChange={s => set({ steps: s })}
              toolOptions={toolOptions} toolsLoading={toolsLoading} />
          </>
        )}
      </div>

      {/* ── System Prompt ── */}
      <div className="field" style={{ margin: 0 }}>
        <label>
          System Prompt
          <span style={{ fontWeight: 400, opacity: 0.5, marginLeft: 5 }}>(optional)</span>
        </label>
        <textarea
          className="admin-input"
          value={job.system_prompt ?? ""}
          onChange={e => set({ system_prompt: e.target.value || null })}
          placeholder={promptMode === "steps"
            ? "Leave blank to use the built-in step executor prompt.\nOverride here to customise the agent's behaviour for this job."
            : "Leave blank to use the default assistant prompt.\nOverride here to customise the agent's behaviour for this job."}
          style={{ minHeight: 70, resize: "vertical", fontSize: 11, fontFamily: "monospace" }}
        />
        {job.system_prompt && (
          <div style={{ fontSize: 10, color: "var(--accent)", marginTop: 2 }}>
            Custom system prompt active — the profile's system prompt will not be used.
          </div>
        )}
        {!job.system_prompt && (
          <div style={{ fontSize: 10, color: "var(--text-tertiary)", marginTop: 2 }}>
            {promptMode === "steps"
              ? "Using built-in step executor prompt (autonomous, no confirmation, strict step order)."
              : "Using default assistant prompt."}
          </div>
        )}
      </div>

      {/* ── Advanced (collapsible) ── */}
      <div style={{ border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden" }}>
        <button
          onClick={() => setAdvancedOpen(o => !o)}
          style={{ width: "100%", display: "flex", alignItems: "center", gap: 8, padding: "8px 12px",
            background: "var(--surface2)", border: "none", cursor: "pointer", textAlign: "left" }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>Advanced</span>
          <span style={{ fontSize: 10, color: "var(--text-tertiary)", fontWeight: 400 }}>
            Tools: {toolSummary}{job.output_file ? " · output file set" : ""}
          </span>
          <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{advancedOpen ? "▲" : "▼"}</span>
        </button>

        {advancedOpen && (
          <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 12 }}>

            {/* Tools — read-only summary in steps mode, full selector in freeform */}
            <div className="field" style={{ margin: 0 }}>
              <label>Tools</label>
              {stepsToolNames !== null ? (
                // Steps mode: tools are derived automatically from step tool_names
                <div style={{ marginTop: 4 }}>
                  {stepsToolNames.length > 0 ? (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 4 }}>
                      {stepsToolNames.map(n => (
                        <span key={n} style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 600, padding: "2px 7px", borderRadius: 5, background: "var(--purple-bg)", color: "var(--purple)", border: "1px solid var(--purple-border)" }}>{n}</span>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 12, color: "var(--text-tertiary)", fontStyle: "italic" }}>Add tool call steps above to specify tools.</div>
                  )}
                  <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>
                    Tools are derived automatically from the steps above — only tools named in steps are sent to the model.
                  </div>
                </div>
              ) : (
                // Freeform mode: full manual tool selector
                <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: "6px 10px", marginTop: 4 }}>
                  <ToolSection title="Built-in tools" defaultOpen>
                    <div className="job-tool-grid">
                      {BUILTIN_TOOLS.filter(t => effectiveBuiltins[t.name] !== false).map(t => (
                        <label key={t.name} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, cursor: "pointer" }}>
                          <input type="checkbox" checked={job.enabled_builtin_tools.includes(t.name)}
                            onChange={() => toggleTool(t.name)} style={{ accentColor: "var(--purple)" }} />
                          {t.icon} {t.label}
                        </label>
                      ))}
                    </div>
                  </ToolSection>
                  {effectiveOpenAPI.length > 0 && (
                    <ToolSection title="OpenAPI / REST services" defaultOpen>
                      {effectiveOpenAPI.map(sp => (
                        <label key={sp.id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, cursor: "pointer", padding: "3px 0" }}>
                          <input type="checkbox" checked={selSpecIds.includes(sp.id)}
                            onChange={e => setSelSpecIds(prev => e.target.checked ? [...prev, sp.id] : prev.filter(x => x !== sp.id))}
                            style={{ accentColor: "var(--accent)" }} />
                          <span style={{ fontWeight: 600 }}>{sp.title}</span>
                          <span style={{ fontSize: 10, color: "var(--text-tertiary)" }}>{sp.base_url}</span>
                        </label>
                      ))}
                    </ToolSection>
                  )}
                  {effectiveMcp.length > 0 && (
                    <ToolSection title="MCP servers">
                      {effectiveMcp.map(srv => (
                        <MCPServerToolToggle key={srv.id} srv={srv}
                          checked={selMcpIds.includes(srv.id)} disabledTools={disabledMcp}
                          onToggleServer={c => setSelMcpIds(prev => c ? [...prev, srv.id] : prev.filter(x => x !== srv.id))}
                          onToggleTool={(n, en) => setDisabledMcp(prev => en ? prev.filter(x => x !== n) : [...prev, n])}
                        />
                      ))}
                    </ToolSection>
                  )}
                </div>
              )}
            </div>


            {/* Output file */}
            <div className="field" style={{ margin: 0 }}>
              <label>Save output to file <span style={{ fontWeight: 400, opacity: 0.5 }}>(optional)</span></label>
              <div style={{ display: "flex", gap: 8 }}>
                <input className="admin-input" style={{ flex: 1, fontFamily: "monospace", fontSize: 11 }}
                  value={job.output_file ?? ""} onChange={e => set({ output_file: e.target.value || null })} placeholder="/path/to/output.md" />
                <button className="btn" style={{ flexShrink: 0 }} onClick={pickOutputFile}>Browse…</button>
              </div>
            </div>

          </div>
        )}
      </div>

      {/* ── Actions ── */}
      <div style={{ display: "flex", gap: 8, paddingTop: 2 }}>
        <button className="btn" onClick={onCancel}>Cancel</button>
        <button className="btn primary" disabled={!canSave} onClick={handleSave}>Save Job</button>
      </div>
    </div>
    </div>
  );
}

// ── History tab ───────────────────────────────────────────────────────────────

function HistoryTab({ runs, jobs, filterJobId, onFilterChange, onClear }: {
  runs: JobRun[];
  jobs: ScheduledJob[];
  filterJobId: string | null;
  onFilterChange: (id: string | null) => void;
  onClear: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Controls bar */}
      <div style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: 8, borderBottom: "1px solid var(--border-light)", flexShrink: 0 }}>
        <select className="admin-input" style={{ flex: 1 }} value={filterJobId ?? ""}
          onChange={e => onFilterChange(e.target.value || null)}>
          <option value="">All jobs</option>
          {jobs.map(j => <option key={j.id} value={j.id}>{j.name}</option>)}
        </select>
        <button className="btn" style={{ fontSize: 11, padding: "3px 8px", color: "#f87171" }} onClick={onClear}>
          Clear
        </button>
      </div>

      {/* Runs list */}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {runs.length === 0 && (
          <div className="admin-empty">No run history yet. Run a job to see results here.</div>
        )}
        {runs.map(run => <RunRow key={run.id} run={run} />)}
      </div>
    </div>
  );
}

function buildDebugReport(run: JobRun): string {
  const lines: string[] = [];
  lines.push("=== Job Run Debug Report ===");
  lines.push(`Job:      ${run.job_name}`);
  lines.push(`Status:   ${run.status}`);
  lines.push(`Profile:  ${run.profile_name ?? "Global"}`);
  lines.push(`Started:  ${run.started_at}`);
  lines.push(`Finished: ${run.finished_at}`);
  lines.push(`Duration: ${run.duration_ms}ms`);
  lines.push("");

  if (run.trace.length > 0) {
    lines.push("=== EXECUTION TRACE ===");
    for (const step of run.trace) {
      lines.push("");
      lines.push(`STEP ${step.step + 1}`);
      if (step.llm_text) {
        lines.push("  LLM:");
        for (const l of step.llm_text.split("\n")) lines.push(`    ${l}`);
      }
      for (const tc of step.tool_calls) {
        lines.push(`  TOOL: ${tc.name}`);
        if (tc.args && tc.args !== "{}") {
          lines.push("  ARGS:");
          for (const l of tc.args.split("\n")) lines.push(`    ${l}`);
        }
        if (tc.result) {
          lines.push("  RESULT:");
          for (const l of tc.result.split("\n")) lines.push(`    ${l}`);
        }
      }
    }
    lines.push("");
  }

  lines.push("=== FINAL OUTPUT ===");
  lines.push(run.output || "(no output)");

  if (run.error) {
    lines.push("");
    lines.push("=== ERROR ===");
    lines.push(run.error);
  }

  return lines.join("\n");
}

function RunRow({ run }: { run: JobRun }) {
  const [expanded,  setExpanded]  = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [copied,    setCopied]    = useState(false);
  const [copiedDebug, setCopiedDebug] = useState(false);
  const toolCount = run.trace.reduce((n, s) => n + s.tool_calls.length, 0);
  const stepCount = run.trace.length;
  const preview   = run.output.slice(0, 120) + (run.output.length > 120 ? "…" : "");
  const isSuccess = run.status === "Success";

  const copyOutput = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(run.output);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const copyDebugReport = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await navigator.clipboard.writeText(buildDebugReport(run));
    setCopiedDebug(true);
    setTimeout(() => setCopiedDebug(false), 2000);
  };

  return (
    <div className="job-run-row">
      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}
           onClick={() => setExpanded(e => !e)}>
        {/* Status badge */}
        <span style={{
          fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 8, flexShrink: 0,
          background: isSuccess ? "rgba(74,222,128,0.12)" : "rgba(248,113,113,0.12)",
          color: isSuccess ? "#4ade80" : "#f87171",
          border: `1px solid ${isSuccess ? "rgba(74,222,128,0.25)" : "rgba(248,113,113,0.25)"}`,
        }}>
          {isSuccess ? "✓ OK" : "✕ Error"}
        </span>

        {/* Job name */}
        <span style={{ fontSize: 12, fontWeight: 600, flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {run.job_name}
        </span>

        {/* Profile badge */}
        {run.profile_name && (
          <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 8, background: "var(--surface2)", color: "var(--text-secondary)", flexShrink: 0 }}>
            {run.profile_name}
          </span>
        )}

        {/* Timing */}
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div style={{ fontSize: 10, color: "var(--text-secondary)" }}>{formatDateTime(run.started_at)}</div>
          <div style={{ fontSize: 10, color: "var(--text-tertiary)" }}>
            → {formatDateTime(run.finished_at)} · {formatDuration(run.duration_ms)}
          </div>
        </div>

        <span style={{ fontSize: 10, opacity: 0.4, flexShrink: 0 }}>{expanded ? "▲" : "▼"}</span>
      </div>

      {/* ── Collapsed: output preview ── */}
      {!expanded && run.output && (
        <div className="job-run-preview">{preview}</div>
      )}

      {/* ── Expanded ── */}
      {expanded && (
        <div style={{ paddingTop: 8 }} onClick={e => e.stopPropagation()}>

          {/* Copy full debug report — prominent, always visible when expanded */}
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
            <button
              className={`copy-btn${copiedDebug ? " copied" : ""}`}
              style={{ fontSize: 11, gap: 5 }}
              onClick={copyDebugReport}
            >
              {copiedDebug ? "✓ Copied" : "⧉ Copy debug report"}
            </button>
          </div>

          {/* Error */}
          {run.error && <div className="job-run-error">{run.error}</div>}

          {/* Debug trace — collapsible section */}
          {stepCount > 0 && (
            <div style={{ marginBottom: 8 }}>
              <button
                className="job-trace-toggle"
                onClick={() => setShowTrace(t => !t)}
              >
                <span>🔍 Debug trace</span>
                <span style={{ fontSize: 10, color: "var(--text-tertiary)", marginLeft: 6 }}>
                  {stepCount} step{stepCount !== 1 ? "s" : ""} · {toolCount} tool call{toolCount !== 1 ? "s" : ""}
                </span>
                <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>
                  {showTrace ? "▲" : "▼"}
                </span>
              </button>
              {showTrace && (
                <div className="job-trace" style={{ marginTop: 6 }}>
                  {run.trace.map(step => <TraceStepView key={step.step} step={step} />)}
                </div>
              )}
            </div>
          )}

          {/* Final output — always shown */}
          <div className="job-trace-label">Output</div>
          <div className="job-run-expanded">
            {run.output
              ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{run.output}</ReactMarkdown>
              : <span style={{ color: "var(--text-tertiary)", fontStyle: "italic" }}>No text output recorded.</span>
            }
          </div>
          {run.output && (
            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 6 }}>
              <button
                className={`copy-btn${copied ? " copied" : ""}`}
                onClick={copyOutput}
              >
                {copied ? "✓ Copied" : "⧉ Copy output"}
              </button>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

function TraceStepView({ step }: { step: TraceStep }) {
  const [openTool, setOpenTool] = useState<number | null>(null);

  return (
    <div className="job-trace-step">
      <div className="job-trace-step-header">
        <span className="job-trace-step-num">Step {step.step + 1}</span>
      </div>

      {/* LLM text */}
      {step.llm_text && (
        <div className="job-trace-llm">
          <span className="job-trace-tag llm">LLM</span>
          <span className="job-trace-llm-text">{step.llm_text}</span>
        </div>
      )}

      {/* Tool calls */}
      {step.tool_calls.map((tc, i) => (
        <div key={i} className="job-trace-tool">
          <div className="job-trace-tool-header"
               onClick={e => { e.stopPropagation(); setOpenTool(openTool === i ? null : i); }}>
            <span className="job-trace-tag tool">TOOL</span>
            <span className="job-trace-tool-name">{tc.name}</span>
            <span style={{ marginLeft: "auto", fontSize: 10, opacity: 0.4 }}>{openTool === i ? "▲" : "▼"}</span>
          </div>
          {openTool === i && (
            <div className="job-trace-tool-body">
              {tc.args && tc.args !== "{}" && (
                <div className="job-trace-section">
                  <div className="job-trace-section-label">Args</div>
                  <pre className="job-trace-pre">{tc.args}</pre>
                </div>
              )}
              {tc.result && (
                <div className="job-trace-section">
                  <div className="job-trace-section-label">Result</div>
                  <pre className="job-trace-pre">
                    {tc.result.slice(0, 1500)}{tc.result.length > 1500 ? "\n…[truncated]" : ""}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
