use chrono::{DateTime, Datelike, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::Manager;

// ── Types ──────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum JobSchedule {
    Daily    { hour: u32, minute: u32 },
    Interval { hours: u32 },
    /// weekday: 0=Mon … 6=Sun
    Weekly   { weekday: u32, hour: u32, minute: u32 },
    Manual,
}

/// A single step in a structured job. Compiles to one numbered section in the prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStep {
    pub id: String,
    pub step_type: String,             // "text" or "tool"
    #[serde(default)]
    pub instruction: Option<String>,   // LLM instruction for this step
    #[serde(default)]
    pub tool_name: Option<String>,     // exact tool name (tool steps only)
    #[serde(default)]
    pub tool_hint: Option<String>,     // argument hints (tool steps only)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub schedule: JobSchedule,
    pub prompt: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub enabled_builtin_tools: Vec<String>,
    pub output_file: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_run_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub profile_name: Option<String>,
    #[serde(default)]
    pub profile_context: Option<JobProfileContext>,
    #[serde(default)]
    pub steps: Vec<JobStep>,
    /// Run a Daily/Weekly slot that was missed because the machine was asleep or busy,
    /// as soon as it wakes. Defaults to true, so jobs saved before this existed are
    /// fixed on load. Turn it off for jobs that are only useful on time (a morning
    /// briefing delivered at 6pm is noise).
    #[serde(default = "default_catch_up")]
    pub catch_up: bool,
    /// Allow this job to run `run_python`. Code execution is otherwise gated behind an
    /// interactive approval that a background job can never answer, so without this a job
    /// given run_python fails on every call. Off by default: switching it on means this job
    /// may execute model-written Python unattended, so it is opted into per job rather than
    /// inherited by every job on the machine.
    #[serde(default)]
    pub allow_code_exec: bool,
}

fn default_catch_up() -> bool { true }

/// NOTE: serialised into scheduled_jobs.json — may contain API keys/tokens from the
/// profile's auth configs. Acceptable for a single-user desktop app (same as localStorage).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JobProfileContext {
    pub ollama_host:        String,
    pub allowed_dirs:       Vec<String>,
    pub openapi_specs:      Vec<JobOpenAPISpec>,
    pub mcp_servers:        Vec<JobMCPServer>,
    pub disabled_mcp_tools: Vec<String>,
    pub snapshot_at:        String,  // ISO-8601, shown in UI for staleness check
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobOpenAPISpec {
    pub id: String,
    pub title: String,
    pub base_url: String,
    pub spec_json: String,
    #[serde(default)]
    pub auth: crate::mcp::AuthConfig,
    #[serde(default)]
    pub response_exclude: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobMCPServer {
    pub id: String,
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub auth: crate::mcp::AuthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RunStatus { Success, Error }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceToolCall {
    pub name: String,
    pub args: String,          // pretty-printed JSON
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    pub step: usize,
    pub llm_text: Option<String>,
    pub tool_calls: Vec<TraceToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    pub id: String,
    pub job_id: String,
    pub job_name: String,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub status: RunStatus,
    pub output: String,
    pub error: Option<String>,
    #[serde(default)]
    pub trace: Vec<TraceStep>,
    #[serde(default)]
    pub profile_name: Option<String>,
    // Legacy field alias so old records still deserialise
    #[serde(default, rename = "ran_at", skip_serializing)]
    pub _ran_at_legacy: Option<DateTime<Utc>>,
}

/// Build an execution trace from the conversation history after the agent loop completes.
pub fn build_trace(conv: &[crate::ollama::WireMessage]) -> Vec<TraceStep> {
    let mut steps: Vec<TraceStep> = Vec::new();
    let mut i = 0;

    // Skip leading user messages (the initial prompt)
    while i < conv.len() && conv[i].role == "user" {
        i += 1;
    }

    while i < conv.len() {
        let msg = &conv[i];
        if msg.role != "assistant" {
            i += 1;
            continue;
        }

        let llm_text = msg.content.clone().filter(|s| !s.is_empty());
        let mut tool_calls: Vec<TraceToolCall> = Vec::new();

        if let Some(calls) = &msg.tool_calls {
            // Advance past this assistant message
            i += 1;
            for call in calls {
                let name = call.function.name.clone();
                let args = serde_json::to_string_pretty(&call.function.arguments)
                    .unwrap_or_default();
                // The next message should be the tool result for this call
                let result = if i < conv.len() && conv[i].role == "tool" {
                    let r = conv[i].content.clone().unwrap_or_default();
                    i += 1;
                    Some(r)
                } else {
                    None
                };
                tool_calls.push(TraceToolCall { name, args, result });
            }
        } else {
            i += 1;
        }

        steps.push(TraceStep {
            step: steps.len(),
            llm_text,
            tool_calls,
        });
    }

    steps
}

/// Compile a steps list into a structured numbered prompt the LLM reliably follows.
pub fn compile_steps(steps: &[JobStep]) -> String {
    if steps.is_empty() { return String::new(); }
    let n = steps.len();
    let mut out = format!(
        "Execute the following {} step{} IN ORDER. Do not skip any step.\n\
         After each step output: checkmark Step N complete: [one line summary]\n\n",
        n, if n == 1 { "" } else { "s" }
    );
    for (i, step) in steps.iter().enumerate() {
        let instr = step.instruction.as_deref().unwrap_or("").trim();
        out.push_str(&format!("STEP {} — {}\n", i + 1, instr));
        if step.step_type == "tool" {
            if let Some(ref name) = step.tool_name {
                out.push_str(&format!("Call tool: {}\n", name));
            }
            if let Some(ref hint) = step.tool_hint {
                let h = hint.trim();
                if !h.is_empty() { out.push_str(&format!("Arguments hint: {}\n", h)); }
            }
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

// ── Persistence ────────────────────────────────────────────────────────────────

pub fn jobs_path() -> std::path::PathBuf { crate::dirs_path().join("scheduled_jobs.json") }
pub fn runs_path() -> std::path::PathBuf { crate::dirs_path().join("job_runs.json") }

pub fn load_jobs() -> Vec<ScheduledJob> {
    std::fs::read_to_string(jobs_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_jobs(jobs: &[ScheduledJob]) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(jobs)?;
    std::fs::write(jobs_path(), json)?;
    Ok(())
}

pub fn load_runs() -> Vec<JobRun> {
    std::fs::read_to_string(runs_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

/// Prepend the new run, keep at most 100 entries per job_id, persist.
pub fn append_run(run: JobRun) -> anyhow::Result<()> {
    let mut runs = load_runs();
    runs.insert(0, run);
    let mut counts: HashMap<String, usize> = HashMap::new();
    runs.retain(|r| {
        let c = counts.entry(r.job_id.clone()).or_insert(0);
        *c += 1;
        *c <= 100
    });
    let json = serde_json::to_string_pretty(&runs)?;
    std::fs::write(runs_path(), json)?;
    Ok(())
}

// ── Schedule due-time check ────────────────────────────────────────────────────

/// How late a job with catch-up disabled may still start. Replaces the old exact
/// minute-equality test, which a tick delayed by load could miss even while awake.
const NO_CATCHUP_GRACE_MINS: i64 = 2;

/// Gap between catch-up runs that come due in the same tick, so a lid opened after a
/// weekend doesn't start every stale job against the model server simultaneously.
const STAGGER_SECS: u64 = 30;

type InFlight = std::sync::Arc<std::sync::Mutex<std::collections::HashSet<String>>>;

/// Releases a job's in-flight claim on drop, so a run that panics doesn't leave the
/// job permanently claimed — which would silently stop it running until restart.
struct InFlightGuard {
    set: InFlight,
    id:  String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if let Ok(mut set) = self.set.lock() {
            set.remove(&self.id);
        }
    }
}

/// The most recent instant this schedule should have fired, at or before `now`.
/// `None` for schedules with no wall-clock slot (Interval, Manual).
///
/// Only ever returns the LATEST slot, which is what keeps catch-up from stampeding:
/// a machine that was off for a week comes back to one outstanding run, not seven.
fn prev_occurrence(
    schedule: &JobSchedule,
    now: DateTime<chrono::Local>,
) -> Option<NaiveDateTime> {
    match schedule {
        JobSchedule::Daily { hour, minute } => {
            let today = now.date_naive().and_hms_opt(*hour, *minute, 0)?;
            Some(if today <= now.naive_local() {
                today
            } else {
                today - chrono::Duration::days(1)
            })
        }
        JobSchedule::Weekly { weekday, hour, minute } => {
            let back = (now.weekday().num_days_from_monday() as i64 - *weekday as i64).rem_euclid(7);
            let cand = (now.date_naive() - chrono::Duration::days(back))
                .and_hms_opt(*hour, *minute, 0)?;
            Some(if cand <= now.naive_local() {
                cand
            } else {
                cand - chrono::Duration::days(7)
            })
        }
        _ => None,
    }
}

pub fn is_due(
    schedule: &JobSchedule,
    last_run: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    catch_up: bool,
    now: DateTime<Utc>,
) -> bool {
    match schedule {
        JobSchedule::Manual => false,

        JobSchedule::Interval { hours } => match last_run {
            None     => true,
            Some(lr) => now.signed_duration_since(lr).num_hours() >= *hours as i64,
        },

        // Daily/Weekly fire on a wall-clock slot. Testing for the exact minute meant a
        // machine asleep at 07:00 never observed minute 07:00 at all and silently skipped
        // to the next day. Compare against the most recent slot instead: if it has passed
        // and we have not run since, the job is outstanding and runs on wake.
        JobSchedule::Daily { .. } | JobSchedule::Weekly { .. } => {
            let local = now.with_timezone(&chrono::Local);
            let Some(prev) = prev_occurrence(schedule, local) else { return false };
            // Fall back to creation time so a job added at 3pm with a 07:00 slot does not
            // immediately count this morning's slot as missed.
            let baseline = last_run.unwrap_or(created_at)
                .with_timezone(&chrono::Local)
                .naive_local();
            if baseline >= prev {
                return false;
            }
            catch_up
                || local.naive_local().signed_duration_since(prev).num_minutes()
                    < NO_CATCHUP_GRACE_MINS
        }
    }
}

// ── Job executor ───────────────────────────────────────────────────────────────

/// Wrap a job's flat tool list as a single discoverable group so the agent loop's per-step
/// selection can narrow it when large; small sets (e.g. explicit step tools) are sent whole.
fn job_tool_groups(tools: Vec<crate::ollama::ToolSchema>) -> Vec<crate::ollama::ToolGroup> {
    if tools.is_empty() {
        Vec::new()
    } else {
        vec![crate::ollama::ToolGroup {
            label: "Job tools".into(),
            description: "Tools available to this scheduled job.".into(),
            tools,
        }]
    }
}

pub async fn execute_job(
    job: &ScheduledJob,
    state: &crate::AppState,
    app: &tauri::AppHandle,
) -> (DateTime<Utc>, String, Vec<TraceStep>, Option<String>) {
    use crate::ollama::{agent_loop, ToolSchema, WireMessage};
    use std::sync::Mutex;

    let started_at = Utc::now();

    // If the job has structured steps, compile them into a deterministic prompt.
    let effective_prompt = if !job.steps.is_empty() {
        compile_steps(&job.steps)
    } else {
        job.prompt.clone()
    };

    // Isolated conversation — never touches AppState::conversation
    let conversation: Mutex<Vec<WireMessage>> = Mutex::new(vec![WireMessage {
        role: "user".into(),
        content: Some(effective_prompt),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        images: None,
    }]);

    // Resolve backend: a per-job host override is treated as an Ollama URL (legacy config);
    // otherwise inherit the active AppState backend (which may be an OpenAI-compatible endpoint).
    let backend = job.profile_context.as_ref()
        .map(|c| c.ollama_host.clone())
        .filter(|h| !h.is_empty())
        .map(crate::ollama::Backend::ollama)
        .unwrap_or_else(|| state.backend.lock().unwrap().clone());

    // When using steps, derive the active tool set from step tool_names — only those
    // tools are sent to the LLM, keeping the context tight and reducing token usage.
    // In free-form mode, use job.enabled_builtin_tools as before.
    let step_tool_names: std::collections::HashSet<String> = job.steps.iter()
        .filter(|s| s.step_type == "tool")
        .filter_map(|s| s.tool_name.clone())
        .collect();
    let use_step_tools = !step_tool_names.is_empty();

    let mut all_tools: Vec<ToolSchema> = crate::tools::all_builtin_schemas()
        .iter()
        .filter_map(|s| serde_json::from_value::<ToolSchema>(s.clone()).ok())
        .filter(|t| {
            if use_step_tools { step_tool_names.contains(&t.function.name) }
            else              { job.enabled_builtin_tools.contains(&t.function.name) }
        })
        .collect();

    // `run_python` is frontend-provided (Pyodide runs in the webview), so it is not in
    // `all_builtin_schemas` and a job could never be given it — despite dispatch having always
    // supported it for background runs. Offer it when the job asks for it, so a scheduled job can
    // process an offloaded result instead of judging from a truncated one.
    let wants_python = if use_step_tools { step_tool_names.iter().any(|n| n == "run_python") }
                       else              { job.enabled_builtin_tools.iter().any(|n| n == "run_python") };
    if wants_python {
        if let Ok(t) = serde_json::from_value::<ToolSchema>(crate::tools::run_python_schema()) {
            all_tools.push(t);
        }
    }
    // Same story for the wiki: implemented in Rust, but schema-less on this side.
    for v in crate::tools::wiki_schemas_for_jobs() {
        let Ok(t) = serde_json::from_value::<ToolSchema>(v) else { continue };
        let wanted = if use_step_tools { step_tool_names.contains(&t.function.name) }
                     else              { job.enabled_builtin_tools.contains(&t.function.name) };
        if wanted && !all_tools.iter().any(|e| e.function.name == t.function.name) {
            all_tools.push(t);
        }
    }

    // Inject current date/time
    let now_local = chrono::Local::now();
    let date_prefix = format!(
        "Current date and time: {} ({})\n\n",
        now_local.format("%A, %d %B %Y at %H:%M"),
        now_local.format("%Y-%m-%d_%H-%M")
    );
    // Resolve allowed dirs early so we can inject them into the system prompt
    let allowed_dirs_for_prompt: Vec<String> = if let Some(ref ctx) = job.profile_context {
        ctx.allowed_dirs.clone()
    } else {
        state.allowed_dirs.lock().unwrap().clone()
    };

    let base_prompt = job.system_prompt.clone().unwrap_or_else(|| {
        if !job.steps.is_empty() {
            "You are a fully autonomous automated job executor. There is no human present — do not ask for confirmation, permission, or input at any point.\n\
             Rules:\n\
             - Execute ALL steps in order, one after another, without stopping or waiting.\n\
             - NEVER ask for confirmation. NEVER say 'please confirm' or 'shall I continue'. Just continue.\n\
             - After each step output: checkmark Step N complete: [one line summary] — then immediately proceed to the next step.\n\
             - When a tool returns a value you must pass to the next step, copy it EXACTLY as returned — no paraphrasing or modification.\n\
             - If a step fails, output: X Step N failed: [reason] — then continue with the remaining steps.\n\
             - Keep going until all steps are done.".into()
        } else {
            "You are a helpful assistant. Complete the user's request thoroughly.".into()
        }
    });

    let dirs_note = if !allowed_dirs_for_prompt.is_empty() {
        format!(
            "\n\nYou have access to these folders on the user's computer: {}. \
             Always use full absolute paths when accessing files. Never use '.' or '~'.",
            allowed_dirs_for_prompt.join(", ")
        )
    } else {
        String::new()
    };

    let system_prompt = format!(
        "{date_prefix}{base_prompt}{dirs_note}\n\nIMPORTANT: Always finish with a written summary or report of what you found or did. Never end on a tool call alone — always produce a final text response."
    );

    let result = if let Some(ref ctx) = job.profile_context {
        // ── Profile-specific isolated path ────────────────────────────────────
        // Parse OpenAPI specs from snapshot, build isolated MCP connections.
        // Nothing here touches the global AppState — no interference with main chat.

        // Per-step tool selection now happens inside the agent loop (see agent_loop).
        let mut all_tools = all_tools;

        let mut registered_specs: Vec<crate::openapi::RegisteredSpec> = Vec::new();
        for sp in &ctx.openapi_specs {
            if let Ok(tools) = crate::openapi::parse_spec(&sp.title, &sp.base_url, &sp.spec_json) {
                let extra: Vec<ToolSchema> = tools.iter()
                    .filter(|t| !use_step_tools || step_tool_names.contains(&t.name))
                    .filter_map(|t| serde_json::from_value::<ToolSchema>(t.schema.clone()).ok())
                    .collect();
                all_tools.extend(extra);
                registered_specs.push(crate::openapi::RegisteredSpec {
                    id:       sp.id.clone(),
                    title:    sp.title.clone(),
                    base_url: sp.base_url.clone(),
                    auth:     sp.auth.clone(),
                    tools,
                    response_exclude: sp.response_exclude.clone(),
                });
            }
        }

        // Connect MCP servers on-demand — isolated, dropped (and killed) after run
        let mut temp_conns: std::collections::HashMap<String, crate::mcp::MCPConnection> = Default::default();
        for srv in &ctx.mcp_servers {
            let cfg = crate::mcp::MCPServerConfig {
                id:      srv.id.clone(),
                name:    srv.name.clone(),
                command: srv.command.clone(),
                args:    srv.args.clone(),
                env:     srv.env.clone(),
                enabled: true,
                auth:    srv.auth.clone(),
                enable_apps: false,
            };
            if let Ok(conn) = crate::mcp::MCPConnection::connect(cfg).await {
                let extra: Vec<ToolSchema> = conn.tools.iter()
                    .filter(|t| !ctx.disabled_mcp_tools.contains(&t.name))
                    .filter(|t| !use_step_tools || step_tool_names.contains(&t.name))
                    .filter_map(|t| serde_json::from_value::<ToolSchema>(t.schema.clone()).ok())
                    .collect();
                all_tools.extend(extra);
                temp_conns.insert(srv.id.clone(), conn);
            }
        }

        let temp_mcp = tokio::sync::Mutex::new(temp_conns);

        let job_groups = job_tool_groups(all_tools);
        let r = agent_loop(
            &backend, &job.model, &system_prompt, &[], &job_groups, 0,
            None, None, &conversation,
            registered_specs, Vec::new(), &temp_mcp, ctx.allowed_dirs.clone(),
            Vec::new(), // no attached-file sandbox paths in jobs
            10, 0, app, true, job.allow_code_exec, false, 25, 0, // web_search=10, tool_result_limit=default; 25 steps; web_tool_cap=default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), // jobs aren't user-cancellable
            false, // discover_tools: jobs keep the deterministic LLM pre-flight
            Vec::new(), // skills: interactive only
        ).await;
        // temp_mcp drops here → MCPConnections drop → kill_on_drop kills stdio processes
        r

    } else {
        // ── Fallback: use active AppState (global jobs / backward compat) ──────
        let specs        = state.openapi_specs.lock().unwrap().clone();
        let allowed_dirs = state.allowed_dirs.lock().unwrap().clone();

        // Per-step tool selection now happens inside the agent loop (see agent_loop).
        let mut all_tools = all_tools;

        let extra: Vec<ToolSchema> = specs.iter()
            .flat_map(|s| s.tools.iter().filter_map(|t| serde_json::from_value::<ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(extra);

        let mcp_extra: Vec<ToolSchema> = state.mcp_connections.lock().await.values()
            .flat_map(|c| c.tools.iter().filter_map(|t| serde_json::from_value::<ToolSchema>(t.schema.clone()).ok()))
            .collect();
        all_tools.extend(mcp_extra);

        let job_groups = job_tool_groups(all_tools);
        agent_loop(
            &backend, &job.model, &system_prompt, &[], &job_groups, 0,
            None, None, &conversation,
            specs, Vec::new(), &state.mcp_connections, allowed_dirs,
            Vec::new(), // no attached-file sandbox paths in jobs
            10, 0, app, true, job.allow_code_exec, false, 25, 0, // web_search=10, tool_result_limit=default; 25 steps; web_tool_cap=default
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)), // jobs aren't user-cancellable
            false, // discover_tools: jobs keep the deterministic LLM pre-flight
            Vec::new(), // skills: interactive only
        ).await
    };

    let conv = conversation.lock().unwrap().clone();
    let trace = build_trace(&conv);
    let output = conv.iter().rev()
        .find(|m| m.role == "assistant" && m.content.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false))
        .and_then(|m| m.content.clone())
        .unwrap_or_default();

    if let Some(ref path) = job.output_file {
        if !output.is_empty() {
            let _ = crate::tools::save_document(path, &output);
        }
    }

    (started_at, output, trace, result.err().map(|e| e.to_string()))
}

// ── Background scheduler ───────────────────────────────────────────────────────

pub fn spawn_scheduler(app: tauri::AppHandle) {
    tauri::async_runtime::spawn(async move {
        // 30s tick — catches every target-minute window even near boundaries
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(30));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Jobs currently executing. `last_run_at` is only written once a run finishes,
        // and a catch-up slot stays due until then, so without this guard any job that
        // outlives one tick would be started again every 30s.
        let in_flight: InFlight = Default::default();

        loop {
            interval.tick().await;
            let now = Utc::now();

            // Read jobs from in-memory state (always consistent with what was saved)
            let state = app.state::<crate::AppState>();
            let jobs: Vec<ScheduledJob> = state.jobs.lock().unwrap().clone();
            drop(state);

            let mut spawned = 0u64;
            for job in jobs.iter().filter(|j| j.enabled) {
                if !is_due(&job.schedule, job.last_run_at, job.created_at, job.catch_up, now) {
                    continue;
                }
                // Claim it for this run, or leave it alone if the last one is still going.
                if !in_flight.lock().unwrap().insert(job.id.clone()) {
                    continue;
                }
                // Waking from sleep can leave several catch-up jobs outstanding at once;
                // stagger them so they don't all hit the model server together.
                let delay = std::time::Duration::from_secs(STAGGER_SECS * spawned);
                spawned += 1;

                let job = job.clone();
                let app = app.clone();
                let in_flight = in_flight.clone();
                tauri::async_runtime::spawn(async move {
                    let _claim = InFlightGuard { set: in_flight, id: job.id.clone() };
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    let state: tauri::State<crate::AppState> = app.state();
                    let (started_at, output, trace, error) = execute_job(&job, &state, &app).await;
                    let finished_at = Utc::now();
                    let duration_ms = (finished_at - started_at).num_milliseconds().max(0) as u64;
                    let run = JobRun {
                        id:          crate::uuid_v4(),
                        job_id:      job.id.clone(),
                        job_name:    job.name.clone(),
                        profile_name: job.profile_name.clone(),
                        started_at,
                        finished_at,
                        duration_ms,
                        status:      if error.is_none() { RunStatus::Success } else { RunStatus::Error },
                        output,
                        error,
                        trace,
                        _ran_at_legacy: None,
                    };
                    let _ = append_run(run.clone());

                    // Update last_run_at in both memory and disk
                    {
                        let mut stored = state.jobs.lock().unwrap();
                        if let Some(j) = stored.iter_mut().find(|j| j.id == job.id) {
                            j.last_run_at = Some(Utc::now());
                        }
                        let list = stored.clone();
                        drop(stored);
                        let _ = save_jobs(&list);
                    }

                    use tauri::Emitter;
                    let _ = app.emit("job-run-done", &run);
                    crate::update_tray_tooltip(&app);
                });
            }
        }
    });
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {

    /// Switching a job's schedule type in the UI used to MERGE the new type over the old fields,
    /// producing e.g. Weekly with a leftover `hours` and no `weekday`. That fails to deserialize,
    /// so save_job rejected the whole job and the Save button looked dead. Guard both halves.
    #[test]
    fn schedule_type_switch_must_carry_its_own_fields() {
        // What the merging UI produced when going Interval -> Weekly: no weekday/hour/minute.
        let broken = r#"{"type":"Weekly","hours":336}"#;
        assert!(serde_json::from_str::<JobSchedule>(broken).is_err(),
            "a Weekly schedule without weekday must be rejected, not silently accepted");

        // What the fixed UI sends: a complete variant.
        let fixed = r#"{"type":"Weekly","weekday":5,"hour":7,"minute":0}"#;
        assert_eq!(
            serde_json::from_str::<JobSchedule>(fixed).unwrap(),
            JobSchedule::Weekly { weekday: 5, hour: 7, minute: 0 },
        );

        // Daily and Interval likewise.
        assert!(serde_json::from_str::<JobSchedule>(r#"{"type":"Daily"}"#).is_err());
        assert_eq!(
            serde_json::from_str::<JobSchedule>(r#"{"type":"Interval","hours":336}"#).unwrap(),
            JobSchedule::Interval { hours: 336 },
        );
    }

    use super::*;
    use chrono::{Duration, Local, Datelike, Timelike};

    /// Most cases don't care about the creation-time guard, so default it to the
    /// distant past and leave catch-up on.
    fn due(sched: &JobSchedule, last_run: Option<DateTime<Utc>>, now: DateTime<Utc>) -> bool {
        is_due(sched, last_run, now - Duration::days(60), true, now)
    }

    /// A Daily schedule whose slot sits `mins` minutes before `now`, in local time.
    fn daily_slot_mins_ago(now: DateTime<Utc>, mins: i64) -> JobSchedule {
        let slot = now.with_timezone(&Local) - Duration::minutes(mins);
        JobSchedule::Daily { hour: slot.hour(), minute: slot.minute() }
    }

    #[test]
    fn manual_is_never_due() {
        assert!(!due(&JobSchedule::Manual, None, Utc::now()));
        assert!(!due(&JobSchedule::Manual, Some(Utc::now()), Utc::now()));
    }

    #[test]
    fn interval_due_when_never_run() {
        assert!(due(&JobSchedule::Interval { hours: 6 }, None, Utc::now()));
    }

    #[test]
    fn interval_due_after_enough_time() {
        let now = Utc::now();
        assert!(due(&JobSchedule::Interval { hours: 6 }, Some(now - Duration::hours(7)), now));
    }

    #[test]
    fn interval_not_due_too_soon() {
        let now = Utc::now();
        assert!(!due(&JobSchedule::Interval { hours: 6 }, Some(now - Duration::minutes(30)), now));
    }

    // Daily/Weekly compare against local wall-clock; derive the schedule from `now`
    // so the assertions hold regardless of the machine's timezone.
    #[test]
    fn daily_due_at_matching_minute_then_not_after_running() {
        let now = Utc::now();
        let local = now.with_timezone(&Local);
        let sched = JobSchedule::Daily { hour: local.hour(), minute: local.minute() };
        assert!(due(&sched, None, now), "should be due at the matching minute");
        assert!(!due(&sched, Some(now), now), "should not re-run after running today");
    }

    /// The bug this catch-up work exists for: asleep at 07:00, opened at 07:30.
    #[test]
    fn daily_catches_up_on_a_slot_missed_while_asleep() {
        let now = Utc::now();
        let sched = daily_slot_mins_ago(now, 30);
        assert!(due(&sched, Some(now - Duration::hours(24)), now),
                "a slot that passed 30 minutes ago should still run");
    }

    #[test]
    fn daily_not_due_again_once_that_slot_has_run() {
        let now = Utc::now();
        let sched = daily_slot_mins_ago(now, 30);
        assert!(!due(&sched, Some(now - Duration::minutes(10)), now),
                "already ran after this slot");
    }

    #[test]
    fn daily_not_due_when_the_next_slot_is_still_ahead() {
        let now = Utc::now();
        let slot = now.with_timezone(&Local) + Duration::hours(1);
        let sched = JobSchedule::Daily { hour: slot.hour(), minute: slot.minute() };
        assert!(!due(&sched, Some(now - Duration::minutes(5)), now));
    }

    /// A job created at 3pm with a 07:00 slot must not treat this morning as missed.
    #[test]
    fn daily_created_after_todays_slot_does_not_fire_immediately() {
        let now = Utc::now();
        let sched = daily_slot_mins_ago(now, 30);
        assert!(!is_due(&sched, None, now - Duration::minutes(10), true, now));
    }

    /// Three weeks away yields ONE outstanding run, not one per missed day.
    #[test]
    fn long_absence_produces_a_single_catch_up_run() {
        let now = Utc::now();
        let created = now - Duration::days(60);
        let sched = daily_slot_mins_ago(now, 30);
        assert!(is_due(&sched, Some(now - Duration::days(21)), created, true, now));
        // ...and once that run lands, nothing further is outstanding.
        assert!(!is_due(&sched, Some(now), created, true, now));
    }

    #[test]
    fn catch_up_disabled_skips_a_missed_slot_but_still_runs_on_time() {
        let now = Utc::now();
        let created = now - Duration::days(60);
        let missed = daily_slot_mins_ago(now, 30);
        assert!(!is_due(&missed, Some(now - Duration::hours(24)), created, false, now),
                "30 minutes late is outside the grace window");
        let fresh = daily_slot_mins_ago(now, 0);
        assert!(is_due(&fresh, Some(now - Duration::hours(24)), created, false, now),
                "the current slot is still within the grace window");
    }

    #[test]
    fn weekly_due_on_matching_weekday_and_time() {
        let now = Utc::now();
        let local = now.with_timezone(&Local);
        let sched = JobSchedule::Weekly {
            weekday: local.weekday().num_days_from_monday(),
            hour: local.hour(),
            minute: local.minute(),
        };
        assert!(due(&sched, None, now));
        assert!(!due(&sched, Some(now), now));
    }

    /// A missed Monday runs on Wednesday rather than waiting a further week.
    #[test]
    fn weekly_catches_up_later_in_the_week() {
        let now = Utc::now();
        let slot = now.with_timezone(&Local) - Duration::hours(25);
        let sched = JobSchedule::Weekly {
            weekday: slot.weekday().num_days_from_monday(),
            hour: slot.hour(),
            minute: slot.minute(),
        };
        assert!(due(&sched, Some(now - Duration::days(8)), now));
    }

    #[test]
    fn compile_steps_numbers_sections() {
        let steps = vec![
            JobStep { id: "1".into(), step_type: "text".into(),
                      instruction: Some("First thing".into()), tool_name: None, tool_hint: None },
            JobStep { id: "2".into(), step_type: "tool".into(),
                      instruction: Some("Second thing".into()),
                      tool_name: Some("read_file".into()), tool_hint: Some("path=/tmp/x".into()) },
        ];
        let out = compile_steps(&steps);
        assert!(out.contains("STEP 1"));
        assert!(out.contains("First thing"));
        assert!(out.contains("STEP 2"));
        assert!(out.contains("Call tool: read_file"));
        assert!(out.contains("path=/tmp/x"));
    }

    #[test]
    fn compile_steps_empty_is_empty() {
        assert_eq!(compile_steps(&[]), "");
    }
}
