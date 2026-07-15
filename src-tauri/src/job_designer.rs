//! AI job designer — turns a plain-English goal into a draft `ScheduledJob`.
//!
//! This is deliberately a *generator*, not an actuator: it returns a job the user reviews
//! and saves through the normal Job form. It never persists or enables anything itself.
//! Drafts come back `enabled: false` so even after saving, a background job stays inert
//! until the user turns it on.

use serde::{Deserialize, Serialize};

use crate::jobs::{JobSchedule, JobStep, ScheduledJob};

/// A tool the drafted job is allowed to reference. Mirrors the frontend job tool catalog,
/// passed in so validation uses the same source of truth (jobs can't call wiki/python tools).
#[derive(Debug, Clone, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct DraftJobArgs {
    pub goal: String,
    pub model: String,
    #[serde(default)]
    pub allowed_dirs: Vec<String>,
    #[serde(default)]
    pub tool_catalog: Vec<ToolInfo>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub profile_name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DraftedJob {
    pub job: ScheduledJob,
    /// Non-fatal issues for the user to eyeball before saving.
    pub warnings: Vec<String>,
}

// ── What we ask the model to emit ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct DraftStep {
    step_type: String,
    #[serde(default)]
    instruction: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_hint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DraftSpec {
    name: String,
    schedule: JobSchedule,
    #[serde(default)]
    output_file: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    steps: Vec<DraftStep>,
}

const DESIGNER_SYSTEM: &str = r#"You are a job designer for LexiChat, a local AI assistant. Turn the user's goal into a scheduled job that an agent will run unattended, step by step.

Reply with ONLY a JSON object, no prose, no code fence, in exactly this shape:
{
  "name": "short job name",
  "schedule": {"type":"Daily","hour":7,"minute":0},
  "output_file": "/absolute/path/with/extension" or null,
  "system_prompt": "role/voice/judgement instructions for the agent, or null",
  "steps": [
    {"step_type":"tool","instruction":"what this step achieves","tool_name":"web_search","tool_hint":"argument guidance e.g. the query"},
    {"step_type":"text","instruction":"a reasoning/composition step with no tool"}
  ]
}

schedule.type is one of: "Daily" (needs hour, minute), "Weekly" (needs weekday 0=Mon..6=Sun, hour, minute), "Interval" (needs hours), "Manual".

HARD RULES — follow every one:
1. tool_name MUST be one of the ALLOWED TOOLS listed below. Never invent a tool. A step needing no tool is step_type "text" with no tool_name.
2. The LAST step MUST be step_type "text" that composes the final deliverable — the job saves the final text message to output_file, so the deliverable must be produced last.
3. To avoid repeating work across runs (deduplication), read a memory file near the start with read_file, and near the end (second-to-last step) rewrite it with write_file as its previous contents plus a new dated section. There is no append tool, so write_file must include the prior contents.
4. Any path in output_file or in a read_file/write_file hint MUST sit inside one of the ALLOWED DIRECTORIES listed below. If none are provided, still write a sensible absolute path but keep it under the user's home.
5. output_file extension picks the format: .pdf and .docx render documents, anything else is plain text.
6. Keep it to the fewest steps that achieve the goal. Prefer concrete tool_hints (exact search queries, exact file paths)."#;

/// Pull the outermost `{ ... }` object out of a possibly-fenced/prosey reply.
fn extract_json_object(s: &str) -> &str {
    let start = s.find('{');
    let end = s.rfind('}');
    match (start, end) {
        (Some(a), Some(b)) if b >= a => &s[a..=b],
        _ => s,
    }
}

fn path_in_allowed(path: &str, allowed: &[String]) -> bool {
    allowed.iter().any(|d| {
        let d = d.trim_end_matches('/');
        path == d || path.starts_with(&format!("{d}/"))
    })
}

pub async fn draft_job(host: &str, args: DraftJobArgs) -> Result<DraftedJob, String> {
    let tool_lines = args.tool_catalog.iter()
        .map(|t| format!("- {}: {}", t.name, t.description))
        .collect::<Vec<_>>()
        .join("\n");
    let dir_lines = if args.allowed_dirs.is_empty() {
        "(none configured — the user must add a folder in Settings → Sandbox before file steps will work)".to_string()
    } else {
        args.allowed_dirs.iter().map(|d| format!("- {d}")).collect::<Vec<_>>().join("\n")
    };

    let user_msg = format!(
        "GOAL:\n{}\n\nALLOWED TOOLS:\n{}\n\nALLOWED DIRECTORIES:\n{}",
        args.goal.trim(), tool_lines, dir_lines,
    );

    let raw = crate::ollama::complete(host, &args.model, DESIGNER_SYSTEM, &user_msg)
        .await
        .map_err(|e| format!("The model could not draft a job: {e}"))?;

    let json = extract_json_object(&raw);
    let spec: DraftSpec = serde_json::from_str(json)
        .map_err(|e| format!("The model's reply was not a valid job spec ({e}). Try rephrasing the goal."))?;

    assemble(spec, &args)
}

/// Validate a parsed spec and turn it into a reviewable, disabled job. Pure (no I/O) so the
/// guardrails — unknown-tool rejection, sandbox-path warnings, deliverable-last, never-enabled
/// — can be tested deterministically without an LLM.
fn assemble(spec: DraftSpec, args: &DraftJobArgs) -> Result<DraftedJob, String> {
    let mut warnings: Vec<String> = Vec::new();
    let tool_names: std::collections::HashSet<&str> =
        args.tool_catalog.iter().map(|t| t.name.as_str()).collect();

    // Build validated steps.
    let mut steps: Vec<JobStep> = Vec::new();
    for (i, ds) in spec.steps.iter().enumerate() {
        let mut step_type = if ds.step_type == "tool" { "tool" } else { "text" }.to_string();
        let mut tool_name = ds.tool_name.clone();

        if step_type == "tool" {
            match &tool_name {
                Some(name) if tool_names.contains(name.as_str()) => {}
                Some(name) => {
                    warnings.push(format!(
                        "Step {} referenced unknown tool \"{}\" — converted to a plain instruction. Pick a tool in the form.",
                        i + 1, name
                    ));
                    step_type = "text".into();
                    tool_name = None;
                }
                None => { step_type = "text".into(); }
            }
        }

        // Flag file-path steps that fall outside the sandbox.
        if let Some(hint) = &ds.tool_hint {
            if matches!(tool_name.as_deref(), Some("read_file") | Some("write_file")) {
                if let Some(p) = hint.split_whitespace().find(|w| w.starts_with('/')) {
                    if !path_in_allowed(p, &args.allowed_dirs) {
                        warnings.push(format!(
                            "Step {} reads/writes \"{}\", which is outside your sandbox — add its folder in Settings → Sandbox or the step will fail.",
                            i + 1, p
                        ));
                    }
                }
            }
        }

        steps.push(JobStep {
            id: crate::uuid_v4(),
            step_type,
            instruction: ds.instruction.clone(),
            tool_name,
            tool_hint: ds.tool_hint.clone(),
        });
    }

    if steps.is_empty() {
        return Err("The model produced a job with no steps. Try a more specific goal.".into());
    }
    if steps.last().map(|s| s.step_type.as_str()) == Some("tool") {
        warnings.push(
            "The last step calls a tool. The job saves the final text message, so add a closing text step that composes the deliverable.".into(),
        );
    }

    // output_file lives outside the sandbox check (it's written directly), but warn if the
    // extension won't render as the user probably expects.
    if let Some(of) = &spec.output_file {
        let lower = of.to_lowercase();
        if !(lower.ends_with(".pdf") || lower.ends_with(".docx")
            || lower.ends_with(".txt") || lower.ends_with(".md")) {
            warnings.push(format!(
                "Output file \"{of}\" has an unusual extension — .pdf/.docx render documents, anything else is saved as plain text.",
            ));
        }
    }

    // enabled_builtin_tools = the builtin tools the steps actually use.
    let enabled_builtin_tools: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        steps.iter()
            .filter_map(|s| s.tool_name.clone())
            .filter(|n| tool_names.contains(n.as_str()))
            .filter(|n| seen.insert(n.clone()))
            .collect()
    };

    let job = ScheduledJob {
        id: crate::uuid_v4(),
        name: if spec.name.trim().is_empty() { "New AI-designed job".into() } else { spec.name },
        enabled: false, // never auto-enable an unattended job
        schedule: spec.schedule,
        prompt: String::new(),
        model: args.model.clone(),
        system_prompt: spec.system_prompt.filter(|s| !s.trim().is_empty()),
        enabled_builtin_tools,
        output_file: spec.output_file.filter(|s| !s.trim().is_empty()),
        created_at: chrono::Utc::now(),
        last_run_at: None,
        profile_id: args.profile_id.clone(),
        profile_name: args.profile_name.clone(),
        profile_context: None,
        steps,
    };

    Ok(DraftedJob { job, warnings })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> DraftJobArgs {
        DraftJobArgs {
            goal: "daily brief".into(),
            model: "qwen3.6:latest".into(),
            allowed_dirs: vec!["/Users/dio/Briefings".into()],
            tool_catalog: ["web_search", "read_file", "write_file", "get_current_datetime"]
                .iter().map(|n| ToolInfo { name: (*n).into(), description: String::new() }).collect(),
            profile_id: None,
            profile_name: None,
        }
    }

    fn spec(json: &str) -> DraftSpec {
        serde_json::from_str(json).expect("valid spec json")
    }

    #[test]
    fn extract_object_survives_fences_and_prose() {
        let s = "Sure!\n```json\n{\"a\":1}\n```\nhope that helps";
        assert_eq!(extract_json_object(s), "{\"a\":1}");
    }

    #[test]
    fn drafts_are_never_enabled_and_derive_their_tools() {
        let d = assemble(spec(r#"{
            "name":"Morning Brief","schedule":{"type":"Daily","hour":7,"minute":0},
            "output_file":"/Users/dio/Briefings/brief.pdf","system_prompt":"be terse",
            "steps":[
              {"step_type":"tool","instruction":"date","tool_name":"get_current_datetime"},
              {"step_type":"tool","instruction":"search","tool_name":"web_search","tool_hint":"UK energy"},
              {"step_type":"text","instruction":"write the brief"}
            ]}"#), &args()).unwrap();
        assert!(!d.job.enabled, "a drafted background job must start disabled");
        assert_eq!(d.job.enabled_builtin_tools, vec!["get_current_datetime", "web_search"]);
        assert!(d.warnings.is_empty(), "clean spec should have no warnings: {:?}", d.warnings);
    }

    #[test]
    fn unknown_tool_is_downgraded_not_kept() {
        // wiki_write isn't in the job catalog — the exact footgun this guards.
        let d = assemble(spec(r#"{
            "name":"x","schedule":{"type":"Manual"},
            "steps":[
              {"step_type":"tool","instruction":"save","tool_name":"wiki_write","tool_hint":"p"},
              {"step_type":"text","instruction":"done"}
            ]}"#), &args()).unwrap();
        assert_eq!(d.job.steps[0].step_type, "text");
        assert!(d.job.steps[0].tool_name.is_none());
        assert!(d.warnings.iter().any(|w| w.contains("wiki_write")));
    }

    #[test]
    fn warns_on_path_outside_sandbox_and_tool_last() {
        let d = assemble(spec(r#"{
            "name":"x","schedule":{"type":"Daily","hour":8,"minute":0},
            "steps":[
              {"step_type":"tool","instruction":"read","tool_name":"read_file","tool_hint":"path /etc/secret.md"},
              {"step_type":"tool","instruction":"search","tool_name":"web_search","tool_hint":"q"}
            ]}"#), &args()).unwrap();
        assert!(d.warnings.iter().any(|w| w.contains("outside your sandbox")));
        assert!(d.warnings.iter().any(|w| w.contains("final text message")));
    }

    #[test]
    fn path_inside_sandbox_is_not_flagged() {
        let d = assemble(spec(r#"{
            "name":"x","schedule":{"type":"Daily","hour":8,"minute":0},
            "steps":[
              {"step_type":"tool","instruction":"read","tool_name":"read_file","tool_hint":"/Users/dio/Briefings/covered.md"},
              {"step_type":"text","instruction":"done"}
            ]}"#), &args()).unwrap();
        assert!(!d.warnings.iter().any(|w| w.contains("outside your sandbox")), "{:?}", d.warnings);
    }

    // Live end-to-end against local Ollama. Ignored by default (needs a running model);
    // run with:  cargo test designer_live -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn designer_live_drafts_a_runnable_job() {
        let a = args();
        let goal = "Every weekday at 7am, search the web for UK energy policy news, skip anything \
                    already covered by reading and rewriting /Users/dio/Briefings/covered.md, and \
                    save a one-page PDF brief to /Users/dio/Briefings/brief.pdf.";
        let d = draft_job("http://localhost:11434", DraftJobArgs { goal: goal.into(), ..a })
            .await.expect("draft");
        eprintln!("name={:?} schedule={:?} out={:?}", d.job.name, d.job.schedule, d.job.output_file);
        for s in &d.job.steps { eprintln!("  [{}] {:?} tool={:?}", s.step_type, s.instruction, s.tool_name); }
        eprintln!("warnings={:?}", d.warnings);
        assert!(!d.job.enabled);
        assert!(!d.job.steps.is_empty());
    }
}
