//! Sandboxed Python execution via Pydantic's Monty interpreter.
//!
//! The `run_python` tool lets the LLM execute Python to compute answers. Monty is
//! a from-scratch Python-subset interpreter written in Rust with a default-deny
//! security model: no filesystem, network, or environment access unless the host
//! explicitly grants it. We grant nothing automatically.
//!
//! File access is offered through three host functions — `read_file`,
//! `write_file`, `list_files` — that route every path through the same
//! [`crate::tools::check_path`] sandbox enforcement used by the built-in file
//! tools. Native Python file/OS I/O (`open()`, `pathlib`, env vars) is denied,
//! so every path the sandbox touches goes through that single audited choke point.
//!
//! Resource limits (wall-clock time, heap memory) are enforced by Monty's
//! `LimitedTracker`, so runaway code dies deterministically.

use std::time::Duration;

use monty::{
    ExcType, ExtFunctionResult, LimitedTracker, MontyException, MontyObject, MontyRun,
    NameLookupResult, PrintWriter, ResourceLimits, RunProgress,
};

use crate::tools::check_path;

/// Maximum wall-clock execution time per run.
const MAX_DURATION: Duration = Duration::from_secs(10);
/// Approximate maximum heap memory (256 MB).
const MAX_MEMORY: usize = 256 * 1024 * 1024;
/// Cap on the result string returned to the model (matches the agent loop's cap).
const MAX_OUTPUT: usize = 6000;

/// Execute `code` in the Monty sandbox.
///
/// `allowed` is the set of paths the sandbox may read/write — the run's allowed
/// directories plus any files the user attached to the message. Monty's
/// interpreter is synchronous, so we run it on a blocking thread to avoid
/// stalling the async runtime.
pub async fn run_python(code: String, allowed: Vec<String>) -> String {
    match tokio::task::spawn_blocking(move || run_blocking(&code, &allowed)).await {
        Ok(s) => s,
        Err(e) => format!("Error: sandbox task failed: {e}"),
    }
}

fn run_blocking(code: &str, allowed: &[String]) -> String {
    let runner = match MontyRun::new(code.to_owned(), "sandbox.py", vec![]) {
        Ok(r) => r,
        // Parse/compile errors surface here (e.g. SyntaxError, unsupported syntax).
        Err(e) => return format!("Error: {e}"),
    };

    let limits = ResourceLimits::new()
        .max_duration(MAX_DURATION)
        .max_memory(MAX_MEMORY);
    let tracker = LimitedTracker::new(limits);

    // Collect everything printed by the code, in order.
    let mut out = String::new();

    let mut progress = match runner.start(vec![], tracker, PrintWriter::CollectString(&mut out)) {
        Ok(p) => p,
        Err(e) => return finalize(out, Err(e)),
    };

    // Drive the start/resume loop: whenever the VM pauses for an external call we
    // service it and resume, until the program completes or errors.
    loop {
        progress = match progress {
            RunProgress::Complete(value) => return finalize(out, Ok(value)),

            // A call to one of our host functions (read_file / write_file / list_files).
            RunProgress::FunctionCall(call) => {
                let result = dispatch_host_fn(&call.function_name, &call.args, &call.kwargs, allowed);
                match call.resume(result, PrintWriter::CollectString(&mut out)) {
                    Ok(p) => p,
                    Err(e) => return finalize(out, Err(e)),
                }
            }

            // Native file/OS I/O (open(), pathlib, os.*) — denied. We force the
            // model onto the host functions so all access goes via check_path.
            RunProgress::OsCall(os) => {
                let exc = MontyException::new(
                    ExcType::RuntimeError,
                    Some(
                        "Direct file/OS access is disabled in this sandbox. Use \
                         read_file(path), write_file(path, content), or list_files(dir)."
                            .into(),
                    ),
                );
                match os.resume(ExtFunctionResult::Error(exc), PrintWriter::CollectString(&mut out)) {
                    Ok(p) => p,
                    Err(e) => return finalize(out, Err(e)),
                }
            }

            // An undefined name referenced as a value → raise NameError, the
            // normal Python behaviour.
            RunProgress::NameLookup(nl) => {
                match nl.resume(NameLookupResult::Undefined, PrintWriter::CollectString(&mut out)) {
                    Ok(p) => p,
                    Err(e) => return finalize(out, Err(e)),
                }
            }

            // No async host functions are registered, so this should not occur.
            RunProgress::ResolveFutures(_) => {
                return finalize_msg(out, "async/await is not supported in the sandbox.");
            }
        };
    }
}

/// Route a host-function call to the right handler, enforcing the sandbox.
fn dispatch_host_fn(
    name: &str,
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    allowed: &[String],
) -> ExtFunctionResult {
    match name {
        "read_file" => {
            let Some(path) = arg_str(args, kwargs, 0, "path") else {
                return runtime_err("read_file(path) requires a string path".into());
            };
            match guarded_read(&path, allowed) {
                Ok(content) => ExtFunctionResult::Return(MontyObject::String(content)),
                Err(e) => runtime_err(e),
            }
        }
        "write_file" => {
            let Some(path) = arg_str(args, kwargs, 0, "path") else {
                return runtime_err("write_file(path, content) requires a string path".into());
            };
            let Some(content) = arg_str(args, kwargs, 1, "content") else {
                return runtime_err("write_file(path, content) requires string content".into());
            };
            match guarded_write(&path, &content, allowed) {
                Ok(n) => ExtFunctionResult::Return(MontyObject::Int(n as i64)),
                Err(e) => runtime_err(e),
            }
        }
        "list_files" => {
            let path = arg_str(args, kwargs, 0, "path").unwrap_or_else(|| ".".into());
            match guarded_list(&path, allowed) {
                Ok(items) => ExtFunctionResult::Return(MontyObject::List(
                    items.into_iter().map(MontyObject::String).collect(),
                )),
                Err(e) => runtime_err(e),
            }
        }
        // Any other undefined function → NameError.
        other => ExtFunctionResult::NotFound(other.to_string()),
    }
}

// ── Guarded filesystem operations (all go through check_path) ──────────────────

fn guarded_read(path: &str, allowed: &[String]) -> Result<String, String> {
    check_path(path, allowed)?;
    std::fs::read_to_string(path).map_err(|e| format!("could not read '{path}': {e}"))
}

fn guarded_write(path: &str, content: &str, allowed: &[String]) -> Result<usize, String> {
    check_path(path, allowed)?;
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(path, content).map_err(|e| format!("could not write '{path}': {e}"))?;
    Ok(content.len())
}

fn guarded_list(path: &str, allowed: &[String]) -> Result<Vec<String>, String> {
    check_path(path, allowed)?;
    let entries = std::fs::read_dir(path).map_err(|e| format!("could not list '{path}': {e}"))?;
    let mut items: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            if is_dir { format!("{name}/") } else { name }
        })
        .collect();
    items.sort();
    Ok(items)
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Extract a string argument by positional index, falling back to a keyword.
fn arg_str(
    args: &[MontyObject],
    kwargs: &[(MontyObject, MontyObject)],
    idx: usize,
    key: &str,
) -> Option<String> {
    if let Some(MontyObject::String(s)) = args.get(idx) {
        return Some(s.clone());
    }
    for (k, v) in kwargs {
        if let (MontyObject::String(k), MontyObject::String(v)) = (k, v) {
            if k == key {
                return Some(v.clone());
            }
        }
    }
    None
}

fn runtime_err(msg: String) -> ExtFunctionResult {
    ExtFunctionResult::Error(MontyException::new(ExcType::RuntimeError, Some(msg)))
}

/// Build the final tool result: captured stdout, plus the final expression value
/// when it is meaningful (not `None`), or a runtime error with any prior output.
fn finalize(out: String, result: Result<MontyObject, MontyException>) -> String {
    let mut s = out.trim_end().to_string();
    match result {
        Ok(value) => {
            if !matches!(value, MontyObject::None) {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&value.to_string());
            }
            if s.is_empty() {
                return "(code ran successfully with no output)".into();
            }
        }
        Err(e) => {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(&format!("Error: {e}"));
        }
    }
    truncate(s)
}

fn finalize_msg(out: String, msg: &str) -> String {
    let mut s = out.trim_end().to_string();
    if !s.is_empty() {
        s.push('\n');
    }
    s.push_str("Error: ");
    s.push_str(msg);
    truncate(s)
}

/// Truncate to MAX_OUTPUT on a UTF-8 char boundary.
fn truncate(s: String) -> String {
    if s.len() <= MAX_OUTPUT {
        return s;
    }
    let mut end = MAX_OUTPUT;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n…[truncated: {} chars total]", &s[..end], s.len())
}

#[cfg(test)]
mod tests {
    use super::run_blocking;

    #[test]
    fn computes_and_prints() {
        assert_eq!(run_blocking("print(2 + 2)", &[]), "4");
    }

    #[test]
    fn returns_final_expression_value() {
        // No print, but the final expression value is surfaced.
        assert_eq!(run_blocking("sum(range(10))", &[]), "45");
    }

    #[test]
    fn read_file_host_fn_reads_allowed_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.txt");
        std::fs::write(&path, "hello sandbox").unwrap();
        let allowed = vec![dir.path().to_string_lossy().to_string()];

        let code = format!("print(read_file({:?}))", path.to_string_lossy());
        let out = run_blocking(&code, &allowed);
        assert!(out.contains("hello sandbox"), "got: {out}");
    }

    #[test]
    fn write_file_host_fn_writes_allowed_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let allowed = vec![dir.path().to_string_lossy().to_string()];

        let code = format!("write_file({:?}, 'written by sandbox')", path.to_string_lossy());
        let out = run_blocking(&code, &allowed);
        assert!(!out.contains("Error"), "unexpected error: {out}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "written by sandbox");
    }

    #[test]
    fn denies_file_outside_allowed_dirs() {
        let allowed = vec!["/nonexistent/allowed/only".to_string()];
        let out = run_blocking("print(read_file('/etc/hosts'))", &allowed);
        assert!(out.contains("Access denied"), "got: {out}");
    }

    #[test]
    fn native_open_is_disabled() {
        let out = run_blocking("open('/etc/hosts')", &[]);
        assert!(out.contains("disabled"), "got: {out}");
    }

    #[test]
    fn runtime_error_is_surfaced() {
        let out = run_blocking("1 / 0", &[]);
        assert!(out.contains("Error"), "got: {out}");
    }
}
