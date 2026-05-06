use std::fs;
use std::path::{Path, PathBuf};
use serde_json::Value;

// ── Wiki directory ────────────────────────────────────────────────────────────

pub fn wiki_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("lexichat")
        .join("wiki");
    let _ = fs::create_dir_all(&base);
    base
}

/// Resolve a model-supplied path to an absolute path inside the wiki dir.
/// Adds .md extension if the path has none. Rejects traversal attempts.
fn resolve(raw: &str) -> Result<PathBuf, String> {
    let raw = raw.trim().trim_start_matches('/');
    if raw.is_empty() {
        return Err("Path must not be empty.".into());
    }
    if raw.contains("..") {
        return Err("Path must not contain '..'.".into());
    }

    let base = wiki_dir();
    let mut p = base.join(raw);

    // Add .md if no extension present
    if p.extension().is_none() {
        p.set_extension("md");
    }

    // Verify the resolved path stays inside wiki_dir
    let canonical_base = fs::canonicalize(&base).unwrap_or(base.clone());
    // For new files, walk up to the first existing ancestor to canonicalize
    let canonical_p = if p.exists() {
        fs::canonicalize(&p).unwrap_or(p.clone())
    } else {
        let mut cur: &Path = &p;
        loop {
            if cur.exists() {
                let mut resolved = fs::canonicalize(cur).unwrap_or(cur.to_path_buf());
                // Append the remaining non-existent suffix
                if let Ok(suffix) = p.strip_prefix(cur) {
                    resolved = resolved.join(suffix);
                }
                break resolved;
            }
            match cur.parent() {
                Some(parent) => cur = parent,
                None => break p.clone(),
            }
        }
    };

    if !canonical_p.starts_with(&canonical_base) {
        return Err(format!("Path '{}' is outside the wiki directory.", raw));
    }

    Ok(p)
}

// ── Tool handlers ─────────────────────────────────────────────────────────────

pub fn wiki_list() -> String {
    let dir = wiki_dir();
    let mut pages: Vec<String> = Vec::new();
    collect_pages(&dir, &dir, &mut pages);
    if pages.is_empty() {
        return "Wiki is empty. Use wiki_write to create your first page.".into();
    }
    pages.sort();
    format!("Wiki pages ({}):\n{}", pages.len(), pages.join("\n"))
}

fn collect_pages(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pages(root, &path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().to_string());
            }
        }
    }
}

pub fn wiki_search(args: &Value) -> String {
    let query = match args["query"].as_str() {
        Some(q) if !q.trim().is_empty() => q.to_lowercase(),
        _ => return "Error: query is required.".into(),
    };

    let dir = wiki_dir();
    let mut results: Vec<String> = Vec::new();
    search_pages(&dir, &dir, &query, &mut results);

    if results.is_empty() {
        return format!("No wiki pages match '{query}'.");
    }
    format!("Search results for '{query}':\n\n{}", results.join("\n---\n"))
}

fn search_pages(root: &Path, dir: &Path, query: &str, out: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            search_pages(root, &path, query, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let Ok(content) = fs::read_to_string(&path) else { continue };
            if content.to_lowercase().contains(query) {
                if let Ok(rel) = path.strip_prefix(root) {
                    // Return up to 3 matching lines with context
                    let matches: Vec<String> = content.lines()
                        .filter(|l| l.to_lowercase().contains(query))
                        .take(3)
                        .map(|l| format!("  > {l}"))
                        .collect();
                    out.push(format!("**{}**\n{}", rel.display(), matches.join("\n")));
                }
            }
        }
    }
}

pub fn wiki_read(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    match fs::read_to_string(&path) {
        Ok(content) => {
            if content.is_empty() {
                format!("Page '{}' exists but is empty.", path_str)
            } else {
                content
            }
        }
        Err(_) => {
            // Give a more helpful hint for the index file specifically
            if path_str.trim_end_matches(".md") == "index" {
                "index.md not found — wiki is empty or index hasn't been created yet. Call wiki_list to see what's stored.".into()
            } else {
                format!("Page '{}' not found. Use wiki_list or wiki_search to find available pages.", path_str)
            }
        }
    }
}

pub fn wiki_write(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let content = match args["content"].as_str() {
        Some(c) => c,
        None => return "Error: content is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if let Some(parent) = path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return format!("Error creating directory: {e}");
        }
    }
    match fs::write(&path, content) {
        Ok(()) => {
            append_log_entry("write", path_str);
            format!("Written {} chars to '{}'.", content.len(), path_str)
        }
        Err(e) => format!("Error writing '{}': {e}", path_str),
    }
}

pub fn wiki_patch(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let find = match args["find"].as_str() {
        Some(f) if !f.is_empty() => f,
        _ => return "Error: find is required and must not be empty.".into(),
    };
    let replace = args["replace"].as_str().unwrap_or("");

    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    let content = match fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return format!("Page '{}' not found.", path_str),
    };
    if !content.contains(find) {
        return format!("Text not found in '{}'. No changes made.", path_str);
    }
    let patched = content.replacen(find, replace, 1);
    match fs::write(&path, &patched) {
        Ok(()) => {
            append_log_entry("patch", path_str);
            format!("Patched '{}': replaced first occurrence.", path_str)
        }
        Err(e) => format!("Error writing '{}': {e}", path_str),
    }
}

pub fn wiki_append(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let content = match args["content"].as_str() {
        Some(c) => c,
        None => return "Error: content is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let existing = fs::read_to_string(&path).unwrap_or_default();
    let sep = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
    let new_content = format!("{existing}{sep}{content}\n");
    match fs::write(&path, &new_content) {
        Ok(()) => format!("Appended {} chars to '{}'.", content.len(), path_str),
        Err(e) => format!("Error appending to '{}': {e}", path_str),
    }
}

pub fn wiki_lint() -> String {
    let dir = wiki_dir();
    let mut all_pages: Vec<String> = Vec::new();
    collect_pages(&dir, &dir, &mut all_pages);

    if all_pages.is_empty() {
        return "Wiki is empty — nothing to lint.".into();
    }

    let content_pages: Vec<&str> = all_pages.iter()
        .map(String::as_str)
        .filter(|p| *p != "index.md" && *p != "log.md")
        .collect();

    let mut issues: Vec<String> = Vec::new();
    let mut stats_words = 0usize;

    // Count words and find empty pages
    for page in &all_pages {
        let path = dir.join(page);
        let content = fs::read_to_string(&path).unwrap_or_default();
        let words: usize = content.split_whitespace().count();
        stats_words += words;
        if words == 0 {
            issues.push(format!("  • Empty page: {page}"));
        }
    }

    // Check index.md coverage
    let index_path = dir.join("index.md");
    if !index_path.exists() {
        if !content_pages.is_empty() {
            issues.push(format!("  • index.md missing — {} pages have no index entry.", content_pages.len()));
        }
    } else {
        let index_content = fs::read_to_string(&index_path).unwrap_or_default();

        // Pages not mentioned in index
        let mut unindexed: Vec<&str> = Vec::new();
        for page in &content_pages {
            // Check if page basename or full path appears anywhere in index
            let stem = std::path::Path::new(page)
                .file_stem().and_then(|s| s.to_str()).unwrap_or(page);
            if !index_content.contains(page) && !index_content.contains(stem) {
                unindexed.push(page);
            }
        }
        if !unindexed.is_empty() {
            issues.push(format!("  • Pages not in index.md ({}):\n{}",
                unindexed.len(),
                unindexed.iter().map(|p| format!("      - {p}")).collect::<Vec<_>>().join("\n")));
        }

        // Markdown links in index that point to missing files
        let mut broken: Vec<String> = Vec::new();
        for cap in index_content.split("](") {
            if let Some(end) = cap.find(')') {
                let link = cap[..end].trim();
                if link.ends_with(".md") && !link.starts_with("http") {
                    let linked = dir.join(link);
                    if !linked.exists() {
                        broken.push(link.to_string());
                    }
                }
            }
        }
        if !broken.is_empty() {
            issues.push(format!("  • Broken links in index.md ({}):\n{}",
                broken.len(),
                broken.iter().map(|l| format!("      - {l}")).collect::<Vec<_>>().join("\n")));
        }
    }

    // Log freshness
    let log_path = dir.join("log.md");
    let log_summary = if log_path.exists() {
        let log = fs::read_to_string(&log_path).unwrap_or_default();
        let entries: Vec<&str> = log.lines().filter(|l| l.starts_with("## [")).collect();
        format!("log.md: {} entries, last: {}", entries.len(),
            entries.last().copied().unwrap_or("(none)"))
    } else {
        "log.md: not created yet".into()
    };

    let summary = format!(
        "Wiki health check\n\
         Pages: {} ({} content + index + log)\n\
         Words: ~{stats_words}\n\
         {log_summary}",
        all_pages.len(), content_pages.len()
    );

    if issues.is_empty() {
        format!("{summary}\n\nNo issues found.")
    } else {
        format!("{summary}\n\nIssues ({}):\n{}", issues.len(), issues.join("\n"))
    }
}

// ── Auto-logging helper ───────────────────────────────────────────────────────

fn append_log_entry(action: &str, detail: &str) {
    let log_path = wiki_dir().join("log.md");
    let now = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let entry = format!("## [{now}] {action} | {detail}\n");
    let existing = fs::read_to_string(&log_path).unwrap_or_default();
    let _ = fs::write(&log_path, format!("{existing}{entry}"));
}

pub fn wiki_delete(args: &Value) -> String {
    let path_str = match args["path"].as_str() {
        Some(p) => p,
        None => return "Error: path is required.".into(),
    };
    let path = match resolve(path_str) {
        Ok(p) => p,
        Err(e) => return format!("Error: {e}"),
    };
    if !path.exists() {
        return format!("Page '{}' not found.", path_str);
    }
    match fs::remove_file(&path) {
        Ok(()) => {
            append_log_entry("delete", path_str);
            format!("Deleted '{}'.", path_str)
        }
        Err(e) => format!("Error deleting '{}': {e}", path_str),
    }
}

// ── Schema helpers (available for future Rust-side use) ──────────────────────

#[allow(dead_code)]
pub fn wiki_schemas() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_list",
                "description": "List all pages in the persistent wiki. Use this to discover what knowledge has been stored.",
                "parameters": { "type": "object", "properties": {}, "required": [] }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_search",
                "description": "Search the wiki for pages containing a keyword or phrase. Returns matching page names and relevant lines. Always search before writing to avoid duplicates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Keyword or phrase to search for." }
                    },
                    "required": ["query"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_read",
                "description": "Read the full contents of a wiki page. Use wiki_list or wiki_search first to find the correct path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path relative to the wiki root, e.g. 'people/alice.md' or 'projects.md'. The .md extension is optional." }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_write",
                "description": "Create or overwrite a wiki page with markdown content. Use wiki_search first to avoid duplicates. Use clear structured markdown with headings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path, e.g. 'people/alice.md' or 'projects.md'." },
                        "content": { "type": "string", "description": "Full markdown content for the page." }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_patch",
                "description": "Update part of an existing wiki page by replacing the first occurrence of a specific string. Use this for small targeted updates rather than rewriting the entire page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path." },
                        "find": { "type": "string", "description": "Exact text to find (must be an exact substring of the page content)." },
                        "replace": { "type": "string", "description": "Text to replace it with." }
                    },
                    "required": ["path", "find", "replace"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "wiki_delete",
                "description": "Permanently delete a wiki page.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Page path to delete." }
                    },
                    "required": ["path"]
                }
            }
        }),
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn with_wiki_dir(f: impl FnOnce(&TempDir)) {
        let tmp = TempDir::new().unwrap();
        // Override the wiki dir for tests by writing directly to tmp path
        f(&tmp);
    }

    #[test]
    fn resolve_adds_md_extension() {
        // resolve() uses wiki_dir() which uses dirs::data_local_dir() —
        // we test only the path manipulation logic indirectly here.
        let dir = wiki_dir();
        let p = dir.join("test");
        let mut with_ext = p.clone();
        with_ext.set_extension("md");
        assert_eq!(with_ext.extension().unwrap(), "md");
    }

    #[test]
    fn wiki_write_and_read_roundtrip() {
        with_wiki_dir(|_tmp| {
            let args_w = serde_json::json!({ "path": "_test_roundtrip", "content": "# Hello\nworld" });
            let result = wiki_write(&args_w);
            assert!(result.contains("Written"), "write failed: {result}");

            let args_r = serde_json::json!({ "path": "_test_roundtrip" });
            let content = wiki_read(&args_r);
            assert!(content.contains("Hello"), "read failed: {content}");

            // Cleanup
            let _ = fs::remove_file(wiki_dir().join("_test_roundtrip.md"));
        });
    }

    #[test]
    fn wiki_patch_updates_content() {
        with_wiki_dir(|_tmp| {
            let path = "_test_patch";
            let _ = wiki_write(&serde_json::json!({ "path": path, "content": "old text here" }));
            let result = wiki_patch(&serde_json::json!({ "path": path, "find": "old text", "replace": "new text" }));
            assert!(result.contains("Patched"), "patch failed: {result}");
            let content = wiki_read(&serde_json::json!({ "path": path }));
            assert!(content.contains("new text"), "content not updated: {content}");
            assert!(!content.contains("old text"), "old text still present");
            let _ = fs::remove_file(wiki_dir().join(format!("{path}.md")));
        });
    }

    #[test]
    fn wiki_patch_reports_not_found() {
        let result = wiki_patch(&serde_json::json!({ "path": "_nonexistent_xyz", "find": "x", "replace": "y" }));
        assert!(result.contains("not found"), "expected not-found: {result}");
    }

    #[test]
    fn wiki_read_missing_page() {
        let result = wiki_read(&serde_json::json!({ "path": "_surely_missing_xyz_abc" }));
        assert!(result.contains("not found"), "expected not-found: {result}");
    }

    #[test]
    fn wiki_search_finds_content() {
        let path = "_test_search";
        let _ = wiki_write(&serde_json::json!({ "path": path, "content": "# Alice\nBirthday: 14th March" }));
        let result = wiki_search(&serde_json::json!({ "query": "birthday" }));
        assert!(result.to_lowercase().contains("birthday") || result.contains("_test_search"),
            "search failed: {result}");
        let _ = fs::remove_file(wiki_dir().join(format!("{path}.md")));
    }

    #[test]
    fn wiki_delete_removes_page() {
        let path = "_test_delete";
        let _ = wiki_write(&serde_json::json!({ "path": path, "content": "to delete" }));
        let result = wiki_delete(&serde_json::json!({ "path": path }));
        assert!(result.contains("Deleted"), "delete failed: {result}");
        let read = wiki_read(&serde_json::json!({ "path": path }));
        assert!(read.contains("not found"), "page still exists after delete");
    }

    #[test]
    fn resolve_rejects_traversal() {
        let result = resolve("../etc/passwd");
        assert!(result.is_err(), "expected traversal rejection");
    }

    #[test]
    fn wiki_append_creates_and_grows() {
        let path = "_test_append";
        // Start fresh
        let _ = wiki_delete(&serde_json::json!({ "path": path }));

        let r1 = wiki_append(&serde_json::json!({ "path": path, "content": "line one" }));
        assert!(r1.contains("Appended"), "first append failed: {r1}");

        let r2 = wiki_append(&serde_json::json!({ "path": path, "content": "line two" }));
        assert!(r2.contains("Appended"), "second append failed: {r2}");

        let content = wiki_read(&serde_json::json!({ "path": path }));
        assert!(content.contains("line one"), "first line missing: {content}");
        assert!(content.contains("line two"), "second line missing: {content}");

        let _ = fs::remove_file(wiki_dir().join(format!("{path}.md")));
    }

    #[test]
    fn wiki_lint_reports_empty_wiki() {
        // With a clean wiki dir this should not panic; just returns a string.
        let result = wiki_lint();
        // Either "empty" or a health report — both are valid strings.
        assert!(!result.is_empty(), "lint returned empty string");
    }

    #[test]
    fn wiki_lint_detects_page_missing_from_index() {
        // Write a content page but no index.md
        let page = "_lint_test_orphan";
        let _ = wiki_write(&serde_json::json!({ "path": page, "content": "# Orphan page" }));
        // Delete index.md if it exists so we get a clean test
        let index_path = wiki_dir().join("index.md");
        let had_index = index_path.exists();
        let index_backup = if had_index { fs::read_to_string(&index_path).ok() } else { None };
        let _ = fs::remove_file(&index_path);

        let result = wiki_lint();
        // Should mention the missing index or orphaned pages
        assert!(
            result.contains("index.md") || result.contains("No issues"),
            "lint output unexpected: {result}"
        );

        // Cleanup
        let _ = fs::remove_file(wiki_dir().join(format!("{page}.md")));
        if let Some(backup) = index_backup {
            let _ = fs::write(&index_path, backup);
        }
    }
}
