use std::path::Path;
use serde_json::Value;

/// Dispatch a tool call by name. Returns the result string.
pub async fn dispatch(name: &str, args: &Value) -> String {
    match name {
        "read_file" => {
            let path = args["path"].as_str().unwrap_or("");
            read_file(path)
        }
        "list_files" => {
            let path = args["path"].as_str().unwrap_or(".");
            list_files(path)
        }
        "web_search" => {
            let query = args["query"].as_str().unwrap_or("");
            web_search(query).await
        }
        _ => format!("Unknown tool: {name}"),
    }
}

// ── Built-in tool implementations ─────────────────────────────────────────────

fn read_file(path: &str) -> String {
    match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => format!("Error reading file '{path}': {e}"),
    }
}

fn list_files(path: &str) -> String {
    let p = Path::new(path);
    match std::fs::read_dir(p) {
        Ok(entries) => {
            let mut lines: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if is_dir { format!("{name}/") } else { name }
                })
                .collect();
            lines.sort();
            lines.join("\n")
        }
        Err(e) => format!("Error listing '{path}': {e}"),
    }
}

async fn web_search(query: &str) -> String {
    // DuckDuckGo instant answer API (no key required)
    let url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding::encode(query)
    );
    match reqwest::get(&url).await {
        Ok(resp) => match resp.json::<serde_json::Value>().await {
            Ok(json) => {
                let mut parts: Vec<String> = Vec::new();
                if let Some(abstract_text) = json["AbstractText"].as_str() {
                    if !abstract_text.is_empty() {
                        parts.push(format!("Summary: {abstract_text}"));
                    }
                }
                if let Some(answer) = json["Answer"].as_str() {
                    if !answer.is_empty() {
                        parts.push(format!("Answer: {answer}"));
                    }
                }
                if let Some(related) = json["RelatedTopics"].as_array() {
                    let topics: Vec<String> = related
                        .iter()
                        .take(5)
                        .filter_map(|t| t["Text"].as_str().map(String::from))
                        .collect();
                    if !topics.is_empty() {
                        parts.push(format!("Related:\n{}", topics.join("\n")));
                    }
                }
                if parts.is_empty() {
                    format!("No instant answer found for: {query}")
                } else {
                    parts.join("\n\n")
                }
            }
            Err(e) => format!("Search parse error: {e}"),
        },
        Err(e) => format!("Search error: {e}"),
    }
}

/// Returns the JSON schemas for all built-in tools.
pub fn builtin_schemas() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a local file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative file path." }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories at a given path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory path to list." }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query." }
                    },
                    "required": ["query"]
                }
            }
        }),
    ]
}
