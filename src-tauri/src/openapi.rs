use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use crate::mcp::AuthConfig;

/// A registered OpenAPI spec with its parsed tools and auth config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredSpec {
    pub id: String,
    pub title: String,
    pub base_url: String,
    #[serde(default)]
    pub auth: AuthConfig,
    pub tools: Vec<APITool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APITool {
    pub name: String,
    pub description: String,
    pub method: String,
    pub path: String,
    pub parameters: Vec<APIParam>,
    pub schema: Value, // JSON schema for the tool function
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIParam {
    pub name: String,
    pub location: String, // "path", "query", "body"
    pub required: bool,
    pub description: String,
}

/// Sanitize a string into a valid Ollama tool-name segment: [a-z0-9_], no leading/trailing _.
pub fn sanitize_tool_name(s: &str) -> String { sanitize_ident(s) }

fn sanitize_ident(s: &str) -> String {
    let mut out = String::new();
    let mut last_under = true; // avoid leading underscore
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_under = false;
        } else if !last_under {
            out.push('_');
            last_under = true;
        }
    }
    out.trim_end_matches('_').to_string()
}

/// Produce a short snake_case prefix from a human service name.
/// "Google Drive API" → "google_drive_"   "My MCP Server" → "my_mcp_server_"
pub fn tool_prefix(name: &str) -> String {
    let lower = name.to_lowercase();
    // Strip common noise suffixes
    let s = lower
        .trim_end_matches(" api")
        .trim_end_matches(" service")
        .trim_end_matches(" server")
        .trim()
        .to_string();
    let sanitized = sanitize_ident(&s);
    if sanitized.is_empty() { return "svc_".into(); }
    format!("{sanitized}_")
}

/// Parse an OpenAPI 3.0 JSON spec into a list of API tools.
pub fn parse_spec(title: &str, _base_url: &str, spec_json: &str) -> Result<Vec<APITool>, String> {
    let spec: Value = serde_json::from_str(spec_json).map_err(|e| format!("JSON parse error: {e}"))?;

    let paths = spec["paths"].as_object().ok_or("No 'paths' in spec")?;
    let prefix = tool_prefix(title);
    let mut tools = Vec::new();

    for (path_str, path_item) in paths {
        let path_obj = match path_item.as_object() {
            Some(o) => o,
            None => continue,
        };

        for (method, operation) in path_obj {
            let method_upper = method.to_uppercase();
            if !["GET","POST","PUT","PATCH","DELETE"].contains(&method_upper.as_str()) { continue; }

            let raw_op_id = operation["operationId"].as_str()
                .map(|s| sanitize_ident(s))
                .unwrap_or_else(|| sanitize_ident(&format!("{}_{}", method_upper.to_lowercase(),
                    path_str.replace('/', "_").trim_matches('_'))));
            // Prefix with service name so model can distinguish tools from different services.
            // Cap at 64 chars — Ollama rejects longer names.
            let combined = format!("{prefix}{raw_op_id}");
            let op_id = if combined.len() > 64 { combined[..64].trim_end_matches('_').to_string() } else { combined };

            let description = operation["summary"].as_str()
                .or_else(|| operation["description"].as_str())
                .unwrap_or(&op_id)
                .to_string();

            // Collect parameters
            let mut params: Vec<APIParam> = Vec::new();
            let mut json_props: HashMap<String, Value> = HashMap::new();
            let mut required_params: Vec<String> = Vec::new();

            // Path + query parameters
            if let Some(param_arr) = operation["parameters"].as_array() {
                for param in param_arr {
                    let name = param["name"].as_str().unwrap_or("").to_string();
                    let location = param["in"].as_str().unwrap_or("query").to_string();
                    let required = param["required"].as_bool().unwrap_or(location == "path");
                    let desc = param["description"].as_str().unwrap_or("").to_string();
                    let ptype = param["schema"]["type"].as_str().unwrap_or("string").to_string();

                    if name.is_empty() { continue; }
                    if required { required_params.push(name.clone()); }
                    json_props.insert(name.clone(), serde_json::json!({ "type": ptype, "description": desc }));
                    params.push(APIParam { name, location, required, description: desc });
                }
            }

            // Request body (simplified — top-level properties only)
            if let Some(body) = operation["requestBody"]["content"]["application/json"]["schema"]["properties"].as_object() {
                let body_required: Vec<String> = operation["requestBody"]["content"]["application/json"]["schema"]["required"]
                    .as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                for (name, prop) in body {
                    let desc = prop["description"].as_str().unwrap_or("").to_string();
                    let ptype = prop["type"].as_str().unwrap_or("string").to_string();
                    let required = body_required.contains(name);
                    if required { required_params.push(name.clone()); }
                    json_props.insert(name.clone(), serde_json::json!({ "type": ptype, "description": desc }));
                    params.push(APIParam { name: name.clone(), location: "body".into(), required, description: desc });
                }
            }

            let schema = serde_json::json!({
                "type": "function",
                "function": {
                    "name": op_id,
                    "description": format!("[{title}] {description}"),
                    "parameters": {
                        "type": "object",
                        "properties": json_props,
                        "required": required_params
                    }
                }
            });

            tools.push(APITool {
                name: op_id,
                description,
                method: method_upper,
                path: path_str.clone(),
                parameters: params,
                schema,
            });
        }
    }

    Ok(tools)
}

/// Execute an OpenAPI tool call.
pub async fn execute(spec: &RegisteredSpec, tool: &APITool, args: &Value) -> String {
    let client = reqwest::Client::new();

    // Build URL: substitute path parameters
    let mut url_path = tool.path.clone();
    for param in tool.parameters.iter().filter(|p| p.location == "path") {
        if let Some(val) = args[&param.name].as_str() {
            url_path = url_path.replace(&format!("{{{}}}", param.name), val);
        }
    }
    let base = spec.base_url.trim_end_matches('/');
    let mut url = format!("{base}{url_path}");

    // Query parameters
    let query_params: Vec<(String,String)> = tool.parameters.iter()
        .filter(|p| p.location == "query")
        .filter_map(|p| {
            args[&p.name].as_str().map(|v| (p.name.clone(), v.to_string()))
                .or_else(|| args[&p.name].as_i64().map(|v| (p.name.clone(), v.to_string())))
        })
        .collect();

    if !query_params.is_empty() {
        let qs = query_params.iter().map(|(k,v)| format!("{k}={}", urlencoding::encode(v))).collect::<Vec<_>>().join("&");
        url = format!("{url}?{qs}");
    }

    // Build request
    let base_req = match tool.method.as_str() {
        "GET"    => client.get(&url),
        "POST"   => client.post(&url),
        "PUT"    => client.put(&url),
        "PATCH"  => client.patch(&url),
        "DELETE" => client.delete(&url),
        _        => client.get(&url),
    };

    let mut req = spec.auth.apply_async(&client, base_req).await;
    req = req.header("User-Agent", concat!("LexiChat/", env!("CARGO_PKG_VERSION"), " (https://github.com/kulbinderdio/lexichat)"));
    req = req.header("Content-Type", "application/json");
    req = req.header("Accept", "application/json");

    // Body
    let body_params: serde_json::Map<String,Value> = tool.parameters.iter()
        .filter(|p| p.location == "body")
        .filter_map(|p| args.get(&p.name).map(|v| (p.name.clone(), v.clone())))
        .collect();
    if !body_params.is_empty() {
        req = req.json(&body_params);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => {
                    // Try to pretty-print JSON
                    if let Ok(json) = serde_json::from_str::<Value>(&body) {
                        format!("HTTP {status}\n{}", serde_json::to_string_pretty(&json).unwrap_or(body))
                    } else {
                        format!("HTTP {status}\n{body}")
                    }
                }
                Err(e) => format!("HTTP {status} — read error: {e}"),
            }
        }
        Err(e) => format!("Request error: {e}"),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sanitize_tool_name ────────────────────────────────────────────────────

    #[test]
    fn sanitize_lowercase_alphanumeric_unchanged() {
        assert_eq!(sanitize_tool_name("list_files"), "list_files");
    }

    #[test]
    fn sanitize_uppercase_lowercased() {
        assert_eq!(sanitize_tool_name("ListFiles"), "listfiles");
    }

    #[test]
    fn sanitize_dots_become_underscores() {
        assert_eq!(sanitize_tool_name("drive.files.list"), "drive_files_list");
    }

    #[test]
    fn sanitize_spaces_become_underscores() {
        assert_eq!(sanitize_tool_name("get file info"), "get_file_info");
    }

    #[test]
    fn sanitize_leading_trailing_underscores_stripped() {
        assert_eq!(sanitize_tool_name(".leading"), "leading");
        assert_eq!(sanitize_tool_name("trailing."), "trailing");
        assert_eq!(sanitize_tool_name(".both."), "both");
    }

    #[test]
    fn sanitize_consecutive_separators_collapse() {
        assert_eq!(sanitize_tool_name("a..b"), "a_b");
        assert_eq!(sanitize_tool_name("a - b"), "a_b");
    }

    #[test]
    fn sanitize_numbers_preserved() {
        assert_eq!(sanitize_tool_name("get2Factor"), "get2factor");
    }

    #[test]
    fn sanitize_empty_string() {
        assert_eq!(sanitize_tool_name(""), "");
    }

    // ── tool_prefix ───────────────────────────────────────────────────────────

    #[test]
    fn prefix_strips_api_suffix() {
        assert_eq!(tool_prefix("Google Drive API"), "google_drive_");
    }

    #[test]
    fn prefix_strips_service_suffix() {
        assert_eq!(tool_prefix("Payment Service"), "payment_");
    }

    #[test]
    fn prefix_strips_server_suffix() {
        assert_eq!(tool_prefix("My MCP Server"), "my_mcp_");
    }

    #[test]
    fn prefix_simple_name() {
        assert_eq!(tool_prefix("Gmail"), "gmail_");
    }

    #[test]
    fn prefix_empty_falls_back() {
        assert_eq!(tool_prefix(""), "svc_");
    }

    #[test]
    fn prefix_only_noise_falls_back() {
        // trim_end_matches removes " api" (with leading space) but bare "API" → "api" has no space,
        // so it stays as "api_". Only a truly empty result triggers the "svc_" fallback.
        assert_eq!(tool_prefix("API"), "api_");
        assert_eq!(tool_prefix(""), "svc_");
    }

    // ── parse_spec ────────────────────────────────────────────────────────────

    fn minimal_spec(path: &str, method: &str, op_id: &str) -> String {
        serde_json::json!({
            "paths": {
                path: {
                    method: {
                        "operationId": op_id,
                        "summary": "Test operation",
                        "parameters": []
                    }
                }
            }
        }).to_string()
    }

    #[test]
    fn parse_spec_generates_prefixed_tool_name() {
        let spec = minimal_spec("/v1/messages", "get", "listMessages");
        let tools = parse_spec("Gmail", "https://gmail.googleapis.com", &spec).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "gmail_listmessages");
    }

    #[test]
    fn parse_spec_method_uppercased() {
        let spec = minimal_spec("/v1/items", "post", "createItem");
        let tools = parse_spec("My API", "https://example.com", &spec).unwrap();
        assert_eq!(tools[0].method, "POST");
    }

    #[test]
    fn parse_spec_path_parameter_detected() {
        let spec = serde_json::json!({
            "paths": {
                "/v1/messages/{messageId}": {
                    "get": {
                        "operationId": "getMessage",
                        "summary": "Get message",
                        "parameters": [{
                            "name": "messageId",
                            "in": "path",
                            "required": true,
                            "schema": { "type": "string" }
                        }]
                    }
                }
            }
        }).to_string();
        let tools = parse_spec("Gmail", "https://gmail.googleapis.com", &spec).unwrap();
        let param = tools[0].parameters.iter().find(|p| p.name == "messageId").unwrap();
        assert_eq!(param.location, "path");
        assert!(param.required);
    }

    #[test]
    fn parse_spec_query_parameter_optional() {
        let spec = serde_json::json!({
            "paths": {
                "/v1/messages": {
                    "get": {
                        "operationId": "listMessages",
                        "parameters": [{
                            "name": "maxResults",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "integer" }
                        }]
                    }
                }
            }
        }).to_string();
        let tools = parse_spec("Gmail", "https://gmail.googleapis.com", &spec).unwrap();
        let param = tools[0].parameters.iter().find(|p| p.name == "maxResults").unwrap();
        assert_eq!(param.location, "query");
        assert!(!param.required);
    }

    #[test]
    fn parse_spec_request_body_properties_extracted() {
        let spec = serde_json::json!({
            "paths": {
                "/v1/messages/send": {
                    "post": {
                        "operationId": "sendMessage",
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "raw": { "type": "string", "description": "Base64 message" }
                                        },
                                        "required": ["raw"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }).to_string();
        let tools = parse_spec("Gmail", "https://gmail.googleapis.com", &spec).unwrap();
        let param = tools[0].parameters.iter().find(|p| p.name == "raw").unwrap();
        assert_eq!(param.location, "body");
        assert!(param.required);
    }

    #[test]
    fn parse_spec_tool_name_capped_at_64_chars() {
        let long_op_id = "a".repeat(80);
        let spec = minimal_spec("/v1/op", "get", &long_op_id);
        let tools = parse_spec("Gmail", "https://gmail.googleapis.com", &spec).unwrap();
        assert!(tools[0].name.len() <= 64);
    }

    #[test]
    fn parse_spec_ignores_non_http_methods() {
        let spec = serde_json::json!({
            "paths": {
                "/v1/items": {
                    "get": { "operationId": "listItems" },
                    "x-custom": { "operationId": "customOp" }
                }
            }
        }).to_string();
        let tools = parse_spec("Test", "https://example.com", &spec).unwrap();
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn parse_spec_synthesises_op_id_when_missing() {
        let spec = serde_json::json!({
            "paths": {
                "/v1/users": {
                    "get": { "summary": "List users" }
                }
            }
        }).to_string();
        let tools = parse_spec("Test API", "https://example.com", &spec).unwrap();
        assert_eq!(tools.len(), 1);
        // Name should be non-empty and start with prefix
        assert!(tools[0].name.starts_with("test_"));
    }

    #[test]
    fn parse_spec_invalid_json_returns_error() {
        let result = parse_spec("Test", "https://example.com", "not json");
        assert!(result.is_err());
    }

    #[test]
    fn parse_spec_missing_paths_returns_error() {
        let result = parse_spec("Test", "https://example.com", r#"{"info":{}}"#);
        assert!(result.is_err());
    }
}
