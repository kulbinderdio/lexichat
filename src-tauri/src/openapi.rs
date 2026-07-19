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

/// Convert a JSON scalar arg (string, number, or bool) into its string form for
/// use in a URL path or query string. Numbers — including floats such as lat/lng —
/// and booleans are stringified; null/array/object yield `None` so the parameter
/// is simply omitted. Models emit numeric params as JSON numbers, not strings, so
/// without this they would be silently dropped and the request sent malformed.
fn arg_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Build the `(key, value)` query-string pairs for a tool call, skipping any parameter the
/// caller didn't supply. Arrays are expanded to repeated pairs (`?k=a&k=b`) — the convention
/// these APIs accept — rather than being dropped.
fn build_query_params(params: &[APIParam], args: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for p in params.iter().filter(|p| p.location == "query") {
        match &args[&p.name] {
            Value::Array(items) => {
                for it in items {
                    if let Some(v) = arg_to_string(it) { out.push((p.name.clone(), v)); }
                }
            }
            other => {
                if let Some(v) = arg_to_string(other) { out.push((p.name.clone(), v)); }
            }
        }
    }
    out
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

/// Resolve a schema node, following a `$ref` into `components/schemas` and unwrapping the
/// `allOf: [{ $ref }]` pattern Swashbuckle emits for enums-with-descriptions. Bounded depth
/// so a circular `$ref` can't loop forever. Returns the concrete schema (with its `enum`,
/// `type`, `items`, etc.) or the input unchanged.
fn resolve_ref<'a>(schema: &'a Value, root: &'a Value, depth: u8) -> std::borrow::Cow<'a, Value> {
    use std::borrow::Cow;
    if depth == 0 { return Cow::Borrowed(schema); }
    if let Some(r) = schema.get("$ref").and_then(|v| v.as_str()) {
        if let Some(name) = r.strip_prefix("#/components/schemas/") {
            if let Some(def) = root.get("components").and_then(|c| c.get("schemas")).and_then(|s| s.get(name)) {
                return Cow::Owned(resolve_ref(def, root, depth - 1).into_owned());
            }
        }
    }
    if let Some(first) = schema.get("allOf").and_then(|v| v.as_array()).and_then(|a| a.first()) {
        return Cow::Owned(resolve_ref(first, root, depth - 1).into_owned());
    }
    Cow::Borrowed(schema)
}

/// Build a JSON-Schema property for one parameter, carrying through the details a model needs
/// to call the API correctly — notably `enum` values and array `items` — which are lost if
/// only `type` is copied. `$ref`/`allOf` are resolved against the spec's components.
fn param_json_schema(pschema: &Value, root: &Value, desc: &str) -> Value {
    let resolved = resolve_ref(pschema, root, 8);
    let mut out = serde_json::Map::new();
    let ty = resolved.get("type").and_then(|v| v.as_str()).unwrap_or("string");
    out.insert("type".into(), Value::String(ty.into()));
    if !desc.is_empty() { out.insert("description".into(), Value::String(desc.into())); }
    if let Some(en) = resolved.get("enum") { out.insert("enum".into(), en.clone()); }
    if let Some(fmt) = resolved.get("format").and_then(|v| v.as_str()) {
        out.insert("format".into(), Value::String(fmt.into()));
    }
    if ty == "array" {
        if let Some(items) = resolved.get("items") {
            out.insert("items".into(), resolve_ref(items, root, 8).into_owned());
        }
    }
    Value::Object(out)
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

                    if name.is_empty() { continue; }
                    if required { required_params.push(name.clone()); }
                    // Resolve the param schema (following $ref/allOf) so enum values, array
                    // items, etc. reach the model — otherwise it guesses enum values and the
                    // API rejects them with a 400 validation error.
                    json_props.insert(name.clone(), param_json_schema(&param["schema"], &spec, &desc));
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
/// `app` is optional: when provided, a refreshed OAuth2 access_token is persisted
/// back to AppState and emitted to the frontend for localStorage persistence.
pub async fn execute<R: tauri::Runtime>(
    spec: &RegisteredSpec,
    tool: &APITool,
    args: &Value,
    app: Option<&tauri::AppHandle<R>>,
) -> String {
    // Repair arguments a model wrongly stringified (an object/array passed as a JSON string,
    // or a number/bool as a string) — same fix as for MCP tools — before building the request.
    let coerced = tool.schema.get("function").and_then(|f| f.get("parameters"))
        .filter(|s| s.is_object())
        .map(|schema| crate::mcp::coerce_args_to_schema(args, schema));
    let args = coerced.as_ref().unwrap_or(args);

    let client = reqwest::Client::builder()
        .use_rustls_tls()
        .build()
        .unwrap_or_default();

    // Build URL: substitute path parameters
    let mut url_path = tool.path.clone();
    for param in tool.parameters.iter().filter(|p| p.location == "path") {
        if let Some(val) = arg_to_string(&args[&param.name]) {
            url_path = url_path.replace(&format!("{{{}}}", param.name), &val);
        }
    }
    let base = spec.base_url.trim_end_matches('/');
    let mut url = format!("{base}{url_path}");

    // Query parameters
    let query_params = build_query_params(&tool.parameters, args);

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
    req = req.header("User-Agent", concat!("LexiChat/", env!("CARGO_PKG_VERSION")));
    req = req.header("Accept", "application/json");

    // Body — only set Content-Type for methods that carry a request body
    let body_params: serde_json::Map<String,Value> = tool.parameters.iter()
        .filter(|p| p.location == "body")
        .filter_map(|p| args.get(&p.name).map(|v| (p.name.clone(), v.clone())))
        .collect();
    if !body_params.is_empty() {
        req = req.json(&body_params);
    } else if ["POST", "PUT", "PATCH"].contains(&tool.method.as_str()) {
        req = req.header("Content-Type", "application/json");
    }

    let result = send_and_format(req).await;

    // On 401, try refreshing the OAuth2 token and retry once
    if result.starts_with("HTTP 401") {
        if let Some(new_token) = spec.auth.try_refresh(&client).await {
            // Persist the new token so subsequent calls in this run succeed
            if let Some(app) = app {
                persist_refreshed_token(app, &spec.id, &new_token).await;
            }

            // Rebuild request with the refreshed token
            let retry_base = match tool.method.as_str() {
                "GET"    => client.get(&url),
                "POST"   => client.post(&url),
                "PUT"    => client.put(&url),
                "PATCH"  => client.patch(&url),
                "DELETE" => client.delete(&url),
                _        => client.get(&url),
            };
            let mut retry_req = retry_base
                .header("Authorization", format!("Bearer {new_token}"))
                .header("User-Agent", concat!("LexiChat/", env!("CARGO_PKG_VERSION")))
                .header("Accept", "application/json");
            if !body_params.is_empty() {
                retry_req = retry_req.json(&body_params);
            } else if ["POST", "PUT", "PATCH"].contains(&tool.method.as_str()) {
                retry_req = retry_req.header("Content-Type", "application/json");
            }
            return send_and_format(retry_req).await;
        }
    }

    result
}

async fn send_and_format(req: reqwest::RequestBuilder) -> String {
    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => {
                    if let Ok(json) = serde_json::from_str::<Value>(&body) {
                        format!("HTTP {status}\n{}", serde_json::to_string_pretty(&json).unwrap_or(body))
                    } else {
                        format!("HTTP {status}\n{body}")
                    }
                }
                Err(e) => format!("HTTP {status} — read error: {e}"),
            }
        }
        Err(e) => {
            let mut msg = format!("Request error: {e}");
            let mut src: Option<&dyn std::error::Error> = std::error::Error::source(&e);
            while let Some(cause) = src {
                msg.push_str(&format!("\n  caused by: {cause}"));
                src = cause.source();
            }
            msg
        }
    }
}

/// Update the in-memory AppState with the new access_token and notify the frontend to persist it.
async fn persist_refreshed_token<R: tauri::Runtime>(app: &tauri::AppHandle<R>, spec_id: &str, new_token: &str) {
    use tauri::Manager;
    let state = app.state::<crate::AppState>();
    {
        let mut specs = state.openapi_specs.lock().unwrap();
        for spec in specs.iter_mut() {
            if spec.id == spec_id {
                if let crate::mcp::AuthConfig::OAuth2 { ref mut access_token, .. } = spec.auth {
                    *access_token = new_token.to_string();
                }
                break;
            }
        }
    }
    use tauri::Emitter;
    let _ = app.emit("openapi-token-refreshed", serde_json::json!({
        "spec_id": spec_id,
        "access_token": new_token,
    }));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A $ref'd enum query param must surface its allowed values in the tool schema, or the
    /// model guesses and the API 400s (the Committees API CommitteeStatus bug).
    #[test]
    fn ref_enum_param_carries_its_values() {
        let spec = r##"{
          "openapi": "3.0.0",
          "paths": {
            "/api/Committees": {
              "get": {
                "operationId": "getCommittees",
                "parameters": [
                  { "name": "CommitteeStatus", "in": "query",
                    "schema": { "$ref": "#/components/schemas/CommitteeStatus" } },
                  { "name": "CommitteeIds", "in": "query",
                    "schema": { "type": "array", "items": { "type": "integer" } } }
                ]
              }
            }
          },
          "components": { "schemas": {
            "CommitteeStatus": { "type": "string", "enum": ["Current", "Former", "All"] }
          } }
        }"##;
        let tools = parse_spec("Committees", "https://x", spec).unwrap();
        let props = &tools[0].schema["function"]["parameters"]["properties"];
        assert_eq!(props["CommitteeStatus"]["enum"],
            serde_json::json!(["Current", "Former", "All"]));
        assert_eq!(props["CommitteeStatus"]["type"], "string");
        assert_eq!(props["CommitteeIds"]["type"], "array");
        assert_eq!(props["CommitteeIds"]["items"]["type"], "integer");
    }

    #[test]
    fn array_query_param_expands_to_repeated_pairs() {
        let params = vec![
            APIParam { name: "CommitteeIds".into(), location: "query".into(), required: false, description: String::new() },
            APIParam { name: "Take".into(), location: "query".into(), required: false, description: String::new() },
        ];
        let args = serde_json::json!({ "CommitteeIds": [1, 2, 3], "Take": 5 });
        let pairs = build_query_params(&params, &args);
        assert_eq!(pairs, vec![
            ("CommitteeIds".into(), "1".into()),
            ("CommitteeIds".into(), "2".into()),
            ("CommitteeIds".into(), "3".into()),
            ("Take".into(), "5".into()),
        ]);
    }

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

    // ── query param building ──────────────────────────────────────────────────

    /// Regression: models emit numeric params (e.g. lat/lng) as JSON numbers, not
    /// strings. Floats, ints and bools must all reach the query string; absent
    /// params must be omitted. Previously floats were silently dropped, sending a
    /// malformed request (e.g. the UK Police API returning HTTP 400 for no coords).
    #[test]
    fn query_params_include_non_string_scalars() {
        let p = |name: &str| APIParam {
            name: name.into(), location: "query".into(), required: false, description: String::new(),
        };
        let params = vec![p("lat"), p("lng"), p("limit"), p("active"), p("q"), p("missing")];
        let args = serde_json::json!({
            "lat": 51.438066, "lng": 0.361697, "limit": 25, "active": true, "q": "hello"
            // "missing" intentionally absent
        });
        let qp = build_query_params(&params, &args);
        assert!(qp.contains(&("lat".into(), "51.438066".into())), "float lat dropped: {qp:?}");
        assert!(qp.contains(&("lng".into(), "0.361697".into())), "float lng dropped: {qp:?}");
        assert!(qp.contains(&("limit".into(), "25".into())));
        assert!(qp.contains(&("active".into(), "true".into())));
        assert!(qp.contains(&("q".into(), "hello".into())));
        assert!(!qp.iter().any(|(k, _)| k == "missing"), "absent param must be omitted: {qp:?}");
        assert_eq!(qp.len(), 5);
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

    // ── HTTP integration (wiremock) ───────────────────────────────────────────

    fn get_spec(base_url: &str) -> RegisteredSpec {
        let spec_json = serde_json::json!({
            "paths": { "/search": { "get": {
                "operationId": "search",
                "summary": "Search",
                "parameters": [{ "name": "q", "in": "query", "required": true, "schema": { "type": "string" } }]
            }}}
        }).to_string();
        let tools = parse_spec("Test", base_url, &spec_json).unwrap();
        RegisteredSpec { id: "id".into(), title: "Test".into(), base_url: base_url.into(),
                         auth: crate::mcp::AuthConfig::None, tools }
    }

    #[tokio::test]
    async fn execute_get_substitutes_query_and_returns_body() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::{method, path, query_param};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search"))
            .and(query_param("q", "hello"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "ok": true })))
            .mount(&server).await;

        let spec = get_spec(&server.uri());
        let out = execute::<tauri::Wry>(&spec, &spec.tools[0], &serde_json::json!({ "q": "hello" }), None).await;
        assert!(out.contains("HTTP 200"), "got: {out}");
        assert!(out.contains("\"ok\""), "got: {out}");
    }

    #[tokio::test]
    async fn execute_coerces_stringified_array_arg() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::{method, path, query_param};
        let server = MockServer::start().await;
        // Coercion turns "[1,2,3]" into a real array, which build_query_params expands to
        // repeated `ids=` params. Without it, the whole "[1,2,3]" string is sent as one value.
        Mock::given(method("GET")).and(path("/items"))
            .and(query_param("ids", "1")).and(query_param("ids", "3"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "ok": true })))
            .mount(&server).await;

        let spec_json = serde_json::json!({
            "paths": { "/items": { "get": {
                "operationId": "items",
                "parameters": [{ "name": "ids", "in": "query", "required": false,
                                 "schema": { "type": "array", "items": { "type": "integer" } } }]
            }}}
        }).to_string();
        let tools = parse_spec("T", &server.uri(), &spec_json).unwrap();
        let spec = RegisteredSpec { id: "id".into(), title: "T".into(), base_url: server.uri(),
                                    auth: crate::mcp::AuthConfig::None, tools };
        // Model wrongly stringified the array.
        let out = execute::<tauri::Wry>(&spec, &spec.tools[0],
            &serde_json::json!({ "ids": "[1,2,3]" }), None).await;
        assert!(out.contains("HTTP 200"), "stringified array should have been coerced: {out}");
    }

    #[tokio::test]
    async fn execute_applies_bearer_auth_header() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::{method, header};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("authorization", "Bearer tok-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "ok": true })))
            .mount(&server).await;

        let mut spec = get_spec(&server.uri());
        spec.auth = crate::mcp::AuthConfig::Bearer { bearer_token: "tok-123".into() };
        let out = execute::<tauri::Wry>(&spec, &spec.tools[0], &serde_json::json!({ "q": "x" }), None).await;
        assert!(out.contains("HTTP 200"), "auth header not accepted: {out}");
    }
}
