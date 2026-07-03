use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use crate::mcp::AuthConfig;
use crate::openapi::{APITool, APIParam, tool_prefix};

/// A single curated example query shown to the model as documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleQuery {
    pub label: String,
    pub query: String,
}

/// A registered SPARQL endpoint and the two tools synthesised from it
/// (a raw-query tool the model authors SPARQL for, and an on-demand schema tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredSparqlEndpoint {
    pub id: String,
    pub title: String,
    pub endpoint_url: String,
    /// PREFIX declarations the model should reuse (one per line).
    #[serde(default)]
    pub prefixes: String,
    /// Free-text or ontology summary of the available classes/properties.
    #[serde(default)]
    pub schema_summary: String,
    /// Curated example queries — the single highest-value context for the model.
    #[serde(default)]
    pub example_queries: Vec<ExampleQuery>,
    /// What topics/questions this endpoint is best for — surfaced to help the model
    /// pick this tool over web search.
    #[serde(default)]
    pub usage_hint: String,
    #[serde(default)]
    pub auth: AuthConfig,
    /// When true (default), reject SPARQL Update operations before sending.
    #[serde(default = "default_true")]
    pub read_only: bool,
    /// Synthesised tools — populated by `build_tools`. Not user-supplied.
    #[serde(default)]
    pub tools: Vec<APITool>,
}

fn default_true() -> bool { true }

/// Result of best-effort endpoint auto-discovery (the "Test & discover" button).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    pub live: bool,
    pub message: String,
    pub suggested_prefixes: String,
    pub suggested_schema: String,
}

/// Cap a tool name at 64 chars (Ollama limit), trimming a trailing underscore.
fn cap64(s: String) -> String {
    if s.len() > 64 { s[..64].trim_end_matches('_').to_string() } else { s }
}

/// Char-safe truncation with an ellipsis marker.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max { return s.to_string(); }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

/// Build the `query` + `schema` tools for an endpoint. Reads config only (not `tools`),
/// so callers assign the result back into `endpoint.tools`.
pub fn build_tools(ep: &RegisteredSparqlEndpoint) -> Vec<APITool> {
    let prefix = tool_prefix(&ep.title);
    let query_name = cap64(format!("{prefix}query"));
    let schema_name = cap64(format!("{prefix}schema"));

    // Always-on query tool: keep the description compact (prefixes + short schema + one example).
    let mut desc = String::new();
    if !ep.usage_hint.trim().is_empty() {
        desc.push_str(&format!(
            "USE THIS TOOL when the user asks about: {}. Prefer it over web_search for these topics — it returns authoritative structured data. ",
            ep.usage_hint.trim()
        ));
    }
    desc.push_str(&format!(
        "Runs a SPARQL query against the {} endpoint ({}). Provide the complete SPARQL query string including any PREFIX lines. ",
        ep.title, ep.endpoint_url
    ));
    if !ep.prefixes.trim().is_empty() {
        desc.push_str(&format!("\nCommon prefixes:\n{}\n", truncate(ep.prefixes.trim(), 600)));
    }
    if !ep.schema_summary.trim().is_empty() {
        desc.push_str(&format!("\nVocabulary:\n{}\n", truncate(ep.schema_summary.trim(), 1200)));
    }
    if let Some(first) = ep.example_queries.first() {
        desc.push_str(&format!("\nExample ({}):\n{}\n", first.label, truncate(first.query.trim(), 600)));
    }
    desc.push_str(&format!(
        "\nCall `{schema_name}` for the full vocabulary and more example queries before writing a query if you are unsure of the data model."
    ));

    let query_schema = serde_json::json!({
        "type": "function",
        "function": {
            "name": query_name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "The full SPARQL query (SELECT/ASK/CONSTRUCT/DESCRIBE), including PREFIX declarations." }
                },
                "required": ["query"]
            }
        }
    });

    let schema_desc = format!(
        "Return the namespace prefixes, vocabulary (classes/properties), and example queries for the {} SPARQL endpoint. Call this before authoring a query when unsure of the data model.",
        ep.title
    );
    let schema_schema = serde_json::json!({
        "type": "function",
        "function": {
            "name": schema_name,
            "description": schema_desc,
            "parameters": { "type": "object", "properties": {}, "required": [] }
        }
    });

    vec![
        APITool {
            name: query_name,
            description: "Run a SPARQL query".into(),
            method: "POST".into(),
            path: ep.endpoint_url.clone(),
            parameters: vec![APIParam {
                name: "query".into(),
                location: "body".into(),
                required: true,
                description: "The full SPARQL query string.".into(),
            }],
            schema: query_schema,
        },
        APITool {
            name: schema_name,
            description: "Describe the SPARQL endpoint schema".into(),
            method: "GET".into(),
            path: ep.endpoint_url.clone(),
            parameters: vec![],
            schema: schema_schema,
        },
    ]
}

/// SPARQL Update keywords rejected when the endpoint is read-only.
fn forbidden_keyword(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    let tokens: std::collections::HashSet<&str> =
        upper.split(|c: char| !c.is_ascii_alphanumeric()).collect();
    for kw in ["INSERT", "DELETE", "DROP", "CLEAR", "LOAD", "CREATE", "COPY", "MOVE"] {
        if tokens.contains(kw) { return Some(kw.to_string()); }
    }
    None
}

/// Execute a SPARQL query against the endpoint, applying auth and a single 401 refresh+retry.
pub async fn execute(
    ep: &RegisteredSparqlEndpoint,
    query: &str,
    app: Option<&tauri::AppHandle>,
) -> String {
    if ep.read_only {
        if let Some(kw) = forbidden_keyword(query) {
            return format!(
                "Rejected: this endpoint is read-only and the query contains a '{kw}' operation. Only SELECT/ASK/CONSTRUCT/DESCRIBE queries are allowed."
            );
        }
    }

    let client = reqwest::Client::builder().use_rustls_tls().build().unwrap_or_default();
    let result = run_query(&client, ep, query, None).await;

    if result.starts_with("HTTP 401") {
        if let Some(new_token) = ep.auth.try_refresh(&client).await {
            if let Some(app) = app {
                persist_refreshed_token(app, &ep.id, &new_token).await;
            }
            return run_query(&client, ep, query, Some(&new_token)).await;
        }
    }
    result
}

async fn run_query(
    client: &reqwest::Client,
    ep: &RegisteredSparqlEndpoint,
    query: &str,
    bearer_override: Option<&str>,
) -> String {
    let base_req = client
        .post(&ep.endpoint_url)
        .header("Accept", "application/sparql-results+json, application/rdf+xml;q=0.8, text/turtle;q=0.7, */*;q=0.5")
        .header("User-Agent", concat!("LexiChat/", env!("CARGO_PKG_VERSION")))
        .form(&[("query", query)]);

    let req = match bearer_override {
        Some(tok) => base_req.header("Authorization", format!("Bearer {tok}")),
        None => ep.auth.apply_async(client, base_req).await,
    };

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            match resp.text().await {
                Ok(body) => format_results(status, &body),
                Err(e) => format!("HTTP {status} — read error: {e}"),
            }
        }
        Err(e) => format!("Request error: {e}"),
    }
}

/// Flatten SPARQL JSON results into a compact text table; pass other bodies through.
fn format_results(status: u16, body: &str) -> String {
    if let Ok(json) = serde_json::from_str::<Value>(body) {
        // SELECT → head.vars + results.bindings
        if let Some(bindings) = json["results"]["bindings"].as_array() {
            let vars: Vec<String> = json["head"]["vars"].as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if bindings.is_empty() {
                return format!("HTTP {status}\n(0 rows). Columns: {}", vars.join(", "));
            }
            let mut out = format!("HTTP {status}\n{} row(s). Columns: {}\n", bindings.len(), vars.join(", "));
            for (i, b) in bindings.iter().enumerate() {
                if i >= 100 {
                    out.push_str(&format!("…({} more rows)\n", bindings.len() - 100));
                    break;
                }
                let row: Vec<String> = vars.iter()
                    .map(|v| b[v]["value"].as_str().unwrap_or("").to_string())
                    .collect();
                out.push_str(&format!("- {}\n", row.join(" | ")));
            }
            return out;
        }
        // ASK → boolean
        if let Some(b) = json.get("boolean") {
            return format!("HTTP {status}\nboolean: {b}");
        }
        return format!("HTTP {status}\n{}", serde_json::to_string_pretty(&json).unwrap_or_else(|_| body.to_string()));
    }
    // CONSTRUCT/DESCRIBE → RDF (turtle/xml) or an error page
    format!("HTTP {status}\n{body}")
}

/// Local formatter backing the on-demand schema tool — no HTTP.
pub fn schema_text(ep: &RegisteredSparqlEndpoint) -> String {
    let mut out = format!("SPARQL endpoint: {}\nURL: {}\n", ep.title, ep.endpoint_url);
    if !ep.prefixes.trim().is_empty() {
        out.push_str(&format!("\nPrefixes:\n{}\n", ep.prefixes.trim()));
    }
    if !ep.schema_summary.trim().is_empty() {
        out.push_str(&format!("\nVocabulary / schema:\n{}\n", ep.schema_summary.trim()));
    }
    if !ep.example_queries.is_empty() {
        out.push_str("\nExample queries:\n");
        for ex in &ep.example_queries {
            out.push_str(&format!("\n# {}\n{}\n", ex.label, ex.query.trim()));
        }
    }
    if ep.prefixes.trim().is_empty() && ep.schema_summary.trim().is_empty() && ep.example_queries.is_empty() {
        out.push_str("\n(No schema documentation was provided. Introspect with: SELECT DISTINCT ?class WHERE { ?s a ?class } LIMIT 100)\n");
    }
    out
}

// ── Auto-discovery ──────────────────────────────────────────────────────────────

/// Best-effort probe: confirm liveness, then sample classes/properties and derive prefixes.
pub async fn probe(endpoint_url: &str, auth: &AuthConfig) -> DiscoveryResult {
    let client = reqwest::Client::builder().use_rustls_tls().build().unwrap_or_default();

    // 1. Liveness
    let live = run_select_var(&client, endpoint_url, auth, "SELECT * WHERE { ?s ?p ?o } LIMIT 1", "s")
        .await.is_ok();
    if !live {
        return DiscoveryResult {
            live: false,
            message: "Could not reach the endpoint or it did not return SPARQL results. Check the URL and auth, then fill the schema in manually.".into(),
            suggested_prefixes: String::new(),
            suggested_schema: String::new(),
        };
    }

    // 2. Sample classes and properties
    let classes = run_select_var(&client, endpoint_url, auth,
        "SELECT DISTINCT ?c WHERE { ?s a ?c } LIMIT 60", "c").await.unwrap_or_default();
    let props = run_select_var(&client, endpoint_url, auth,
        "SELECT DISTINCT ?p WHERE { ?s ?p ?o } LIMIT 100", "p").await.unwrap_or_default();

    // 3. Derive namespace → prefix suggestions from the URIs we saw
    let mut namespaces: BTreeMap<String, String> = BTreeMap::new();
    for uri in classes.iter().chain(props.iter()) {
        if let Some((ns, _local)) = split_namespace(uri) {
            if !namespaces.contains_key(&ns) {
                let label = suggest_prefix(&ns, namespaces.len());
                namespaces.insert(ns, label);
            }
        }
    }
    let suggested_prefixes = namespaces.iter()
        .map(|(ns, p)| format!("PREFIX {p}: <{ns}>"))
        .collect::<Vec<_>>()
        .join("\n");

    // 4. Schema summary listing the sampled vocabulary
    let mut schema = String::new();
    if !classes.is_empty() {
        schema.push_str("Classes (sampled):\n");
        for c in classes.iter().take(40) { schema.push_str(&format!("- {c}\n")); }
    }
    if !props.is_empty() {
        schema.push_str("\nProperties (sampled):\n");
        for p in props.iter().take(60) { schema.push_str(&format!("- {p}\n")); }
    }

    DiscoveryResult {
        live: true,
        message: format!("Endpoint is live. Sampled {} classes and {} properties — review and edit before saving.", classes.len(), props.len()),
        suggested_prefixes,
        suggested_schema: schema,
    }
}

/// Run a SELECT and collect the string values bound to `var`.
async fn run_select_var(
    client: &reqwest::Client,
    endpoint_url: &str,
    auth: &AuthConfig,
    query: &str,
    var: &str,
) -> Result<Vec<String>, String> {
    let base_req = client
        .post(endpoint_url)
        .header("Accept", "application/sparql-results+json")
        .header("User-Agent", concat!("LexiChat/", env!("CARGO_PKG_VERSION")))
        .form(&[("query", query)]);
    let req = auth.apply_async(client, base_req).await;
    let resp = req.send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    let json: Value = resp.json().await.map_err(|e| e.to_string())?;
    let bindings = json["results"]["bindings"].as_array().ok_or("no bindings")?;
    Ok(bindings.iter()
        .filter_map(|b| b[var]["value"].as_str().map(String::from))
        .collect())
}

/// Split a URI into (namespace, local) at the last `#` or `/`.
fn split_namespace(uri: &str) -> Option<(String, String)> {
    let idx = uri.rfind('#').or_else(|| uri.rfind('/'))?;
    let (ns, local) = uri.split_at(idx + 1);
    if ns.is_empty() || local.is_empty() { return None; }
    Some((ns.to_string(), local.to_string()))
}

/// Map well-known namespaces to their conventional prefix; otherwise generate one.
fn suggest_prefix(ns: &str, n: usize) -> String {
    let known: &[(&str, &str)] = &[
        ("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf"),
        ("http://www.w3.org/2000/01/rdf-schema#", "rdfs"),
        ("http://www.w3.org/2002/07/owl#", "owl"),
        ("http://www.w3.org/2001/XMLSchema#", "xsd"),
        ("http://www.w3.org/2004/02/skos/core#", "skos"),
        ("http://xmlns.com/foaf/0.1/", "foaf"),
        ("http://purl.org/dc/terms/", "dct"),
        ("http://purl.org/dc/elements/1.1/", "dc"),
    ];
    for (k, v) in known {
        if ns == *k { return (*v).to_string(); }
    }
    format!("ns{n}")
}

/// Update in-memory AppState with a refreshed OAuth2 token and notify the frontend.
async fn persist_refreshed_token(app: &tauri::AppHandle, ep_id: &str, new_token: &str) {
    use tauri::Manager;
    let state = app.state::<crate::AppState>();
    {
        let mut eps = state.sparql_endpoints.lock().unwrap();
        for ep in eps.iter_mut() {
            if ep.id == ep_id {
                if let crate::mcp::AuthConfig::OAuth2 { ref mut access_token, .. } = ep.auth {
                    *access_token = new_token.to_string();
                }
                break;
            }
        }
    }
    use tauri::Emitter;
    let _ = app.emit("openapi-token-refreshed", serde_json::json!({
        "spec_id": ep_id,
        "access_token": new_token,
    }));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(title: &str) -> RegisteredSparqlEndpoint {
        RegisteredSparqlEndpoint {
            id: "id1".into(),
            title: title.into(),
            endpoint_url: "https://example.org/sparql".into(),
            prefixes: "PREFIX ex: <https://example.org/>".into(),
            schema_summary: "Class ex:Thing with property ex:name".into(),
            example_queries: vec![ExampleQuery { label: "all".into(), query: "SELECT * WHERE {?s ?p ?o} LIMIT 1".into() }],
            usage_hint: "test data".into(),
            auth: AuthConfig::default(),
            read_only: true,
            tools: vec![],
        }
    }

    #[test]
    fn build_tools_makes_query_and_schema() {
        let ep = endpoint("HM Land Registry");
        let tools = build_tools(&ep);
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "hm_land_registry_query");
        assert_eq!(tools[1].name, "hm_land_registry_schema");
    }

    #[test]
    fn query_tool_requires_query_param() {
        let ep = endpoint("Land Registry");
        let tools = build_tools(&ep);
        let params = &tools[0].schema["function"]["parameters"];
        assert_eq!(params["required"][0], "query");
        assert!(params["properties"]["query"].is_object());
    }

    #[test]
    fn schema_tool_takes_no_params() {
        let ep = endpoint("Land Registry");
        let tools = build_tools(&ep);
        let props = &tools[1].schema["function"]["parameters"]["properties"];
        assert_eq!(props.as_object().unwrap().len(), 0);
    }

    #[test]
    fn query_description_embeds_prefix_and_example() {
        let ep = endpoint("Land Registry");
        let tools = build_tools(&ep);
        let desc = tools[0].schema["function"]["description"].as_str().unwrap();
        assert!(desc.contains("PREFIX ex:"));
        assert!(desc.contains("Example"));
    }

    #[test]
    fn tool_name_capped_at_64() {
        let ep = endpoint(&"x".repeat(80));
        let tools = build_tools(&ep);
        assert!(tools[0].name.len() <= 64);
    }

    #[test]
    fn read_only_guard_blocks_updates() {
        assert_eq!(forbidden_keyword("DELETE WHERE { ?s ?p ?o }"), Some("DELETE".into()));
        assert_eq!(forbidden_keyword("INSERT DATA { <a> <b> <c> }"), Some("INSERT".into()));
        assert_eq!(forbidden_keyword("DROP GRAPH <g>"), Some("DROP".into()));
    }

    #[test]
    fn read_only_guard_allows_reads() {
        assert_eq!(forbidden_keyword("SELECT * WHERE { ?s ?p ?o } LIMIT 10"), None);
        assert_eq!(forbidden_keyword("ASK { ?s a ?c }"), None);
        assert_eq!(forbidden_keyword("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"), None);
    }

    #[test]
    fn format_results_flattens_select() {
        let body = serde_json::json!({
            "head": { "vars": ["s", "label"] },
            "results": { "bindings": [
                { "s": { "type": "uri", "value": "https://ex/1" }, "label": { "type": "literal", "value": "One" } },
                { "s": { "type": "uri", "value": "https://ex/2" }, "label": { "type": "literal", "value": "Two" } }
            ] }
        }).to_string();
        let out = format_results(200, &body);
        assert!(out.contains("2 row(s)"));
        assert!(out.contains("https://ex/1 | One"));
        assert!(out.contains("Columns: s, label"));
    }

    #[test]
    fn format_results_handles_ask() {
        let body = serde_json::json!({ "head": {}, "boolean": true }).to_string();
        let out = format_results(200, &body);
        assert!(out.contains("boolean: true"));
    }

    #[test]
    fn format_results_passes_through_non_json() {
        let out = format_results(200, "@prefix ex: <https://ex/> .");
        assert!(out.contains("@prefix"));
    }

    #[test]
    fn split_namespace_hash_and_slash() {
        assert_eq!(split_namespace("http://x/ns#Thing"), Some(("http://x/ns#".into(), "Thing".into())));
        assert_eq!(split_namespace("http://x/ns/Thing"), Some(("http://x/ns/".into(), "Thing".into())));
    }

    #[test]
    fn suggest_prefix_known_and_generated() {
        assert_eq!(suggest_prefix("http://www.w3.org/2000/01/rdf-schema#", 3), "rdfs");
        assert_eq!(suggest_prefix("https://novel.example/ns#", 2), "ns2");
    }

    // ── HTTP integration (wiremock) ───────────────────────────────────────────

    fn endpoint_at(url: &str) -> RegisteredSparqlEndpoint {
        RegisteredSparqlEndpoint {
            id: "id".into(), title: "Test".into(), endpoint_url: url.into(),
            prefixes: String::new(), schema_summary: String::new(),
            example_queries: vec![], usage_hint: String::new(),
            auth: AuthConfig::default(), read_only: true, tools: vec![],
        }
    }

    #[tokio::test]
    async fn execute_flattens_select_results() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::method;
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "head": { "vars": ["s", "label"] },
            "results": { "bindings": [
                { "s": { "type": "uri", "value": "https://ex/1" }, "label": { "type": "literal", "value": "One" } }
            ] }
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server).await;

        let ep = endpoint_at(&server.uri());
        let out = execute(&ep, "SELECT ?s ?label WHERE { ?s ?p ?label } LIMIT 1", None).await;
        assert!(out.contains("1 row(s)"), "got: {out}");
        assert!(out.contains("https://ex/1 | One"), "got: {out}");
    }

    #[tokio::test]
    async fn execute_rejects_update_without_calling_server() {
        // read_only guard must reject before any HTTP call — an unroutable URL proves it.
        let ep = endpoint_at("http://127.0.0.1:9/never");
        let out = execute(&ep, "DELETE WHERE { ?s ?p ?o }", None).await;
        assert!(out.starts_with("Rejected"), "got: {out}");
    }

    #[tokio::test]
    async fn probe_reports_live_and_derives_prefixes() {
        use wiremock::{MockServer, Mock, ResponseTemplate};
        use wiremock::matchers::method;
        let server = MockServer::start().await;
        // One response satisfies the liveness (?s), class (?c) and property (?p) probes.
        let body = serde_json::json!({
            "head": { "vars": ["s", "c", "p"] },
            "results": { "bindings": [{
                "s": { "type": "uri", "value": "https://ex/1" },
                "c": { "type": "uri", "value": "http://www.w3.org/2000/01/rdf-schema#Class" },
                "p": { "type": "uri", "value": "https://ex/prop#name" }
            }] }
        });
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server).await;

        let res = probe(&server.uri(), &AuthConfig::default()).await;
        assert!(res.live);
        assert!(res.suggested_prefixes.contains("rdfs"), "prefixes: {}", res.suggested_prefixes);
        assert!(res.suggested_schema.contains("Class"), "schema: {}", res.suggested_schema);
    }
}
