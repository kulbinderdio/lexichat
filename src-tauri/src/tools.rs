use serde_json::Value;
use std::path::Path;
use std::fs;

/// Dispatch a built-in tool call. Returns the result string.
pub async fn dispatch_builtin(name: &str, args: &Value, allowed_dirs: &[String]) -> String {
    match name {
        "read_file"          => read_file(args, allowed_dirs),
        "write_file"         => write_file(args, allowed_dirs),
        "list_files"         => list_files(args, allowed_dirs),
        "search_files"       => search_files(args, allowed_dirs),
        "search_in_files"    => search_in_files(args, allowed_dirs),
        "get_file_info"      => get_file_info(args, allowed_dirs),
        "list_directory_tree"=> list_directory_tree(args, allowed_dirs),
        "create_directory"   => create_directory(args, allowed_dirs),
        "move_file"          => move_file(args, allowed_dirs),
        "delete_file"        => delete_file(args, allowed_dirs),
        "find_old_files"     => find_old_files(args, allowed_dirs),
        "web_search"         => web_search(args).await,
        "compose_email"      => compose_email(args),
        _                    => format!("Unknown tool: {name}"),
    }
}

// ── Sandbox check ─────────────────────────────────────────────────────────────

fn check_path(path: &str, allowed_dirs: &[String]) -> Result<(), String> {
    if allowed_dirs.is_empty() {
        return Ok(());
    }
    // Try to canonicalize (resolves symlinks, "..", etc.)
    // On Windows canonicalize() adds a \\?\ prefix — we must canonicalize
    // both sides so the comparison is apples-to-apples.
    let canonical = std::fs::canonicalize(path)
        .unwrap_or_else(|_| {
            // File may not exist yet (e.g. write_file). Walk parent chain.
            let p = std::path::Path::new(path);
            let mut cur = p;
            loop {
                if cur.exists() {
                    return std::fs::canonicalize(cur).unwrap_or_else(|_| cur.to_path_buf());
                }
                match cur.parent() {
                    Some(parent) => cur = parent,
                    None => return p.to_path_buf(),
                }
            }
        });

    let is_allowed = allowed_dirs.iter().any(|dir| {
        // Canonicalize the allowed dir too so \\?\ prefixes match on Windows
        let canonical_dir = std::fs::canonicalize(dir)
            .unwrap_or_else(|_| std::path::PathBuf::from(dir));
        // Path::starts_with does component-level matching (avoids /foo/bar matching /foo/barbaz)
        canonical.starts_with(&canonical_dir)
    });

    if is_allowed {
        Ok(())
    } else {
        Err(format!(
            "Access denied: '{}' is outside the allowed folders. \
            You MUST use an absolute path within one of these allowed folders: {}. \
            Do not use relative paths like '.' or '~'.",
            path,
            allowed_dirs.join(", ")
        ))
    }
}

// ── File tools ────────────────────────────────────────────────────────────────

fn read_file(args: &Value, allowed_dirs: &[String]) -> String {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let limit  = args["limit"].as_u64().map(|v| v as usize);

    let lower = path.to_lowercase();
    if lower.ends_with(".pdf")  { return read_pdf(path, offset, limit); }
    if lower.ends_with(".docx") { return read_docx(path, offset, limit); }

    // Detect image extensions before attempting a text read
    let image_exts = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".heic", ".avif"];
    if image_exts.iter().any(|ext| lower.ends_with(ext)) {
        return format!(
            "[Image file — cannot read as text. \
            To describe or analyse this image you must ask the user to attach it \
            via the paperclip button and send it to you directly. \
            Do NOT call read_file on image files.]"
        );
    }

    match fs::read_to_string(path) {
        Ok(content) => slice_lines(&content, offset, limit),
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            format!("[Binary file — cannot read as text. Use get_file_info to inspect metadata.]")
        }
        Err(e) => format!("Error reading '{path}': {e}"),
    }
}

fn read_pdf(path: &str, offset: usize, limit: Option<usize>) -> String {
    match pdf_extract::extract_text(path) {
        Ok(text) if !text.trim().is_empty() => slice_lines(&text, offset, limit),
        Ok(_) => format!("[PDF '{path}' contains no extractable text — likely a scanned/image PDF. Try describing it with a vision model instead.]"),
        Err(e) => format!("Error extracting text from PDF '{path}': {e}"),
    }
}

fn read_docx(path: &str, offset: usize, limit: Option<usize>) -> String {
    use std::io::Read;
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return format!("Error opening '{path}': {e}"),
    };
    let mut archive = match zip::ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) => return format!("Error reading DOCX '{path}': {e}"),
    };
    let xml = match archive.by_name("word/document.xml") {
        Ok(mut entry) => {
            let mut s = String::new();
            if entry.read_to_string(&mut s).is_ok() { s }
            else { return format!("Could not decode word/document.xml in '{path}'"); }
        }
        Err(_) => return format!("Invalid DOCX: missing document.xml in '{path}'"),
    };
    let text = extract_docx_text(&xml);
    if text.trim().is_empty() {
        return format!("[DOCX '{path}' contains no extractable text]");
    }
    slice_lines(&text, offset, limit)
}

/// Extract plain text from DOCX word/document.xml by scanning <w:t> and <w:p> elements.
fn extract_docx_text(xml: &str) -> String {
    let mut out = String::new();
    let mut rest = xml;
    loop {
        // Find next interesting tag
        let para_pos = rest.find("<w:p").filter(|&p| {
            rest.as_bytes().get(p + 4).map(|&b| b == b'>' || b == b' ').unwrap_or(false)
        });
        let wt_pos = rest.find("<w:t");
        let br_pos = rest.find("<w:br");

        match (para_pos, wt_pos, br_pos) {
            (None, None, None) => break,
            _ => {
                // Pick the earliest tag
                let min_pos = [para_pos, wt_pos, br_pos]
                    .iter().filter_map(|&p| p).min().unwrap();

                if Some(min_pos) == wt_pos {
                    // Extract text content of <w:t ...>...</w:t>
                    rest = &rest[min_pos + 4..];
                    if let Some(gt) = rest.find('>') {
                        rest = &rest[gt + 1..];
                        if let Some(end) = rest.find("</w:t>") {
                            out.push_str(&rest[..end]);
                            rest = &rest[end + 6..];
                        }
                    }
                } else if Some(min_pos) == br_pos {
                    out.push('\n');
                    rest = &rest[min_pos + 5..];
                } else {
                    // paragraph — add newline between paragraphs
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push('\n');
                    }
                    rest = &rest[min_pos + 4..];
                }
            }
        }
    }
    out
}

fn slice_lines(content: &str, offset: usize, limit: Option<usize>) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.saturating_sub(1);
    let slice = if let Some(lim) = limit {
        &lines[start.min(lines.len())..((start + lim).min(lines.len()))]
    } else {
        &lines[start.min(lines.len())..]
    };
    if slice.is_empty() { "(empty)".into() } else { slice.join("\n") }
}

fn write_file(args: &Value, allowed_dirs: &[String]) -> String {
    let path    = args["path"].as_str().unwrap_or("");
    let content = args["content"].as_str().unwrap_or("");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    if let Some(parent) = Path::new(path).parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return format!("Error creating directories: {e}");
        }
    }
    let lower = path.to_lowercase();
    if lower.ends_with(".docx") { return write_docx(path, content); }
    if lower.ends_with(".pdf")  { return write_pdf(path, content); }
    match fs::write(path, content) {
        Ok(_)  => format!("Written {} bytes to '{path}'", content.len()),
        Err(e) => format!("Error writing '{path}': {e}"),
    }
}

fn write_pdf(path: &str, content: &str) -> String {
    use printpdf::*;
    use std::io::BufWriter;

    const PAGE_W: f32 = 210.0; // A4 mm
    const PAGE_H: f32 = 297.0;
    const MARGIN_X: f32 = 20.0;
    const MARGIN_Y: f32 = 22.0;
    const FONT_SIZE: f32 = 11.0;
    const LINE_H: f32 = 6.2; // mm per line

    // Helvetica 11pt: available width ≈ 170mm ≈ 481pt, avg char ≈ 5.8pt → ~83 chars
    // Use 80 to be safe with wider characters (W, M, etc.)
    const MAX_CHARS: usize = 80;

    // Word-wrap all lines to fit within the printable width
    let wrapped: Vec<String> = content
        .lines()
        .flat_map(|line| pdf_wrap(line, MAX_CHARS))
        .collect();

    let (doc, first_page, first_layer) =
        PdfDocument::new("Document", Mm(PAGE_W), Mm(PAGE_H), "text");

    let font = match doc.add_builtin_font(BuiltinFont::Helvetica) {
        Ok(f) => f,
        Err(e) => return format!("PDF font error: {e}"),
    };

    let mut current_page  = first_page;
    let mut current_layer = first_layer;
    let mut y: f32 = PAGE_H - MARGIN_Y;

    for line in &wrapped {
        if y < MARGIN_Y + LINE_H {
            let (new_page, new_layer) = doc.add_page(Mm(PAGE_W), Mm(PAGE_H), "text");
            current_page  = new_page;
            current_layer = new_layer;
            y = PAGE_H - MARGIN_Y;
        }
        if !line.is_empty() {
            doc.get_page(current_page)
               .get_layer(current_layer)
               .use_text(line.as_str(), FONT_SIZE, Mm(MARGIN_X), Mm(y), &font);
        }
        y -= LINE_H;
    }

    let file = match fs::File::create(path) {
        Ok(f) => f,
        Err(e) => return format!("Error creating '{path}': {e}"),
    };
    match doc.save(&mut BufWriter::new(file)) {
        Ok(_)  => format!("Written PDF '{path}' ({} lines, {} pages)",
                          wrapped.len(),
                          1 + wrapped.len() * LINE_H as usize / (PAGE_H - MARGIN_Y * 2.0) as usize),
        Err(e) => format!("Error writing PDF '{path}': {e}"),
    }
}

/// Word-wrap a single line into segments of at most `max_chars` characters.
fn pdf_wrap(line: &str, max_chars: usize) -> Vec<String> {
    if line.is_empty() {
        return vec![String::new()];
    }
    // Detect leading indent (spaces/dashes/bullets) to preserve on continuation lines
    let indent_len = line.len() - line.trim_start_matches(|c: char| c == ' ' || c == '\t').len();
    let indent = if indent_len > 0 && indent_len < 8 {
        "  ".to_string() // continuation indent
    } else {
        String::new()
    };

    let mut segments = Vec::new();
    let mut current = String::new();

    for word in line.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            segments.push(current);
            current = format!("{indent}{word}");
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    if segments.is_empty() {
        segments.push(String::new());
    }
    segments
}

fn write_docx(path: &str, content: &str) -> String {
    use std::io::Write as IoWrite;
    use zip::write::SimpleFileOptions;

    let file = match fs::File::create(path) {
        Ok(f) => f,
        Err(e) => return format!("Error creating '{path}': {e}"),
    };
    let mut zip = zip::ZipWriter::new(file);
    let opts = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Build paragraph XML from content lines
    let paragraphs: String = content.lines().map(|line| {
        let escaped = line
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;");
        format!(r#"<w:p><w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>"#)
    }).collect::<Vec<_>>().join("");

    let document_xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{paragraphs}<w:sectPr/></w:body>
</w:document>"#
    );

    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"#;

    let rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#;

    let word_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"#;

    let mut ok = true;
    ok &= zip.start_file("[Content_Types].xml", opts).is_ok();
    ok &= zip.write_all(content_types.as_bytes()).is_ok();
    ok &= zip.start_file("_rels/.rels", opts).is_ok();
    ok &= zip.write_all(rels.as_bytes()).is_ok();
    ok &= zip.start_file("word/_rels/document.xml.rels", opts).is_ok();
    ok &= zip.write_all(word_rels.as_bytes()).is_ok();
    ok &= zip.start_file("word/document.xml", opts).is_ok();
    ok &= zip.write_all(document_xml.as_bytes()).is_ok();

    if !ok || zip.finish().is_err() {
        return format!("Error writing DOCX '{path}'");
    }
    format!("Written DOCX '{path}' ({} chars)", content.len())
}

fn list_files(args: &Value, allowed_dirs: &[String]) -> String {
    let path = args["path"].as_str().unwrap_or(".");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    match fs::read_dir(path) {
        Ok(entries) => {
            let mut items: Vec<String> = entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    if is_dir { format!("{name}/") } else { name }
                })
                .collect();
            items.sort();
            if items.is_empty() { "(empty directory)".into() } else { items.join("\n") }
        }
        Err(e) => format!("Error listing '{path}': {e}"),
    }
}

fn search_files(args: &Value, allowed_dirs: &[String]) -> String {
    let pattern   = args["pattern"].as_str().unwrap_or("*");
    let directory = args["directory"].as_str().unwrap_or(".");
    if let Err(e) = check_path(directory, allowed_dirs) { return e; }
    let mut results = Vec::new();
    walk_dir(Path::new(directory), &mut |path| {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if glob_matches(&name, pattern) {
            results.push(path.to_string_lossy().to_string());
        }
    }, 0, 10);
    if results.is_empty() {
        format!("No files matching '{pattern}' found in '{directory}'")
    } else {
        results.join("\n")
    }
}

fn search_in_files(args: &Value, allowed_dirs: &[String]) -> String {
    let query        = args["query"].as_str().unwrap_or("").to_lowercase();
    let directory    = args["directory"].as_str().unwrap_or(".");
    let file_pattern = args["file_pattern"].as_str();
    if let Err(e) = check_path(directory, allowed_dirs) { return e; }
    let mut results  = Vec::new();

    walk_dir(Path::new(directory), &mut |path| {
        if path.is_dir() { return; }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if let Some(pat) = file_pattern {
            if !glob_matches(&name, pat) { return; }
        }
        if let Ok(content) = fs::read_to_string(path) {
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&query) {
                    results.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                    if results.len() >= 50 { return; }
                }
            }
        }
    }, 0, 8);

    if results.is_empty() {
        format!("No matches for '{}' found in '{}'", query, directory)
    } else {
        results.join("\n")
    }
}

fn get_file_info(args: &Value, allowed_dirs: &[String]) -> String {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    match fs::metadata(path) {
        Ok(meta) => {
            let size    = meta.len();
            let is_dir  = meta.is_dir();
            let modified = meta.modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| format!("{}", d.as_secs()))
                .unwrap_or_default();
            format!("path: {path}\ntype: {}\nsize: {size} bytes\nmodified: {modified}",
                    if is_dir { "directory" } else { "file" })
        }
        Err(e) => format!("Error: {e}"),
    }
}

fn list_directory_tree(args: &Value, allowed_dirs: &[String]) -> String {
    let path      = args["path"].as_str().unwrap_or(".");
    let max_depth = args["max_depth"].as_u64().unwrap_or(3).min(6) as usize;
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    let mut out   = Vec::new();
    tree_recurse(Path::new(path), 0, max_depth, "", &mut out);
    out.join("\n")
}

fn tree_recurse(dir: &Path, depth: usize, max_depth: usize, prefix: &str, out: &mut Vec<String>) {
    if depth > max_depth { return; }
    let entries = match fs::read_dir(dir) {
        Ok(e) => { let mut v: Vec<_> = e.filter_map(|x| x.ok()).collect(); v.sort_by_key(|e| e.file_name()); v }
        Err(_) => return,
    };
    for (i, entry) in entries.iter().enumerate() {
        let is_last = i == entries.len() - 1;
        let connector = if is_last { "└── " } else { "├── " };
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        out.push(format!("{prefix}{connector}{name}{}", if is_dir { "/" } else { "" }));
        if is_dir && depth < max_depth {
            let new_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });
            tree_recurse(&entry.path(), depth + 1, max_depth, &new_prefix, out);
        }
    }
}

fn create_directory(args: &Value, allowed_dirs: &[String]) -> String {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    match fs::create_dir_all(path) {
        Ok(_)  => format!("Created directory '{path}'"),
        Err(e) => format!("Error: {e}"),
    }
}

fn move_file(args: &Value, allowed_dirs: &[String]) -> String {
    let src  = args["source"].as_str().unwrap_or("");
    let dest = args["destination"].as_str().unwrap_or("");
    if let Err(e) = check_path(src, allowed_dirs) { return e; }
    if let Err(e) = check_path(dest, allowed_dirs) { return e; }
    match fs::rename(src, dest) {
        Ok(_)  => format!("Moved '{src}' → '{dest}'"),
        Err(e) => format!("Error: {e}"),
    }
}

fn delete_file(args: &Value, allowed_dirs: &[String]) -> String {
    let path = args["path"].as_str().unwrap_or("");
    if let Err(e) = check_path(path, allowed_dirs) { return e; }
    if Path::new(path).is_dir() {
        return "Cannot delete directories with this tool.".into();
    }
    match fs::remove_file(path) {
        Ok(_)  => format!("Deleted '{path}'"),
        Err(e) => format!("Error: {e}"),
    }
}

fn find_old_files(args: &Value, allowed_dirs: &[String]) -> String {
    let directory      = args["directory"].as_str().unwrap_or(".");
    let older_than_days = args["older_than_days"].as_u64().unwrap_or(30);
    let pattern        = args["pattern"].as_str();
    if let Err(e) = check_path(directory, allowed_dirs) { return e; }
    let cutoff_secs    = older_than_days * 86400;
    let now            = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut results = Vec::new();
    walk_dir(Path::new(directory), &mut |path| {
        if path.is_dir() { return; }
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if let Some(pat) = pattern {
            if !glob_matches(&name, pat) { return; }
        }
        if let Ok(meta) = fs::metadata(path) {
            if let Ok(modified) = meta.modified() {
                if let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                    if now.saturating_sub(dur.as_secs()) > cutoff_secs {
                        results.push(format!("{} ({} days old)", path.display(),
                            now.saturating_sub(dur.as_secs()) / 86400));
                    }
                }
            }
        }
    }, 0, 6);

    if results.is_empty() {
        format!("No files older than {older_than_days} days found in '{directory}'")
    } else {
        results.join("\n")
    }
}

// ── Email composition ─────────────────────────────────────────────────────────

/// Build a base64url-encoded RFC 2822 email string ready for the Gmail API `raw` field.
fn compose_email(args: &Value) -> String {
    let to      = args["to"].as_str().unwrap_or("");
    let from    = args["from"].as_str().unwrap_or("");
    let subject = args["subject"].as_str().unwrap_or("");
    let body    = args["body"].as_str().unwrap_or("");
    let reply_to_id = args["reply_to_message_id"].as_str().unwrap_or("");

    if to.is_empty() {
        return "Error: 'to' is required".into();
    }

    let mut msg = String::new();
    if !from.is_empty()    { msg.push_str(&format!("From: {from}\r\n")); }
    msg.push_str(&format!("To: {to}\r\n"));
    msg.push_str(&format!("Subject: {subject}\r\n"));
    if !reply_to_id.is_empty() {
        msg.push_str(&format!("In-Reply-To: {reply_to_id}\r\n"));
        msg.push_str(&format!("References: {reply_to_id}\r\n"));
    }
    msg.push_str("MIME-Version: 1.0\r\n");
    msg.push_str("Content-Type: text/plain; charset=UTF-8\r\n");
    msg.push_str("Content-Transfer-Encoding: quoted-printable\r\n");
    msg.push_str("\r\n");
    msg.push_str(body);

    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    let encoded = URL_SAFE_NO_PAD.encode(msg.as_bytes());
    format!("base64url encoded email (pass this as the 'raw' field to gmail_sendmessage):\n{encoded}")
}

// ── Web search ────────────────────────────────────────────────────────────────

async fn web_search(args: &Value) -> String {
    let query = args["query"].as_str().unwrap_or("");
    if query.is_empty() { return "No query provided".into(); }

    let client = match reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("Search error: {e}"),
    };

    // 1. Try DDG Instant Answer API first for factual queries (fast, no scraping)
    let instant_url = format!(
        "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
        urlencoding::encode(query)
    );
    if let Ok(resp) = client.get(&instant_url).send().await {
        if let Ok(json) = resp.json::<Value>().await {
            let mut parts = Vec::new();
            if let Some(t) = json["AbstractText"].as_str() {
                if !t.is_empty() { parts.push(format!("Summary: {t}")); }
            }
            if let Some(a) = json["Answer"].as_str() {
                if !a.is_empty() { parts.push(format!("Answer: {a}")); }
            }
            if let Some(url) = json["AbstractURL"].as_str() {
                if !url.is_empty() { parts.push(format!("Source: {url}")); }
            }
            if let Some(related) = json["RelatedTopics"].as_array() {
                let topics: Vec<_> = related.iter().take(3)
                    .filter_map(|t| {
                        let text = t["Text"].as_str()?;
                        let url  = t["FirstURL"].as_str().unwrap_or("");
                        if url.is_empty() { Some(text.to_string()) }
                        else { Some(format!("{text}\n   {url}")) }
                    })
                    .collect();
                if !topics.is_empty() { parts.push(format!("Related:\n{}", topics.join("\n\n"))); }
            }
            if !parts.is_empty() {
                return parts.join("\n\n");
            }
        }
    }

    // 2. Fall back to DDG HTML search for real web results
    let html_url = format!(
        "https://html.duckduckgo.com/html/?q={}",
        urlencoding::encode(query)
    );
    match client.get(&html_url).send().await {
        Ok(resp) => {
            let html = match resp.text().await {
                Ok(h) => h,
                Err(e) => return format!("Search error: {e}"),
            };
            let results = extract_ddg_results(&html);
            if results.is_empty() {
                format!("No results found for: {query}")
            } else {
                format!(
                    "Search results for '{query}':\n\n{}",
                    results.iter().enumerate()
                        .map(|(i, (title, snippet, url))| {
                            format!("{}. {}\n   {}\n   {}", i + 1, title, snippet, url)
                        })
                        .collect::<Vec<_>>()
                        .join("\n\n")
                )
            }
        }
        Err(e) => format!("Search error: {e}"),
    }
}

/// Decode a DuckDuckGo redirect href into a real destination URL.
/// DDG wraps all outbound links as: //duckduckgo.com/l/?uddg=<url-encoded-url>&...
fn decode_ddg_href(href: &str) -> String {
    if let Some(pos) = href.find("uddg=") {
        let encoded = &href[pos + 5..];
        let end = encoded.find('&').unwrap_or(encoded.len());
        if let Ok(decoded) = urlencoding::decode(&encoded[..end]) {
            let s = decoded.into_owned();
            if s.starts_with("http") { return s; }
        }
    }
    // Already a plain URL
    if href.starts_with("http") { return href.to_string(); }
    String::new()
}

/// Extract (title, snippet, url) tuples from DuckDuckGo HTML results page.
fn extract_ddg_results(html: &str) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut pos = 0;

    while results.len() < 6 {
        // Find next result block by locating result__a (title link)
        let Some(title_idx) = html[pos..].find("class=\"result__a\"") else { break };
        let block_start = pos + title_idx;

        // Extract href from the title anchor
        let href = {
            // Search backwards a bit for href=
            let search_back = if block_start > 200 { block_start - 200 } else { 0 };
            let chunk = &html[search_back..block_start + 50];
            extract_attr(chunk, "href").unwrap_or_default()
        };

        // Extract title text
        let title = {
            let after = block_start + 17; // skip past class="result__a"
            let Some(tag_end) = html[after..].find('>') else { pos = block_start + 1; continue };
            let text_start = after + tag_end + 1;
            let Some(close) = html[text_start..].find("</a>") else { pos = block_start + 1; continue };
            decode_html_entities(&strip_tags(&html[text_start..text_start + close]))
        };

        // Extract snippet — find result__snippet after current position
        let snippet = {
            let search_from = block_start;
            let window = 2000.min(html.len().saturating_sub(search_from));
            if let Some(snip_idx) = html[search_from..search_from + window].find("class=\"result__snippet\"") {
                let snip_start = search_from + snip_idx + 22;
                if let Some(tag_end) = html[snip_start..].find('>') {
                    let text_start = snip_start + tag_end + 1;
                    let close = html[text_start..].find("</a>").unwrap_or(300.min(html.len().saturating_sub(text_start)));
                    decode_html_entities(&strip_tags(&html[text_start..text_start + close]))
                } else { String::new() }
            } else {
                String::new()
            }
        };

        pos = block_start + 1;

        let title = title.trim().to_string();
        if title.is_empty() { continue; }
        let url = decode_ddg_href(&href);
        if url.is_empty() { continue; }
        results.push((title, snippet.trim().to_string(), url));
    }
    results
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn decode_html_entities(s: &str) -> String {
    s.replace("&amp;", "&")
     .replace("&lt;", "<")
     .replace("&gt;", ">")
     .replace("&quot;", "\"")
     .replace("&#x27;", "'")
     .replace("&#39;", "'")
     .replace("&nbsp;", " ")
     .replace("&mdash;", "—")
     .replace("&ndash;", "–")
}

fn extract_attr(html: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = html.find(&needle)? + needle.len();
    let end = html[start..].find('"')? + start;
    Some(html[start..end].to_string())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn walk_dir(dir: &Path, f: &mut impl FnMut(&Path), depth: usize, max_depth: usize) {
    if depth > max_depth { return; }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        f(&path);
        if path.is_dir() {
            walk_dir(&path, f, depth + 1, max_depth);
        }
    }
}

fn glob_matches(s: &str, pattern: &str) -> bool {
    // Simple glob: * matches any sequence, ? matches one char
    let s = s.as_bytes();
    let p = pattern.as_bytes();
    let mut si = 0usize;
    let mut pi = 0usize;
    let mut star_pi = usize::MAX;
    let mut star_si = 0usize;
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == s[si]) {
            si += 1; pi += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_pi = pi; star_si = si; pi += 1;
        } else if star_pi != usize::MAX {
            star_si += 1; si = star_si; pi = star_pi + 1;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' { pi += 1; }
    pi == p.len()
}

// ── Schema definitions ────────────────────────────────────────────────────────

pub fn all_builtin_schemas() -> Vec<serde_json::Value> {
    vec![
        schema("list_files", "List files and folders at a path on this computer.",
            vec![("path","string","Directory path to list. Omit or use '.' for current directory.")], vec![]),
        schema("read_file", "Read a local file. Supports plain text, PDF, and DOCX (Word) files.",
            vec![("path","string","Absolute file path. PDFs and DOCX files have their text extracted automatically."),
                 ("offset","integer","Line number to start reading from (1-based, optional)."),
                 ("limit","integer","Max lines to return (optional).")], vec!["path"]),
        schema("write_file", "Create or overwrite a local file. Supports plain text (.txt, .md, etc.), PDF (.pdf), and Word (.docx).",
            vec![("path","string","Absolute file path. Extension determines format: .pdf, .docx, or any text extension."),
                 ("content","string","Text content to write.")], vec!["path","content"]),
        schema("search_files", "Find files by name pattern (glob). Use * to match any characters.",
            vec![("pattern","string","Glob pattern e.g. '*.txt', 'report_*'."),
                 ("directory","string","Directory to search in (optional, default '.').")], vec!["pattern"]),
        schema("search_in_files", "Search for text inside files. Returns matching lines.",
            vec![("query","string","Text to search for."),
                 ("directory","string","Directory to search in (optional)."),
                 ("file_pattern","string","Only search files matching this glob, e.g. '*.py' (optional).")], vec!["query"]),
        schema("get_file_info", "Get metadata for a file or directory: size, type, modification date.",
            vec![("path","string","Path to the file or directory.")], vec!["path"]),
        schema("list_directory_tree", "Show a recursive directory tree.",
            vec![("path","string","Root directory path."),
                 ("max_depth","integer","Depth limit (default 3, max 6).")], vec!["path"]),
        schema("create_directory", "Create a new directory (and any missing parents).",
            vec![("path","string","Directory path to create.")], vec!["path"]),
        schema("move_file", "Move or rename a file or directory.",
            vec![("source","string","Source path."),
                 ("destination","string","Destination path.")], vec!["source","destination"]),
        schema("delete_file", "Delete a file permanently.",
            vec![("path","string","File path to delete.")], vec!["path"]),
        schema("find_old_files", "Find files not modified in N days.",
            vec![("directory","string","Directory to search."),
                 ("older_than_days","integer","Return files older than this many days."),
                 ("pattern","string","Optional filename glob filter.")], vec!["directory","older_than_days"]),
        schema("web_search", "Search the web for current information.",
            vec![("query","string","Search query (2-6 words work best).")], vec!["query"]),
        schema("compose_email", "Build a base64url-encoded RFC 2822 email string for use with the Gmail API sendMessage tool. Call this first, then pass the result as the 'raw' field to gmail_sendmessage.",
            vec![("to","string","Recipient email address(es), comma-separated."),
                 ("from","string","Sender email address (optional, Gmail will use the account address if omitted)."),
                 ("subject","string","Email subject line."),
                 ("body","string","Plain text email body."),
                 ("reply_to_message_id","string","Message-ID to reply to, for threading (optional).")],
            vec!["to","subject","body"]),
    ]
}

fn schema(name: &str, desc: &str, params: Vec<(&str,&str,&str)>, required: Vec<&str>) -> serde_json::Value {
    let props: serde_json::Map<String,Value> = params.iter().map(|(pname, ptype, pdesc)| {
        (pname.to_string(), serde_json::json!({ "type": ptype, "description": pdesc }))
    }).collect();
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required
            }
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── strip_tags ────────────────────────────────────────────────────────────

    #[test]
    fn strip_tags_removes_html() {
        assert_eq!(strip_tags("<b>hello</b>"), "hello");
    }

    #[test]
    fn strip_tags_plain_text_unchanged() {
        assert_eq!(strip_tags("no tags here"), "no tags here");
    }

    #[test]
    fn strip_tags_nested() {
        assert_eq!(strip_tags("<a href=\"x\"><b>text</b></a>"), "text");
    }

    // ── decode_html_entities ──────────────────────────────────────────────────

    #[test]
    fn decode_amp() { assert_eq!(decode_html_entities("a &amp; b"), "a & b"); }

    #[test]
    fn decode_lt_gt() { assert_eq!(decode_html_entities("&lt;div&gt;"), "<div>"); }

    #[test]
    fn decode_quot() { assert_eq!(decode_html_entities("say &quot;hi&quot;"), "say \"hi\""); }

    #[test]
    fn decode_nbsp() { assert_eq!(decode_html_entities("a&nbsp;b"), "a b"); }

    #[test]
    fn decode_apostrophe() {
        assert_eq!(decode_html_entities("it&#x27;s"), "it's");
        assert_eq!(decode_html_entities("it&#39;s"), "it's");
    }

    #[test]
    fn decode_no_entities_unchanged() {
        assert_eq!(decode_html_entities("plain text"), "plain text");
    }

    // ── extract_attr ──────────────────────────────────────────────────────────

    #[test]
    fn extract_attr_finds_href() {
        let html = r#"<a href="https://example.com">link</a>"#;
        assert_eq!(extract_attr(html, "href"), Some("https://example.com".into()));
    }

    #[test]
    fn extract_attr_missing_returns_none() {
        let html = r#"<a class="foo">link</a>"#;
        assert_eq!(extract_attr(html, "href"), None);
    }

    // ── extract_ddg_results ───────────────────────────────────────────────────

    #[test]
    fn extract_ddg_results_parses_title_and_snippet() {
        let html = r##"
            <a href="https://example.com" class="result__a">Example Title</a>
            <a class="result__snippet" href="https://example.com">This is the snippet text</a>
        "##;
        let results = extract_ddg_results(html);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Example Title");
        assert!(results[0].1.contains("snippet text"));
    }

    #[test]
    fn extract_ddg_results_empty_html() {
        let results = extract_ddg_results("<html><body></body></html>");
        assert!(results.is_empty());
    }

    #[test]
    fn extract_ddg_results_caps_at_six() {
        let block = |i: usize| format!(
            "<a href=\"https://example.com/{i}\" class=\"result__a\">Title {i}</a>\
             <a class=\"result__snippet\" href=\"https://example.com/{i}\">Snippet {i}</a>"
        );
        let html: String = (0..10).map(block).collect();
        let results = extract_ddg_results(&html);
        assert!(results.len() <= 6);
    }

    fn tmp() -> TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    // ── check_path ────────────────────────────────────────────────────────────

    #[test]
    fn check_path_allowed_when_no_restrictions() {
        assert!(check_path("/any/path", &[]).is_ok());
    }

    // Canonicalize the temp dir so symlinks (e.g. /var → /private/var on macOS) are resolved,
    // matching what check_path does internally.
    fn canonical_dir_str(dir: &TempDir) -> String {
        fs::canonicalize(dir.path()).unwrap().to_string_lossy().to_string()
    }

    #[test]
    fn check_path_allowed_within_dir() {
        let dir = tmp();
        let file = dir.path().join("test.txt");
        fs::write(&file, "hi").unwrap();
        let allowed = vec![canonical_dir_str(&dir)];
        assert!(check_path(file.to_str().unwrap(), &allowed).is_ok());
    }

    #[test]
    fn check_path_denied_outside_allowed() {
        let dir = tmp();
        let allowed = vec![canonical_dir_str(&dir)];
        assert!(check_path("/etc/passwd", &allowed).is_err());
    }

    #[test]
    fn check_path_denied_traversal_attempt() {
        let dir = tmp();
        let inner = dir.path().join("inner");
        fs::create_dir(&inner).unwrap();
        let allowed = vec![fs::canonicalize(&inner).unwrap().to_string_lossy().to_string()];
        // Try to escape via ../
        let escape = format!("{}/../../etc/passwd", inner.to_string_lossy());
        assert!(check_path(&escape, &allowed).is_err());
    }

    #[test]
    fn check_path_nonexistent_file_uses_parent() {
        let dir = tmp();
        let allowed = vec![canonical_dir_str(&dir)];
        let nonexistent = dir.path().join("new_file.txt");
        // File doesn't exist yet but parent is allowed
        assert!(check_path(nonexistent.to_str().unwrap(), &allowed).is_ok());
    }

    // ── glob_matches ──────────────────────────────────────────────────────────

    #[test]
    fn glob_star_matches_any_sequence() {
        assert!(glob_matches("hello.txt", "*.txt"));
        assert!(glob_matches("report_2024.csv", "report_*"));
    }

    #[test]
    fn glob_star_matches_empty() {
        assert!(glob_matches("file", "file*"));
    }

    #[test]
    fn glob_question_matches_single_char() {
        assert!(glob_matches("file1.txt", "file?.txt"));
        assert!(!glob_matches("file12.txt", "file?.txt"));
    }

    #[test]
    fn glob_exact_match() {
        assert!(glob_matches("exact", "exact"));
        assert!(!glob_matches("exact", "other"));
    }

    #[test]
    fn glob_star_star_not_special() {
        // Our glob doesn't support ** — * matches any flat sequence including slashes
        assert!(glob_matches("a/b/c.txt", "*.txt"));
    }

    #[test]
    fn glob_no_match() {
        assert!(!glob_matches("hello.rs", "*.txt"));
        assert!(!glob_matches("", "?.txt"));
    }

    #[test]
    fn glob_empty_pattern_matches_empty_string() {
        assert!(glob_matches("", ""));
    }

    #[test]
    fn glob_only_stars() {
        assert!(glob_matches("anything", "*"));
        assert!(glob_matches("", "*"));
    }

    // ── all_builtin_schemas ───────────────────────────────────────────────────

    #[test]
    fn all_builtin_schemas_returns_expected_count() {
        let schemas = all_builtin_schemas();
        assert_eq!(schemas.len(), 13);
    }

    #[test]
    fn all_builtin_schemas_have_required_fields() {
        for schema in all_builtin_schemas() {
            let func = &schema["function"];
            assert!(func["name"].as_str().is_some(), "Missing name in schema");
            assert!(func["description"].as_str().is_some(), "Missing description in schema");
            assert!(func["parameters"].is_object(), "Missing parameters in schema");
        }
    }

    #[test]
    fn all_builtin_schemas_names_are_valid_tool_names() {
        for schema in all_builtin_schemas() {
            let name = schema["function"]["name"].as_str().unwrap();
            assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                "Invalid tool name: {name}");
            assert!(!name.starts_with('_'), "Tool name starts with _: {name}");
            assert!(!name.ends_with('_'), "Tool name ends with _: {name}");
            assert!(name.len() <= 64, "Tool name too long: {name}");
        }
    }
}
