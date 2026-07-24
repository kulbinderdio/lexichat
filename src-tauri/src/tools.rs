use serde_json::Value;
use std::path::Path;
use std::fs;

/// Dispatch a built-in tool call. Returns the result string.
pub async fn dispatch_builtin(name: &str, args: &Value, allowed_dirs: &[String], web_search_results: usize) -> String {
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
        "web_search"         => web_search(args, web_search_results).await,
        "fetch_webpage"      => fetch_webpage(args).await,
        // Some models call this instead of web_search — treat as an alias.
        // The schema uses "queries" (array); we take the first element.
        "google:search" | "google_search" => {
            let query = args["queries"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .or_else(|| args["query"].as_str())
                .unwrap_or("");
            let patched = serde_json::json!({ "query": query });
            web_search(&patched, web_search_results).await
        },
        "compose_email"         => compose_email(args),
        "get_current_datetime"  => get_current_datetime(),
        // Wiki memory tools
        "wiki_list"             => crate::wiki::wiki_list(),
        "wiki_search"           => crate::wiki::wiki_search(args),
        "wiki_read"             => crate::wiki::wiki_read(args),
        "wiki_write"            => crate::wiki::wiki_write(args),
        "wiki_patch"            => crate::wiki::wiki_patch(args),
        "wiki_delete"           => crate::wiki::wiki_delete(args),
        "wiki_append"           => crate::wiki::wiki_append(args),
        "wiki_lint"             => crate::wiki::wiki_lint(),
        _                       => format!("Unknown tool: {name}"),
    }
}

// ── Sandbox check ─────────────────────────────────────────────────────────────

pub(crate) fn check_path(path: &str, allowed_dirs: &[String]) -> Result<(), String> {
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

/// Full extracted text of a PDF or DOCX for staging into the code sandbox (Pyodide has no
/// PDF/Word parser). Returns None for other file types or on failure — the caller then stages
/// the raw bytes instead. No truncation.
pub fn extract_document_text(path: &str) -> Option<String> {
    let lower = path.to_lowercase();
    let text = if lower.ends_with(".pdf") {
        pdf_extract::extract_text(path).ok()?
    } else if lower.ends_with(".docx") {
        let t = read_docx(path, 0, None);
        // read_docx returns an error/notice string on failure — drop those.
        if t.starts_with("Error") || t.starts_with("Invalid") || t.starts_with("Could not") || t.starts_with('[') {
            return None;
        }
        t
    } else {
        return None;
    };
    if text.trim().is_empty() { None } else { Some(text) }
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

fn pdf_draw_line(layer: &printpdf::PdfLayerReference, x1: f32, y1: f32, x2: f32, y2: f32) {
    use printpdf::*;
    layer.set_outline_color(Color::Rgb(Rgb::new(0.35, 0.35, 0.35, None)));
    layer.set_outline_thickness(0.4);
    layer.add_line(Line {
        points: vec![
            (Point::new(Mm(x1), Mm(y1)), false),
            (Point::new(Mm(x2), Mm(y2)), false),
        ],
        is_closed: false,
    });
}

fn write_pdf(path: &str, content: &str) -> String {
    use printpdf::*;
    use std::io::BufWriter;

    // The built-in PDF font renders WinAnsi/Latin-1 only; transliterate anything
    // outside it (arrows, Greek, subscripts, math) to ASCII so it isn't dropped.
    let content = pdf_safe_text(content);
    let content = content.as_str();

    const W: f32 = 210.0_f32;
    const H: f32 = 297.0_f32;
    const ML: f32 = 22.0_f32;
    const MT: f32 = 22.0_f32;
    const MB: f32 = 20.0_f32;

    let (doc, mut pg, mut ly) = PdfDocument::new("Document", Mm(W), Mm(H), "text");
    let fn_ = match doc.add_builtin_font(BuiltinFont::Helvetica)    { Ok(f) => f, Err(e) => return format!("PDF font error: {e}") };
    let fb  = match doc.add_builtin_font(BuiltinFont::HelveticaBold) { Ok(f) => f, Err(e) => return format!("PDF font error: {e}") };
    let fc  = match doc.add_builtin_font(BuiltinFont::Courier)       { Ok(f) => f, Err(e) => return format!("PDF font error: {e}") };

    let mut y: f32 = H - MT;
    let mut pages: usize = 1;
    let mut total_lines: usize = 0;
    let mut in_code = false;
    // Buffered table rows: Vec<(is_header, cells)>
    let mut table_buf: Vec<(bool, Vec<String>)> = Vec::new();

    // Helper: advance y, starting a new page at the bottom margin.
    macro_rules! adv {
        ($h:expr) => {{
            y -= $h as f32;
            if y < MB {
                let (p, l) = doc.add_page(Mm(W), Mm(H), "text");
                pg = p; ly = l; y = H - MT; pages += 1;
            }
        }};
    }
    macro_rules! put {
        ($text:expr, $x:expr, $sz:expr, $font:expr) => {
            if !($text as &str).is_empty() {
                doc.get_page(pg).get_layer(ly).use_text($text as &str, $sz as f32, Mm($x as f32), Mm(y), &$font);
            }
        };
    }

    // Flush buffered table rows to PDF with borders.
    // Uses a macro so it can access pg/ly/y/pages/doc in scope.
    macro_rules! flush_table {
        () => {
            if !table_buf.is_empty() {
                let avail: f32 = W - ML - 22.0;
                let n_cols = table_buf.iter().map(|(_, r)| r.len()).max().unwrap_or(1).max(1);

                // Proportional column widths based on max cell char count
                let col_widths: Vec<usize> = (0..n_cols).map(|c| {
                    table_buf.iter()
                        .map(|(_, r)| r.get(c).map(|s| s.chars().count()).unwrap_or(0))
                        .max().unwrap_or(1).max(3)
                }).collect();
                let total_w: usize = col_widths.iter().sum::<usize>().max(1);

                // Column left-edge x positions (mm) and char wrap limits
                // Helvetica 10pt ≈ 0.55 * 10 * 0.353 mm/char ≈ 1.94 mm/char; use 2.0 conservatively
                let mut col_x: Vec<f32> = Vec::with_capacity(n_cols);
                let mut col_maxc: Vec<usize> = Vec::with_capacity(n_cols);
                let mut cx = ML;
                for &cw in &col_widths {
                    let mm = (avail * cw as f32 / total_w as f32).max(18.0);
                    col_x.push(cx);
                    let pad = 3.0_f32; // 1.5mm padding each side
                    col_maxc.push(((mm - pad).max(8.0) / 2.0_f32 as f32) as usize);
                    cx += mm;
                }
                let table_right = cx;

                adv!(3.0); // space before table

                for (is_hdr, cells) in table_buf.iter() {
                    let font = if *is_hdr { &fb } else { &fn_ };
                    let size: f32 = if *is_hdr { 10.5 } else { 10.0 };
                    let line_h: f32 = 5.5; // mm per text line
                    let pad_top: f32 = 4.5; // baseline sits 4.5mm below top border (cap height ~2.5mm leaves 2mm clearance)
                    let pad_bot: f32 = 2.0;

                    let cell_lines: Vec<Vec<String>> = (0..n_cols).map(|c| {
                        let cell = cells.get(c).map(String::as_str).unwrap_or("");
                        pdf_word_wrap(cell, *col_maxc.get(c).unwrap_or(&20))
                    }).collect();
                    let row_h_lines = cell_lines.iter().map(|l| l.len()).max().unwrap_or(1);
                    let row_h: f32 = row_h_lines as f32 * line_h + pad_top + pad_bot;

                    if y - row_h < MB {
                        let (p, l) = doc.add_page(Mm(W), Mm(H), "text");
                        pg = p; ly = l; y = H - MT; pages += 1;
                    }

                    let row_top = y;
                    let row_bot = y - row_h;

                    // Top border of this row (also serves as table top for first row)
                    pdf_draw_line(&doc.get_page(pg).get_layer(ly), ML, row_top, table_right, row_top);

                    // Cell text, 1.5mm left padding, pad_top below top border
                    for (ci, lines) in cell_lines.iter().enumerate() {
                        let cx = col_x.get(ci).copied().unwrap_or(ML);
                        let mut cy = row_top - pad_top;
                        for line in lines {
                            if !line.is_empty() {
                                doc.get_page(pg).get_layer(ly)
                                   .use_text(line.as_str(), size, Mm(cx + 1.5), Mm(cy), font);
                            }
                            cy -= line_h;
                        }
                    }

                    // Bottom border
                    pdf_draw_line(&doc.get_page(pg).get_layer(ly), ML, row_bot, table_right, row_bot);
                    // Left, column dividers, right border
                    pdf_draw_line(&doc.get_page(pg).get_layer(ly), ML, row_top, ML, row_bot);
                    for &vx in col_x.iter().skip(1) {
                        pdf_draw_line(&doc.get_page(pg).get_layer(ly), vx, row_top, vx, row_bot);
                    }
                    pdf_draw_line(&doc.get_page(pg).get_layer(ly), table_right, row_top, table_right, row_bot);

                    total_lines += row_h_lines;
                    y = row_bot;
                }
                adv!(4.0); // space after table
                table_buf.clear();
            }
        };
    }

    // Returns true if `line` is a Markdown table row (starts/ends with | or contains |)
    let is_table_row = |line: &str| -> bool {
        let t = line.trim();
        t.starts_with('|') && t.len() > 1
    };
    // Returns true if `line` is a separator row like |---|---|
    let is_table_sep = |line: &str| -> bool {
        line.trim().trim_matches('|').split('|')
            .all(|c| c.trim().chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
    };
    // Parse cells from a | cell | cell | row
    let parse_row = |line: &str| -> Vec<String> {
        let t = line.trim().trim_matches('|');
        t.split('|').map(|c| md_strip_inline(c.trim())).collect()
    };

    for raw in content.lines() {
        // ── Code fence ───────────────────────────────────────────────────────
        if raw.trim_start().starts_with("```") {
            flush_table!();
            in_code = !in_code;
            adv!(if in_code { 2.0 } else { 3.0 });
            continue;
        }
        if in_code {
            flush_table!();
            for part in pdf_word_wrap(raw, 90) {
                put!(&part, ML, 9.5, fc);
                adv!(5.5);
                total_lines += 1;
            }
            continue;
        }

        // ── Table rows ───────────────────────────────────────────────────────
        if is_table_row(raw) {
            if is_table_sep(raw) {
                // Mark the previous row as header if this is the first separator
                if let Some(last) = table_buf.last_mut() {
                    last.0 = true;
                }
            } else {
                table_buf.push((false, parse_row(raw)));
            }
            continue;
        }
        // Non-table line: flush any buffered table first
        flush_table!();

        // ── Headings ─────────────────────────────────────────────────────────
        let heading = if let Some(r) = raw.strip_prefix("# ")    { Some((r, 22.0_f32, 11.5_f32, 40_usize, 5.0_f32)) }
                 else if let Some(r) = raw.strip_prefix("## ")   { Some((r, 17.0_f32,  9.0_f32, 52_usize, 4.0_f32)) }
                 else if let Some(r) = raw.strip_prefix("### ")  { Some((r, 13.5_f32,  7.5_f32, 65_usize, 3.0_f32)) }
                 else if let Some(r) = raw.strip_prefix("#### ") { Some((r, 11.5_f32,  6.5_f32, 75_usize, 2.0_f32)) }
                 else { None };
        if let Some((rest, size, lh, max_c, before)) = heading {
            adv!(before);
            let text = md_strip_inline(rest);
            for part in pdf_word_wrap(&text, max_c) {
                put!(&part, ML, size, fb);
                adv!(lh);
                total_lines += 1;
            }
            adv!(2.0);
            continue;
        }

        // ── Horizontal rule ───────────────────────────────────────────────────
        if matches!(raw.trim(), "---" | "***" | "___" | "- - -") {
            adv!(3.0);
            put!("-----------------------------------------------------------------------", ML, 8.0, fn_);
            adv!(5.0);
            continue;
        }

        // ── Blank line ────────────────────────────────────────────────────────
        if raw.trim().is_empty() { adv!(3.5); continue; }

        // ── Blockquote ────────────────────────────────────────────────────────
        if let Some(rest) = raw.strip_prefix("> ") {
            let text = md_strip_inline(rest);
            for part in pdf_word_wrap(&text, 76) {
                let prefixed = format!("  | {part}");
                put!(&prefixed, ML, 10.5, fn_);
                adv!(6.0);
                total_lines += 1;
            }
            continue;
        }

        // ── Bullet list ───────────────────────────────────────────────────────
        let bullet = raw.strip_prefix("    - ").or_else(|| raw.strip_prefix("    * ")).map(|r| (r, ML + 8.0_f32))
            .or_else(|| raw.strip_prefix("  - ").or_else(|| raw.strip_prefix("  * ")).map(|r| (r, ML + 4.0_f32)))
            .or_else(|| raw.strip_prefix("- ").or_else(|| raw.strip_prefix("* ").or_else(|| raw.strip_prefix("+ "))).map(|r| (r, ML)));
        if let Some((rest, bx)) = bullet {
            let tx = bx + 5.0;
            let text = md_strip_inline(rest);
            let max_c = (76.0_f32 * (W - ML - 20.0 - (tx - ML)) / (W - ML - 20.0)) as usize;
            put!("-", bx, 11.0, fn_);
            let parts = pdf_word_wrap(&text, max_c.max(30));
            for (i, part) in parts.iter().enumerate() {
                if i > 0 { adv!(6.0); }
                doc.get_page(pg).get_layer(ly).use_text(part.as_str(), 11.0, Mm(tx), Mm(y), &fn_);
                total_lines += 1;
            }
            adv!(6.0);
            continue;
        }

        // ── Numbered list ─────────────────────────────────────────────────────
        let numbered = {
            let t = raw.trim_start();
            t.find(". ").and_then(|dot| {
                if dot > 0 && dot <= 3 && t[..dot].chars().all(|c| c.is_ascii_digit()) {
                    Some((&t[..dot+2], t[dot+2..].trim()))
                } else { None }
            })
        };
        if let Some((marker, rest)) = numbered {
            let tx = ML + 8.0;
            put!(marker, ML, 11.0, fn_);
            for (i, part) in pdf_word_wrap(&md_strip_inline(rest), 74).iter().enumerate() {
                if i > 0 { adv!(6.0); }
                doc.get_page(pg).get_layer(ly).use_text(part.as_str(), 11.0, Mm(tx), Mm(y), &fn_);
                total_lines += 1;
            }
            adv!(6.0);
            continue;
        }

        // ── Normal / bold line ────────────────────────────────────────────────
        let trimmed = raw.trim();
        let (text, font) =
            if (trimmed.starts_with("**") && trimmed.ends_with("**") && trimmed.len() > 4)
            || (trimmed.starts_with("__") && trimmed.ends_with("__") && trimmed.len() > 4) {
                (md_strip_inline(&trimmed[2..trimmed.len()-2]), true)
            } else {
                (md_strip_inline(trimmed), false)
            };
        for part in pdf_word_wrap(&text, 82) {
            if font { put!(&part, ML, 11.0, fb); } else { put!(&part, ML, 11.0, fn_); }
            adv!(6.0);
            total_lines += 1;
        }
    }

    flush_table!(); // flush any table at end of document

    let file = match fs::File::create(path) {
        Ok(f) => f,
        Err(e) => return format!("Error creating '{path}': {e}"),
    };
    match doc.save(&mut BufWriter::new(file)) {
        Ok(_)  => format!("Written PDF '{path}' ({total_lines} lines, {pages} page(s))"),
        Err(e) => format!("Error writing PDF '{path}': {e}"),
    }
}

/// Map characters the built-in PDF font (WinAnsi/Latin-1) can't render into safe
/// ASCII equivalents, so scientific text (arrows, Greek letters, subscript digits,
/// math symbols) isn't silently dropped from the page. Characters already in
/// Latin-1 are kept as-is; unknown non-Latin-1 glyphs are dropped (rare).
fn pdf_safe_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let c = ch as u32;
        if c < 0x100 { out.push(ch); continue; } // ASCII + Latin-1 render fine
        match ch {
            '→' => out.push_str("->"),
            '←' => out.push_str("<-"),
            '↔' => out.push_str("<->"),
            '⇒' => out.push_str("=>"),
            '−' => out.push('-'),
            '≤' => out.push_str("<="),
            '≥' => out.push_str(">="),
            '≠' => out.push_str("!="),
            '≈' => out.push('~'),
            '…' => out.push_str("..."),
            '•' | '●' | '▪' | '◦' => out.push('-'),
            'α' => out.push_str("alpha"),
            'β' => out.push_str("beta"),
            'γ' => out.push_str("gamma"),
            'δ' => out.push_str("delta"),
            'μ' => out.push('u'),
            '\u{2080}'..='\u{2089}' => out.push(char::from(b'0' + (c - 0x2080) as u8)), // ₀-₉
            '⁰' => out.push('0'),
            '\u{2074}'..='\u{2079}' => out.push(char::from(b'0' + (c - 0x2070) as u8)), // ⁴-⁹
            '\u{2009}' | '\u{200a}' | '\u{202f}' | '\u{2007}' => out.push(' '), // thin/nbsp spaces
            _ => {} // unknown glyph: drop
        }
    }
    out
}

/// Word-wrap text into segments of at most `max_chars` characters. A single token
/// longer than the line width is hard-broken so it can't overflow the page/cell.
fn pdf_word_wrap(text: &str, max_chars: usize) -> Vec<String> {
    if text.is_empty() { return vec![String::new()]; }
    let max_chars = max_chars.max(1);
    if text.chars().count() <= max_chars { return vec![text.to_string()]; }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if word.chars().count() > max_chars {
            // Hard-break an over-long token into fixed-width chunks.
            if !current.is_empty() { lines.push(std::mem::take(&mut current)); }
            let chars: Vec<char> = word.chars().collect();
            for chunk in chars.chunks(max_chars) {
                lines.push(chunk.iter().collect());
            }
            current = lines.pop().unwrap_or_default(); // keep last chunk open for following words
        } else if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= max_chars {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::replace(&mut current, word.to_string()));
        }
    }
    if !current.is_empty() { lines.push(current); }
    if lines.is_empty() { lines.push(String::new()); }
    lines
}

/// Remove common inline Markdown markers, returning clean display text.
fn md_strip_inline(s: &str) -> String {
    // Remove paired bold/italic markers (**..** __..__ *.* _._ `..`)
    let mut r = s.trim().to_string();
    for marker in &["**", "__", "*", "`"] {
        let m = *marker;
        let ml = m.len();
        loop {
            if let Some(start) = r.find(m) {
                if let Some(rel) = r[start + ml..].find(m) {
                    let inner = r[start + ml .. start + ml + rel].to_string();
                    let tail  = r[start + ml + rel + ml..].to_string();
                    r = format!("{}{}{}", &r[..start], inner, tail);
                } else { break; }
            } else { break; }
        }
    }
    // Convert simple [text](url) links → just the text
    loop {
        if let Some(ob) = r.find('[') {
            if let Some(rel_cb) = r[ob..].find("](") {
                let cb = ob + rel_cb;
                let text = r[ob+1..cb].to_string();
                if let Some(rel_cp) = r[cb+2..].find(')') {
                    let cp = cb + 2 + rel_cp;
                    r = format!("{}{}{}", &r[..ob], text, &r[cp+1..]);
                    continue;
                }
            }
        }
        break;
    }
    r.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
}

// ── DOCX markdown rendering helpers ──────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

struct InlineRun { text: String, bold: bool, italic: bool, code: bool }

fn flush_run(buf: &mut String, runs: &mut Vec<InlineRun>, bold: bool, italic: bool, code: bool) {
    if !buf.is_empty() {
        runs.push(InlineRun { text: std::mem::take(buf), bold, italic, code });
    }
}

// Parse inline markdown (* ** *** ` ) into styled runs.
// Deliberately ignores _ / __ to avoid false positives in snake_case identifiers.
fn parse_inline(text: &str) -> Vec<InlineRun> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut runs: Vec<InlineRun> = Vec::new();
    let mut buf = String::new();
    let mut bold = false;
    let mut italic = false;
    let mut code = false;
    let mut i = 0;

    while i < n {
        if code {
            if chars[i] == '`' {
                flush_run(&mut buf, &mut runs, bold, italic, true);
                code = false;
            } else {
                buf.push(chars[i]);
            }
            i += 1;
            continue;
        }
        if chars[i] == '*' {
            let mut count = 0;
            while i < n && chars[i] == '*' { count += 1; i += 1; }
            flush_run(&mut buf, &mut runs, bold, italic, code);
            match count {
                1 => italic = !italic,
                2 => bold = !bold,
                _ => { bold = !bold; italic = !italic; }
            }
        } else if chars[i] == '`' {
            flush_run(&mut buf, &mut runs, bold, italic, code);
            code = true;
            i += 1;
        } else {
            buf.push(chars[i]);
            i += 1;
        }
    }
    flush_run(&mut buf, &mut runs, bold, italic, code);
    runs
}

fn inline_runs_xml(text: &str) -> String {
    let mut out = String::new();
    for run in parse_inline(text) {
        if run.text.is_empty() { continue; }
        let escaped = xml_escape(&run.text);
        let mut rpr = String::new();
        if run.bold   { rpr.push_str("<w:b/>"); }
        if run.italic { rpr.push_str("<w:i/>"); }
        if run.code   { rpr.push_str(r#"<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="20"/>"#); }
        if rpr.is_empty() {
            out.push_str(&format!(r#"<w:r><w:t xml:space="preserve">{escaped}</w:t></w:r>"#));
        } else {
            out.push_str(&format!(r#"<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{escaped}</w:t></w:r>"#));
        }
    }
    out
}

fn heading_para(text: &str, style: &str) -> String {
    let runs = inline_runs_xml(text);
    format!(r#"<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{runs}</w:p>"#)
}

fn is_numbered_item(s: &str) -> bool {
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    i > 0 && b.get(i) == Some(&b'.') && b.get(i + 1) == Some(&b' ')
}

fn md_to_docx_body(content: &str) -> String {
    let mut out = String::new();
    let mut in_code_block = false;

    for line in content.lines() {
        let s = line.trim();

        // Fenced code block toggle
        if s.starts_with("```") {
            in_code_block = !in_code_block;
            // Emit an empty para when closing to add spacing
            if !in_code_block {
                out.push_str(r#"<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>"#);
            }
            continue;
        }

        if in_code_block {
            let escaped = xml_escape(line); // preserve original indentation
            out.push_str(&format!(
                r#"<w:p><w:pPr><w:pStyle w:val="CodeBlock"/></w:pPr><w:r><w:t xml:space="preserve">{escaped}</w:t></w:r></w:p>"#
            ));
            continue;
        }

        // Headings (check most hashes first)
        if let Some(t) = s.strip_prefix("#### ").or_else(|| s.strip_prefix("##### ")).or_else(|| s.strip_prefix("###### ")) {
            out.push_str(&heading_para(t, "Heading3"));
        } else if let Some(t) = s.strip_prefix("### ") {
            out.push_str(&heading_para(t, "Heading3"));
        } else if let Some(t) = s.strip_prefix("## ") {
            out.push_str(&heading_para(t, "Heading2"));
        } else if let Some(t) = s.strip_prefix("# ") {
            out.push_str(&heading_para(t, "Heading1"));
        }
        // Horizontal rule
        else if s == "---" || s == "***" || s == "___" || s == "- - -" || s == "* * *" {
            out.push_str(r#"<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" w:color="auto"/></w:pBdr></w:pPr></w:p>"#);
        }
        // Blockquote
        else if s.starts_with('>') {
            let t = s[1..].trim_start_matches(' ');
            let runs = inline_runs_xml(t);
            out.push_str(&format!(
                r#"<w:p><w:pPr><w:ind w:left="720"/></w:pPr>{runs}</w:p>"#
            ));
        }
        // Bullet list
        else if let Some(t) = s.strip_prefix("- ").or_else(|| s.strip_prefix("* ")).or_else(|| s.strip_prefix("+ ")) {
            let runs = inline_runs_xml(t);
            let bullet = xml_escape("•");
            out.push_str(&format!(
                r#"<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr><w:r><w:t xml:space="preserve">{bullet}   </w:t></w:r>{runs}</w:p>"#
            ));
        }
        // Numbered list
        else if is_numbered_item(s) {
            let runs = inline_runs_xml(s);
            out.push_str(&format!(
                r#"<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>{runs}</w:p>"#
            ));
        }
        // Empty line
        else if s.is_empty() {
            out.push_str(r#"<w:p><w:pPr><w:spacing w:after="60"/></w:pPr></w:p>"#);
        }
        // Normal paragraph
        else {
            let runs = inline_runs_xml(s);
            out.push_str(&format!(r#"<w:p>{runs}</w:p>"#));
        }
    }
    out
}

fn docx_styles() -> &'static str {
    r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="240" w:after="60"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="40"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="200" w:after="60"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="32"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:pPr><w:spacing w:before="160" w:after="60"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="CodeBlock">
    <w:name w:val="Code Block"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr><w:ind w:left="720"/><w:spacing w:before="0" w:after="0"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="20"/></w:rPr>
  </w:style>
</w:styles>"#
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

    let body = md_to_docx_body(content);
    let document_xml = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body}<w:sectPr/></w:body>
</w:document>"#
    );

    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"#;

    let rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"#;

    let word_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"#;

    let mut ok = true;
    ok &= zip.start_file("[Content_Types].xml", opts).is_ok();
    ok &= zip.write_all(content_types.as_bytes()).is_ok();
    ok &= zip.start_file("_rels/.rels", opts).is_ok();
    ok &= zip.write_all(rels.as_bytes()).is_ok();
    ok &= zip.start_file("word/_rels/document.xml.rels", opts).is_ok();
    ok &= zip.write_all(word_rels.as_bytes()).is_ok();
    ok &= zip.start_file("word/styles.xml", opts).is_ok();
    ok &= zip.write_all(docx_styles().as_bytes()).is_ok();
    ok &= zip.start_file("word/document.xml", opts).is_ok();
    ok &= zip.write_all(document_xml.as_bytes()).is_ok();

    if !ok || zip.finish().is_err() {
        return format!("Error writing DOCX '{path}'");
    }
    format!("Written DOCX '{path}' ({} chars)", content.len())
}

/// Save content to path, routing by extension (.docx / .pdf / anything else → plain text).
/// No sandbox check — caller is responsible for using a user-chosen path.
pub fn save_document(path: &str, content: &str) -> Result<(), String> {
    let lower = path.to_lowercase();
    let result = if lower.ends_with(".docx") {
        write_docx(path, content)
    } else if lower.ends_with(".pdf") {
        write_pdf(path, content)
    } else if lower.ends_with(".html") || lower.ends_with(".htm") {
        // Styled, self-contained HTML report (the "artifact" house style) instead of raw markdown.
        match fs::write(path, crate::report::render_report_html(content, None, None, &[])) {
            Ok(_) => "ok".to_string(),
            Err(e) => format!("Error: {e}"),
        }
    } else {
        match fs::write(path, content) {
            Ok(_)  => format!("ok"),
            Err(e) => format!("Error: {e}"),
        }
    };
    if result.starts_with("Error") { Err(result) } else { Ok(()) }
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
    let to          = args["to"].as_str().unwrap_or("");
    let from        = args["from"].as_str().unwrap_or("");
    let subject     = args["subject"].as_str().unwrap_or("");
    let body        = args["body"].as_str().unwrap_or("");
    let reply_to_id = args["reply_to_message_id"].as_str().unwrap_or("");

    if to.is_empty() {
        return "Error: 'to' is required".into();
    }

    use base64::{Engine as _, engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD}};

    // Base64-encode the body so any UTF-8 content and long lines are handled correctly
    let body_b64 = STANDARD.encode(body.as_bytes());

    let mut msg = String::new();
    if !from.is_empty() { msg.push_str(&format!("From: {from}\r\n")); }
    msg.push_str(&format!("To: {to}\r\n"));
    msg.push_str(&format!("Subject: {subject}\r\n"));
    if !reply_to_id.is_empty() {
        msg.push_str(&format!("In-Reply-To: {reply_to_id}\r\n"));
        msg.push_str(&format!("References: {reply_to_id}\r\n"));
    }
    msg.push_str("MIME-Version: 1.0\r\n");
    msg.push_str("Content-Type: text/plain; charset=UTF-8\r\n");
    msg.push_str("Content-Transfer-Encoding: base64\r\n");
    msg.push_str("\r\n");
    msg.push_str(&body_b64);

    // Outer envelope: base64url (no padding) for the Gmail API 'raw' field
    URL_SAFE_NO_PAD.encode(msg.as_bytes())
}

// ── Date / time ───────────────────────────────────────────────────────────────

fn get_current_datetime() -> String {
    let now = chrono::Local::now();
    format!(
        "Current date and time:\n\
         - Human readable: {}\n\
         - ISO 8601: {}\n\
         - Filename-safe: {}\n\
         - Day: {}\n\
         - Unix timestamp: {}",
        now.format("%A, %d %B %Y at %H:%M:%S"),
        now.format("%Y-%m-%dT%H:%M:%S%z"),
        now.format("%Y-%m-%d_%H-%M"),
        now.format("%A"),
        now.timestamp(),
    )
}

// ── Web search ────────────────────────────────────────────────────────────────

async fn web_search(args: &Value, max_results: usize) -> String {
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
            let results = extract_ddg_results(&html, max_results);
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

/// How `fetch_webpage` should treat a response based on its Content-Type.
#[derive(Debug, PartialEq)]
enum FetchKind { Pdf, Html, Binary }

/// Classify a Content-Type so fetch_webpage can extract PDFs, parse HTML/text,
/// and avoid decoding binary files (images, archives, …) as garbage "HTML".
fn classify_content_type(ctype: &str) -> FetchKind {
    let c = ctype.to_ascii_lowercase();
    if c.contains("pdf") {
        FetchKind::Pdf
    } else if c.is_empty()
        || c.starts_with("text/")
        || c.contains("html")
        || c.contains("xml")
        || c.contains("json")
        || c.contains("javascript")
    {
        FetchKind::Html
    } else {
        FetchKind::Binary
    }
}

async fn fetch_webpage(args: &Value) -> String {
    let url = args["url"].as_str().unwrap_or("");
    if url.is_empty() { return "No URL provided".into(); }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return format!("Invalid URL '{url}': must start with http:// or https://");
    }

    use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, CACHE_CONTROL, CONTENT_TYPE, REFERER, UPGRADE_INSECURE_REQUESTS};

    let mut headers = HeaderMap::new();
    headers.insert(ACCEPT, HeaderValue::from_static(
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    ));
    headers.insert(ACCEPT_LANGUAGE, HeaderValue::from_static("en-US,en;q=0.9"));
    headers.insert(CACHE_CONTROL,   HeaderValue::from_static("max-age=0"));
    headers.insert(UPGRADE_INSECURE_REQUESTS, HeaderValue::from_static("1"));
    // Simulate arriving from a Google search — many sites (BBC, Guardian, etc.)
    // allow more content to apparent search-engine referrals.
    headers.insert(REFERER, HeaderValue::from_static("https://www.google.com/"));
    headers.insert("Sec-Fetch-Dest", HeaderValue::from_static("document"));
    headers.insert("Sec-Fetch-Mode", HeaderValue::from_static("navigate"));
    headers.insert("Sec-Fetch-Site", HeaderValue::from_static("cross-site"));
    headers.insert("Sec-Ch-Ua", HeaderValue::from_static(
        r#""Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99""#,
    ));
    headers.insert("Sec-Ch-Ua-Mobile",   HeaderValue::from_static("?0"));
    headers.insert("Sec-Ch-Ua-Platform", HeaderValue::from_static("\"macOS\""));

    let client = match reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        .default_headers(headers)
        .cookie_store(true)
        .timeout(std::time::Duration::from_secs(20))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("Error building HTTP client: {e}"),
    };

    let resp = match client.get(url).send().await {
        Ok(r) => r,
        Err(e) => return format!("Error fetching '{url}': {e}"),
    };

    let status = resp.status();
    let ctype = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Route by content type: extract PDFs, refuse other binaries, else parse HTML.
    match classify_content_type(&ctype) {
        FetchKind::Pdf => {
            let bytes = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => return format!("Error downloading PDF from '{url}': {e}"),
            };
            return match pdf_extract::extract_text_from_mem(&bytes) {
                Ok(t) if !t.trim().is_empty() => format!("[PDF document at {url} — extracted text]\n\n{t}"),
                Ok(_) => format!("[The document at '{url}' is a PDF with no extractable text — likely a scanned/image-only PDF.]"),
                Err(e) => format!("[Could not extract text from the PDF at '{url}': {e}]"),
            };
        }
        FetchKind::Binary => {
            let kind = if ctype.is_empty() { "unknown type".to_string() } else { ctype };
            return format!(
                "[The URL '{url}' returned non-text content ({kind}), which fetch_webpage cannot read. \
                 It is a binary file (image, archive, etc.) — skip it or find an HTML/text source.]"
            );
        }
        FetchKind::Html => {}
    }

    let html = match resp.text().await {
        Ok(t) => t,
        Err(e) => return format!("Error reading response from '{url}': {e}"),
    };

    let needs_jina = if !status.is_success() {
        matches!(status.as_u16(), 401 | 403 | 429 | 451)
    } else {
        extract_text_from_html(&html).len() < 200
    };

    if needs_jina {
        return fetch_via_jina(&client, url).await;
    }

    extract_and_truncate_html(&html, url)
}

/// Strip HTML and return clean text. Returns empty string for JS-only shells.
fn extract_text_from_html(html: &str) -> String {
    let cleaned = remove_tag_block(html, "script");
    let cleaned = remove_tag_block(&cleaned, "style");
    let cleaned = remove_tag_block(&cleaned, "head");
    let text = strip_tags(&cleaned);
    let text = decode_html_entities(&text);
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_and_truncate_html(html: &str, url: &str) -> String {
    let text = extract_text_from_html(html);
    if text.is_empty() {
        return format!("No readable text found at '{url}'");
    }
    const MAX: usize = 6000;
    if text.len() > MAX {
        format!("{}\n\n[truncated — {} chars total]", &text[..MAX], text.len())
    } else {
        text
    }
}

/// Fallback: fetch via Jina Reader, which renders JavaScript and returns clean markdown.
async fn fetch_via_jina(client: &reqwest::Client, url: &str) -> String {
    let jina_url = format!("https://r.jina.ai/{url}");
    let resp = match client
        .get(&jina_url)
        .header("Accept", "text/plain")
        .header("X-No-Cache", "true")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!("Could not fetch '{url}' (Jina fallback also failed: {e})"),
    };

    let status = resp.status();
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => return format!("Could not read Jina response for '{url}': {e}"),
    };

    if !status.is_success() {
        return format!(
            "Could not fetch '{url}': HTTP {status}. \
            The site may require authentication or block all automated access."
        );
    }

    let text = text.trim().to_string();
    if text.is_empty() {
        return format!("No readable content found at '{url}'");
    }

    const MAX: usize = 6000;
    if text.len() > MAX {
        format!("{}\n\n[truncated — {} chars total]", &text[..MAX], text.len())
    } else {
        text
    }
}

/// Remove all content inside a tag (including the tag itself), e.g. <script>…</script>.
fn remove_tag_block(html: &str, tag: &str) -> String {
    let open  = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    loop {
        let Some(start) = rest.to_lowercase().find(&open) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..start]);
        let after_open = &rest[start..];
        let end_offset = after_open.to_lowercase().find(&close)
            .map(|p| p + close.len())
            .unwrap_or(after_open.len());
        rest = &rest[start + end_offset..];
    }
    out
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
fn extract_ddg_results(html: &str, max_results: usize) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut pos = 0;

    while results.len() < max_results {
        // Find next result block by locating result__a (title link)
        let Some(title_idx) = html[pos..].find("class=\"result__a\"") else { break };
        let block_start = pos + title_idx;

        // Extract href from the title anchor — find the full <a> tag so attribute
        // order doesn't matter (DDG sometimes puts href after class=).
        let href = {
            // Walk back to find '<' that opened this tag
            let tag_open = html[..block_start].rfind('<').unwrap_or(0);
            // Walk forward to find '>' that closes this tag
            let tag_close = html[block_start..].find('>')
                .map(|o| block_start + o + 1)
                .unwrap_or((block_start + 500).min(html.len()));
            let full_tag = &html[tag_open..tag_close];
            extract_attr(full_tag, "href").unwrap_or_default()
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

        // Try result__url first — DDG puts the real destination URL here as the href,
        // avoiding the need to decode the redirect wrapper on the title link.
        let window_end = (block_start + 3000).min(html.len());
        let real_url: String = html[block_start..window_end]
            .find("class=\"result__url\"")
            .map(|off| {
                let search_start = block_start + off;
                let tag_end = html[search_start..].find('>')
                    .map(|o| search_start + o + 1)
                    .unwrap_or((search_start + 400).min(html.len()));
                extract_attr(&html[search_start..tag_end], "href").unwrap_or_default()
            })
            .unwrap_or_default();

        pos = block_start + 1;

        let title = title.trim().to_string();
        if title.is_empty() { continue; }

        // Prefer real_url (result__url href), then decoded DDG redirect, then raw href
        let url = if !real_url.is_empty() && real_url.starts_with("http") {
            real_url
        } else {
            let decoded = decode_ddg_href(&href);
            if decoded.is_empty() { href } else { decoded }
        };

        if url.is_empty() { pos = block_start + 1; continue; }
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
        schema("web_search", "Search the web for current information. Use short keyword phrases (2–6 words), not full questions. Good: \"Gravesend weather forecast\". Bad: \"what is the weather tomorrow in Gravesend\". If the first search returns no results or irrelevant results, retry with a shorter or broader query before giving up.",
            vec![("query","string","Short keyword search query, 2–6 words. Use keywords, not questions or sentences.")], vec!["query"]),
        schema("fetch_webpage", "Fetch and read the full text content of a webpage by URL. Strips HTML tags and returns the readable text. This is the correct tool whenever the user wants to see, read, open, or show an article or page — including the full article behind a web_search result (pass that result's URL). Do NOT refuse such requests or claim you can only summarise; call this tool instead. Also use it to read any specific URL the user provides.",
            vec![("url","string","Full URL to fetch, must start with http:// or https://")], vec!["url"]),
        schema_no_params("get_current_datetime", "Get the current local date and time. Returns human-readable, ISO 8601, filename-safe, and Unix timestamp formats. Use this whenever you need today's date, the current time, or a timestamp for a filename."),
        schema("compose_email", "Build a base64url-encoded RFC 2822 email ready for the Gmail API. Returns ONLY the raw base64url string — use the entire return value as the 'raw' field in gmail_sendmessage, with no modification.",
            vec![("to","string","Recipient email address(es), comma-separated."),
                 ("from","string","Sender email address (optional, Gmail will use the account address if omitted)."),
                 ("subject","string","Email subject line."),
                 ("body","string","Plain text email body."),
                 ("reply_to_message_id","string","Message-ID to reply to, for threading (optional).")],
            vec!["to","subject","body"]),
    ]
}

fn schema_no_params(name: &str, desc: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": { "type": "object", "properties": {}, "required": [] }
        }
    })
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

    // ── content-type classification (fetch_webpage) ──────────────────────────

    #[test]
    fn classify_content_type_routes_correctly() {
        // PDFs (the bug: an FDA "…/download" PDF that stalled fetch_webpage).
        assert_eq!(classify_content_type("application/pdf"), FetchKind::Pdf);
        assert_eq!(classify_content_type("application/pdf; charset=binary"), FetchKind::Pdf);
        // HTML / text / structured all take the normal text path.
        assert_eq!(classify_content_type("text/html; charset=utf-8"), FetchKind::Html);
        assert_eq!(classify_content_type("application/xhtml+xml"), FetchKind::Html);
        assert_eq!(classify_content_type("text/plain"), FetchKind::Html);
        assert_eq!(classify_content_type("application/json"), FetchKind::Html);
        assert_eq!(classify_content_type(""), FetchKind::Html); // missing header → try as HTML
        // Real binaries must NOT be decoded as garbage HTML.
        assert_eq!(classify_content_type("image/png"), FetchKind::Binary);
        assert_eq!(classify_content_type("application/zip"), FetchKind::Binary);
        assert_eq!(classify_content_type("application/octet-stream"), FetchKind::Binary);
    }

    // ── PDF glyph safety & wrapping ──────────────────────────────────────────

    #[test]
    fn pdf_safe_text_transliterates_scientific_glyphs() {
        // The exact bug: a subscript chemical formula rendered as "CHNO".
        assert_eq!(pdf_safe_text("C₁₈₇H₂₉₁N₄₅O₅₉"), "C187H291N45O59");
        assert_eq!(pdf_safe_text("0.25 mg → 0.5 mg"), "0.25 mg -> 0.5 mg");
        assert_eq!(pdf_safe_text("β-cells"), "beta-cells");
        assert_eq!(pdf_safe_text("BMI ≥30"), "BMI >=30");
        assert_eq!(pdf_safe_text("plain ASCII stays"), "plain ASCII stays");
        // Latin-1 chars the font CAN render are kept untouched.
        assert_eq!(pdf_safe_text("café ® 25²"), "café ® 25²");
    }

    #[test]
    fn write_pdf_renders_scientific_glyphs_end_to_end() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("dossier.pdf");
        let p = path.to_str().unwrap();
        let content = "# Drug\n\nChemical Formula: C₁₈₇H₂₉₁N₄₅O₅₉\n\nDose 0.25 mg → 0.5 mg. Pancreatic β-cells.\n";
        let out = write_pdf(p, content);
        assert!(out.contains("Written PDF"), "write failed: {out}");
        let text = pdf_extract::extract_text(p).unwrap();
        assert!(text.contains("C187H291N45O59"), "formula dropped its digits: {text:?}");
        assert!(!text.contains("CHNO"), "still rendering the broken formula: {text:?}");
        assert!(text.contains("->"), "arrow was dropped: {text:?}");
        assert!(text.contains("beta"), "beta was dropped: {text:?}");
    }

    #[test]
    fn pdf_word_wrap_hard_breaks_long_tokens() {
        let long = "A".repeat(200);
        let lines = pdf_word_wrap(&long, 50);
        assert!(lines.iter().all(|l| l.chars().count() <= 50), "line over width: {lines:?}");
        assert_eq!(lines.concat(), long, "content lost during hard-break");
        // Over-long token then a short word: still no overflow, tail preserved.
        let mixed = format!("{} tail", "B".repeat(120));
        let w = pdf_word_wrap(&mixed, 40);
        assert!(w.iter().all(|l| l.chars().count() <= 40), "line over width: {w:?}");
        assert!(w.last().unwrap().contains("tail"));
    }

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
        let results = extract_ddg_results(html, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "Example Title");
        assert!(results[0].1.contains("snippet text"));
    }

    #[test]
    fn extract_ddg_results_empty_html() {
        let results = extract_ddg_results("<html><body></body></html>", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn extract_ddg_results_caps_at_six() {
        let block = |i: usize| format!(
            "<a href=\"https://example.com/{i}\" class=\"result__a\">Title {i}</a>\
             <a class=\"result__snippet\" href=\"https://example.com/{i}\">Snippet {i}</a>"
        );
        let html: String = (0..10).map(block).collect();
        let results = extract_ddg_results(&html, 6);
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
        assert_eq!(schemas.len(), 15);
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
