// Renders a model's markdown report into a self-contained, styled HTML document — the "artifact"
// house style. Used by the chat "Export as report" action (via `render_report_html`) and by
// `tools::save_document` when a job's output_file ends in `.html`. No external assets: the CSS is
// inlined, so the file opens (and prints to PDF) in any browser offline.

/// Minimal HTML-escape for text nodes / attributes.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
     .replace('"', "&quot;").replace('\'', "&#x27;")
}

/// Inline markdown → HTML: links, bold, italic (`*`/`_`), inline code. Escapes first.
fn inline(s: &str) -> String {
    let mut out = esc(s);
    // strip stray LaTeX the models sometimes emit, e.g. $\text{12,039}$
    out = out.replace("\\text", "");
    // images ![alt](src) — must run before links so the leading `!` isn't left behind. Only
    // `data:` sources are kept (self-contained); remote URLs are dropped (CSP-blocked, unportable).
    out = replace_pattern(&out, "![", "](", ")", |alt, src| {
        if src.starts_with("data:") {
            format!("<img class=\"report-img\" alt=\"{}\" src=\"{}\">", alt, src)
        } else {
            String::new() // drop remote images
        }
    });
    // links [text](url)
    out = replace_pattern(&out, "[", "](", ")", |text, url| {
        format!("<a href=\"{}\" target=\"_blank\" rel=\"noopener\">{}</a>", esc_attr(url), text)
    });
    out = wrap_pairs(&out, "**", "strong");
    out = wrap_pairs(&out, "`", "code");
    out = wrap_single(&out, '*', "em");
    out = wrap_single(&out, '_', "em");
    out
}

fn esc_attr(s: &str) -> String { s.replace('"', "%22").replace(' ', "%20") }

/// Replace `open TEXT mid URL close` (used for markdown links) via a formatter.
fn replace_pattern(s: &str, open: &str, mid: &str, close: &str, f: impl Fn(&str, &str) -> String) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(o) = rest.find(open) {
        if let Some(m) = rest[o + open.len()..].find(mid) {
            let text = &rest[o + open.len()..o + open.len() + m];
            let after = &rest[o + open.len() + m + mid.len()..];
            if let Some(c) = after.find(close) {
                let url = &after[..c];
                out.push_str(&rest[..o]);
                out.push_str(&f(text, url));
                rest = &after[c + close.len()..];
                continue;
            }
        }
        out.push_str(&rest[..o + open.len()]);
        rest = &rest[o + open.len()..];
    }
    out.push_str(rest);
    out
}

/// Wrap text delimited by a multi-char marker (e.g. `**bold**`) in a tag.
fn wrap_pairs(s: &str, marker: &str, tag: &str) -> String {
    let parts: Vec<&str> = s.split(marker).collect();
    if parts.len() < 3 { return s.to_string(); }
    let mut out = String::new();
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            if i % 2 == 1 && i < parts.len() - 1 && i + 1 <= parts.len() {
                out.push_str(&format!("<{tag}>"));
            } else if i % 2 == 0 {
                out.push_str(&format!("</{tag}>"));
            } else {
                out.push_str(marker);
            }
        }
        out.push_str(p);
    }
    out
}

/// Wrap text delimited by a single char (`*` / `_`) in a tag, pairwise.
fn wrap_single(s: &str, marker: char, tag: &str) -> String {
    let count = s.matches(marker).count();
    if count < 2 { return s.to_string(); }
    let mut out = String::new();
    let mut open = false;
    let mut pairs_left = count - (count % 2);
    for ch in s.chars() {
        if ch == marker && pairs_left > 0 {
            let t = if open { format!("</{tag}>") } else { format!("<{tag}>") };
            out.push_str(&t);
            open = !open;
            pairs_left -= 1;
        } else {
            out.push(ch);
        }
    }
    out
}

/// Convert report markdown (the subset the models emit) into an HTML body fragment.
pub fn markdown_to_html(md: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut out = String::new();
    let mut i = 0;
    let mut in_list = false;
    let close_list = |out: &mut String, in_list: &mut bool| {
        if *in_list { out.push_str("</ul>\n"); *in_list = false; }
    };
    while i < lines.len() {
        let ln = lines[i].trim_end();
        let t = ln.trim();
        if t == "***" || t == "---" || t == "___" {
            close_list(&mut out, &mut in_list); out.push_str("<hr>\n"); i += 1; continue;
        }
        if let Some(h) = ln.strip_prefix("### ") {
            close_list(&mut out, &mut in_list); out.push_str(&format!("<h3>{}</h3>\n", inline(h.trim_matches('*')))); i += 1; continue;
        }
        if let Some(h) = ln.strip_prefix("## ") {
            close_list(&mut out, &mut in_list); out.push_str(&format!("<h2>{}</h2>\n", inline(h))); i += 1; continue;
        }
        if let Some(h) = ln.strip_prefix("# ") {
            close_list(&mut out, &mut in_list); out.push_str(&format!("<h2 class=\"doc\">{}</h2>\n", inline(h))); i += 1; continue;
        }
        // table: header row then a |---|---| separator
        if t.starts_with('|') && i + 1 < lines.len() && is_table_sep(lines[i + 1].trim()) {
            close_list(&mut out, &mut in_list);
            let header = split_row(t);
            out.push_str("<div class=\"tw\"><table><thead><tr>");
            for c in &header { out.push_str(&format!("<th>{}</th>", inline(c))); }
            out.push_str("</tr></thead><tbody>");
            let mut j = i + 2;
            while j < lines.len() && lines[j].trim().starts_with('|') {
                out.push_str("<tr>");
                for c in split_row(lines[j].trim()) { out.push_str(&format!("<td>{}</td>", inline(&c))); }
                out.push_str("</tr>");
                j += 1;
            }
            out.push_str("</tbody></table></div>\n");
            i = j; continue;
        }
        // list item
        if is_list_item(t) {
            if !in_list { out.push_str("<ul>\n"); in_list = true; }
            let item = strip_list_marker(t);
            out.push_str(&format!("<li>{}</li>\n", inline(&item)));
            i += 1; continue;
        }
        if t.is_empty() { close_list(&mut out, &mut in_list); i += 1; continue; }
        close_list(&mut out, &mut in_list);
        out.push_str(&format!("<p>{}</p>\n", inline(t)));
        i += 1;
    }
    close_list(&mut out, &mut in_list);
    out
}

fn is_table_sep(s: &str) -> bool {
    let s = s.trim_matches('|');
    !s.is_empty() && s.chars().all(|c| matches!(c, '-' | ':' | ' ' | '|'))
        && s.contains('-')
}
fn split_row(s: &str) -> Vec<String> {
    s.trim().trim_matches('|').split('|').map(|c| c.trim().to_string()).collect()
}
fn is_list_item(t: &str) -> bool {
    t.starts_with("- ") || t.starts_with("* ")
        || t.chars().take_while(|c| c.is_ascii_digit()).count() > 0
            && t.trim_start_matches(|c: char| c.is_ascii_digit()).starts_with(". ")
}
fn strip_list_marker(t: &str) -> String {
    if let Some(r) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")) { return r.to_string(); }
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    t[digits.len()..].trim_start_matches(". ").to_string()
}

/// First `# ` heading, else a default.
fn extract_title(md: &str) -> String {
    for ln in md.lines() {
        if let Some(h) = ln.trim().strip_prefix("# ") { return h.trim().to_string(); }
    }
    "Report".to_string()
}

const REPORT_CSS: &str = r#"
:root{--ground:#f4f6f8;--surface:#fff;--ink:#1b1e25;--muted:#59606d;--faint:#878e9c;--rule:#e0e4ea;--rule-soft:#eef1f5;--accent:#2f6f7e}
@media(prefers-color-scheme:dark){:root{--ground:#12151a;--surface:#191d24;--ink:#e6e9ef;--muted:#a2aab8;--faint:#6b7382;--rule:#292f39;--rule-soft:#20252d;--accent:#5aa9ba}}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);line-height:1.62;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased;padding:clamp(20px,5vw,64px) 20px 80px}
.report{max-width:800px;margin:0 auto;background:var(--surface);border:1px solid var(--rule);border-top:3px solid var(--accent);border-radius:4px;padding:clamp(26px,5vw,56px)}
.report h1{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;font-size:clamp(26px,4.4vw,38px);line-height:1.14;letter-spacing:-.01em;margin:0 0 18px;text-wrap:balance}
.report h2{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-weight:600;font-size:21px;margin:34px 0 12px;padding-top:16px;border-top:1px solid var(--rule-soft);letter-spacing:-.01em}
.report h2.doc{border-top:0;padding-top:0;font-size:26px}
.report h3{font-family:"Iowan Old Style",Palatino,Georgia,serif;font-weight:600;font-size:18px;margin:26px 0 9px}
.report p{margin:0 0 13px;font-size:15.5px;max-width:68ch}
.report ul{margin:0 0 15px;padding-left:0;list-style:none;max-width:68ch}
.report li{position:relative;padding-left:20px;margin:5px 0;font-size:15.5px}
.report li::before{content:"";position:absolute;left:4px;top:.62em;width:5px;height:5px;border-radius:50%;background:var(--accent);opacity:.55}
.report strong{font-weight:650}
.report em{color:var(--muted)}
.report code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.9em;background:var(--rule-soft);padding:.1em .35em;border-radius:4px}
.report a{color:var(--accent);text-underline-offset:2px}
.report hr{border:0;border-top:1px solid var(--rule-soft);margin:24px 0}
.report .report-img{max-width:100%;height:auto;display:block;margin:14px 0;border:1px solid var(--rule);border-radius:8px}
.figs{display:flex;flex-direction:column;gap:16px;margin-top:8px}
.figs figure{margin:0;border:1px solid var(--rule);border-radius:8px;overflow:hidden;background:#fff}
.figs img{display:block;width:100%;height:auto}
.tw{overflow-x:auto;margin:6px 0 20px;border:1px solid var(--rule);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
thead th{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--faint);font-weight:600;padding:10px 14px;background:var(--rule-soft);border-bottom:1px solid var(--rule)}
tbody td{padding:9px 14px;border-bottom:1px solid var(--rule-soft)}
tbody tr:last-child td{border-bottom:0}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:12px}
.colophon{max-width:800px;margin:22px auto 0;font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--faint);text-align:center}
@media print{body{background:#fff;padding:0}.report{border:0;max-width:none}}
"#;

/// Wrap a markdown report in the full self-contained HTML document (inlined CSS). `figures` are
/// `data:` image URLs (charts/maps generated in the chat) appended as a Figures section — used for
/// images the report text didn't already place inline via a `{{figure:N}}` token.
pub fn render_report_html(markdown: &str, title: Option<&str>, subtitle: Option<&str>, figures: &[String]) -> String {
    let title = title.map(str::to_string).unwrap_or_else(|| extract_title(markdown));
    let body = markdown_to_html(markdown);
    let eyebrow = subtitle.map(|s| format!("<div class=\"eyebrow\">{}</div>", esc(s))).unwrap_or_default();
    let figs: Vec<&String> = figures.iter().filter(|s| s.starts_with("data:")).collect();
    let figures_section = if figs.is_empty() {
        String::new()
    } else {
        let items: String = figs.iter()
            .map(|src| format!("<figure><img src=\"{src}\" alt=\"figure\"></figure>"))
            .collect();
        format!("<h2>Figures</h2><div class=\"figs\">{items}</div>")
    };
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>{t}</title><style>{css}</style></head>\
         <body><main class=\"report\">{eyebrow}{body}{figures_section}</main>\
         <footer class=\"colophon\">Generated by LexiChat</footer></body></html>",
        t = esc(&title), css = REPORT_CSS, eyebrow = eyebrow, body = body, figures_section = figures_section,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_headings_lists_tables_inline() {
        let md = "# Title\n\n## Section\n\n- one **bold**\n- two\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nA [link](https://x.com) and *em*.";
        let h = markdown_to_html(md);
        assert!(h.contains("<h2 class=\"doc\">Title</h2>"));
        assert!(h.contains("<h2>Section</h2>"));
        assert!(h.contains("<li>one <strong>bold</strong></li>"));
        assert!(h.contains("<table>") && h.contains("<th>A</th>") && h.contains("<td>1</td>"));
        assert!(h.contains("<a href=\"https://x.com\""));
        assert!(h.contains("<em>em</em>"));
    }

    #[test]
    fn escapes_html_and_wraps_document() {
        let doc = render_report_html("# R\n\nA <script> & 'quote'", Some("My Report"), Some("Local Area Checker"), &[]);
        assert!(doc.starts_with("<!doctype html>"));
        assert!(doc.contains("<title>My Report</title>"));
        assert!(doc.contains("Local Area Checker"));
        assert!(doc.contains("&lt;script&gt;"));   // escaped, not live
        assert!(!doc.contains("<script>"));
    }

    #[test]
    fn title_falls_back_to_first_heading() {
        assert_eq!(extract_title("# Hello World\ntext"), "Hello World");
        assert_eq!(extract_title("no heading"), "Report");
    }

    #[test]
    fn renders_data_images_drops_remote_and_appends_figures() {
        // data: image kept, remote image dropped
        let h = markdown_to_html("![chart](data:image/png;base64,AAAA)\n\n![x](https://evil/x.png)");
        assert!(h.contains("<img class=\"report-img\" alt=\"chart\" src=\"data:image/png;base64,AAAA\">"));
        assert!(!h.contains("https://evil"));
        // figures section appended for data: figures, remote ignored
        let doc = render_report_html("# R", None, None,
            &["data:image/png;base64,BBBB".into(), "https://nope/y.png".into()]);
        assert!(doc.contains("<h2>Figures</h2>"));
        assert!(doc.contains("src=\"data:image/png;base64,BBBB\""));
        assert!(!doc.contains("https://nope"));
    }
}
