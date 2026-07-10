// Chat history: on-disk persistence of past conversations.
//
// Each conversation is stored as its own JSON file under `<data>/lexichat/
// conversations/{id}.json`, holding both the authoritative Ollama `wire`
// history and the opaque frontend `display` messages. A lightweight
// `index.json` of metadata lets the sidebar list load without parsing every
// (potentially image-heavy) conversation file. Mirrors the load/save idiom in
// `jobs.rs`.

use serde::{Deserialize, Serialize};
use crate::ollama::WireMessage;

/// Lightweight per-conversation metadata — this is what the history list shows.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub model: String,
    pub created_at: i64, // unix seconds
    pub updated_at: i64,
    #[serde(default)]
    pub message_count: usize,
}

/// A full saved conversation: metadata + backend wire history + frontend display.
#[derive(Clone, Serialize, Deserialize)]
pub struct Conversation {
    #[serde(flatten)]
    pub meta: ConversationMeta,
    /// Authoritative message history sent to Ollama — restored verbatim on load.
    #[serde(default)]
    pub wire: Vec<WireMessage>,
    /// Opaque frontend `ChatMessage[]` used only for rendering.
    #[serde(default)]
    pub display: serde_json::Value,
}

fn conversations_dir() -> std::path::PathBuf {
    let dir = crate::dirs_path().join("conversations");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn index_path() -> std::path::PathBuf {
    conversations_dir().join("index.json")
}

fn conversation_path(id: &str) -> std::path::PathBuf {
    conversations_dir().join(format!("{id}.json"))
}

/// Mint a new conversation id. Microsecond timestamp is unique enough given a
/// new conversation is created at most once per "New chat".
pub fn new_id() -> String {
    format!("conv-{}", chrono::Utc::now().timestamp_micros())
}

pub fn now_secs() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Load the metadata index (newest first). Missing/corrupt → empty list.
pub fn load_index() -> Vec<ConversationMeta> {
    std::fs::read_to_string(index_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_index(index: &[ConversationMeta]) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(index)?;
    std::fs::write(index_path(), json)?;
    Ok(())
}

pub fn load_one(id: &str) -> Option<Conversation> {
    std::fs::read_to_string(conversation_path(id))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Write the conversation file and upsert its meta into the index (newest first).
pub fn save_one(conv: &Conversation) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(conv)?;
    std::fs::write(conversation_path(&conv.meta.id), json)?;

    let mut index = load_index();
    index.retain(|m| m.id != conv.meta.id);
    index.push(conv.meta.clone());
    index.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    save_index(&index)
}

pub fn delete_one(id: &str) -> anyhow::Result<()> {
    let _ = std::fs::remove_file(conversation_path(id));
    let mut index = load_index();
    index.retain(|m| m.id != id);
    save_index(&index)
}

pub fn rename(id: &str, title: &str) -> anyhow::Result<()> {
    if let Some(mut conv) = load_one(id) {
        conv.meta.title = title.to_string();
        save_one(&conv)?;
    }
    Ok(())
}

/// Title = first line of the first user message, trimmed to ~40 chars.
pub fn derive_title(wire: &[WireMessage]) -> String {
    let raw = wire
        .iter()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
        .unwrap_or("")
        .trim();
    if raw.is_empty() {
        return "New conversation".to_string();
    }
    let first_line = raw.lines().next().unwrap_or(raw).trim();
    let mut title: String = first_line.chars().take(40).collect();
    if first_line.chars().count() > 40 {
        title.push('…');
    }
    title
}
