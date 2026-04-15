mod ollama;
mod tools;

use std::sync::Mutex;
use tauri::{AppHandle, State};
use serde::Deserialize;

// ── Shared app state ──────────────────────────────────────────────────────────

pub struct AppState {
    pub ollama_host: Mutex<String>,
    pub conversation: Mutex<Vec<ollama::WireMessage>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            ollama_host: Mutex::new("http://localhost:11434".into()),
            conversation: Mutex::new(Vec::new()),
        }
    }
}

// ── Commands ──────────────────────────────────────────────────────────────────

#[tauri::command]
async fn get_models(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let host = state.ollama_host.lock().unwrap().clone();
    ollama::list_models(&host).await.map_err(|e| e.to_string())
}

#[tauri::command]
async fn set_ollama_host(host: String, state: State<'_, AppState>) -> Result<(), String> {
    *state.ollama_host.lock().unwrap() = host;
    Ok(())
}

#[tauri::command]
async fn reset_conversation(state: State<'_, AppState>) -> Result<(), String> {
    state.conversation.lock().unwrap().clear();
    Ok(())
}

#[derive(Deserialize)]
pub struct SendMessageArgs {
    pub model: String,
    pub message: String,
    pub system_prompt: String,
    pub tools: Vec<ollama::ToolSchema>,
}

#[tauri::command]
async fn send_message(
    args: SendMessageArgs,
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    let host = state.ollama_host.lock().unwrap().clone();

    // Append user message to conversation history
    {
        let mut conv = state.conversation.lock().unwrap();
        conv.push(ollama::WireMessage {
            role: "user".into(),
            content: Some(args.message.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
    }

    // Run the agentic loop
    ollama::agent_loop(
        &host,
        &args.model,
        &args.system_prompt,
        &args.tools,
        &state.conversation,
        &app,
    )
    .await
    .map_err(|e| e.to_string())
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_models,
            set_ollama_host,
            reset_conversation,
            send_message,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
