import { useState, useEffect, useRef, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Send, Settings, RotateCcw, Bot, User, Wrench, Terminal } from "lucide-react";
import "./App.css";

// ── Types ─────────────────────────────────────────────────────────────────────

type MessageKind = "user" | "assistant" | "tool-call" | "tool-result" | "error";

interface ChatMessage {
  id: string;
  kind: MessageKind;
  text: string;
  streaming?: boolean;
  label?: string; // tool name for tool messages
}

interface ToolSchema {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: unknown;
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const uid = () => Math.random().toString(36).slice(2);

const SYSTEM_PROMPT = `You are a personal AI assistant running locally for a single authorised user.
You have tools to read local files and search the web.
Rules:
- Use list_files and read_file for questions about local files and folders.
- Use web_search for current events, weather, or live data.
- Once you have enough information, give a direct answer without more tool calls.`;

const BUILTIN_TOOLS: ToolSchema[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read the contents of a local file.",
      parameters: {
        type: "object",
        properties: { path: { type: "string", description: "File path." } },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_files",
      description: "List files and directories at a given path.",
      parameters: {
        type: "object",
        properties: { path: { type: "string", description: "Directory path." } },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for current information.",
      parameters: {
        type: "object",
        properties: { query: { type: "string", description: "Search query." } },
        required: ["query"],
      },
    },
  },
];

// ── Settings panel ────────────────────────────────────────────────────────────

function SettingsPanel({
  host,
  onSave,
  onClose,
}: {
  host: string;
  onSave: (host: string) => void;
  onClose: () => void;
}) {
  const [value, setValue] = useState(host);
  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>
        <div className="field">
          <label>Ollama host</label>
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="http://localhost:11434"
          />
        </div>
        <div className="settings-actions">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn primary" onClick={() => { onSave(value); onClose(); }}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────

function Bubble({ msg }: { msg: ChatMessage }) {
  const icons: Record<MessageKind, React.ReactNode> = {
    user: <User size={12} />,
    assistant: <Bot size={12} />,
    "tool-call": <Wrench size={12} />,
    "tool-result": <Terminal size={12} />,
    error: <span>⚠</span>,
  };
  const labels: Record<MessageKind, string> = {
    user: "You",
    assistant: "Assistant",
    "tool-call": msg.label ? `Tool: ${msg.label}` : "Tool call",
    "tool-result": msg.label ? `Result: ${msg.label}` : "Tool result",
    error: "Error",
  };
  return (
    <div className={`msg ${msg.kind}`}>
      <div className="msg-header">
        {icons[msg.kind]}
        {labels[msg.kind]}
      </div>
      <div className="msg-body">
        {msg.text}
        {msg.streaming && <span className="streaming-dot" />}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [host, setHost] = useState("http://localhost:11434");
  const [isRunning, setIsRunning] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Fetch models on mount and when host changes
  const fetchModels = useCallback(async () => {
    try {
      const list = await invoke<string[]>("get_models");
      setModels(list);
      if (list.length > 0 && !selectedModel) setSelectedModel(list[0]);
    } catch {
      setModels([]);
    }
  }, [host, selectedModel]);

  useEffect(() => { fetchModels(); }, [host]);

  // Scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Listen to agent events from Rust
  useEffect(() => {
    const unlisten: Array<() => void> = [];

    listen<{ delta: string }>("agent-token", (e) => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.kind === "assistant" && last.streaming) {
          return [
            ...prev.slice(0, -1),
            { ...last, text: last.text + e.payload.delta },
          ];
        }
        // First token — create new streaming bubble
        return [
          ...prev,
          { id: uid(), kind: "assistant", text: e.payload.delta, streaming: true },
        ];
      });
    }).then((u) => unlisten.push(u));

    listen<{ name: string; args: string }>("agent-tool-call", (e) => {
      setMessages((prev) => {
        // Mark current assistant bubble as done streaming
        const updated = prev.map((m) =>
          m.streaming ? { ...m, streaming: false } : m
        );
        return [
          ...updated,
          {
            id: uid(),
            kind: "tool-call" as MessageKind,
            label: e.payload.name,
            text: e.payload.args,
          },
        ];
      });
    }).then((u) => unlisten.push(u));

    listen<{ name: string; result: string }>("agent-tool-result", (e) => {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          kind: "tool-result" as MessageKind,
          label: e.payload.name,
          text: e.payload.result,
        },
      ]);
    }).then((u) => unlisten.push(u));

    listen<{ error: string | null }>("agent-done", (e) => {
      setIsRunning(false);
      setMessages((prev) => {
        const updated = prev.map((m) =>
          m.streaming ? { ...m, streaming: false } : m
        );
        if (e.payload.error) {
          return [
            ...updated,
            { id: uid(), kind: "error" as MessageKind, text: e.payload.error },
          ];
        }
        return updated;
      });
    }).then((u) => unlisten.push(u));

    return () => { unlisten.forEach((u) => u()); };
  }, []);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isRunning || !selectedModel) return;

    setMessages((prev) => [
      ...prev,
      { id: uid(), kind: "user", text },
    ]);
    setInput("");
    setIsRunning(true);

    try {
      await invoke("send_message", {
        args: {
          model: selectedModel,
          message: text,
          system_prompt: SYSTEM_PROMPT,
          tools: BUILTIN_TOOLS,
        },
      });
    } catch (e) {
      setIsRunning(false);
      setMessages((prev) => [
        ...prev,
        { id: uid(), kind: "error", text: String(e) },
      ]);
    }
  };

  const handleReset = async () => {
    await invoke("reset_conversation");
    setMessages([]);
  };

  const handleSaveHost = async (newHost: string) => {
    setHost(newHost);
    await invoke("set_ollama_host", { host: newHost });
    fetchModels();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  };

  return (
    <div className="app">
      {/* Toolbar */}
      <div className="toolbar">
        <span className="toolbar-title">AI Agent</span>

        <select
          className="select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={models.length === 0}
        >
          {models.length === 0 ? (
            <option>No models found</option>
          ) : (
            models.map((m) => <option key={m}>{m}</option>)
          )}
        </select>

        <button className="btn" onClick={handleReset} disabled={isRunning} title="New chat">
          <RotateCcw size={13} />
          New chat
        </button>

        <button className="btn" onClick={() => setShowSettings(true)} title="Settings">
          <Settings size={13} />
        </button>
      </div>

      {/* Chat */}
      <div className="chat-scroll">
        {messages.length === 0 ? (
          <div className="welcome">
            <h2>AI Agent</h2>
            <p>Ask me anything — I can read local files and search the web.</p>
          </div>
        ) : (
          messages.map((msg) => <Bubble key={msg.id} msg={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-bar">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Message AI Agent… (Enter to send, Shift+Enter for newline)"
          disabled={isRunning}
          rows={1}
        />
        <button className="send-btn" onClick={handleSend} disabled={isRunning || !input.trim()}>
          <Send size={15} />
          Send
        </button>
      </div>

      {showSettings && (
        <SettingsPanel
          host={host}
          onSave={handleSaveHost}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}
