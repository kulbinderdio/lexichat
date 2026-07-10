import { useState } from "react";
import { Plus, Trash2, PanelLeftClose } from "lucide-react";

// Mirrors the Rust `history::ConversationMeta` (snake_case over the wire).
export interface ConversationMeta {
  id: string;
  title: string;
  profile_id: string | null;
  model: string;
  created_at: number; // unix seconds
  updated_at: number;
  message_count: number;
}

interface Props {
  visible: boolean;
  conversations: ConversationMeta[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onHide: () => void;
}

function relativeTime(unixSecs: number): string {
  const diff = Date.now() / 1000 - unixSecs;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(unixSecs * 1000).toLocaleDateString();
}

export function HistoryPanel({ visible, conversations, activeId, onSelect, onNew, onDelete, onRename, onHide }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  if (!visible) return null;

  const startRename = (c: ConversationMeta) => {
    setEditingId(c.id);
    setDraft(c.title);
  };

  const commitRename = () => {
    if (editingId) {
      const title = draft.trim();
      if (title) onRename(editingId, title);
    }
    setEditingId(null);
  };

  return (
    <div className="history-panel">
      <div className="history-header">
        <button className="history-collapse" onClick={onHide} title="Hide history">
          <PanelLeftClose size={15} />
        </button>
        <span>Chat history</span>
        <button className="history-new" onClick={onNew} title="New chat">
          <Plus size={13} /> New
        </button>
      </div>

      <div className="history-list">
        {conversations.length === 0 && (
          <div className="history-empty">No saved conversations yet.</div>
        )}
        {conversations.map(c => (
          <div
            key={c.id}
            className={`history-item ${c.id === activeId ? "active" : ""}`}
            onClick={() => editingId !== c.id && onSelect(c.id)}
          >
            <div className="history-item-main">
              {editingId === c.id ? (
                <input
                  className="history-rename-input"
                  autoFocus
                  value={draft}
                  onChange={e => setDraft(e.target.value)}
                  onClick={e => e.stopPropagation()}
                  onBlur={commitRename}
                  onKeyDown={e => {
                    if (e.key === "Enter") commitRename();
                    if (e.key === "Escape") setEditingId(null);
                  }}
                />
              ) : (
                <div
                  className="history-title"
                  title={c.title}
                  onDoubleClick={e => { e.stopPropagation(); startRename(c); }}
                >
                  {c.title}
                </div>
              )}
              <div className="history-meta">{relativeTime(c.updated_at)}</div>
            </div>
            <button
              className="history-del"
              title="Delete conversation"
              onClick={e => { e.stopPropagation(); onDelete(c.id); }}
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
