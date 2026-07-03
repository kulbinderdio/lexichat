// Injected into the page (via page.addInitScript) BEFORE the app boots, to stand
// in for the Tauri IPC bridge so the React app runs in a plain browser:
//   • window.__TAURI_INTERNALS__.invoke — canned command responses
//   • the event system (plugin:event|listen + transformCallback)
//   • window.__mockEmit(event, payload) — lets a test push backend events
//
// Must be self-contained (Playwright serializes it into the page).
export function tauriMockInit() {
  const CANNED: Record<string, unknown> = {
    get_models: ["llama3"],
    get_allowed_dirs: [],
    get_jobs: [],
    get_job_runs: [],
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const internals: any = {
    callbacks: {} as Record<number, (e: unknown) => void>,
    listeners: {} as Record<string, number[]>,
    nextId: 1,
    transformCallback(cb: (e: unknown) => void) {
      const id = internals.nextId++;
      internals.callbacks[id] = cb;
      return id;
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async invoke(cmd: string, args: any) {
      if (cmd === "plugin:event|listen") {
        (internals.listeners[args.event] ||= []).push(args.handler);
        return 0;
      }
      if (cmd.startsWith("plugin:event|")) return undefined;
      if (cmd === "mcp_ui_call_tool") return { text: "ok", content: [], structured: null };
      if (cmd in CANNED) return CANNED[cmd];
      return null;
    },
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).__TAURI_INTERNALS__ = internals;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (window as any).__mockEmit = (event: string, payload: unknown) => {
    for (const id of internals.listeners[event] || []) {
      internals.callbacks[id]?.({ event, id, payload });
    }
  };
}
