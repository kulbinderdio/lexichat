import _React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// StrictMode disabled: Tauri event listeners fire twice in dev mode under StrictMode,
// causing duplicate streaming tokens. Remove StrictMode to avoid this.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <App />
);
