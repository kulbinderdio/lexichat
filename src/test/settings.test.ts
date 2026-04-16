import { describe, it, expect, beforeEach } from "vitest";
import { loadSettings, saveSettings } from "../App";

const DEFAULT_HOST = "http://localhost:11434";

beforeEach(() => {
  localStorage.clear();
});

describe("loadSettings", () => {
  it("returns defaults when localStorage is empty", () => {
    const s = loadSettings();
    expect(s.host).toBe(DEFAULT_HOST);
    expect(s.models).toEqual([]);
    expect(s.profiles).toEqual([]);
    expect(s.activeProfileId).toBeNull();
    expect(s.maxTools).toBe(30);
  });

  it("merges saved values over defaults", () => {
    localStorage.setItem("lexi_settings", JSON.stringify({ host: "http://localhost:1234" }));
    const s = loadSettings();
    expect(s.host).toBe("http://localhost:1234");
    // Other defaults still present
    expect(s.maxTools).toBe(30);
    expect(s.models).toEqual([]);
  });

  it("returns defaults for corrupted JSON", () => {
    localStorage.setItem("lexi_settings", "not valid json {{{");
    const s = loadSettings();
    expect(s.host).toBe(DEFAULT_HOST);
  });

  it("preserves saved profiles", () => {
    const profiles = [{ id: "p1", name: "Work", systemPrompt: "", enabledTools: {}, mcpServers: [], openapiSpecs: [] }];
    localStorage.setItem("lexi_settings", JSON.stringify({ profiles }));
    const s = loadSettings();
    expect(s.profiles).toHaveLength(1);
    expect(s.profiles[0].id).toBe("p1");
  });

  it("preserves saved enabledTools", () => {
    localStorage.setItem("lexi_settings", JSON.stringify({
      enabledTools: { read_file: false, list_files: true, web_search: false },
    }));
    const s = loadSettings();
    expect(s.enabledTools.read_file).toBe(false);
    expect(s.enabledTools.web_search).toBe(false);
  });
});

describe("saveSettings + loadSettings round-trip", () => {
  it("persists host change", () => {
    const s = loadSettings();
    saveSettings({ ...s, host: "http://remote:11434" });
    expect(loadSettings().host).toBe("http://remote:11434");
  });

  it("persists model list", () => {
    const s = loadSettings();
    saveSettings({ ...s, models: ["gemma4:9b", "llama3.2"] });
    expect(loadSettings().models).toEqual(["gemma4:9b", "llama3.2"]);
  });

  it("persists activeProfileId", () => {
    const s = loadSettings();
    saveSettings({ ...s, activeProfileId: "profile-abc" });
    expect(loadSettings().activeProfileId).toBe("profile-abc");
  });

  it("persists maxTools", () => {
    const s = loadSettings();
    saveSettings({ ...s, maxTools: 10 });
    expect(loadSettings().maxTools).toBe(10);
  });

  it("overwrites previous save", () => {
    const s = loadSettings();
    saveSettings({ ...s, host: "http://first:11434" });
    saveSettings({ ...s, host: "http://second:11434" });
    expect(loadSettings().host).toBe("http://second:11434");
  });
});
