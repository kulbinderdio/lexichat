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

  it("defaults include empty toolRegistry", () => {
    const s = loadSettings();
    expect(s.toolRegistry).toBeDefined();
    expect(s.toolRegistry.mcpServers).toEqual([]);
    // openapiSpecs may contain built-in specs (Wikipedia etc.) so just check it's an array
    expect(Array.isArray(s.toolRegistry.openapiSpecs)).toBe(true);
  });

  it("merges saved values over defaults", () => {
    localStorage.setItem("lexi_settings", JSON.stringify({ host: "http://localhost:1234" }));
    const s = loadSettings();
    expect(s.host).toBe("http://localhost:1234");
    expect(s.maxTools).toBe(30);
    expect(s.models).toEqual([]);
  });

  it("returns defaults for corrupted JSON", () => {
    localStorage.setItem("lexi_settings", "not valid json {{{");
    const s = loadSettings();
    expect(s.host).toBe(DEFAULT_HOST);
  });

  it("preserves saved profiles (new format)", () => {
    const profiles = [{
      id: "p1", name: "Work", systemPrompt: "", model: "llama3", maxTools: 30,
      enabledTools: {}, enabledMcpServerIds: [], enabledOpenapiSpecIds: [],
    }];
    localStorage.setItem("lexi_settings", JSON.stringify({
      profiles,
      toolRegistry: { mcpServers: [], openapiSpecs: [] },
    }));
    const s = loadSettings();
    expect(s.profiles).toHaveLength(1);
    expect(s.profiles[0].id).toBe("p1");
    expect(s.profiles[0].enabledMcpServerIds).toEqual([]);
    expect(s.profiles[0].enabledOpenapiSpecIds).toEqual([]);
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

describe("migration from legacy format", () => {
  it("migrates top-level mcpServers/openapiSpecs into toolRegistry", () => {
    const legacy = {
      host: DEFAULT_HOST,
      mcpServers: [{ id: "mcp1", name: "My Server", command: "node server.js", args: [], env: {} }],
      openapiSpecs: [{ id: "spec1", title: "My API", base_url: "https://api.example.com", spec_json: "{}", enabled: true }],
      profiles: [],
    };
    localStorage.setItem("lexi_settings", JSON.stringify(legacy));
    const s = loadSettings();
    expect(s.toolRegistry.mcpServers).toHaveLength(1);
    expect(s.toolRegistry.mcpServers[0].id).toBe("mcp1");
    // spec1 plus any built-in specs
    expect(s.toolRegistry.openapiSpecs.some(sp => sp.id === "spec1")).toBe(true);
    expect(s.mcpServers).toBeUndefined();
    expect(s.openapiSpecs).toBeUndefined();
  });

  it("migrates per-profile mcpServers/openapiSpecs into registry and ID lists", () => {
    const legacy = {
      host: DEFAULT_HOST,
      mcpServers: [],
      openapiSpecs: [],
      profiles: [{
        id: "p1", name: "Work", systemPrompt: "", model: "", maxTools: 30, enabledTools: {},
        mcpServers: [{ id: "mcp1", name: "S1", command: "s1", args: [], env: {} }],
        openapiSpecs: [{ id: "spec1", title: "T1", base_url: "https://t1.example.com", spec_json: "{}", enabled: true }],
      }],
    };
    localStorage.setItem("lexi_settings", JSON.stringify(legacy));
    const s = loadSettings();
    expect(s.toolRegistry.mcpServers.some(m => m.id === "mcp1")).toBe(true);
    expect(s.toolRegistry.openapiSpecs.some(sp => sp.id === "spec1")).toBe(true);
    expect(s.profiles[0].enabledMcpServerIds).toContain("mcp1");
    expect(s.profiles[0].enabledOpenapiSpecIds).toContain("spec1");
    expect((s.profiles[0] as any).mcpServers).toBeUndefined();
    expect((s.profiles[0] as any).openapiSpecs).toBeUndefined();
  });

  it("deduplicates tools that appear in both global and profile", () => {
    const sharedMcp = { id: "mcp1", name: "Shared", command: "s", args: [], env: {} };
    const legacy = {
      host: DEFAULT_HOST,
      mcpServers: [sharedMcp],
      openapiSpecs: [],
      profiles: [{
        id: "p1", name: "Work", systemPrompt: "", model: "", maxTools: 30, enabledTools: {},
        mcpServers: [sharedMcp],
        openapiSpecs: [],
      }],
    };
    localStorage.setItem("lexi_settings", JSON.stringify(legacy));
    const s = loadSettings();
    expect(s.toolRegistry.mcpServers.filter(m => m.id === "mcp1")).toHaveLength(1);
  });

  it("skips migration when toolRegistry already present", () => {
    const modern = {
      host: DEFAULT_HOST,
      toolRegistry: {
        mcpServers: [{ id: "mcp1", name: "S", command: "s", args: [], env: {} }],
        openapiSpecs: [],
      },
      profiles: [{ id: "p1", name: "Work", systemPrompt: "", model: "", maxTools: 30,
        enabledTools: {}, enabledMcpServerIds: ["mcp1"], enabledOpenapiSpecIds: [] }],
    };
    localStorage.setItem("lexi_settings", JSON.stringify(modern));
    const s = loadSettings();
    expect(s.toolRegistry.mcpServers).toHaveLength(1);
    expect(s.profiles[0].enabledMcpServerIds).toEqual(["mcp1"]);
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

  it("round-trips toolRegistry with servers and specs", () => {
    const s = loadSettings();
    const updated = {
      ...s,
      toolRegistry: {
        mcpServers: [{ id: "m1", name: "Test", command: "test", args: [], env: {} }],
        openapiSpecs: [{ id: "sp1", title: "API", base_url: "https://api.example.com", spec_json: "{}", enabled: true }],
      },
    };
    saveSettings(updated);
    const loaded = loadSettings();
    expect(loaded.toolRegistry.mcpServers).toHaveLength(1);
    expect(loaded.toolRegistry.mcpServers[0].id).toBe("m1");
    expect(loaded.toolRegistry.openapiSpecs.some(sp => sp.id === "sp1")).toBe(true);
  });

  it("round-trips profile enabledMcpServerIds and enabledOpenapiSpecIds", () => {
    const s = loadSettings();
    const profile = {
      id: "p1", name: "Dev", systemPrompt: "", model: "", maxTools: 30,
      enabledTools: {}, enabledMcpServerIds: ["m1"], enabledOpenapiSpecIds: ["sp1"],
    };
    saveSettings({ ...s, profiles: [profile] });
    const loaded = loadSettings();
    expect(loaded.profiles[0].enabledMcpServerIds).toEqual(["m1"]);
    expect(loaded.profiles[0].enabledOpenapiSpecIds).toEqual(["sp1"]);
  });
});
