import { describe, it, expect } from "vitest";
import { buildExportEnvelope, parseImport, mergeImport } from "../profileIO";
import type { Profile, AppSettings, StoredOpenAPISpec, StoredMCPServer } from "../AdminPanel";

const spec = (id: string): StoredOpenAPISpec => ({
  id, title: `Spec ${id}`, base_url: "https://api.example.com", spec_json: "{}",
  auth: { type: "apikey", api_key_header: "X-Key", api_key_value: "SECRET-123" },
});
const mcp = (id: string): StoredMCPServer => ({
  id, name: `Srv ${id}`, command: "/usr/local/bin/thing", args: [],
  env: { API_TOKEN: "SECRET-ENV" },
  auth: { type: "bearer", bearer_token: "SECRET-BEARER" },
});
const profile = (over: Partial<Profile> = {}): Profile => ({
  id: "p1", name: "Research", systemPrompt: "you research", model: "qwen3.6:latest",
  enabledTools: {}, enabledMcpServerIds: ["m1"], enabledOpenapiSpecIds: ["s1", "builtin-wikipedia"],
  enabledSparqlEndpointIds: [], maxTools: 30, allowedDirs: ["/Users/dio/secret"], ...over,
});
const settings = (over: Partial<AppSettings> = {}): AppSettings => ({
  host: "http://localhost:11434", models: [], profiles: [], activeProfileId: null,
  maxTools: 30, maxSteps: 20, webSearchResults: 10, enabledTools: {},
  toolRegistry: { openapiSpecs: [spec("s1")], sparqlEndpoints: [], mcpServers: [mcp("m1")] },
  ...over,
} as AppSettings);

describe("buildExportEnvelope", () => {
  it("bundles enabled APIs with all secrets stripped and no built-ins or local paths", () => {
    const env = buildExportEnvelope(profile(), settings().toolRegistry);
    const json = JSON.stringify(env);
    expect(json).not.toContain("SECRET-123");
    expect(json).not.toContain("SECRET-ENV");
    expect(json).not.toContain("SECRET-BEARER");
    expect(json).not.toContain("/Users/dio/secret");
    // The API definition is still there (just without the key), built-in id is not bundled.
    expect(env.toolRegistry.openapiSpecs.map(s => s.id)).toEqual(["s1"]);
    expect(env.toolRegistry.openapiSpecs[0].auth?.type).toBe("apikey");   // shape kept
    expect(env.toolRegistry.openapiSpecs[0].auth?.api_key_value).toBeUndefined(); // secret gone
    expect(env.toolRegistry.mcpServers[0].env).toEqual({ API_TOKEN: "" }); // key kept, value blanked
    expect(env.profile.allowedDirs).toEqual([]);
  });
});

describe("parseImport", () => {
  it("accepts v2, v1 (profile-only), and rejects junk", () => {
    const v2 = buildExportEnvelope(profile(), settings().toolRegistry);
    expect(parseImport(v2)?.openapiSpecs).toHaveLength(1);
    expect(parseImport({ lexichat_export_version: 1, profile: profile() })?.openapiSpecs).toEqual([]);
    expect(parseImport({ foo: "bar" })).toBeNull();
    expect(parseImport({ lexichat_export_version: 2, profile: { name: "x" } })).toBeNull(); // no systemPrompt
  });
});

describe("mergeImport", () => {
  it("adds bundled APIs, keeps enabled-id references resolving, and flags credentials", () => {
    const env = buildExportEnvelope(profile(), settings().toolRegistry);
    const bundle = parseImport(env)!;
    // Fresh install: empty registry, model not installed.
    const target = settings({ toolRegistry: { openapiSpecs: [], sparqlEndpoints: [], mcpServers: [] }, models: [] });
    const res = mergeImport(bundle, target, "new-id");

    // The spec the profile references is now in the registry (referential integrity).
    expect(res.settings.toolRegistry.openapiSpecs.map(s => s.id)).toContain("s1");
    const imported = res.settings.profiles.at(-1)!;
    expect(imported.enabledOpenapiSpecIds).toContain("s1");
    expect(imported.id).toBe("new-id");
    expect(imported.allowedDirs).toEqual([]);
    expect(res.warnings.some(w => w.includes("Re-enter credentials"))).toBe(true);
    expect(res.warnings.some(w => w.includes("qwen3.6"))).toBe(true); // model missing
  });

  it("does not duplicate an API already in the registry", () => {
    const env = buildExportEnvelope(profile(), settings().toolRegistry);
    const bundle = parseImport(env)!;
    // Target already has s1.
    const target = settings();
    const res = mergeImport(bundle, target, "new-id");
    expect(res.settings.toolRegistry.openapiSpecs.filter(s => s.id === "s1")).toHaveLength(1);
  });
});
