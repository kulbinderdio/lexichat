import { describe, it, expect } from "vitest";
import { extractSpecMeta } from "../AdminPanel";

// ── OpenAPI 3.x ───────────────────────────────────────────────────────────────

describe("extractSpecMeta — OpenAPI 3.x", () => {
  const base3 = (extra = {}) =>
    JSON.stringify({
      info: { title: "Gmail API" },
      servers: [{ url: "https://gmail.googleapis.com" }],
      ...extra,
    });

  it("extracts title from info.title", () => {
    const { title } = extractSpecMeta(base3());
    expect(title).toBe("Gmail API");
  });

  it("extracts baseUrl from servers[0].url", () => {
    const { baseUrl } = extractSpecMeta(base3());
    expect(baseUrl).toBe("https://gmail.googleapis.com");
  });

  it("strips trailing slash from baseUrl", () => {
    const { baseUrl } = extractSpecMeta(
      JSON.stringify({ servers: [{ url: "https://example.com/" }] })
    );
    expect(baseUrl).toBe("https://example.com");
  });

  it("extracts tokenUrl from authorizationCode flow", () => {
    const spec = base3({
      components: {
        securitySchemes: {
          oauth2: {
            type: "oauth2",
            flows: {
              authorizationCode: {
                authorizationUrl: "https://accounts.google.com/o/oauth2/auth",
                tokenUrl: "https://oauth2.googleapis.com/token",
                scopes: { "https://www.googleapis.com/auth/gmail.readonly": "Read Gmail" },
              },
            },
          },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.tokenUrl).toBe("https://oauth2.googleapis.com/token");
    expect(meta.authorizationUrl).toBe("https://accounts.google.com/o/oauth2/auth");
  });

  it("extracts scopes as space-separated string", () => {
    const spec = base3({
      components: {
        securitySchemes: {
          oauth2: {
            type: "oauth2",
            flows: {
              authorizationCode: {
                authorizationUrl: "https://auth.example.com",
                tokenUrl: "https://token.example.com",
                scopes: { "read:data": "Read", "write:data": "Write" },
              },
            },
          },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.scopes).toContain("read:data");
    expect(meta.scopes).toContain("write:data");
  });

  it("prefers authorizationCode over clientCredentials", () => {
    const spec = base3({
      components: {
        securitySchemes: {
          oauth2: {
            type: "oauth2",
            flows: {
              clientCredentials: { tokenUrl: "https://cc.example.com/token", scopes: {} },
              authorizationCode: {
                authorizationUrl: "https://auth.example.com",
                tokenUrl: "https://ac.example.com/token",
                scopes: {},
              },
            },
          },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.tokenUrl).toBe("https://ac.example.com/token");
  });

  it("returns empty object for spec with no security schemes", () => {
    const meta = extractSpecMeta(base3());
    expect(meta.tokenUrl).toBeUndefined();
    expect(meta.authorizationUrl).toBeUndefined();
  });

  it("ignores non-oauth2 security schemes", () => {
    const spec = base3({
      components: {
        securitySchemes: {
          apiKey: { type: "apiKey", in: "header", name: "X-API-Key" },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.tokenUrl).toBeUndefined();
  });
});

// ── Swagger 2.0 ───────────────────────────────────────────────────────────────

describe("extractSpecMeta — Swagger 2.0", () => {
  it("extracts baseUrl from host + basePath + schemes", () => {
    const spec = JSON.stringify({
      host: "www.googleapis.com",
      basePath: "/drive/v3",
      schemes: ["https"],
    });
    const { baseUrl } = extractSpecMeta(spec);
    expect(baseUrl).toBe("https://www.googleapis.com/drive/v3");
  });

  it("defaults to https when schemes missing", () => {
    const spec = JSON.stringify({ host: "api.example.com", basePath: "/v1" });
    const { baseUrl } = extractSpecMeta(spec);
    expect(baseUrl).toBe("https://api.example.com/v1");
  });

  it("extracts tokenUrl and authorizationUrl from accessCode flow", () => {
    const spec = JSON.stringify({
      info: { title: "Drive API" },
      host: "www.googleapis.com",
      securityDefinitions: {
        Oauth2c: {
          type: "oauth2",
          flow: "accessCode",
          authorizationUrl: "https://accounts.google.com/o/oauth2/auth",
          tokenUrl: "https://accounts.google.com/o/oauth2/token",
          scopes: { "https://www.googleapis.com/auth/drive": "Drive access" },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.tokenUrl).toBe("https://accounts.google.com/o/oauth2/token");
    expect(meta.authorizationUrl).toBe("https://accounts.google.com/o/oauth2/auth");
  });

  it("extracts authorizationUrl from implicit flow (no tokenUrl)", () => {
    const spec = JSON.stringify({
      securityDefinitions: {
        Oauth2: {
          type: "oauth2",
          flow: "implicit",
          authorizationUrl: "https://accounts.google.com/o/oauth2/auth",
          scopes: { "https://www.googleapis.com/auth/drive": "Drive" },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.authorizationUrl).toBe("https://accounts.google.com/o/oauth2/auth");
  });

  it("extracts scopes from Swagger 2.0 definitions", () => {
    const spec = JSON.stringify({
      securityDefinitions: {
        oauth: {
          type: "oauth2",
          flow: "accessCode",
          authorizationUrl: "https://auth.example.com",
          tokenUrl: "https://token.example.com",
          scopes: { "scope:a": "A", "scope:b": "B" },
        },
      },
    });
    const meta = extractSpecMeta(spec);
    expect(meta.scopes).toContain("scope:a");
    expect(meta.scopes).toContain("scope:b");
  });
});

// ── Edge cases ────────────────────────────────────────────────────────────────

describe("extractSpecMeta — edge cases", () => {
  it("returns empty object for invalid JSON", () => {
    const meta = extractSpecMeta("not json at all");
    expect(meta).toEqual({});
  });

  it("returns empty object for empty string", () => {
    const meta = extractSpecMeta("");
    expect(meta).toEqual({});
  });

  it("returns empty object for empty spec", () => {
    const meta = extractSpecMeta("{}");
    expect(meta).toEqual({});
  });

  it("handles missing title gracefully", () => {
    const meta = extractSpecMeta(JSON.stringify({ servers: [{ url: "https://example.com" }] }));
    expect(meta.title).toBeUndefined();
    expect(meta.baseUrl).toBe("https://example.com");
  });
});
