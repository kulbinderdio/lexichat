import type {
  Profile, AuthConfig, AppSettings,
  StoredOpenAPISpec, StoredSparqlEndpoint, StoredMCPServer,
} from "./AdminPanel";
import { BUILTIN_OPENAPI_SPEC_IDS, BUILTIN_SPARQL_ENDPOINT_IDS } from "./AdminPanel";

// Envelope v2 bundles the profile AND the definitions of every API it enables, with all
// secrets stripped, so an import wires the profile up identically on another machine.
// v1 (profile only) is still accepted on import for back-compat.
export interface ProfileExportEnvelope {
  lexichat_export_version: 2;
  exported_at: string;
  profile: Profile;
  toolRegistry: {
    openapiSpecs: StoredOpenAPISpec[];
    sparqlEndpoints: StoredSparqlEndpoint[];
    mcpServers: StoredMCPServer[];
  };
}

export interface ImportBundle {
  profile: Profile;
  openapiSpecs: StoredOpenAPISpec[];
  sparqlEndpoints: StoredSparqlEndpoint[];
  mcpServers: StoredMCPServer[];
}

/** Keep only non-secret auth fields (shape/URLs/usernames), never credentials. */
export function stripAuth(auth: AuthConfig): AuthConfig {
  return {
    type:              auth.type,
    api_key_header:    auth.api_key_header,
    token_url:         auth.token_url,
    client_id:         auth.client_id,
    scope:             auth.scope,
    authorization_url: auth.authorization_url,
    basic_username:    auth.basic_username,
    // Intentionally omitted: bearer_token, api_key_value, basic_password,
    // access_token, refresh_token, client_secret
  };
}

const stripSpec = (s: StoredOpenAPISpec): StoredOpenAPISpec =>
  ({ ...s, auth: s.auth ? stripAuth(s.auth) : undefined });
const stripEndpoint = (e: StoredSparqlEndpoint): StoredSparqlEndpoint =>
  ({ ...e, auth: e.auth ? stripAuth(e.auth) : undefined });
const stripServer = (m: StoredMCPServer): StoredMCPServer =>
  ({ ...m,
     auth: m.auth ? stripAuth(m.auth) : undefined,
     // env values are frequently secrets; keep the keys so the importer knows what to fill.
     env: Object.fromEntries(Object.keys(m.env ?? {}).map(k => [k, ""])) });

export function buildExportEnvelope(
  profile: Profile,
  registry: AppSettings["toolRegistry"],
): ProfileExportEnvelope {
  const specIds   = new Set(profile.enabledOpenapiSpecIds ?? []);
  const sparqlIds = new Set(profile.enabledSparqlEndpointIds ?? []);
  const mcpIds    = new Set(profile.enabledMcpServerIds ?? []);
  return {
    lexichat_export_version: 2,
    exported_at: new Date().toISOString(),
    profile: {
      ...profile,
      // Local absolute paths — useless on another machine, and don't leak the author's tree.
      allowedDirs: [],
      toolAuthOverrides: profile.toolAuthOverrides
        ? Object.fromEntries(Object.entries(profile.toolAuthOverrides).map(([id, a]) => [id, stripAuth(a)]))
        : undefined,
    },
    toolRegistry: {
      // Built-ins ship with every install — don't bundle them; the enabled-id still resolves
      // to the target's own built-in.
      openapiSpecs: registry.openapiSpecs
        .filter(s => specIds.has(s.id) && !BUILTIN_OPENAPI_SPEC_IDS.has(s.id))
        .map(stripSpec),
      sparqlEndpoints: registry.sparqlEndpoints
        .filter(e => sparqlIds.has(e.id) && !BUILTIN_SPARQL_ENDPOINT_IDS.has(e.id))
        .map(stripEndpoint),
      mcpServers: registry.mcpServers
        .filter(m => mcpIds.has(m.id))
        .map(stripServer),
    },
  };
}

/** Parse either a v1 or v2 envelope into a bundle, or null if it isn't a LexiChat profile. */
export function parseImport(raw: unknown): ImportBundle | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (obj.lexichat_export_version !== 1 && obj.lexichat_export_version !== 2) return null;
  const p = obj.profile as Partial<Profile> | undefined;
  if (typeof p?.name !== "string" || typeof p?.systemPrompt !== "string") return null;
  const reg = (obj.toolRegistry ?? {}) as Partial<ProfileExportEnvelope["toolRegistry"]>;
  return {
    profile: p as Profile,
    openapiSpecs:    Array.isArray(reg.openapiSpecs)    ? reg.openapiSpecs    : [],
    sparqlEndpoints: Array.isArray(reg.sparqlEndpoints) ? reg.sparqlEndpoints : [],
    mcpServers:      Array.isArray(reg.mcpServers)      ? reg.mcpServers      : [],
  };
}

export function resolveImportName(name: string, existing: Profile[]): string {
  return existing.some(p => p.name === name) ? `${name} (imported)` : name;
}

export interface ImportResult {
  settings: AppSettings;
  profileId: string;
  warnings: string[];
}

/**
 * Merge an imported bundle into settings: add any bundled APIs the registry doesn't already
 * have (preserving their ids so the profile's enabled-id references resolve), append the new
 * profile, and return warnings about what the user must still set up (credentials, model,
 * MCP command paths, sandbox).
 */
export function mergeImport(bundle: ImportBundle, settings: AppSettings, newProfileId: string): ImportResult {
  const reg = settings.toolRegistry;
  const haveSpec = new Set(reg.openapiSpecs.map(s => s.id));
  const haveEp   = new Set(reg.sparqlEndpoints.map(e => e.id));
  const haveSrv  = new Set(reg.mcpServers.map(m => m.id));

  const openapiSpecs    = [...reg.openapiSpecs,    ...bundle.openapiSpecs.filter(s => !haveSpec.has(s.id))];
  const sparqlEndpoints = [...reg.sparqlEndpoints, ...bundle.sparqlEndpoints.filter(e => !haveEp.has(e.id))];
  const mcpServers      = [...reg.mcpServers,      ...bundle.mcpServers.filter(m => !haveSrv.has(m.id))];

  const profile: Profile = {
    ...bundle.profile,
    id: newProfileId,
    name: resolveImportName(bundle.profile.name, settings.profiles),
    allowedDirs: [],
    enabledMcpServerIds:      bundle.profile.enabledMcpServerIds      ?? [],
    enabledOpenapiSpecIds:    bundle.profile.enabledOpenapiSpecIds    ?? [],
    enabledSparqlEndpointIds: bundle.profile.enabledSparqlEndpointIds ?? [],
    enabledTools:             bundle.profile.enabledTools             ?? {},
    maxTools:                 bundle.profile.maxTools                 ?? settings.maxTools,
  };

  const warnings: string[] = [];
  const needCreds = [
    ...bundle.openapiSpecs.filter(s => s.auth && s.auth.type !== "none").map(s => s.title),
    ...bundle.sparqlEndpoints.filter(e => e.auth && e.auth.type !== "none").map(e => e.title),
    ...bundle.mcpServers.filter(m => (m.auth && m.auth.type !== "none") || Object.keys(m.env ?? {}).length > 0).map(m => m.name),
  ];
  if (needCreds.length) warnings.push(`Re-enter credentials for: ${needCreds.join(", ")}.`);
  const stdio = bundle.mcpServers.filter(m => m.command && !/^https?:\/\//i.test(m.command)).map(m => m.name);
  if (stdio.length) warnings.push(`Set the local command path for MCP server(s): ${stdio.join(", ")}.`);
  if (profile.model && !settings.models.includes(profile.model)) {
    warnings.push(`Model "${profile.model}" isn't installed in Ollama on this machine.`);
  }
  warnings.push("Set this profile's sandbox folders if it uses file tools.");

  return {
    settings: { ...settings, toolRegistry: { openapiSpecs, sparqlEndpoints, mcpServers }, profiles: [...settings.profiles, profile] },
    profileId: newProfileId,
    warnings,
  };
}
