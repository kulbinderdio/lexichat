import type { Profile, AuthConfig } from "./AdminPanel";

export interface ProfileExportEnvelope {
  lexichat_export_version: 1;
  exported_at: string;
  profile: Profile;
}

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

export function buildExportEnvelope(profile: Profile): ProfileExportEnvelope {
  return {
    lexichat_export_version: 1,
    exported_at: new Date().toISOString(),
    profile: {
      ...profile,
      // Strip auth from any per-profile auth overrides
      toolAuthOverrides: profile.toolAuthOverrides
        ? Object.fromEntries(
            Object.entries(profile.toolAuthOverrides).map(([id, auth]) => [id, stripAuth(auth)])
          )
        : undefined,
    },
  };
}

export function validateImport(raw: unknown): Profile | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (obj.lexichat_export_version !== 1) return null;
  const p = obj.profile as Partial<Profile>;
  if (typeof p?.name !== "string" || typeof p?.systemPrompt !== "string") return null;
  return p as Profile;
}

export function resolveImportName(name: string, existing: Profile[]): string {
  return existing.some(p => p.name === name) ? `${name} (imported)` : name;
}
