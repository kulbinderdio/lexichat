export type JobScheduleType = "Daily" | "Interval" | "Weekly" | "Manual";

export interface JobSchedule {
  type: JobScheduleType;
  hour?: number;
  minute?: number;
  hours?: number;
  weekday?: number;
}

export interface JobOpenAPISpec {
  id: string;
  title: string;
  base_url: string;
  spec_json: string;
  auth?: import("./AdminPanel").AuthConfig;
}

export interface JobMCPServer {
  id: string;
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  auth?: import("./AdminPanel").AuthConfig;
}

export interface JobProfileContext {
  ollama_host: string;
  allowed_dirs: string[];
  openapi_specs: JobOpenAPISpec[];
  mcp_servers: JobMCPServer[];
  disabled_mcp_tools: string[];
  snapshot_at: string;
}

export interface ScheduledJob {
  id: string;
  name: string;
  enabled: boolean;
  schedule: JobSchedule;
  prompt: string;
  model: string;
  system_prompt: string | null;
  enabled_builtin_tools: string[];
  output_file: string | null;
  created_at: string;
  last_run_at: string | null;
  profile_id: string | null;
  profile_name: string | null;
  profile_context: JobProfileContext | null;
}

export interface TraceToolCall {
  name: string;
  args: string;
  result: string | null;
}

export interface TraceStep {
  step: number;
  llm_text: string | null;
  tool_calls: TraceToolCall[];
}

export interface JobRun {
  id: string;
  job_id: string;
  job_name: string;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  status: "Success" | "Error";
  output: string;
  error: string | null;
  trace: TraceStep[];
  profile_name: string | null;
}
