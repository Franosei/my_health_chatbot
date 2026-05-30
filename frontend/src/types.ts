export type Dict<T = unknown> = Record<string, T>;

export type ProductConfig = {
  product_name: string;
  product_tagline: string;
  product_subtitle: string;
  founder_name: string;
  support_email: string;
  terms_version: string;
  role_options: string[];
  role_terms: Record<string, RoleTerms>;
  privacy_notice_points: string[];
};

export type RoleTerms = {
  title: string;
  summary: string;
  bullets: string[];
  acknowledgement: string;
};

export type Profile = {
  username: string;
  display_name?: string;
  email?: string;
  care_context?: string;
  role?: string;
  clinical_role?: string;
  organization?: string;
  follow_up_preferences?: string;
  date_of_birth?: string;
  biological_sex?: string;
  created_at?: string;
  last_login?: string;
};

export type Message = {
  role: "user" | "assistant" | string;
  content: string;
  timestamp?: string;
  sources?: Source[];
  trace_id?: string;
  metadata?: Dict<any>;
  message_id?: string;
};

export type Source = {
  source_id?: string;
  title?: string;
  journal?: string;
  year?: string | number;
  url?: string;
  snippet?: string;
  evidence_tier?: number;
  tier_label?: string;
  tier_description?: string;
};

export type Snapshot = {
  product: {
    name: string;
    tagline: string;
    subtitle: string;
    support_email: string;
  };
  user: string;
  profile: Profile;
  metrics: Record<string, number>;
  latest_triage: Dict<any>;
  chat_history: Message[];
  uploads: Dict<any>[];
  document_summaries: Dict<any>[];
  symptom_logs: Dict<any>[];
  medications: Dict<any>[];
  allergies: Dict<any>[];
  conditions: Dict<any>[];
  vitals: Dict<any>[];
  triage_summaries: Dict<any>[];
  traces: Dict<any>[];
  audit: Dict<any>[];
  memory: Dict<any>;
  trial_search_result?: TrialSearchResult | null;
};

export type AuthResponse = {
  token: string;
  profile: Profile;
  snapshot: Snapshot;
};

export type TrialSearchResult = {
  searched_at?: string;
  trials: Dict<any>[];
  condition_terms: string[];
  medication_terms: string[];
  location: string;
  error?: string;
};

export type ChatStreamEvent =
  | { type: "user_message"; message: Message }
  | { type: "assistant_message"; message: Message }
  | { type: "status"; message: string }
  | { type: "token"; delta: string }
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "error"; message: string; assistant_message?: Message }
  | { type: "done" };
