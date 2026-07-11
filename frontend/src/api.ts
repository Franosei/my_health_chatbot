import type { AuthResponse, CarePlan, ChatStreamEvent, ClinicalNote, FeedbackRating, FeedbackResponse, ProductConfig, Snapshot } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const TOKEN_KEY = "flynnmed_token";

export function getStoredToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setStoredToken(token: string): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

type RequestOptions = RequestInit & {
  auth?: boolean;
};

async function readError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    return JSON.stringify(payload.detail ?? payload);
  } catch {
    return response.statusText || "Request failed.";
  }
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const hasFormData = options.body instanceof FormData;
  if (!hasFormData && options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (options.auth !== false) {
    const token = getStoredToken();
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export function getConfig(): Promise<ProductConfig> {
  return apiRequest<ProductConfig>("/api/config", { auth: false });
}

export function login(identifier: string, password: string): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/api/auth/login", {
    auth: false,
    method: "POST",
    body: JSON.stringify({ identifier, password })
  });
}

export function signup(payload: Record<string, unknown>): Promise<AuthResponse> {
  return apiRequest<AuthResponse>("/api/auth/signup", {
    auth: false,
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchSnapshot(): Promise<Snapshot> {
  return apiRequest<Snapshot>("/api/snapshot");
}

export async function streamChat(
  message: string,
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({ message })
  });

  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as ChatStreamEvent);
    } catch {
      // Ignore unparseable lines (partial flushes, heartbeats)
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) processLine(line);
    }
    if (buffer.trim()) processLine(buffer);
  } catch (err) {
    // The connection dropped mid-stream -- surface a clean error event
    const message = err instanceof Error ? err.message : "The connection was interrupted.";
    onEvent({ type: "error", message: `Stream interrupted: ${message}` });
  }
}

export async function streamImageAnalysis(
  message: string,
  image: File,
  onEvent: (event: ChatStreamEvent) => void
): Promise<void> {
  const token = getStoredToken();
  const form = new FormData();
  form.append("message", message);
  form.append("image", image);

  const response = await fetch(`${API_BASE}/api/chat/image/stream`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: form
  });

  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      onEvent(JSON.parse(trimmed) as ChatStreamEvent);
    } catch {
      // Ignore unparseable lines (partial flushes, heartbeats)
    }
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) processLine(line);
    }
    if (buffer.trim()) processLine(buffer);
  } catch (err) {
    const message = err instanceof Error ? err.message : "The connection was interrupted.";
    onEvent({ type: "error", message: `Stream interrupted: ${message}` });
  }
}

export function uploadDocuments(files: File[], processUnverified: boolean): Promise<{
  processed: { file: string }[];
  pending: unknown[];
  duplicates: { file: string; message: string }[];
  rejected: { file: string; message: string }[];
  snapshot: Snapshot;
}> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("process_unverified", String(processUnverified));
  return apiRequest("/api/uploads", {
    method: "POST",
    body: form
  });
}

export function transcribeAudio(file: File): Promise<{ text: string }> {
  const form = new FormData();
  form.append("audio", file);
  return apiRequest("/api/voice/transcribe", {
    method: "POST",
    body: form
  });
}

export function rateResponse(payload: {
  trace_id: string;
  message_id?: string;
  rating: FeedbackRating;
}): Promise<FeedbackResponse> {
  return apiRequest<FeedbackResponse>("/api/feedback", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

// ── Clinical notes ─────────────────────────────────────────────────────────

export function generateNote(payload: {
  question?: string;
  conversation_summary?: string;
  trace_id?: string;
}): Promise<{ note: ClinicalNote; snapshot: Snapshot }> {
  return apiRequest("/api/notes", { method: "POST", body: JSON.stringify(payload) });
}

export function updateNote(
  noteId: string,
  updates: Partial<Pick<ClinicalNote, "subjective" | "objective" | "assessment" | "plan" | "urgency_level" | "requires_gp_visit" | "gp_visit_reason">>
): Promise<{ note: ClinicalNote; snapshot: Snapshot }> {
  return apiRequest(`/api/notes/${noteId}`, { method: "PUT", body: JSON.stringify(updates) });
}

export function deleteNote(noteId: string): Promise<void> {
  return apiRequest(`/api/notes/${noteId}`, { method: "DELETE" });
}

export function emailNote(noteId: string): Promise<{ ok: boolean; sent_to: string; snapshot: Snapshot }> {
  return apiRequest(`/api/notes/${noteId}/email`, { method: "POST" });
}

export function sendUrgentAlert(reason: string, urgencyLevel: string): Promise<{ ok: boolean; sent_to: string }> {
  return apiRequest("/api/email/urgent", {
    method: "POST",
    body: JSON.stringify({ reason, urgency_level: urgencyLevel })
  });
}

export async function downloadProtectedFile(path: string, filename: string): Promise<void> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      Authorization: `Bearer ${token}`
    }
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// ── Care Plans ───────────────────────────────────────────────────────────────

export function listCarePlans(): Promise<CarePlan[]> {
  return apiRequest<CarePlan[]>("/api/care-plans");
}

export function deleteCarePlan(planId: string): Promise<{ ok: boolean }> {
  return apiRequest(`/api/care-plans/${planId}`, { method: "DELETE" });
}

export function toggleCarePlanTask(
  planId: string,
  taskId: string,
  done: boolean
): Promise<CarePlan> {
  return apiRequest<CarePlan>(`/api/care-plans/${planId}/tasks/${taskId}`, {
    method: "PATCH",
    body: JSON.stringify({ done })
  });
}

export function addAfterVisitNote(planId: string, note: string): Promise<CarePlan> {
  return apiRequest<CarePlan>(`/api/care-plans/${planId}/after-visit`, {
    method: "POST",
    body: JSON.stringify({ note })
  });
}

export function generateGpPrep(planId: string): Promise<{ gp_prep_summary: string; plan: CarePlan }> {
  return apiRequest(`/api/care-plans/${planId}/gp-prep`, { method: "POST" });
}

export async function generateCarePlan(
  condition: string,
  chatSummary: string,
  onProgress: (msg: string) => void
): Promise<CarePlan> {
  const token = getStoredToken();
  const response = await fetch(`${API_BASE}/api/care-plans/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`
    },
    body: JSON.stringify({ condition, chat_summary: chatSummary })
  });

  if (!response.ok || !response.body) {
    throw new Error(await readError(response));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPlan: CarePlan | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const event = JSON.parse(trimmed);
        if (event.type === "progress") onProgress(event.message as string);
        else if (event.type === "done") finalPlan = event.plan as CarePlan;
        else if (event.type === "error") throw new Error(event.message as string);
      } catch {
        /* partial flush */
      }
    }
  }

  if (!finalPlan) throw new Error("Care plan generation did not complete.");
  return finalPlan;
}
