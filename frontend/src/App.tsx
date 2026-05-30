import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  BadgePlus,
  CalendarClock,
  CheckCircle2,
  ClipboardList,
  Download,
  FileDown,
  FileText,
  FlaskConical,
  HeartPulse,
  Home,
  LogOut,
  MessageSquare,
  Mic,
  Pill,
  Plus,
  Search,
  Send,
  ShieldCheck,
  Stethoscope,
  StopCircle,
  Trash2,
  Upload
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  apiRequest,
  downloadProtectedFile,
  fetchSnapshot,
  getConfig,
  getStoredToken,
  login,
  setStoredToken,
  signup,
  streamChat,
  transcribeAudio,
  uploadDocuments
} from "./api";
import type { AuthResponse, Dict, Message, ProductConfig, Snapshot, TrialSearchResult } from "./types";
import {
  buildSeries,
  buildSymptomSeries,
  buildTimelineEvents,
  buildTrendInsights,
  clean,
  formatDate,
  formatTimestamp,
  unique,
  vitalLabel
} from "./utils";

type View = "workspace" | "chat" | "timeline" | "trials";

const STARTER_PROMPTS = [
  "What symptoms would make chest pain an urgent medical review issue?",
  "Summarize the most important themes from my uploaded records in plain language.",
  "What does the recent evidence say about hypertension treatment in older adults?"
];

const COUNTRIES = [
  "United Kingdom",
  "United States",
  "Ghana",
  "Nigeria",
  "Canada",
  "Australia",
  "Germany",
  "France",
  "Netherlands",
  "Ireland",
  "Italy",
  "Spain",
  "India",
  "South Africa",
  "Kenya",
  "Brazil",
  "Mexico",
  "Japan",
  "Singapore",
  "United Arab Emirates"
];

const VITAL_OPTIONS = [
  ["blood_pressure", "Blood pressure", "mmHg", "120/80"],
  ["heart_rate", "Heart rate", "bpm", "72"],
  ["weight", "Weight", "kg", "72"],
  ["blood_glucose", "Blood glucose", "mmol/L", "5.4"],
  ["oxygen_saturation", "Oxygen saturation", "%", "98"],
  ["temperature", "Temperature", "C", "37.0"],
  ["hba1c", "HbA1c", "mmol/mol", "48"],
  ["egfr", "eGFR", "mL/min/1.73m2", "65"]
];

function App() {
  const [config, setConfig] = useState<ProductConfig | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [view, setView] = useState<View>("workspace");
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let mounted = true;
    async function boot() {
      try {
        const loadedConfig = await getConfig();
        if (!mounted) {
          return;
        }
        setConfig(loadedConfig);
        if (getStoredToken()) {
          const loadedSnapshot = await fetchSnapshot();
          if (mounted) {
            setSnapshot(loadedSnapshot);
          }
        }
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "Could not load the app.");
        setStoredToken("");
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    }
    boot();
    return () => {
      mounted = false;
    };
  }, []);

  const handleAuth = (response: AuthResponse) => {
    setStoredToken(response.token);
    setSnapshot(response.snapshot);
    setView("workspace");
    setNotice("");
  };

  const signOut = () => {
    setStoredToken("");
    setSnapshot(null);
    setView("workspace");
  };

  if (loading) {
    return (
      <main className="boot-screen">
        <img src="/assistant.png" alt="" />
        <span>Opening Dr. Charlotte</span>
      </main>
    );
  }

  if (!config) {
    return <main className="boot-screen">Configuration is unavailable.</main>;
  }

  if (!snapshot) {
    return <AuthScreen config={config} onSuccess={handleAuth} />;
  }

  return (
    <Shell snapshot={snapshot} view={view} setView={setView} signOut={signOut} notice={notice}>
      {view === "workspace" && <WorkspaceView snapshot={snapshot} setView={setView} setSnapshot={setSnapshot} />}
      {view === "chat" && <ChatView snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />}
      {view === "timeline" && <TimelineView snapshot={snapshot} />}
      {view === "trials" && <TrialsView snapshot={snapshot} setSnapshot={setSnapshot} setNotice={setNotice} />}
    </Shell>
  );
}

function AuthScreen({ config, onSuccess }: { config: ProductConfig; onSuccess: (response: AuthResponse) => void }) {
  const [mode, setMode] = useState<"Sign in" | "Create account">("Sign in");
  const [role, setRole] = useState(config.role_options[0]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loginForm, setLoginForm] = useState({ identifier: "", password: "" });
  const [signupForm, setSignupForm] = useState({
    full_name: "",
    email: "",
    username: "",
    organization: "",
    date_of_birth: "",
    biological_sex: "",
    password: "",
    confirm_password: "",
    accept_role_terms: false,
    accept_privacy: false
  });

  const terms = config.role_terms[role] ?? config.role_terms[config.role_options[0]];

  async function submitLogin(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      onSuccess(await login(loginForm.identifier, loginForm.password));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Sign in failed.");
    } finally {
      setBusy(false);
    }
  }

  async function submitSignup(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      onSuccess(
        await signup({
          ...signupForm,
          role
        })
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Account creation failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-intro">
        <div className="eyebrow">Secure health workspace</div>
        <h1>{config.product_name}</h1>
        <p>{config.product_tagline}</p>
        <img src="/workspace-preview.png" alt="" className="auth-preview" />
      </section>

      <section className="auth-panel" aria-label="Account access">
        <div className="segmented">
          {(["Sign in", "Create account"] as const).map((item) => (
            <button key={item} className={mode === item ? "active" : ""} onClick={() => setMode(item)} type="button">
              {item}
            </button>
          ))}
        </div>

        {error && <div className="notice error">{error}</div>}

        {mode === "Sign in" ? (
          <form onSubmit={submitLogin} className="stack">
            <label>
              Email or username
              <input
                value={loginForm.identifier}
                onChange={(event) => setLoginForm({ ...loginForm, identifier: event.target.value })}
                autoComplete="username"
              />
            </label>
            <label>
              Password
              <input
                type="password"
                value={loginForm.password}
                onChange={(event) => setLoginForm({ ...loginForm, password: event.target.value })}
                autoComplete="current-password"
              />
            </label>
            <button className="primary full" disabled={busy} type="submit">
              <ShieldCheck size={18} />
              Sign in
            </button>
            <p className="support-line">Support: {config.support_email}</p>
          </form>
        ) : (
          <form onSubmit={submitSignup} className="stack">
            <label>
              Account role
              <select value={role} onChange={(event) => setRole(event.target.value)}>
                {config.role_options.map((option) => (
                  <option key={option}>{option}</option>
                ))}
              </select>
            </label>
            <div className="terms-panel">
              <strong>{terms.title}</strong>
              <p>{terms.summary}</p>
              <ul>
                {terms.bullets.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
            <div className="form-grid">
              <label>
                Full name
                <input
                  value={signupForm.full_name}
                  onChange={(event) => setSignupForm({ ...signupForm, full_name: event.target.value })}
                  autoComplete="name"
                />
              </label>
              <label>
                Email address
                <input
                  type="email"
                  value={signupForm.email}
                  onChange={(event) => setSignupForm({ ...signupForm, email: event.target.value })}
                  autoComplete="email"
                />
              </label>
              <label>
                Username
                <input
                  value={signupForm.username}
                  onChange={(event) => setSignupForm({ ...signupForm, username: event.target.value })}
                  autoComplete="username"
                />
              </label>
              <label>
                Organisation
                <input
                  value={signupForm.organization}
                  onChange={(event) => setSignupForm({ ...signupForm, organization: event.target.value })}
                />
              </label>
              <label>
                Date of birth
                <input
                  type="date"
                  value={signupForm.date_of_birth}
                  onChange={(event) => setSignupForm({ ...signupForm, date_of_birth: event.target.value })}
                />
              </label>
              <label>
                Biological sex
                <select
                  value={signupForm.biological_sex}
                  onChange={(event) => setSignupForm({ ...signupForm, biological_sex: event.target.value })}
                >
                  <option value="">Prefer not to say</option>
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </label>
              <label>
                Password
                <input
                  type="password"
                  value={signupForm.password}
                  onChange={(event) => setSignupForm({ ...signupForm, password: event.target.value })}
                  autoComplete="new-password"
                />
              </label>
              <label>
                Confirm password
                <input
                  type="password"
                  value={signupForm.confirm_password}
                  onChange={(event) => setSignupForm({ ...signupForm, confirm_password: event.target.value })}
                  autoComplete="new-password"
                />
              </label>
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={signupForm.accept_role_terms}
                onChange={(event) => setSignupForm({ ...signupForm, accept_role_terms: event.target.checked })}
              />
              {terms.acknowledgement}
            </label>
            <label className="check-row">
              <input
                type="checkbox"
                checked={signupForm.accept_privacy}
                onChange={(event) => setSignupForm({ ...signupForm, accept_privacy: event.target.checked })}
              />
              I have read the privacy notice and understand account support is available at {config.support_email}.
            </label>
            <button className="primary full" disabled={busy} type="submit">
              <BadgePlus size={18} />
              Create account
            </button>
          </form>
        )}
      </section>
    </main>
  );
}

function Shell({
  children,
  snapshot,
  view,
  setView,
  signOut,
  notice
}: {
  children: React.ReactNode;
  snapshot: Snapshot;
  view: View;
  setView: (view: View) => void;
  signOut: () => void;
  notice: string;
}) {
  const nav = [
    { id: "workspace" as const, label: "Home", icon: Home },
    { id: "chat" as const, label: "Chat", icon: MessageSquare },
    { id: "timeline" as const, label: "Timeline", icon: CalendarClock },
    { id: "trials" as const, label: "Trials", icon: FlaskConical }
  ];
  const name = clean(snapshot.profile.display_name, snapshot.user);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <img src="/assistant.png" alt="" />
          <div>
            <strong>{snapshot.product.name}</strong>
            <span>{clean(snapshot.profile.clinical_role || snapshot.profile.role, "Patient / Individual")}</span>
          </div>
        </div>
        <nav>
          {nav.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}>
                <Icon size={19} />
                {item.label}
              </button>
            );
          })}
        </nav>
        <button className="ghost sidebar-signout" onClick={signOut}>
          <LogOut size={18} />
          Sign out
        </button>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div>
            <span className="eyebrow">Dr. Charlotte</span>
            <h1>{name}</h1>
          </div>
          <button className="icon-button" onClick={signOut} title="Sign out" aria-label="Sign out">
            <LogOut size={20} />
          </button>
        </header>
        {notice && <div className="notice">{notice}</div>}
        {children}
      </main>

      <nav className="mobile-nav" aria-label="Primary navigation">
        {nav.map((item) => {
          const Icon = item.icon;
          return (
            <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}>
              <Icon size={20} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );
}

function WorkspaceView({
  snapshot,
  setView,
  setSnapshot
}: {
  snapshot: Snapshot;
  setView: (view: View) => void;
  setSnapshot: (snapshot: Snapshot) => void;
}) {
  const metrics: Array<[string, number, LucideIcon]> = [
    ["Documents", snapshot.metrics.documents, FileText],
    ["Symptoms", snapshot.metrics.symptoms, Activity],
    ["Conditions", snapshot.metrics.conditions, Stethoscope],
    ["Medications", snapshot.metrics.medications, Pill],
    ["Vitals/labs", snapshot.metrics.vitals, HeartPulse]
  ];

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Workspace</span>
          <h2>Choose where to work today.</h2>
          <p>Chat, review the longitudinal record, or search recruiting studies with saved account context.</p>
        </div>
        {snapshot.latest_triage?.next_step && (
          <div className="triage-strip">
            <strong>{clean(snapshot.latest_triage.urgency_level, "Routine")}</strong>
            <span>{clean(snapshot.latest_triage.next_step, "Self-care")}</span>
          </div>
        )}
      </section>

      <section className="metric-grid">
        {metrics.map(([label, value, Icon]) => (
          <div className="metric-card" key={label}>
            <Icon size={20} />
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <section className="action-grid">
        <ActionButton
          icon={<MessageSquare size={21} />}
          title="Chat"
          body="Ask a question, continue a conversation, upload records, and save structured health context."
          onClick={() => setView("chat")}
        />
        <ActionButton
          icon={<CalendarClock size={21} />}
          title="Health Timeline"
          body="Review conditions, medications, allergies, readings, uploaded records, and trend cards."
          onClick={() => setView("timeline")}
        />
        <ActionButton
          icon={<Search size={21} />}
          title="Find Clinical Trials"
          body="Rank recruiting ClinicalTrials.gov records against your saved account profile."
          onClick={() => setView("trials")}
        />
      </section>

      <RecordPanel snapshot={snapshot} setSnapshot={setSnapshot} compact />
    </div>
  );
}

function ActionButton({
  icon,
  title,
  body,
  onClick
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  onClick: () => void;
}) {
  return (
    <button className="action-card" onClick={onClick}>
      <span className="action-icon">{icon}</span>
      <strong>{title}</strong>
      <p>{body}</p>
    </button>
  );
}

function ChatView({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [messages, setMessages] = useState<Message[]>(snapshot.chat_history);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [streamText, setStreamText] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMessages(snapshot.chat_history);
  }, [snapshot.chat_history]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamText, status]);

  async function sendMessage(text = draft) {
    const message = text.trim();
    if (!message || busy) {
      return;
    }
    setDraft("");
    setBusy(true);
    setStatus("Starting evidence review...");
    setStreamText("");
    try {
      await streamChat(message, (event) => {
        if (event.type === "user_message") {
          setMessages((current) => [...current, event.message]);
        }
        if (event.type === "status") {
          setStatus(event.message);
        }
        if (event.type === "token") {
          setStreamText((current) => current + event.delta);
        }
        if (event.type === "assistant_message") {
          setMessages((current) => [...current, event.message]);
          setStreamText("");
          setStatus("");
        }
        if (event.type === "snapshot") {
          setSnapshot(event.snapshot);
        }
        if (event.type === "error") {
          setNotice(event.message);
          if (event.assistant_message) {
            setMessages((current) => [...current, event.assistant_message as Message]);
          }
          setStreamText("");
          setStatus("");
        }
      });
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Message failed.");
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  return (
    <div className="chat-layout">
      <section className="chat-panel">
        <div className="chat-head">
          <div>
            <span className="eyebrow">Evidence chat</span>
            <h2>Conversation</h2>
          </div>
          <button
            className="ghost"
            onClick={async () => {
              setSnapshot(await apiRequest<Snapshot>("/api/chat", { method: "DELETE" }));
              setMessages([]);
            }}
            title="Clear chat"
          >
            <Trash2 size={17} />
            Clear
          </button>
        </div>

        {!messages.length && (
          <div className="starter-row">
            {STARTER_PROMPTS.map((prompt) => (
              <button key={prompt} onClick={() => sendMessage(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        )}

        <div className="message-list" ref={listRef}>
          {messages.map((message, index) => (
            <MessageBubble key={message.message_id ?? `${message.role}-${index}`} message={message} onFollowUp={sendMessage} />
          ))}
          {streamText && (
            <MessageBubble
              message={{
                role: "assistant",
                content: `${streamText}\n\n...`,
                metadata: {},
                sources: []
              }}
              transient
            />
          )}
          {status && (
            <div className="status-line">
              <span className="status-dots"><span /><span /><span /></span>
              {status}
            </div>
          )}
        </div>

        <div className="composer">
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask a health question, request an evidence summary, or continue your saved conversation..."
            rows={4}
          />
          <div className="composer-actions">
            <VoiceRecorder onTranscript={(text) => setDraft((current) => `${current}${current ? "\n" : ""}${text}`)} />
            <button className="ghost" onClick={() => setDraft("")} type="button">
              Clear
            </button>
            <button className="primary" onClick={() => sendMessage()} disabled={busy}>
              <Send size={18} />
              Send
            </button>
          </div>
        </div>
      </section>

      <aside className="side-panel">
        <UploadPanel setSnapshot={setSnapshot} setNotice={setNotice} />
        <ExportPanel snapshot={snapshot} setNotice={setNotice} />
        <RecordPanel snapshot={snapshot} setSnapshot={setSnapshot} compact={false} />
      </aside>
    </div>
  );
}

function MessageBubble({ message, transient = false, onFollowUp }: { message: Message; transient?: boolean; onFollowUp?: (text: string) => void }) {
  const isUser = message.role === "user";
  const metadata = message.metadata ?? {};
  const triage = metadata.triage_summary;
  const alerts = metadata.medication_alerts ?? [];
  const imageSrc = metadata.image_b64 ? `data:image/png;base64,${metadata.image_b64}` : metadata.image_url;

  return (
    <article className={`message ${isUser ? "user-message" : "assistant-message"} ${transient ? "transient" : ""}`}>
      <img src={isUser ? "/user.png" : "/assistant.png"} alt="" />
      <div className="message-body">
        <div className="message-meta">
          <strong>{isUser ? "You" : "Dr. Charlotte"}</strong>
          {message.timestamp && <span>{formatTimestamp(message.timestamp)}</span>}
          {message.trace_id && <span>{message.trace_id}</span>}
        </div>
        <div className="markdown">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {imageSrc && (
          <figure className="media-block">
            <img src={imageSrc} alt={clean(metadata.image_caption, "Generated illustration")} />
            <figcaption>{clean(metadata.image_caption, "AI-generated educational illustration")}</figcaption>
          </figure>
        )}
        {metadata.video_url && (
          <figure className="media-block">
            <video src={metadata.video_url} controls />
            <figcaption>{clean(metadata.video_caption, "Generated educational video")}</figcaption>
          </figure>
        )}
        {metadata.video_rate_limit_msg && <div className="notice warn">{metadata.video_rate_limit_msg}</div>}
        {triage && Object.keys(triage).length > 0 && <TriageCard summary={triage} />}
        {alerts.length > 0 && <MedicationAlerts alerts={alerts} />}
        {message.sources && message.sources.length > 0 && <SourceList sources={message.sources} />}
        {!isUser && metadata.follow_up_questions?.length > 0 && (
          <div className="follow-ups">
            {metadata.follow_up_questions.slice(0, 3).map((question: string) => (
              <button key={question} className="follow-up-btn" onClick={() => onFollowUp?.(question)}>
                {question}
              </button>
            ))}
          </div>
        )}
        {!isUser && metadata.trace && Object.keys(metadata.trace).length > 0 && (
          <EvidenceBasis trace={metadata.trace} sourceCount={message.sources?.length ?? 0} />
        )}
      </div>
    </article>
  );
}

function TriageCard({ summary }: { summary: Dict<any> }) {
  const monitor = Array.isArray(summary.what_to_monitor) ? summary.what_to_monitor.slice(0, 3) : [];
  const actions = Array.isArray(summary.immediate_actions) ? summary.immediate_actions.slice(0, 3) : [];
  const escalation = Array.isArray(summary.escalation_triggers) ? summary.escalation_triggers.slice(0, 3) : [];
  return (
    <div className="triage-card">
      <div>
        <span>Structured triage</span>
        <strong>{clean(summary.next_step, "Self-care")}</strong>
      </div>
      <p>
        <b>Urgency:</b> {clean(summary.urgency_level, "Routine")}
      </p>
      {summary.pathway_label && <p>{summary.pathway_label}</p>}
      {!!monitor.length && (
        <ul>
          {monitor.map((item: string) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      )}
      {!!actions.length && <p>{actions.join(" ")}</p>}
      {!!escalation.length && <p className="escalation">Escalate if: {escalation.join("; ")}</p>}
    </div>
  );
}

function MedicationAlerts({ alerts }: { alerts: Dict<any>[] }) {
  return (
    <div className="alert-list">
      <strong>Medication interaction check</strong>
      {alerts.slice(0, 3).map((alert) => (
        <div key={clean(alert.pair)} className={`interaction ${clean(alert.severity, "mentioned")}`}>
          <span>{clean(alert.pair, "Medication pair")}</span>
          <p>{clean(alert.summary)}</p>
        </div>
      ))}
    </div>
  );
}

const TIER_LABELS: Record<number, string> = {
  1: "NHS / NICE guidelines",
  2: "Systematic reviews",
  3: "Research papers",
};

const RISK_PLAIN: Record<string, { label: string; color: string }> = {
  routine:  { label: "Routine assessment",             color: "var(--green)" },
  elevated: { label: "Careful attention recommended",  color: "var(--gold)" },
  high:     { label: "Prompt review advised",          color: "var(--accent)" },
  urgent:   { label: "Urgent review recommended",      color: "var(--danger)" },
  crisis:   { label: "Seek immediate help",            color: "var(--danger)" },
};

function EvidenceBasis({ trace, sourceCount }: { trace: Dict<any>; sourceCount: number }) {
  const tiers: number[] = Array.isArray(trace.evidence_tiers_present) ? trace.evidence_tiers_present : [];
  const risk = RISK_PLAIN[String(trace.risk_level ?? "").toLowerCase()];
  const escalated: boolean = !!trace.escalation_triggered;
  const crisis: boolean = !!trace.crisis_detected;

  const tierBadges = tiers.map((t) => TIER_LABELS[t]).filter(Boolean);

  return (
    <div className="evidence-basis">
      <span className="evidence-label">Evidence basis</span>
      <div className="evidence-chips">
        {tierBadges.map((label) => (
          <span key={label} className="ev-chip ev-chip--source">{label}</span>
        ))}
        {sourceCount > 0 && (
          <span className="ev-chip ev-chip--count">{sourceCount} source{sourceCount === 1 ? "" : "s"} reviewed</span>
        )}
        {risk && (
          <span className="ev-chip ev-chip--risk" style={{ color: risk.color, borderColor: risk.color }}>
            {risk.label}
          </span>
        )}
        {(escalated || crisis) && (
          <span className="ev-chip ev-chip--alert">Safety guidance applied</span>
        )}
      </div>
    </div>
  );
}

function SourceList({ sources }: { sources: NonNullable<Message["sources"]> }) {
  return (
    <details className="details-block">
      <summary>{sources.length} source{sources.length === 1 ? "" : "s"}</summary>
      <div className="source-grid">
        {sources.map((source, index) => (
          <a key={source.source_id ?? index} href={source.url || undefined} target="_blank" rel="noreferrer">
            <span>{source.source_id ?? `S${index + 1}`}</span>
            <strong>{clean(source.title, "Untitled source")}</strong>
            <small>{unique([source.journal, source.year, source.tier_label]).join(" - ")}</small>
          </a>
        ))}
      </div>
    </details>
  );
}

function VoiceRecorder({ onTranscript }: { onTranscript: (text: string) => void }) {
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  async function toggle() {
    if (recording) {
      mediaRecorder.current?.stop();
      setRecording(false);
      return;
    }
    if (!navigator.mediaDevices || typeof MediaRecorder === "undefined") {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const recorder = new MediaRecorder(stream);
    chunks.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) {
        chunks.current.push(event.data);
      }
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunks.current, { type: "audio/webm" });
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      setBusy(true);
      try {
        const result = await transcribeAudio(file);
        if (result.text) {
          onTranscript(result.text);
        }
      } finally {
        setBusy(false);
      }
    };
    mediaRecorder.current = recorder;
    recorder.start();
    setRecording(true);
  }

  return (
    <button className="ghost" onClick={toggle} type="button" disabled={busy} title="Voice input">
      {recording ? <StopCircle size={18} /> : <Mic size={18} />}
      {recording ? "Stop" : busy ? "Transcribing" : "Voice"}
    </button>
  );
}

function UploadPanel({
  setSnapshot,
  setNotice
}: {
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [processUnverified, setProcessUnverified] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!files.length) {
      return;
    }
    setBusy(true);
    try {
      const result = await uploadDocuments(files, processUnverified);
      setSnapshot(result.snapshot);
      setFiles([]);
      if (result.pending.length) {
        setNotice(`${result.pending.length} document needs review before extraction.`);
      } else {
        setNotice(`${result.processed.length} document processed.`);
      }
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="tool-panel">
      <div className="panel-head">
        <Upload size={19} />
        <strong>Documents</strong>
      </div>
      <input type="file" accept="application/pdf" multiple onChange={(event) => setFiles(Array.from(event.target.files ?? []))} />
      <label className="check-row">
        <input type="checkbox" checked={processUnverified} onChange={(event) => setProcessUnverified(event.target.checked)} />
        Process even when the patient name needs review
      </label>
      <button className="primary full" onClick={submit} disabled={busy || !files.length}>
        <Upload size={18} />
        Upload PDFs
      </button>
    </section>
  );
}

function ExportPanel({ snapshot, setNotice }: { snapshot: Snapshot; setNotice: (notice: string) => void }) {
  async function download(path: string, filename: string) {
    try {
      await downloadProtectedFile(path, filename);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Download failed.");
    }
  }

  return (
    <section className="tool-panel">
      <div className="panel-head">
        <FileDown size={19} />
        <strong>Exports</strong>
      </div>
      <button className="ghost full" onClick={() => download("/api/export/summary.pdf", `${snapshot.user}-health-summary.pdf`)}>
        <Download size={17} />
        Health summary PDF
      </button>
      <button className="ghost full" onClick={() => download("/api/export/account", `${snapshot.user}-audit.json`)}>
        <Download size={17} />
        Account export
      </button>
    </section>
  );
}

function RecordPanel({
  snapshot,
  setSnapshot,
  compact
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  compact: boolean;
}) {
  const [profileOpen, setProfileOpen] = useState(false);
  const [profile, setProfile] = useState({
    display_name: clean(snapshot.profile.display_name),
    email: clean(snapshot.profile.email),
    care_context: clean(snapshot.profile.care_context),
    organization: clean(snapshot.profile.organization),
    date_of_birth: clean(snapshot.profile.date_of_birth),
    biological_sex: clean(snapshot.profile.biological_sex)
  });
  const [symptom, setSymptom] = useState({ symptom: "", logged_for: new Date().toISOString().slice(0, 10), severity: 5, triggers: "", notes: "" });
  const [condition, setCondition] = useState({ name: "", status: "active", recorded_on: "", notes: "" });
  const [medication, setMedication] = useState({ name: "", dose: "", schedule: "", reason: "", started_on: "", notes: "" });
  const [allergy, setAllergy] = useState({ name: "", reaction: "", severity: "unknown", allergy_type: "other", confirmed: true, notes: "" });
  const [vitals, setVitals] = useState({ type: "blood_pressure", value: "", unit: "mmHg", recorded_on: new Date().toISOString().slice(0, 10), notes: "" });

  async function save(path: string, payload: Record<string, unknown>, reset?: () => void) {
    const updated = await apiRequest<Snapshot>(path, {
      method: "POST",
      body: JSON.stringify(payload)
    });
    setSnapshot(updated);
    reset?.();
  }

  async function remove(path: string) {
    setSnapshot(await apiRequest<Snapshot>(path, { method: "DELETE" }));
  }

  const recentItems = [
    ...snapshot.conditions.slice(0, 3).map((item) => ({ kind: "Condition", title: clean(item.name), id: clean(item.condition_id), path: `/api/conditions/${item.condition_id}` })),
    ...snapshot.medications.slice(0, 3).map((item) => ({ kind: "Medication", title: clean(item.name), id: clean(item.medication_id), path: `/api/medications/${item.medication_id}` })),
    ...snapshot.allergies.slice(0, 2).map((item) => ({ kind: "Allergy", title: clean(item.name), id: clean(item.allergy_id), path: `/api/allergies/${item.allergy_id}` }))
  ];

  return (
    <section className={compact ? "record-panel compact" : "record-panel"}>
      <div className="panel-head">
        <ClipboardList size={19} />
        <strong>Health record</strong>
      </div>

      <details open={!compact || profileOpen} onToggle={(event) => setProfileOpen(event.currentTarget.open)}>
        <summary>Profile</summary>
        <form
          className="stack"
          onSubmit={async (event) => {
            event.preventDefault();
            setSnapshot(
              await apiRequest<Snapshot>("/api/profile", {
                method: "PUT",
                body: JSON.stringify(profile)
              })
            );
          }}
        >
          <label>
            Display name
            <input value={profile.display_name} onChange={(event) => setProfile({ ...profile, display_name: event.target.value })} />
          </label>
          <label>
            Email
            <input value={profile.email} onChange={(event) => setProfile({ ...profile, email: event.target.value })} />
          </label>
          <label>
            Care context
            <input value={profile.care_context} onChange={(event) => setProfile({ ...profile, care_context: event.target.value })} />
          </label>
          <div className="form-grid">
            <label>
              Date of birth
              <input type="date" value={profile.date_of_birth} onChange={(event) => setProfile({ ...profile, date_of_birth: event.target.value })} />
            </label>
            <label>
              Biological sex
              <select value={profile.biological_sex} onChange={(event) => setProfile({ ...profile, biological_sex: event.target.value })}>
                <option value="">Prefer not to say</option>
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </label>
          </div>
          <button className="primary full" type="submit">
            <CheckCircle2 size={18} />
            Save profile
          </button>
        </form>
      </details>

      <details open={!compact}>
        <summary>Symptoms</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/symptoms", symptom, () => setSymptom({ ...symptom, symptom: "", notes: "", triggers: "" }));
          }}
        >
          <input placeholder="Symptom" value={symptom.symptom} onChange={(event) => setSymptom({ ...symptom, symptom: event.target.value })} />
          <div className="form-grid">
            <input type="date" value={symptom.logged_for} onChange={(event) => setSymptom({ ...symptom, logged_for: event.target.value })} />
            <label className="range-label">
              Severity {symptom.severity}/10
              <input type="range" min="0" max="10" value={symptom.severity} onChange={(event) => setSymptom({ ...symptom, severity: Number(event.target.value) })} />
            </label>
          </div>
          <input placeholder="Triggers" value={symptom.triggers} onChange={(event) => setSymptom({ ...symptom, triggers: event.target.value })} />
          <textarea placeholder="Notes" value={symptom.notes} onChange={(event) => setSymptom({ ...symptom, notes: event.target.value })} rows={2} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save symptom
          </button>
        </form>
        <MiniList items={snapshot.symptom_logs.slice(0, 5)} titleKey="symptom" detailKey="logged_for" deleteKey="log_id" deletePath="/api/symptoms" onDelete={remove} />
      </details>

      <details>
        <summary>Conditions</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/conditions", condition, () => setCondition({ name: "", status: "active", recorded_on: "", notes: "" }));
          }}
        >
          <input placeholder="Condition" value={condition.name} onChange={(event) => setCondition({ ...condition, name: event.target.value })} />
          <div className="form-grid">
            <select value={condition.status} onChange={(event) => setCondition({ ...condition, status: event.target.value })}>
              <option value="active">Active</option>
              <option value="past">Past</option>
              <option value="resolved">Resolved</option>
              <option value="unknown">Unknown</option>
            </select>
            <input type="date" value={condition.recorded_on} onChange={(event) => setCondition({ ...condition, recorded_on: event.target.value })} />
          </div>
          <textarea placeholder="Notes" value={condition.notes} onChange={(event) => setCondition({ ...condition, notes: event.target.value })} rows={2} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save condition
          </button>
        </form>
      </details>

      <details>
        <summary>Medications</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/medications", medication, () => setMedication({ name: "", dose: "", schedule: "", reason: "", started_on: "", notes: "" }));
          }}
        >
          <input placeholder="Medication" value={medication.name} onChange={(event) => setMedication({ ...medication, name: event.target.value })} />
          <div className="form-grid">
            <input placeholder="Dose" value={medication.dose} onChange={(event) => setMedication({ ...medication, dose: event.target.value })} />
            <input placeholder="Schedule" value={medication.schedule} onChange={(event) => setMedication({ ...medication, schedule: event.target.value })} />
          </div>
          <input placeholder="Reason" value={medication.reason} onChange={(event) => setMedication({ ...medication, reason: event.target.value })} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save medication
          </button>
        </form>
      </details>

      <details>
        <summary>Allergies</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/allergies", allergy, () => setAllergy({ name: "", reaction: "", severity: "unknown", allergy_type: "other", confirmed: true, notes: "" }));
          }}
        >
          <input placeholder="Allergy" value={allergy.name} onChange={(event) => setAllergy({ ...allergy, name: event.target.value })} />
          <input placeholder="Reaction" value={allergy.reaction} onChange={(event) => setAllergy({ ...allergy, reaction: event.target.value })} />
          <div className="form-grid">
            <select value={allergy.severity} onChange={(event) => setAllergy({ ...allergy, severity: event.target.value })}>
              <option value="unknown">Unknown</option>
              <option value="mild">Mild</option>
              <option value="moderate">Moderate</option>
              <option value="severe">Severe</option>
            </select>
            <select value={allergy.allergy_type} onChange={(event) => setAllergy({ ...allergy, allergy_type: event.target.value })}>
              <option value="drug">Drug</option>
              <option value="food">Food</option>
              <option value="environmental">Environmental</option>
              <option value="other">Other</option>
            </select>
          </div>
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save allergy
          </button>
        </form>
      </details>

      <details>
        <summary>Vitals and labs</summary>
        <form
          className="stack"
          onSubmit={(event) => {
            event.preventDefault();
            save("/api/vitals", vitals, () => setVitals({ ...vitals, value: "", notes: "" }));
          }}
        >
          <select
            value={vitals.type}
            onChange={(event) => {
              const option = VITAL_OPTIONS.find((item) => item[0] === event.target.value);
              setVitals({ ...vitals, type: event.target.value, unit: option?.[2] ?? "" });
            }}
          >
            {VITAL_OPTIONS.map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
          <div className="form-grid">
            <input placeholder={VITAL_OPTIONS.find((item) => item[0] === vitals.type)?.[3]} value={vitals.value} onChange={(event) => setVitals({ ...vitals, value: event.target.value })} />
            <input placeholder="Unit" value={vitals.unit} onChange={(event) => setVitals({ ...vitals, unit: event.target.value })} />
          </div>
          <input type="date" value={vitals.recorded_on} onChange={(event) => setVitals({ ...vitals, recorded_on: event.target.value })} />
          <button className="primary full" type="submit">
            <Plus size={18} />
            Save reading
          </button>
        </form>
        <MiniList items={snapshot.vitals.slice(0, 5)} titleKey="type" detailKey="value" deleteKey="vitals_id" deletePath="/api/vitals" onDelete={remove} transformTitle={vitalLabel} />
      </details>

      {compact && !!recentItems.length && (
        <div className="mini-list">
          {recentItems.map((item) => (
            <div key={`${item.kind}-${item.id}`}>
              <span>{item.kind}</span>
              <strong>{item.title}</strong>
              <button className="icon-button" onClick={() => remove(item.path)} title="Remove" aria-label="Remove">
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function MiniList({
  items,
  titleKey,
  detailKey,
  deleteKey,
  deletePath,
  onDelete,
  transformTitle
}: {
  items: Dict<any>[];
  titleKey: string;
  detailKey: string;
  deleteKey: string;
  deletePath: string;
  onDelete: (path: string) => Promise<void>;
  transformTitle?: (value: unknown) => string;
}) {
  if (!items.length) {
    return <p className="muted">No saved records yet.</p>;
  }
  return (
    <div className="mini-list">
      {items.map((item) => (
        <div key={clean(item[deleteKey])}>
          <span>{clean(item[detailKey])}</span>
          <strong>{transformTitle ? transformTitle(item[titleKey]) : clean(item[titleKey])}</strong>
          <button className="icon-button" onClick={() => onDelete(`${deletePath}/${item[deleteKey]}`)} title="Remove" aria-label="Remove">
            <Trash2 size={15} />
          </button>
        </div>
      ))}
    </div>
  );
}

function TimelineView({ snapshot }: { snapshot: Snapshot }) {
  const [tab, setTab] = useState<"summary" | "timeline" | "trends">("summary");
  const events = useMemo(() => buildTimelineEvents(snapshot), [snapshot]);
  const insights = useMemo(() => buildTrendInsights(snapshot), [snapshot]);
  const activeConditions = snapshot.conditions.filter((item) => clean(item.status, "unknown").toLowerCase() === "active");
  const latestVitalsByType = unique(snapshot.vitals.map((item) => item.type)).map((type) =>
    snapshot.vitals.find((entry) => clean(entry.type).toLowerCase() === type.toLowerCase())
  );

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Health timeline</span>
          <h2>Saved context and trends.</h2>
          <p>Review the account record separately from the chat thread.</p>
        </div>
      </section>

      <div className="segmented tabs">
        <button className={tab === "summary" ? "active" : ""} onClick={() => setTab("summary")}>
          Summary
        </button>
        <button className={tab === "timeline" ? "active" : ""} onClick={() => setTab("timeline")}>
          Timeline
        </button>
        <button className={tab === "trends" ? "active" : ""} onClick={() => setTab("trends")}>
          Trends
        </button>
      </div>

      {tab === "summary" && (
        <section className="summary-grid">
          <SummaryCard title="Current health snapshot" count={`${activeConditions.length} active`}>
            {activeConditions.length ? activeConditions.slice(0, 6).map((item) => clean(item.name)).join(", ") : "No active conditions saved yet."}
          </SummaryCard>
          <SummaryCard title="Recent readings" count={`${latestVitalsByType.length} types`}>
            {latestVitalsByType.length
              ? latestVitalsByType
                  .slice(0, 8)
                  .map((item) => `${vitalLabel(item?.type)} ${clean(item?.value)} ${clean(item?.unit)}`)
                  .join("; ")
              : "No readings saved yet."}
          </SummaryCard>
          <SummaryCard title="Medicines and allergies" count={`${snapshot.medications.length + snapshot.allergies.length} saved`}>
            {unique([...snapshot.medications.map((item) => item.name), ...snapshot.allergies.map((item) => `Allergy: ${item.name}`)]).slice(0, 8).join("; ") ||
              "No medications or allergies recorded."}
          </SummaryCard>
          <SummaryCard title="Latest triage" count={clean(snapshot.latest_triage.urgency_level, "Routine")}>
            {clean(snapshot.latest_triage.next_step, "No triage summary saved yet.")}
          </SummaryCard>
          <SummaryCard title="Saved history notes" count={clean(snapshot.memory?.updated_at, "") ? formatDate(snapshot.memory.updated_at) : "Not updated"}>
            {clean(snapshot.memory?.summary, "No longitudinal memory saved yet.")}
          </SummaryCard>
          <SummaryCard title="Key patterns" count={`${insights.length} found`}>
            {insights.length ? insights.map((item) => item.title).join("; ") : "Add readings over time to unlock trend cards."}
          </SummaryCard>
        </section>
      )}

      {tab === "timeline" && (
        <section className="timeline-list">
          {events.length ? (
            events.slice(0, 30).map((event, index) => (
              <article key={`${event.type}-${index}`} className="timeline-event">
                <time>{formatDate(event.when)}</time>
                <div>
                  <span>{clean(event.type)}</span>
                  <strong>{clean(event.title)}</strong>
                  <p>{clean(event.detail)}</p>
                </div>
              </article>
            ))
          ) : (
            <div className="empty-state">No timeline events yet.</div>
          )}
        </section>
      )}

      {tab === "trends" && (
        <section className="trend-grid">
          {insights.length ? (
            insights.map((insight) => (
              <article className="trend-card" key={insight.title}>
                <span>Pattern found</span>
                <strong>{insight.title}</strong>
                <p>{insight.body}</p>
                <TrendChart snapshot={snapshot} detail={insight.detail} />
              </article>
            ))
          ) : (
            <div className="empty-state">No trend warnings yet. Add multiple readings over time.</div>
          )}
          <TrendLibrary snapshot={snapshot} />
        </section>
      )}
    </div>
  );
}

function SummaryCard({ title, count, children }: { title: string; count: string; children: React.ReactNode }) {
  return (
    <article className="summary-card">
      <span>{count}</span>
      <strong>{title}</strong>
      <p>{children}</p>
    </article>
  );
}

function TrendLibrary({ snapshot }: { snapshot: Snapshot }) {
  const vitalTypes = unique(snapshot.vitals.map((entry) => entry.type));
  const symptoms = unique(snapshot.symptom_logs.map((entry) => entry.symptom));
  return (
    <>
      {vitalTypes.map((type) => (
        <article className="trend-card" key={type}>
          <span>{buildSeries(snapshot.vitals, type).length} readings</span>
          <strong>{vitalLabel(type)}</strong>
          <TrendChart snapshot={snapshot} detail={type} />
        </article>
      ))}
      {symptoms.map((symptom) => {
        const series = buildSymptomSeries(snapshot.symptom_logs, symptom);
        if (series.length < 2) {
          return null;
        }
        return (
          <article className="trend-card" key={symptom}>
            <span>{series.length} readings</span>
            <strong>{symptom}</strong>
            <TrendChart snapshot={snapshot} detail={`symptom:${symptom}`} />
          </article>
        );
      })}
    </>
  );
}

function TrendChart({ snapshot, detail }: { snapshot: Snapshot; detail: string }) {
  const series = detail.startsWith("symptom:")
    ? buildSymptomSeries(snapshot.symptom_logs, detail.split(":")[1])
    : buildSeries(snapshot.vitals, detail).map((point) => ({ date: point.date, value: point.value, secondValue: point.secondValue }));
  if (series.length < 2) {
    return <p className="muted">No chartable pattern yet.</p>;
  }
  return <TinyChart series={series} />;
}

function TinyChart({ series }: { series: { date: string; value: number; secondValue?: number }[] }) {
  const width = 360;
  const height = 140;
  const values = series.flatMap((point) => [point.value, point.secondValue].filter((value): value is number => typeof value === "number"));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const point = (value: number, index: number) => {
    const x = series.length === 1 ? width / 2 : (index / (series.length - 1)) * (width - 28) + 14;
    const y = height - 22 - ((value - min) / span) * (height - 42);
    return `${x},${y}`;
  };
  const primary = series.map((entry, index) => point(entry.value, index)).join(" ");
  const secondary = series.filter((entry) => typeof entry.secondValue === "number").map((entry, index) => point(entry.secondValue as number, index)).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="tiny-chart" role="img" aria-label="Trend chart">
      <line x1="12" x2={width - 12} y1={height - 20} y2={height - 20} />
      <polyline points={primary} />
      {secondary && <polyline points={secondary} className="secondary" />}
      {series.map((entry, index) => (
        <circle key={`${entry.date}-${index}`} cx={point(entry.value, index).split(",")[0]} cy={point(entry.value, index).split(",")[1]} r="3" />
      ))}
    </svg>
  );
}

function TrialsView({
  snapshot,
  setSnapshot,
  setNotice
}: {
  snapshot: Snapshot;
  setSnapshot: (snapshot: Snapshot) => void;
  setNotice: (notice: string) => void;
}) {
  const [location, setLocation] = useState(snapshot.trial_search_result?.location || "United Kingdom");
  const [busy, setBusy] = useState(false);
  const result = snapshot.trial_search_result as TrialSearchResult | null | undefined;

  async function searchTrials() {
    setBusy(true);
    try {
      const response = await apiRequest<{ result: TrialSearchResult; snapshot: Snapshot }>("/api/trials/search", {
        method: "POST",
        body: JSON.stringify({ location, max_results: 10 })
      });
      setSnapshot(response.snapshot);
      setNotice(`Found ${response.result.trials.length} ranked trial records.`);
    } catch (caught) {
      setNotice(caught instanceof Error ? caught.message : "Trial search failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="view-stack">
      <section className="workspace-band">
        <div>
          <span className="eyebrow">Clinical trials</span>
          <h2>Find recruiting studies.</h2>
          <p>Results use saved health context and need confirmation by a study team or clinician.</p>
        </div>
        <div className="trial-search">
          <select value={location} onChange={(event) => setLocation(event.target.value)}>
            {COUNTRIES.map((country) => (
              <option key={country}>{country}</option>
            ))}
          </select>
          <button className="primary" onClick={searchTrials} disabled={busy}>
            <Search size={18} />
            {busy ? "Searching" : "Find trials"}
          </button>
        </div>
      </section>

      <section className="metric-grid">
        <div className="metric-card">
          <Activity size={20} />
          <strong>{snapshot.metrics.symptoms}</strong>
          <span>Symptoms</span>
        </div>
        <div className="metric-card">
          <Pill size={20} />
          <strong>{snapshot.metrics.medications}</strong>
          <span>Medications</span>
        </div>
        <div className="metric-card">
          <HeartPulse size={20} />
          <strong>{snapshot.metrics.vitals}</strong>
          <span>Vitals/labs</span>
        </div>
        <div className="metric-card">
          <ClipboardList size={20} />
          <strong>{snapshot.metrics.triage_records}</strong>
          <span>Triage records</span>
        </div>
      </section>

      {result?.error && <div className="notice warn">{result.error}</div>}
      {result && !result.error && (
        <section className="trial-results">
          <div className="toolbar-card">
            <span>Recruiting only</span>
            <span>Top {result.trials.length} ranked matches</span>
            <span>Country: {result.location}</span>
            <span>{formatTimestamp(result.searched_at) || "Saved search"}</span>
          </div>
          {(result.condition_terms.length > 0 || result.medication_terms.length > 0) && (
            <p className="muted">
              Searches run for {unique([...result.condition_terms, ...result.medication_terms.map((item) => `${item} (drug)`)]).join(", ")}
            </p>
          )}
          {result.trials.length ? (
            result.trials.map((trial, index) => <TrialCard trial={trial} index={index + 1} key={clean(trial.nct_id, String(index))} />)
          ) : (
            <div className="empty-state">No recruiting trials were returned for this context and location.</div>
          )}
        </section>
      )}
    </div>
  );
}

function TrialCard({ trial, index }: { trial: Dict<any>; index: number }) {
  const score = Number(trial.match_score ?? 0);
  const bestLocation = (trial.best_location ?? {}) as Dict<any>;
  const contacts = Array.isArray(trial.contacts) ? trial.contacts : [];
  const officials = Array.isArray(trial.officials) ? trial.officials : [];
  return (
    <article className="trial-card">
      <div className="trial-head">
        <span>#{index} {score >= 80 ? "Strong record match" : "Needs manual review"}</span>
        <strong>{score}%</strong>
      </div>
      <h3>{clean(trial.title, "Untitled trial")}</h3>
      <p>{Array.isArray(trial.conditions) ? trial.conditions.slice(0, 6).join(", ") : "Condition not listed"}</p>
      <div className="trial-chip-row">
        <span>{clean(trial.status, "Recruiting")}</span>
        <span>{clean(trial.phase, "Phase not listed")}</span>
        <span>{clean(trial.study_type, "Study type not listed")}</span>
      </div>
      <div className="trial-detail-grid">
        <div>
          <strong>Location</strong>
          <p>{unique([bestLocation.city, bestLocation.country]).join(", ") || "No trial site listed"}</p>
        </div>
        <div>
          <strong>Hospital / site</strong>
          <p>{clean(bestLocation.facility, "Not listed")}</p>
        </div>
        <div>
          <strong>Official</strong>
          <p>{officials.length ? unique([officials[0].name, officials[0].role, officials[0].affiliation]).join(" - ") : "Not listed"}</p>
        </div>
        <div>
          <strong>Contact</strong>
          <p>{contacts.length ? unique([contacts[0].name, contacts[0].phone, contacts[0].email]).join(" - ") : "No public contact listed"}</p>
        </div>
      </div>
      {trial.url && (
        <a className="open-link" href={String(trial.url)} target="_blank" rel="noreferrer">
          Open ClinicalTrials.gov record
        </a>
      )}
      <details className="details-block">
        <summary>Why this score?</summary>
        <p>Alignment {trial.alignment_score ?? 0}/50, coverage {trial.coverage_score ?? 0}/30, location {trial.location_score ?? 0}/20.</p>
        {trial.llm_reasoning && <p>{String(trial.llm_reasoning)}</p>}
        {Array.isArray(trial.exclusion_risks) && trial.exclusion_risks.length > 0 && <p>Potential exclusion factors: {trial.exclusion_risks.join("; ")}</p>}
      </details>
    </article>
  );
}

export default App;
