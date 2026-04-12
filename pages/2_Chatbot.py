import base64
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app_ui.theme import format_timestamp, inject_custom_css
from app_ui.uploader import upload_documents
from backend.product_config import PRODUCT_NAME, SUPPORT_EMAIL
from backend.rag_system import RAGEngine
from backend.user_store import UserStore
from backend.voice_transcriber import VoiceTranscriber

_PAGE_DIR = Path(__file__).parent.parent
USER_AVATAR = str(_PAGE_DIR / "app_ui/static/user.png")
ASSISTANT_AVATAR = str(_PAGE_DIR / "app_ui/static/assistant.png")
STARTER_PROMPTS = [
    "What does the recent evidence say about hypertension treatment in older adults?",
    "Summarize the most important themes from my uploaded records in plain language.",
    "What symptoms would make chest pain an urgent medical review issue?",
]


def resolve_image_source(
    image_url: str = "",
    image_bytes: bytes | None = None,
    image_b64: str = "",
) -> str | bytes | None:
    if image_url:
        return image_url
    if image_bytes:
        return image_bytes
    if image_b64:
        try:
            return base64.b64decode(image_b64)
        except Exception:
            return None
    return None


def render_source_links(sources: list[dict]) -> None:
    links = []
    for source in sources:
        source_id = source.get("source_id", "Source")
        url = source.get("url", "")
        if url:
            links.append(f"[{source_id}]({url})")

    if links:
        st.markdown("Sources: " + " | ".join(links))


def render_triage_summary(summary: dict) -> None:
    if not summary:
        return

    monitor_items = summary.get("what_to_monitor", [])
    monitor_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in monitor_items[:3]
        if str(item).strip()
    ) or "<li>No specific monitoring points saved.</li>"

    st.markdown(
        f"""
        <div class="triage-card">
            <div class="triage-card-head">
                <span class="triage-label">Structured triage</span>
                <span class="triage-next-step">{html.escape(summary.get('next_step', 'Self-care'))}</span>
            </div>
            <div class="triage-grid">
                <div>
                    <strong>Urgency</strong>
                    <p>{html.escape(summary.get('urgency_level', 'Routine'))}</p>
                </div>
                <div>
                    <strong>Suggested next step</strong>
                    <p>{html.escape(summary.get('next_step', 'Self-care'))}</p>
                </div>
            </div>
            <div class="triage-monitor">
                <strong>What to monitor</strong>
                <ul>{monitor_html}</ul>
            </div>
            <p class="triage-rationale">{html.escape(summary.get('rationale', ''))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_medication_alerts(alerts: list[dict], resolved_medications: list[dict]) -> None:
    if not alerts and not resolved_medications:
        return

    st.markdown("#### Medication interaction check")
    if alerts:
        for alert in alerts[:3]:
            severity = alert.get("severity", "mentioned")
            severity_label = {
                "high": "High label warning",
                "monitor": "Needs monitoring",
                "mentioned": "Label mention",
            }.get(severity, "Label mention")
            evidence = alert.get("evidence", [])
            source_url = evidence[0].get("source_url", "") if evidence else ""
            st.markdown(
                f"""
                <div class="interaction-card interaction-{severity}">
                    <div class="interaction-head">
                        <strong>{html.escape(alert.get('pair', 'Medication pair'))}</strong>
                        <span>{html.escape(severity_label)}</span>
                    </div>
                    <p>{html.escape(alert.get('summary', ''))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if source_url:
                st.markdown(f"[Open openFDA label evidence]({source_url})")
    else:
        medication_names = ", ".join(
            item.get("canonical_name", item.get("query_name", ""))
            for item in resolved_medications[:6]
            if item.get("canonical_name") or item.get("query_name")
        )
        st.info(
            "No explicit pair-specific warning was found in the queried openFDA label sections"
            + (f" for {medication_names}." if medication_names else ".")
            + " This is helpful but not exhaustive."
        )


def render_message_meta(message: dict) -> None:
    timestamp = format_timestamp(message.get("timestamp", ""))
    source_count = len(message.get("sources", []))
    trace_id = message.get("trace_id")
    triage_summary = message.get("metadata", {}).get("triage_summary", {})
    pills = []
    if timestamp:
        pills.append(timestamp)
    if source_count:
        pills.append(f"{source_count} sources")
    if trace_id:
        pills.append(trace_id)
    if triage_summary.get("next_step"):
        pills.append(f"Next: {triage_summary['next_step']}")

    if pills:
        joined = "".join(f"<span>{pill}</span>" for pill in pills)
        st.markdown(f"<div class='meta-pill-row'>{joined}</div>", unsafe_allow_html=True)


def render_source_trace(message: dict) -> None:
    sources = message.get("sources", [])
    personal_context = message.get("metadata", {}).get("personal_context", [])
    longitudinal_memory = message.get("metadata", {}).get("longitudinal_memory", "")
    trace = message.get("metadata", {}).get("trace", {})

    if not sources and not personal_context and not longitudinal_memory and not trace:
        return

    trace_title_parts = []
    if sources:
        trace_title_parts.append(f"{len(sources)} literature source(s)")
    if personal_context:
        trace_title_parts.append(f"{len(personal_context)} personal context item(s)")
    if longitudinal_memory:
        trace_title_parts.append("longitudinal memory")
    if trace.get("trace_id"):
        trace_title_parts.append(trace["trace_id"])

    expander_title = "Source trace"
    if trace_title_parts:
        expander_title = "Source trace: " + " | ".join(trace_title_parts)

    with st.expander(expander_title, expanded=False):
        if sources:
            for source in sources:
                tier = source.get("evidence_tier", 3)
                tier_label = source.get("tier_label", f"Tier {tier}")
                tier_description = source.get("tier_description", "")
                tier_badge_html = (
                    f'<span class="tier-badge tier-{tier}" title="{tier_description}">'
                    f"{tier_label}</span>"
                ) if tier_label else ""

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-head">
                            <span class="source-badge">{source.get('source_id', 'S')}</span>
                            <div>
                                <strong>{source.get('title', 'Untitled article')}</strong>
                                {tier_badge_html}
                                <br />
                                <span>{source.get('journal', 'Journal unavailable')} {source.get('year', '')}</span>
                            </div>
                        </div>
                        <div class="source-card-body">
                            <p><strong>Section:</strong> {source.get('section', 'Retrieved text')}</p>
                            <p>{source.get('snippet', '')}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if source.get("url"):
                    st.link_button(
                        f"Open {source.get('source_id', 'source')}",
                        source["url"],
                        use_container_width=False,
                    )

        if personal_context:
            st.markdown("#### Personal context considered")
            for item in personal_context:
                st.markdown(
                    f"""
                    <div class="context-card">
                        <strong>{item.get('title', item.get('source', 'Uploaded context'))}</strong>
                        <p>{item.get('snippet', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if longitudinal_memory:
            memory_html = html.escape(longitudinal_memory).replace("\n", "<br />")
            st.markdown("#### Longitudinal memory considered")
            st.markdown(
                f"""
                <div class="context-card">
                    <strong>Persistent account memory</strong>
                    <p>{memory_html}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if trace:
            st.markdown("#### Audit trace")
            audit_display = {
                "trace_id": trace.get("trace_id"),
                "retrieval_mode": trace.get("retrieval_mode"),
                "expanded_queries": trace.get("expanded_queries", []),
                "model": trace.get("model"),
                "created_at": trace.get("created_at"),
            }
            if trace.get("role_key"):
                audit_display["role_key"] = trace.get("role_key")
            if trace.get("intent_category"):
                audit_display["intent_category"] = trace.get("intent_category")
            if trace.get("risk_level"):
                audit_display["risk_level"] = trace.get("risk_level")
            if trace.get("evidence_tiers_present"):
                audit_display["evidence_tiers_present"] = trace.get("evidence_tiers_present")
            if trace.get("pathway_used"):
                audit_display["pathway_used"] = trace.get("pathway_used")
            if trace.get("escalation_triggered"):
                audit_display["escalation_triggered"] = trace.get("escalation_triggered")
            if trace.get("policy_gates_applied"):
                audit_display["policy_gates_applied"] = trace.get("policy_gates_applied")
            if trace.get("medication_alert_count") is not None:
                audit_display["medication_alert_count"] = trace.get("medication_alert_count")
            st.json(audit_display)


def render_chat_history(history: list[dict]) -> None:
    for message in history:
        avatar = USER_AVATAR if message.get("role") == "user" else ASSISTANT_AVATAR
        with st.chat_message(message.get("role", "assistant"), avatar=avatar):
            st.markdown(message.get("content", ""))
            meta = message.get("metadata", {})
            history_image = resolve_image_source(
                image_url=meta.get("image_url", ""),
                image_b64=meta.get("image_b64", ""),
            )
            if history_image and message.get("role") == "assistant":
                st.image(
                    history_image,
                    caption=meta.get("image_caption", "Generated illustration"),
                    width="stretch",
                )

            video_url = meta.get("video_url", "")
            if video_url and message.get("role") == "assistant":
                st.video(video_url)
                st.caption(meta.get("video_caption", "Generated video"))

            if message.get("role") == "assistant":
                render_triage_summary(meta.get("triage_summary", {}))
                render_medication_alerts(
                    meta.get("medication_alerts", []),
                    meta.get("resolved_medications", []),
                )
                render_source_links(message.get("sources", []))
            render_message_meta(message)
            if message.get("role") == "assistant":
                render_source_trace(message)


def queue_prompt(prompt: str) -> None:
    st.session_state.prompt_draft = prompt
    st.rerun()


def clear_prompt_draft() -> None:
    st.session_state.prompt_draft = ""


def submit_prompt_draft() -> None:
    st.session_state.pending_submitted_prompt = st.session_state.get("prompt_draft", "").strip()
    st.session_state.prompt_draft = ""


st.set_page_config(
    page_title=PRODUCT_NAME,
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

current_user = st.session_state.get("current_user")
if not current_user:
    st.warning("Please sign in to continue.")
    st.switch_page("pages/1_Landing.py")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine(embedding_dir="data/uploads")

if "voice_transcriber" not in st.session_state:
    try:
        st.session_state.voice_transcriber = VoiceTranscriber()
    except Exception:
        st.session_state.voice_transcriber = None

rag_engine: RAGEngine = st.session_state.rag_engine
voice_transcriber: VoiceTranscriber | None = st.session_state.voice_transcriber
rag_engine.restore_user_context(current_user)

if st.session_state.get("history_user") != current_user:
    st.session_state.chat_history = UserStore.get_chat_history(current_user)
    st.session_state.history_user = current_user

chat_history = st.session_state.get("chat_history", [])
user_profile = UserStore.get_user_profile(current_user)
uploads = UserStore.get_uploads(current_user)
symptom_logs = UserStore.get_symptom_logs(current_user, limit=None)
medications = UserStore.get_medications(current_user)
traces = UserStore.get_interaction_traces(current_user, limit=5)
audit_records = UserStore.get_audit(current_user, limit=8)
latest_triage = UserStore.get_latest_triage_summary(current_user)

with st.sidebar:
    clinical_role_display = user_profile.get("clinical_role") or user_profile.get("role", "Patient / Individual")
    st.markdown(
        f"""
        <div class="sidebar-profile">
            <div class="feature-eyebrow">Signed in</div>
            <h2>{user_profile.get('display_name', current_user)}</h2>
            <span class="clinical-role-badge">{clinical_role_display}</span>
            <p>{user_profile.get('care_context', 'Personal health guidance')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_actions = st.columns(2, gap="small")
    with sidebar_actions[0]:
        if st.button("Clear conversation", use_container_width=True):
            UserStore.clear_chat_history(current_user)
            st.session_state.chat_history = []
            st.success("Conversation history cleared.")
            st.rerun()
    with sidebar_actions[1]:
        if st.button("Sign out", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.history_user = None
            st.session_state.chat_history = []
            st.switch_page("pages/1_Landing.py")

    with st.expander("Account settings", expanded=False):
        with st.form("profile_form"):
            profile_name = st.text_input("Display name", value=user_profile.get("display_name", ""))
            profile_email = st.text_input("Email", value=user_profile.get("email", ""))
            st.text_input("Account role", value=clinical_role_display, disabled=True)
            st.text_input("Account type", value=user_profile.get("care_context", ""), disabled=True)
            organization = st.text_input("Organization", value=user_profile.get("organization", ""))
            follow_up = st.text_area(
                "Follow-up preferences",
                value=user_profile.get("follow_up_preferences", ""),
                height=90,
            )
            profile_saved = st.form_submit_button("Save changes", type="primary", use_container_width=True)

        if profile_saved:
            updated = UserStore.update_profile(
                current_user,
                {
                    "display_name": profile_name,
                    "email": profile_email,
                    "organization": organization,
                    "follow_up_preferences": follow_up,
                },
            )
            if updated:
                st.success("Account details updated.")
                st.rerun()
            else:
                st.error("That email address is already linked to another account.")

        st.divider()
        st.caption(
            f"Need support or a role change? Contact {SUPPORT_EMAIL}."
        )
        if st.button("Sign out from this account", use_container_width=True, type="secondary"):
            st.session_state.current_user = None
            st.session_state.history_user = None
            st.session_state.chat_history = []
            st.switch_page("pages/1_Landing.py")

    st.markdown("### Documents")
    saved_paths = upload_documents(current_user)
    if saved_paths:
        with st.spinner("Indexing uploaded documents for future questions..."):
            indexed = rag_engine.ingest_documents(user=current_user, file_paths=saved_paths)
        st.success(f"Indexed {len(indexed)} document(s).")
        st.rerun()

    if uploads:
        for upload in uploads[:6]:
            uploaded_at = format_timestamp(upload.get("uploaded_at", ""))
            st.markdown(
                f"""
                <div class="mini-record">
                    <strong>{upload.get('file', 'Document')}</strong>
                    <span>{uploaded_at or 'Saved'}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No uploaded records yet.")

    with st.expander("Symptom timeline tracker", expanded=False):
        with st.form("symptom_tracker_form"):
            symptom_name = st.text_input("Symptom")
            symptom_date = st.date_input(
                "Date noticed",
                value=datetime.now(timezone.utc).date(),
            )
            symptom_severity = st.slider("Severity", min_value=0, max_value=10, value=5)
            symptom_triggers = st.text_input("Possible triggers")
            symptom_notes = st.text_area("Notes", height=80)
            symptom_saved = st.form_submit_button(
                "Log symptom",
                type="primary",
                use_container_width=True,
            )

        if symptom_saved:
            saved_entry = UserStore.add_symptom_log(
                current_user,
                symptom=symptom_name,
                logged_for=symptom_date.isoformat(),
                severity=symptom_severity,
                triggers=symptom_triggers,
                notes=symptom_notes,
            )
            if saved_entry:
                rag_engine.restore_user_context(current_user)
                st.success("Symptom saved to your timeline.")
                st.rerun()
            else:
                st.error("Add a symptom name and date to save a tracker entry.")

        if symptom_logs:
            for entry in symptom_logs[:6]:
                log_label = f"{entry.get('logged_for', '')} | {entry.get('symptom', 'Symptom')}"
                if entry.get("severity"):
                    log_label += f" | {entry['severity']}/10"
                st.markdown(
                    f"""
                    <div class="mini-record">
                        <strong>{html.escape(log_label)}</strong>
                        <span>{html.escape(entry.get('triggers', 'No trigger noted') or 'No trigger noted')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove log", key=f"symptom_delete_{entry['log_id']}", use_container_width=True):
                    UserStore.delete_symptom_log(current_user, entry["log_id"])
                    rag_engine.restore_user_context(current_user)
                    st.rerun()
        else:
            st.caption("No symptom logs yet.")

    with st.expander("Medication list", expanded=False):
        with st.form("medication_list_form"):
            medication_name = st.text_input("Medication name")
            medication_dose = st.text_input("Dose")
            medication_schedule = st.text_input("Schedule")
            medication_reason = st.text_input("Reason / condition")
            medication_saved = st.form_submit_button(
                "Save medication",
                type="primary",
                use_container_width=True,
            )

        if medication_saved:
            saved_medication = UserStore.save_medication(
                current_user,
                {
                    "name": medication_name,
                    "dose": medication_dose,
                    "schedule": medication_schedule,
                    "reason": medication_reason,
                },
            )
            if saved_medication:
                rag_engine.restore_user_context(current_user)
                st.success("Medication list updated.")
                st.rerun()
            else:
                st.error("Enter a medication name to save it.")

        if medications:
            for medication in medications[:8]:
                pieces = [medication.get("name", "Medication")]
                if medication.get("dose"):
                    pieces.append(medication["dose"])
                if medication.get("schedule"):
                    pieces.append(medication["schedule"])
                st.markdown(
                    f"""
                    <div class="mini-record">
                        <strong>{html.escape(' | '.join(pieces))}</strong>
                        <span>{html.escape(medication.get('reason', 'On file') or 'On file')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Remove medication",
                    key=f"medication_delete_{medication['medication_id']}",
                    use_container_width=True,
                ):
                    UserStore.delete_medication(current_user, medication["medication_id"])
                    rag_engine.restore_user_context(current_user)
                    st.rerun()
        else:
            st.caption("No medications saved yet.")

    st.markdown("### GP summary")
    gp_pdf = rag_engine.build_gp_summary_pdf_for_user(current_user)
    has_gp_content = bool(
        symptom_logs
        or medications
        or uploads
        or rag_engine.get_combined_longitudinal_memory(current_user)
        or latest_triage
    )
    st.download_button(
        "Download GP summary PDF",
        data=gp_pdf,
        file_name=f"{current_user}-gp-summary.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=not has_gp_content,
    )
    if latest_triage:
        st.caption(
            f"Latest triage: {latest_triage.get('urgency_level', 'Routine')} -> "
            f"{latest_triage.get('next_step', 'Self-care')}"
        )

    st.markdown("### Audit export")
    export_payload = json.dumps(UserStore.export_user_snapshot(current_user), indent=2)
    st.download_button(
        "Download account export",
        data=export_payload,
        file_name=f"{current_user}-audit.json",
        mime="application/json",
        use_container_width=True,
    )

    if traces:
        st.markdown("### Recent traces")
        for trace in traces:
            st.markdown(
                f"""
                <div class="mini-record">
                    <strong>{trace.get('trace_id', 'trace')}</strong>
                    <span>{trace.get('retrieval_mode', 'trace')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if audit_records:
        with st.expander("Recent audit events", expanded=False):
            for record in audit_records:
                st.markdown(
                    f"""
                    <div class="audit-row">
                        <strong>{record.get('event', 'event')}</strong>
                        <p>{record.get('details', '')}</p>
                        <span>{format_timestamp(record.get('time', ''))}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

top_left, top_right = st.columns([1.4, 1], gap="large")

with top_left:
    st.markdown(
        f"""
        <div class="workspace-hero">
            <div class="feature-eyebrow">{PRODUCT_NAME}</div>
            <h1>Welcome back, {user_profile.get('display_name', current_user)}.</h1>
            <p>
                Continue your conversation, review supporting references, and manage your saved documents and account history in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    metric_columns = st.columns(3, gap="small")
    metric_columns[0].metric("Messages", len(chat_history))
    metric_columns[1].metric("Symptom logs", len(symptom_logs))
    metric_columns[2].metric("Medications", len(medications))

st.markdown(
    """
    <div class="toolbar-card">
        <span>Structured triage</span>
        <span>Symptom timeline</span>
        <span>Medication safety</span>
        <span>GP handover PDF</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if chat_history:
    st.info(f"Resumed {len(chat_history)} saved message(s) from your account.")
else:
    st.markdown("### Start with a strong question")
    starter_cols = st.columns(len(STARTER_PROMPTS), gap="small")
    for index, prompt in enumerate(STARTER_PROMPTS):
        with starter_cols[index]:
            if st.button(prompt, key=f"starter_{index}", use_container_width=True):
                queue_prompt(prompt)

render_chat_history(chat_history)

st.session_state.setdefault("prompt_draft", "")
st.session_state.setdefault("pending_submitted_prompt", "")
voice_audio_hash = None

if voice_transcriber:
    with st.expander("Speak your question", expanded=False):
        st.caption(
            "Use the microphone control below to record your question. "
            "When you stop recording, the transcript will be added to your draft so you can review it before sending."
        )

        audio_bytes = b""
        audio_filename = "recording.wav"

        if hasattr(st, "audio_input"):
            audio_file = st.audio_input(
                "Record your question",
                key="voice_audio_input",
                help="Allow microphone access in your browser when prompted.",
            )
            if audio_file is not None:
                audio_bytes = audio_file.getvalue()
                audio_filename = getattr(audio_file, "name", audio_filename) or audio_filename
        else:
            try:
                from streamlit_mic_recorder import mic_recorder

                legacy_audio = mic_recorder(
                    start_prompt="Start recording",
                    stop_prompt="Stop recording",
                    just_once=True,
                    use_container_width=True,
                    key="mic_recorder",
                )
                if legacy_audio and legacy_audio.get("bytes"):
                    audio_bytes = legacy_audio["bytes"]
                    audio_filename = "recording.webm"
            except ImportError:
                st.info("Voice input is unavailable in this environment.")

        if audio_bytes:
            voice_audio_hash = hashlib.sha1(audio_bytes).hexdigest()
            last_audio_hash = st.session_state.get("last_voice_audio_hash")

            if voice_audio_hash != last_audio_hash:
                with st.spinner("Transcribing..."):
                    transcribed = voice_transcriber.transcribe(
                        audio_bytes,
                        filename=audio_filename,
                    )
                st.session_state.last_voice_audio_hash = voice_audio_hash
                st.session_state.last_voice_transcript = transcribed

            transcribed = st.session_state.get("last_voice_transcript", "")
            if transcribed:
                if voice_audio_hash != st.session_state.get("last_voice_applied_hash"):
                    existing_draft = st.session_state.get("prompt_draft", "").strip()
                    if existing_draft:
                        st.session_state.prompt_draft = f"{existing_draft}\n{transcribed}".strip()
                    else:
                        st.session_state.prompt_draft = transcribed
                    st.session_state.last_voice_applied_hash = voice_audio_hash

                st.success("Transcript added to your draft. Review it below and send when ready.")
                st.caption(transcribed)
            else:
                st.warning("Could not transcribe audio. Please try again or type your question.")

st.markdown("### Your message")
st.text_area(
    "Message",
    key="prompt_draft",
    height=120,
    placeholder="Ask a health question, request an evidence summary, or continue your saved conversation...",
    label_visibility="collapsed",
)

composer_actions = st.columns([1, 1, 4], gap="small")
with composer_actions[0]:
    send_prompt = st.button(
        "Send",
        type="primary",
        use_container_width=True,
        on_click=submit_prompt_draft,
    )
with composer_actions[1]:
    st.button(
        "Clear",
        use_container_width=True,
        on_click=clear_prompt_draft,
    )

active_question = ""
if send_prompt:
    active_question = st.session_state.pop("pending_submitted_prompt", "").strip()
    if not active_question:
        st.warning("Enter or record a message before sending.")

if active_question:
    now = datetime.now(timezone.utc).isoformat()
    user_entry = {
        "role": "user",
        "content": active_question,
        "timestamp": now,
        "sources": [],
        "metadata": {},
    }
    st.session_state.chat_history.append(user_entry)
    UserStore.append_chat(current_user, user_entry)

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(active_question)
        render_message_meta(user_entry)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        progress_panel = (
            st.status("Starting evidence review...", expanded=True)
            if hasattr(st, "status")
            else None
        )
        answer_placeholder = st.empty()

        try:
            payload = None
            streamed_answer_parts: list[str] = []
            for event in rag_engine.stream_user_question_events(
                question=active_question,
                chat_history=st.session_state.chat_history,
                user=current_user,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    message = event.get("message", "Working...")
                    if progress_panel:
                        progress_panel.write(message)
                        progress_panel.update(label=message, state="running")
                elif event_type == "token":
                    streamed_answer_parts.append(event.get("delta", ""))
                    answer_placeholder.markdown("".join(streamed_answer_parts).strip() + "▌")
                elif event_type == "final":
                    payload = event.get("payload")

            if not payload:
                raise RuntimeError("The answer pipeline did not return a payload.")

            if progress_panel:
                progress_panel.update(label="Evidence review complete", state="complete", expanded=False)

            assistant_entry = {
                "role": "assistant",
                "content": payload["answer_markdown"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sources": payload.get("sources", []),
                "trace_id": payload.get("trace", {}).get("trace_id"),
                "metadata": {
                    "personal_context": payload.get("personal_context", []),
                    "longitudinal_memory": payload.get("longitudinal_memory", ""),
                    "triage_summary": payload.get("triage_summary", {}),
                    "medication_alerts": payload.get("medication_alerts", []),
                    "resolved_medications": payload.get("resolved_medications", []),
                    "trace": payload.get("trace", {}),
                    "image_url": payload.get("image_url", ""),
                    "image_b64": base64.b64encode(payload["image_bytes"]).decode()
                    if payload.get("image_bytes")
                    else "",
                    "image_caption": payload.get("image_caption", ""),
                    "video_url": payload.get("video_url", ""),
                    "video_caption": payload.get("video_caption", ""),
                },
            }
            answer_placeholder.markdown(assistant_entry["content"])

            image_src = resolve_image_source(
                image_url=payload.get("image_url", ""),
                image_bytes=payload.get("image_bytes"),
            )
            if image_src:
                st.image(
                    image_src,
                    caption=payload.get("image_caption", "Generated illustration"),
                    width="stretch",
                )
                st.markdown(
                    "<p style='font-size:11px;color:var(--text-soft);margin-top:0.2rem;'>"
                    "AI-generated illustration - for educational reference only. "
                    "Always verify with a qualified clinician or physiotherapist.</p>",
                    unsafe_allow_html=True,
                )

            if payload.get("video_url"):
                st.video(payload["video_url"])
                st.caption(payload.get("video_caption", "Generated video"))
                st.markdown(
                    "<p style='font-size:11px;color:var(--text-soft);margin-top:0.2rem;'>"
                    "AI-generated video - for educational reference only. "
                    "Always verify with a qualified clinician.</p>",
                    unsafe_allow_html=True,
                )
            elif payload.get("video_rate_limit_msg"):
                st.warning(payload["video_rate_limit_msg"])

            try:
                refreshed_memory = rag_engine.refresh_longitudinal_memory_from_turn(
                    user=current_user,
                    user_message=active_question,
                    personal_context=payload.get("personal_context", []),
                )
                if refreshed_memory:
                    assistant_entry["metadata"]["longitudinal_memory"] = refreshed_memory
            except Exception as exc:
                print(f"Longitudinal memory refresh failed: {exc}")

            render_triage_summary(assistant_entry["metadata"].get("triage_summary", {}))
            render_medication_alerts(
                assistant_entry["metadata"].get("medication_alerts", []),
                assistant_entry["metadata"].get("resolved_medications", []),
            )
            render_source_links(assistant_entry.get("sources", []))
            render_message_meta(assistant_entry)
            render_source_trace(assistant_entry)
            st.session_state.chat_history.append(assistant_entry)
            UserStore.append_chat(current_user, assistant_entry)
            st.rerun()
        except Exception as exc:
            if progress_panel:
                progress_panel.update(label="Response unavailable", state="error", expanded=True)
            error_message = (
                "## Response unavailable\n"
                f"I ran into an issue while building the answer: `{exc}`.\n\n"
                "Please try again, or narrow the question if the request is very broad."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sources": [],
                "metadata": {},
            }
            answer_placeholder.markdown(error_message)
            render_message_meta(assistant_entry)
            st.session_state.chat_history.append(assistant_entry)
            UserStore.append_chat(current_user, assistant_entry)
            st.rerun()
