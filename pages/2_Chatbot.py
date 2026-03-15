import html
import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app_ui.theme import format_timestamp, inject_custom_css
from app_ui.uploader import upload_documents
from backend.rag_system import RAGEngine
from backend.user_store import UserStore

USER_AVATAR = str(Path("app_ui/static/user.png"))
ASSISTANT_AVATAR = str(Path("app_ui/static/assistant.png"))
STARTER_PROMPTS = [
    "What does the recent evidence say about hypertension treatment in older adults?",
    "Summarize the most important themes from my uploaded records in plain language.",
    "What symptoms would make chest pain an urgent medical review issue?",
]


def render_message_meta(message: dict) -> None:
    timestamp = format_timestamp(message.get("timestamp", ""))
    source_count = len(message.get("sources", []))
    trace_id = message.get("trace_id")
    pills = []
    if timestamp:
        pills.append(timestamp)
    if source_count:
        pills.append(f"{source_count} sources")
    if trace_id:
        pills.append(trace_id)

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
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-card-head">
                            <span class="source-badge">{source.get('source_id', 'S')}</span>
                            <div>
                                <strong>{source.get('title', 'Untitled article')}</strong><br />
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
                    <strong>Persistent patient memory</strong>
                    <p>{memory_html}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if trace:
            st.markdown("#### Audit trace")
            st.json(
                {
                    "trace_id": trace.get("trace_id"),
                    "retrieval_mode": trace.get("retrieval_mode"),
                    "expanded_queries": trace.get("expanded_queries", []),
                    "model": trace.get("model"),
                    "created_at": trace.get("created_at"),
                }
            )


def render_chat_history(history: list[dict]) -> None:
    for message in history:
        avatar = USER_AVATAR if message.get("role") == "user" else ASSISTANT_AVATAR
        with st.chat_message(message.get("role", "assistant"), avatar=avatar):
            st.markdown(message.get("content", ""))
            render_message_meta(message)
            if message.get("role") == "assistant":
                render_source_trace(message)


def queue_prompt(prompt: str) -> None:
    st.session_state.queued_prompt = prompt
    st.rerun()


st.set_page_config(
    page_title="My Health Checks Workspace",
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

rag_engine: RAGEngine = st.session_state.rag_engine
rag_engine.restore_user_context(current_user)

if st.session_state.get("history_user") != current_user:
    st.session_state.chat_history = UserStore.get_chat_history(current_user)
    st.session_state.history_user = current_user

chat_history = st.session_state.get("chat_history", [])
user_profile = UserStore.get_user_profile(current_user)
uploads = UserStore.get_uploads(current_user)
traces = UserStore.get_interaction_traces(current_user, limit=5)
audit_records = UserStore.get_audit(current_user, limit=8)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-profile">
            <div class="feature-eyebrow">Signed in</div>
            <h2>{user_profile.get('display_name', current_user)}</h2>
            <p>{user_profile.get('role', 'Individual')}</p>
            <p>{user_profile.get('care_context', 'Personal health guidance')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar_actions = st.columns(2, gap="small")
    with sidebar_actions[0]:
        if st.button("Delete chat", use_container_width=True):
            UserStore.clear_chat_history(current_user)
            st.session_state.chat_history = []
            st.success("Chat history deleted.")
            st.rerun()
    with sidebar_actions[1]:
        if st.button("Logout", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.history_user = None
            st.session_state.chat_history = []
            st.switch_page("pages/1_Landing.py")

    with st.expander("Profile settings", expanded=False):
        with st.form("profile_form"):
            profile_name = st.text_input("Display name", value=user_profile.get("display_name", ""))
            profile_email = st.text_input("Email", value=user_profile.get("email", ""))
            care_context = st.text_input("Use case", value=user_profile.get("care_context", ""))
            role = st.text_input("Role", value=user_profile.get("role", ""))
            organization = st.text_input("Organization", value=user_profile.get("organization", ""))
            follow_up = st.text_area(
                "Follow-up preferences",
                value=user_profile.get("follow_up_preferences", ""),
                height=90,
            )
            profile_saved = st.form_submit_button("Save profile", type="primary", use_container_width=True)

        if profile_saved:
            UserStore.update_profile(
                current_user,
                {
                    "display_name": profile_name,
                    "email": profile_email,
                    "care_context": care_context,
                    "role": role,
                    "organization": organization,
                    "follow_up_preferences": follow_up,
                },
            )
            st.success("Profile updated.")
            st.rerun()

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

    st.markdown("### Audit export")
    export_payload = json.dumps(UserStore.export_user_snapshot(current_user), indent=2)
    st.download_button(
        "Download audit JSON",
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
            <div class="feature-eyebrow">Professional workspace</div>
            <h1>{user_profile.get('display_name', current_user)}, your evidence-backed health assistant is ready.</h1>
            <p>
                Ask for personal health explanations, clinician-style evidence summaries, or literature-backed follow-up questions.
                Every substantive answer is designed to surface references and trace data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    metric_columns = st.columns(3, gap="small")
    metric_columns[0].metric("Messages", len(chat_history))
    metric_columns[1].metric("Uploads", len(uploads))
    metric_columns[2].metric("Traces", len(UserStore.get_interaction_traces(current_user, limit=None)))

st.markdown(
    """
    <div class="toolbar-card">
        <span>Evidence-first answers</span>
        <span>PubMed source trace</span>
        <span>Continuity across logins</span>
        <span>Audit-ready session history</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if chat_history:
    st.info(f"Resumed {len(chat_history)} message(s) from your saved workspace.")
else:
    st.markdown("### Start with a strong prompt")
    starter_cols = st.columns(len(STARTER_PROMPTS), gap="small")
    for index, prompt in enumerate(STARTER_PROMPTS):
        with starter_cols[index]:
            if st.button(prompt, key=f"starter_{index}", use_container_width=True):
                queue_prompt(prompt)

render_chat_history(chat_history)

queued_prompt = st.session_state.pop("queued_prompt", None)
user_question = st.chat_input("Ask a health question, request a literature summary, or continue your prior discussion...")
active_question = user_question or queued_prompt

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
                    "trace": payload.get("trace", {}),
                },
            }
            answer_placeholder.markdown(assistant_entry["content"])
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
            render_message_meta(assistant_entry)
            render_source_trace(assistant_entry)
            st.session_state.chat_history.append(assistant_entry)
            UserStore.append_chat(current_user, assistant_entry)
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
