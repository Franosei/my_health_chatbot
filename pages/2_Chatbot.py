import base64
import hashlib
import html
import inspect
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app_ui.theme import format_timestamp, inject_custom_css
from app_ui.uploader import upload_documents
from backend.product_config import PRODUCT_NAME, SUPPORT_EMAIL
from backend.rag_system import RAGEngine
from backend.role_router import RoleRouter
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
CUSTOM_VITALS_OPTION = "Other measurement..."


def selectbox_accepts_new_options() -> bool:
    try:
        return "accept_new_options" in inspect.signature(st.selectbox).parameters
    except (TypeError, ValueError):
        return False


def format_vitals_type(vitals_type: str, labels: dict[str, str]) -> str:
    cleaned = (vitals_type or "").strip()
    if not cleaned:
        return "Measurement"
    if cleaned in labels:
        return labels[cleaned]
    if "_" in cleaned or cleaned.islower():
        return cleaned.replace("_", " ").title()
    return cleaned


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
    pass  # consolidated into render_why_this_answer


def render_triage_summary(summary: dict) -> None:
    if not summary:
        return

    monitor_items = summary.get("what_to_monitor", [])
    immediate_actions = summary.get("immediate_actions", [])
    escalation_items = summary.get("escalation_triggers", [])
    monitor_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in monitor_items[:3]
        if str(item).strip()
    ) or "<li>No specific monitoring points saved.</li>"
    immediate_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in immediate_actions[:4]
        if str(item).strip()
    )
    escalation_html = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in escalation_items[:3]
        if str(item).strip()
    )
    pathway_label = summary.get("pathway_label", "")

    st.markdown(
        f"""
        <div class="triage-card">
            <div class="triage-card-head">
                <span class="triage-label">Structured triage</span>
                <span class="triage-next-step">{html.escape(summary.get('next_step', 'Self-care'))}</span>
            </div>
            {f"<p><strong>Pathway</strong><br />{html.escape(pathway_label)}</p>" if pathway_label else ""}
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
            {f"<div class='triage-monitor'><strong>Immediate actions</strong><ul>{immediate_html}</ul></div>" if immediate_html else ""}
            {f"<div class='triage-monitor'><strong>Escalate immediately if</strong><ul>{escalation_html}</ul></div>" if escalation_html else ""}
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
    trace = message.get("metadata", {}).get("trace", {})
    role_key = trace.get("role_key", "patient")
    is_clinician = role_key in ("doctor", "nurse", "midwife", "physiotherapist")
    triage_summary = message.get("metadata", {}).get("triage_summary", {})
    pills = []
    if timestamp:
        pills.append(timestamp)
    if is_clinician and message.get("trace_id"):
        pills.append(message["trace_id"])
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

    expander_title = "Why this answer?"
    if trace_title_parts:
        expander_title = "Why this answer? " + " | ".join(trace_title_parts)

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

        claim_alignment = trace.get("claim_alignment", []) if trace else []
        if claim_alignment:
            st.markdown("#### Claim-source alignment")
            for item in claim_alignment:
                status = item.get("status", "general_knowledge")
                claim = item.get("claim", "")
                sids = item.get("source_ids", [])
                badge = "supported" if status == "supported" else "general knowledge"
                badge_class = "tier-badge tier-1" if status == "supported" else "tier-badge tier-3"
                sid_str = ", ".join(sids) if sids else "—"
                st.markdown(
                    f"""
                    <div class="source-card" style="margin-bottom:6px">
                        <span class="{badge_class}">{badge}</span>
                        <span style="margin-left:8px">{html.escape(claim)}</span>
                        <span style="float:right;opacity:0.6;font-size:12px">{html.escape(sid_str)}</span>
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
            if trace.get("decision_logic_version"):
                audit_display["decision_logic_version"] = trace.get("decision_logic_version")
            if trace.get("rule_hits"):
                audit_display["rule_hits"] = trace.get("rule_hits")
            if trace.get("guideline_references"):
                audit_display["guideline_references"] = trace.get("guideline_references")
            st.json(audit_display)


_INTENT_LABELS = {
    "symptom_triage": "Symptom triage",
    "medication_query": "Medication query",
    "chronic_condition": "Chronic condition management",
    "maternity": "Maternity and pregnancy",
    "msk": "Musculoskeletal",
    "mental_health": "Mental health",
    "general_health": "General health",
    "education": "Health education",
    "administrative": "Administrative",
}

_RISK_LABELS = {
    "routine": "Routine",
    "elevated": "Elevated",
    "urgent": "Urgent",
    "crisis": "Crisis — immediate action required",
}

_PATHWAY_LABELS = {
    "general_triage": "General triage",
    "maternity": "Maternity pathway",
    "msk": "Musculoskeletal pathway",
    "medications": "Medications pathway",
    "chronic_conditions": "Chronic conditions pathway",
}

_ROLE_LABELS = {
    "patient": "Patient",
    "caregiver": "Caregiver",
    "doctor": "Doctor",
    "nurse": "Nurse",
    "midwife": "Midwife",
    "physiotherapist": "Physiotherapist",
}

_GATE_LABELS = {
    "allergy_contraindication": "Allergy / contraindication check",
    "medication_lay": "Medication safety (patient)",
    "medication_clinical": "Medication guidance (clinician)",
    "no_diagnosis": "No-diagnosis policy",
    "diagnosis_clinical": "Differential discussion (clinician)",
    "crisis": "Crisis escalation",
    "pregnancy_safety": "Pregnancy safety",
    "paediatric_safety": "Paediatric safety",
    "elderly_polypharmacy": "Elderly polypharmacy",
    "mental_health": "Mental health safety",
    "urgent_escalation": "Urgent escalation",
}

_TIER_LABELS_FULL = {
    1: "Tier 1 — NHS / NICE formal guidance",
    2: "Tier 2 — Review evidence",
    3: "Tier 3 — Primary research",
}

_FLAG_LABELS = {
    "elderly": "Older adult",
    "paediatric": "Paediatric",
    "pregnancy": "Pregnancy",
    "postpartum": "Postpartum",
    "newborn": "Newborn",
}


def _fmt(raw: str, lookup: dict) -> str:
    return lookup.get(raw, raw.replace("_", " ").title()) if raw else ""


def render_why_this_answer(message: dict) -> None:
    """Single role-aware 'Why this answer?' expander — replaces reasoning panel + source trace."""
    sources = message.get("sources", [])
    meta = message.get("metadata", {})
    trace = meta.get("trace", {})
    personal_context = meta.get("personal_context", [])
    longitudinal_memory = meta.get("longitudinal_memory", "")

    if not sources and not personal_context and not longitudinal_memory and not trace:
        return

    role_key = trace.get("role_key", "patient")
    is_clinician = role_key in ("doctor", "nurse", "midwife", "physiotherapist")

    tier_map = {1: 0, 2: 0, 3: 0}
    for s in sources:
        t = s.get("evidence_tier", 3)
        if t in tier_map:
            tier_map[t] += 1

    title_parts: list[str] = []
    if tier_map[1]:
        title_parts.append(f"{tier_map[1]} guideline{'s' if tier_map[1] > 1 else ''}")
    total_research = tier_map[2] + tier_map[3]
    if total_research:
        title_parts.append(f"{total_research} research source{'s' if total_research > 1 else ''}")
    expander_title = "Why this answer?" + (" — " + " · ".join(title_parts) if title_parts else "")

    with st.expander(expander_title, expanded=False):
        if is_clinician:
            # ── CLINICIAN: full technical audit ──────────────────────────────
            intent_cat = trace.get("intent_category", "")
            risk_level = trace.get("risk_level", "")
            pathway = trace.get("pathway_used", "")
            gates = trace.get("policy_gates_applied", [])
            expanded_queries = trace.get("expanded_queries", [])
            vulnerable_flags = trace.get("vulnerable_flags", [])
            escalation = trace.get("escalation_triggered", False)
            crisis = trace.get("crisis_detected", False)

            rows: list[tuple[str, str]] = []
            if intent_cat:
                rows.append(("Intent", _fmt(intent_cat, _INTENT_LABELS)))
            if risk_level:
                rows.append(("Risk level", _fmt(risk_level, _RISK_LABELS)))
            if escalation or crisis:
                rows.append(("Escalation", "Yes — escalation notice included" if escalation else "Crisis response"))
            if role_key:
                rows.append(("Role", _fmt(role_key, _ROLE_LABELS)))
            if vulnerable_flags:
                rows.append(("Vulnerable flags", ", ".join(_fmt(f, _FLAG_LABELS) for f in vulnerable_flags)))
            history_used = bool(trace.get("memory_match_count", 0) or any(trace.get("expanded_queries", [])))
            rows.append(("Patient history used", "Yes" if history_used else "No"))
            if expanded_queries and len(expanded_queries) > 1:
                rows.append(("Queries generated", str(len(expanded_queries))))
            if pathway:
                rows.append(("Pathway", _fmt(pathway, _PATHWAY_LABELS)))
            if gates:
                rows.append(("Policy gates", "; ".join(_fmt(g.get("gate_name", ""), _GATE_LABELS) for g in gates[:6])))
            tier_parts = [f"{c} {_TIER_LABELS_FULL[t]}" for t, c in tier_map.items() if c]
            if tier_parts:
                rows.append(("Evidence quality", "; ".join(tier_parts)))

            table_html = "".join(
                f"<tr>"
                f"<td style='padding:6px 14px 6px 0;color:#6b7280;font-size:13px;white-space:nowrap;"
                f"vertical-align:top;border-bottom:1px solid #f0f0f0'>{html.escape(lbl)}</td>"
                f"<td style='padding:6px 0 6px 14px;font-size:13px;font-weight:500;"
                f"vertical-align:top;border-bottom:1px solid #f0f0f0'>{html.escape(val)}</td>"
                f"</tr>"
                for lbl, val in rows
            )
            if table_html:
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;margin-bottom:12px'>{table_html}</table>",
                    unsafe_allow_html=True,
                )

            if gates:
                with st.expander("Policy gate details", expanded=False):
                    for gate in gates[:6]:
                        st.markdown(f"**{_fmt(gate.get('gate_name', ''), _GATE_LABELS)}**")
                        if gate.get("reason"):
                            st.caption(gate["reason"])

            if sources:
                st.markdown("#### Sources")
                for source in sources:
                    tier = source.get("evidence_tier", 3)
                    tier_label = source.get("tier_label", f"Tier {tier}")
                    tier_badge = (
                        f'<span class="tier-badge tier-{tier}" title="{source.get("tier_description","")}">'
                        f"{tier_label}</span>"
                    )
                    st.markdown(
                        f"<div class='source-card'>"
                        f"<div class='source-card-head'>"
                        f"<span class='source-badge'>{source.get('source_id','S')}</span>"
                        f"<div><strong>{source.get('title','Untitled')}</strong> {tier_badge}"
                        f"<br/><span>{source.get('journal','')}{' ' + str(source.get('year','')) if source.get('year') else ''}</span></div>"
                        f"</div>"
                        f"<div class='source-card-body'><p>{source.get('snippet','')}</p></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if source.get("url"):
                        st.link_button(f"Open {source.get('source_id','source')}", source["url"], use_container_width=False)

            if personal_context:
                st.markdown("#### Personal context used")
                for item in personal_context:
                    st.markdown(
                        f"<div class='context-card'><strong>{item.get('title', item.get('source','Context'))}</strong>"
                        f"<p>{item.get('snippet','')}</p></div>",
                        unsafe_allow_html=True,
                    )

            if longitudinal_memory:
                st.markdown("#### Patient history applied")
                st.markdown(
                    f"<div class='context-card'><strong>Persistent account memory</strong>"
                    f"<p>{html.escape(longitudinal_memory).replace(chr(10), '<br/>')}</p></div>",
                    unsafe_allow_html=True,
                )

        else:
            # ── PATIENT / CAREGIVER: simple trust card ────────────────────────
            risk_level = trace.get("risk_level", "")
            triage_next = meta.get("triage_summary", {}).get("next_step", "")
            history_used = bool(trace.get("memory_match_count", 0) or longitudinal_memory)

            trust_lines: list[str] = []
            if tier_map[1]:
                trust_lines.append(
                    f"**{tier_map[1]} NHS / NICE guideline{'s' if tier_map[1] > 1 else ''}** checked"
                )
            if tier_map[2]:
                trust_lines.append(f"**{tier_map[2]} clinical review{'s' if tier_map[2] > 1 else ''}** reviewed")
            if tier_map[3]:
                trust_lines.append(f"**{tier_map[3]} research {'studies' if tier_map[3] > 1 else 'study'}** consulted")
            if history_used:
                trust_lines.append("**Your saved health record** was used to personalise this answer")
            if risk_level:
                trust_lines.append(f"Risk level assessed as **{_fmt(risk_level, _RISK_LABELS).lower()}**")
            if triage_next:
                trust_lines.append(f"Suggested next step: **{triage_next}**")

            for line in trust_lines:
                st.markdown(f"- {line}")

            st.caption(
                "All answers are grounded in current NHS, NICE, and peer-reviewed clinical evidence. "
                "This is not a personal diagnosis — if you are concerned, follow the next step above."
            )

            if sources and any(s.get("url") for s in sources):
                st.markdown("**Read the sources:**")
                for source in sources:
                    if source.get("url"):
                        st.markdown(f"- [{source.get('title', source.get('source_id','Source'))}]({source['url']})")


def render_feedback_buttons(message: dict) -> None:
    trace = message.get("metadata", {}).get("trace", {})
    trace_id = trace.get("trace_id", "")
    if not trace_id:
        return

    rated_key = f"feedback_rated_{trace_id}"
    if st.session_state.get(rated_key):
        rating_given = st.session_state[rated_key]
        icon = "thumbs_up" if rating_given == "thumbs_up" else "thumbs_down"
        label = "Helpful" if rating_given == "thumbs_up" else "Not helpful"
        st.markdown(
            f'<div class="meta-pill-row" style="margin-top:4px">'
            f'<span style="opacity:0.65;font-size:12px">Rated: {label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    cols = st.columns([1, 1, 8], gap="small")
    with cols[0]:
        if st.button("👍", key=f"fb_up_{trace_id}", help="Helpful response", use_container_width=True):
            from backend.feedback_store import save_feedback
            save_feedback("thumbs_up", trace)
            st.session_state[rated_key] = "thumbs_up"
            st.rerun()
    with cols[1]:
        if st.button("👎", key=f"fb_down_{trace_id}", help="Not helpful", use_container_width=True):
            from backend.feedback_store import save_feedback
            save_feedback("thumbs_down", trace)
            st.session_state[rated_key] = "thumbs_down"
            st.rerun()


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
            render_message_meta(message)
            if message.get("role") == "assistant":
                render_feedback_buttons(message)
                render_why_this_answer(message)


def render_follow_up_questions(questions: list) -> None:
    if not questions:
        return
    st.markdown(
        "<div class='followup-container'>"
        "<span class='followup-label'>Also relevant — click if this applies to you</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    for idx, item in enumerate(questions[:5]):
        if isinstance(item, dict):
            display = item.get("display", "")
            prompt = item.get("prompt", display)
        else:
            display = str(item)
            prompt = display
        if not display:
            continue
        if st.button(
            display,
            key=f"followup_{idx}_{abs(hash(display)) % 99999}",
            use_container_width=True,
        ):
            send_follow_up(prompt)


def send_follow_up(question: str) -> None:
    st.session_state.queued_follow_up = question
    st.rerun()


def queue_prompt(prompt: str) -> None:
    st.session_state.prompt_draft = prompt
    st.rerun()


def clear_prompt_draft() -> None:
    st.session_state.prompt_draft = ""


def submit_prompt_draft() -> None:
    st.session_state.pending_submitted_prompt = st.session_state.get("prompt_draft", "").strip()
    st.session_state.prompt_draft = ""


def record_summary_export(username: str, summary_label: str) -> None:
    UserStore.add_audit(username, "summary_generated", f"{summary_label} generated")


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
    st.session_state.auth_panel = "Sign in"
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
allergies = UserStore.get_allergies(current_user)
conditions = UserStore.get_conditions(current_user)
vitals = UserStore.get_vitals(current_user)
traces = UserStore.get_interaction_traces(current_user, limit=5)
audit_records = UserStore.get_audit(current_user, limit=8)
latest_triage = UserStore.get_latest_triage_summary(current_user)

clinical_role_key = RoleRouter().resolve(
    user_profile.get("clinical_role") or user_profile.get("role", "")
).role_key
_is_clinical_user = clinical_role_key in ("doctor", "nurse", "midwife", "physiotherapist")

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

    if st.button("Workspace", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")

    if st.button("Health timeline", use_container_width=True):
        st.switch_page("pages/3_Health_Timeline.py")

    if st.button("Find clinical trials", use_container_width=True):
        st.switch_page("pages/4_Find_Clinical_Trials.py")

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
            st.session_state.auth_panel = "Sign in"
            st.switch_page("pages/1_Landing.py")

    with st.expander("Account settings", expanded=False):
        with st.form("profile_form"):
            profile_name = st.text_input("Full name", value=user_profile.get("display_name", ""))
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
            if len(re.findall(r"[A-Za-z]{2,}", profile_name)) < 2:
                st.error("Enter your full name with at least first and last name.")
            else:
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
            st.session_state.auth_panel = "Sign in"
            st.switch_page("pages/1_Landing.py")

    st.markdown("### Documents")
    profile_name = (user_profile.get("display_name") or "").strip()
    expected_document_name = "" if profile_name.lower() == current_user.lower() else profile_name
    if not expected_document_name:
        st.warning("Add your full name in Account settings before uploading if you want automatic document-name verification.")
    saved_paths = upload_documents(current_user, expected_name=expected_document_name)
    if saved_paths:
        with st.spinner("Indexing and extracting health data from uploaded documents..."):
            indexed = rag_engine.ingest_documents(user=current_user, file_paths=saved_paths)

        # Show what was auto-populated from each new document
        for doc in indexed:
            if not doc.get("is_new"):
                continue
            if doc.get("summary_error"):
                st.warning(f"{doc['file']}: {doc['summary_error']}")
            extracted = doc.get("extracted") or {}
            extraction_errors = extracted.get("extraction_errors", [])
            extracted_vitals = extracted.get("vitals", [])
            extracted_medications = extracted.get("medications", [])
            extracted_allergies = extracted.get("allergies", [])
            extracted_conditions = extracted.get("conditions", [])

            parts = []
            if extracted_vitals:
                parts.append(f"{len(extracted_vitals)} vital/lab result(s)")
            if extracted_medications:
                parts.append(f"{len(extracted_medications)} medication(s)")
            if extracted_allergies:
                parts.append(f"{len(extracted_allergies)} allergy/allergies")
            if extracted_conditions:
                parts.append(f"{len(extracted_conditions)} condition(s) noted")

            for error in extraction_errors:
                st.error(f"{doc['file']}: {error}")

            if parts:
                st.success(
                    f"**{doc['file']}** - auto-populated: {', '.join(parts)}. "
                    "Review in the trackers below."
                )
            elif extraction_errors:
                st.warning(f"**{doc['file']}** was saved, but structured extraction did not run.")
            else:
                st.info(f"**{doc['file']}** indexed. No structured data found to extract.")

        uploads = UserStore.get_uploads(current_user)
        medications = UserStore.get_medications(current_user)
        allergies = UserStore.get_allergies(current_user)
        conditions = UserStore.get_conditions(current_user)
        vitals = UserStore.get_vitals(current_user)

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

    with st.expander("Conditions and medical history", expanded=False):
        with st.form("condition_history_form"):
            condition_name = st.text_input("Condition or diagnosis")
            condition_status = st.selectbox("Status", ["unknown", "active", "past", "resolved"])
            condition_date = st.date_input(
                "Date recorded",
                value=datetime.now(timezone.utc).date(),
                key="condition_recorded_on",
            )
            condition_notes = st.text_input("Notes (optional)")
            condition_saved = st.form_submit_button(
                "Save condition",
                type="primary",
                use_container_width=True,
            )

        if condition_saved:
            saved_condition = UserStore.save_condition(
                current_user,
                {
                    "name": condition_name,
                    "status": condition_status,
                    "recorded_on": condition_date.isoformat(),
                    "notes": condition_notes,
                },
            )
            if saved_condition:
                rag_engine.restore_user_context(current_user)
                st.success("Condition history updated.")
                st.rerun()
            else:
                st.error("Enter a condition or diagnosis to save it.")

        if conditions:
            for condition in conditions[:8]:
                label_parts = [condition.get("name", "Condition")]
                status = condition.get("status", "")
                if status and status != "unknown":
                    label_parts.append(status.title())
                if condition.get("recorded_on"):
                    label_parts.append(condition["recorded_on"])
                st.markdown(
                    f"""
                    <div class="mini-record">
                        <strong>{html.escape(" | ".join(label_parts))}</strong>
                        <span>{html.escape(condition.get('notes', 'On file') or 'On file')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove condition", key=f"condition_delete_{condition['condition_id']}", use_container_width=True):
                    UserStore.delete_condition(current_user, condition["condition_id"])
                    rag_engine.restore_user_context(current_user)
                    st.rerun()
        else:
            st.caption("No conditions or past history saved yet.")

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

    with st.expander("Allergy and contraindication profile", expanded=False):
        with st.form("allergy_form"):
            allergy_name = st.text_input("Allergen name (drug, food, or substance)")
            allergy_reaction = st.text_input("Reaction (e.g. rash, anaphylaxis)")
            allergy_severity = st.selectbox("Severity", ["unknown", "mild", "moderate", "severe"])
            allergy_type = st.selectbox("Type", ["drug", "food", "environmental", "other"])
            allergy_confirmed = st.checkbox("Confirmed allergy (uncheck if suspected only)", value=True)
            allergy_saved = st.form_submit_button(
                "Save allergy", type="primary", use_container_width=True
            )

        if allergy_saved:
            saved_allergy = UserStore.save_allergy(
                current_user,
                {
                    "name": allergy_name,
                    "reaction": allergy_reaction,
                    "severity": allergy_severity,
                    "allergy_type": allergy_type,
                    "confirmed": allergy_confirmed,
                },
            )
            if saved_allergy:
                st.success("Allergy profile updated.")
                st.rerun()
            else:
                st.error("Enter an allergen name to save it.")

        if allergies:
            for allergy in allergies:
                label_parts = [allergy.get("name", "Allergen")]
                if allergy.get("reaction"):
                    label_parts.append(allergy["reaction"])
                severity = allergy.get("severity", "")
                confirmed_str = "" if allergy.get("confirmed", True) else " (suspected)"
                st.markdown(
                    f"""
                    <div class="mini-record">
                        <strong>{html.escape(" | ".join(label_parts))}{html.escape(confirmed_str)}</strong>
                        <span>{html.escape(severity.title() if severity else "Severity not recorded")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove", key=f"allergy_delete_{allergy['allergy_id']}", use_container_width=True):
                    UserStore.delete_allergy(current_user, allergy["allergy_id"])
                    st.rerun()
        else:
            st.caption("No allergies recorded. Add any known drug or food allergies here.")

    _VITALS_ALL = [
        ("blood_pressure", "Blood pressure", "mmHg", "e.g. 120/80"),
        ("heart_rate", "Heart rate", "bpm", "e.g. 72"),
        ("weight", "Weight", "kg", "e.g. 74"),
        ("height", "Height", "cm", "e.g. 170"),
        ("bmi", "BMI", "kg/m2", "e.g. 24.5"),
        ("respiratory_rate", "Respiratory rate", "breaths/min", "e.g. 16"),
        ("blood_glucose", "Blood glucose", "mmol/L", "e.g. 5.4"),
        ("temperature", "Temperature", "°C", "e.g. 37.2"),
        ("oxygen_saturation", "Oxygen saturation (SpO₂)", "%", "e.g. 98"),
        ("peak_flow", "Peak flow", "L/min", "e.g. 420"),
        ("hba1c", "HbA1c", "mmol/mol", "e.g. 48"),
        ("egfr", "eGFR", "mL/min/1.73m²", "e.g. 65"),
        ("creatinine", "Creatinine", "µmol/L", "e.g. 85"),
    ]
    _VITALS_LAY = {
        "blood_pressure", "heart_rate", "weight", "blood_glucose",
        "height", "bmi", "temperature", "oxygen_saturation",
        "respiratory_rate",
    }
    vitals_options = _VITALS_ALL if _is_clinical_user else [
        v for v in _VITALS_ALL if v[0] in _VITALS_LAY
    ]
    vitals_type_labels = {v[0]: v[1] for v in vitals_options}
    vitals_label_to_key = {v[1].lower(): v[0] for v in vitals_options}
    vitals_units = {v[0]: v[2] for v in vitals_options}
    vitals_placeholders = {v[0]: v[3] for v in vitals_options}

    with st.expander("Vitals and lab results", expanded=False):
        with st.form("vitals_form"):
            vitals_type_options = [v[1] for v in vitals_options]
            if selectbox_accepts_new_options():
                selected_vitals_label = st.selectbox(
                    "Measurement type",
                    options=vitals_type_options,
                    index=None,
                    placeholder="Select or type a measurement",
                    accept_new_options=True,
                )
                custom_vitals_label = ""
            else:
                selected_vitals_label = st.selectbox(
                    "Measurement type",
                    options=[*vitals_type_options, CUSTOM_VITALS_OPTION],
                )
                custom_vitals_label = (
                    st.text_input(
                        "Custom measurement type",
                        placeholder="e.g. Waist circumference",
                    )
                    if selected_vitals_label == CUSTOM_VITALS_OPTION
                    else ""
                )
            vitals_label = (custom_vitals_label or selected_vitals_label or "").strip()
            vitals_type = vitals_label_to_key.get(vitals_label.lower(), vitals_label)
            known_vitals_key = vitals_label_to_key.get(vitals_label.lower(), "")
            vitals_value = st.text_input(
                "Value",
                placeholder=vitals_placeholders.get(known_vitals_key, "e.g. 5.4, 120/80, positive"),
            )
            vitals_date = st.date_input("Date recorded", value=datetime.now(timezone.utc).date())
            vitals_notes = st.text_input("Notes (optional)")
            vitals_saved = st.form_submit_button(
                "Save reading", type="primary", use_container_width=True
            )

        if vitals_saved:
            if not vitals_type or not vitals_value.strip():
                st.error("Enter a measurement type and value to save this reading.")
            else:
                saved_vitals = UserStore.save_vitals_entry(
                    current_user,
                    {
                        "type": vitals_type,
                        "value": vitals_value,
                        "unit": vitals_units.get(vitals_type, ""),
                        "recorded_on": vitals_date.isoformat(),
                        "notes": vitals_notes,
                    },
                )
                if saved_vitals:
                    rag_engine.restore_user_context(current_user)
                    st.success("Reading saved.")
                    st.rerun()
                else:
                    st.error("Enter a measurement type and value to save this reading.")

        if vitals:
            for entry in vitals[:8]:
                vtype = format_vitals_type(entry.get("type", ""), vitals_type_labels)
                val = entry.get("value", "")
                unit = entry.get("unit", "")
                date_str = entry.get("recorded_on", "")
                st.markdown(
                    f"""
                    <div class="mini-record">
                        <strong>{html.escape(vtype)}: {html.escape(val)} {html.escape(unit)}</strong>
                        <span>{html.escape(date_str or "Date not recorded")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Remove", key=f"vitals_delete_{entry['vitals_id']}", use_container_width=True):
                    UserStore.delete_vitals_entry(current_user, entry["vitals_id"])
                    rag_engine.restore_user_context(current_user)
                    st.rerun()
        else:
            st.caption("No readings saved yet.")

    _SUMMARY_LABELS = {
        "patient":          ("Personal Health Summary",        "personal-health-summary"),
        "caregiver":        ("Personal Health Summary",        "personal-health-summary"),
        "doctor":           ("GP Summary",                     "gp-summary"),
        "nurse":            ("Nursing Handover Note",          "nursing-handover"),
        "midwife":          ("Maternity Care Summary",         "maternity-care-summary"),
        "physiotherapist":  ("Physiotherapy Assessment",       "physio-assessment"),
    }
    _summary_label, _summary_slug = _SUMMARY_LABELS.get(
        clinical_role_key, ("Health Summary", "health-summary")
    )

    st.markdown(f"### {_summary_label}")
    summary_pdf = rag_engine.build_summary_pdf_for_user(current_user)
    has_summary_content = bool(
        symptom_logs
        or medications
        or uploads
        or allergies
        or conditions
        or vitals
        or rag_engine.get_combined_longitudinal_memory(current_user)
        or latest_triage
    )
    st.download_button(
        f"Download {_summary_label} PDF",
        data=summary_pdf,
        file_name=f"{current_user}-{_summary_slug}.pdf",
        mime="application/pdf",
        use_container_width=True,
        disabled=not has_summary_content,
        on_click=record_summary_export,
        args=(current_user, _summary_label),
    )
    if latest_triage:
        st.caption(
            f"Latest triage: {latest_triage.get('urgency_level', 'Routine')} — "
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

# Show follow-up question chips for the last assistant turn only
if chat_history:
    _last = chat_history[-1]
    if _last.get("role") == "assistant":
        _follow_ups = _last.get("metadata", {}).get("follow_up_questions", [])
        if _follow_ups:
            render_follow_up_questions(_follow_ups)

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
elif st.session_state.get("queued_follow_up"):
    active_question = st.session_state.pop("queued_follow_up", "").strip()

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
                    "follow_up_questions": payload.get("follow_up_questions", []),
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
            render_message_meta(assistant_entry)
            render_why_this_answer(assistant_entry)
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
