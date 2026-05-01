import html
import re
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional

import streamlit as st

from app_ui.theme import inject_custom_css
from backend.product_config import PRODUCT_NAME, SUPPORT_EMAIL
from backend.user_store import UserStore


VITAL_LABELS = {
    "blood_pressure": "Blood pressure",
    "heart_rate": "Heart rate",
    "weight": "Weight",
    "blood_glucose": "Blood glucose",
    "temperature": "Temperature",
    "oxygen_saturation": "Oxygen saturation",
    "peak_flow": "Peak flow",
    "hba1c": "HbA1c",
    "egfr": "eGFR",
    "creatinine": "Creatinine",
}

TREND_ORDER = [
    "blood_pressure",
    "blood_glucose",
    "weight",
    "creatinine",
    "egfr",
]


st.set_page_config(
    page_title=f"Health Timeline - {PRODUCT_NAME}",
    page_icon=":material/timeline:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        try:
            parsed_date = datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
            return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
        except ValueError:
            return None


def display_date(value: str) -> str:
    parsed = parse_datetime(value)
    if parsed:
        return parsed.strftime("%d %b %Y")
    return value or "Date not recorded"


def clean_text(value: object, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def unique_nonempty(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = clean_text(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def first_number(value: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_blood_pressure(value: str) -> tuple[Optional[float], Optional[float]]:
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value or "")
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def vital_label(vital_type: str) -> str:
    return VITAL_LABELS.get(vital_type, vital_type.replace("_", " ").title())


def format_vital(entry: dict) -> str:
    pieces = [clean_text(entry.get("value"))]
    if entry.get("unit"):
        pieces.append(clean_text(entry.get("unit")))
    return " ".join(piece for piece in pieces if piece).strip() or "Value not recorded"


def render_list(items: list[str], empty: str) -> str:
    cleaned = unique_nonempty(items)
    if not cleaned:
        return f"<p>{html.escape(empty)}</p>"
    return "<ul class='summary-list'>" + "".join(
        f"<li>{html.escape(item)}</li>" for item in cleaned[:6]
    ) + "</ul>"


def render_summary_card(title: str, body_html: str, meta: str = "") -> None:
    st.markdown(
        f"""
        <div class="timeline-summary-card">
            <div class="feature-eyebrow">Health Summary</div>
            <h3>{html.escape(title)}</h3>
            {body_html}
            {f"<span class='timeline-card-meta'>{html.escape(meta)}</span>" if meta else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_vital_series(vitals: list[dict], vital_type: str) -> list[dict]:
    rows = []
    for entry in vitals:
        if entry.get("type") != vital_type:
            continue
        recorded = parse_datetime(entry.get("recorded_on") or entry.get("created_at", ""))
        if not recorded:
            continue

        if vital_type == "blood_pressure":
            systolic, diastolic = parse_blood_pressure(entry.get("value", ""))
            if systolic is None:
                continue
            rows.append(
                {
                    "date": recorded.date().isoformat(),
                    "Systolic": systolic,
                    "Diastolic": diastolic,
                }
            )
        else:
            value = first_number(entry.get("value", ""))
            if value is None:
                continue
            rows.append({"date": recorded.date().isoformat(), "value": value})

    rows.sort(key=lambda row: row["date"])
    return rows


def build_symptom_series(symptom_logs: list[dict], symptom_name: str) -> list[dict]:
    rows = []
    for entry in symptom_logs:
        if clean_text(entry.get("symptom")).lower() != symptom_name.lower():
            continue
        recorded = parse_datetime(entry.get("logged_for") or entry.get("created_at", ""))
        if not recorded:
            continue
        rows.append({"date": recorded.date().isoformat(), "severity": int(entry.get("severity", 0))})
    rows.sort(key=lambda row: row["date"])
    return rows


def split_recent_prior(rows: list[dict], value_key: str) -> tuple[list[float], list[float]]:
    dated = []
    for row in rows:
        try:
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        value = row.get(value_key)
        if value is not None:
            dated.append((row_date, float(value)))

    if not dated:
        return [], []

    latest_date = max(item[0] for item in dated)
    recent_start = latest_date - timedelta(days=7)
    recent = [value for row_date, value in dated if row_date >= recent_start]
    prior = [value for row_date, value in dated if row_date < recent_start]
    return recent, prior


def build_trend_insights(vitals: list[dict], symptom_logs: list[dict]) -> list[dict]:
    insights = []

    bp_rows = build_vital_series(vitals, "blood_pressure")
    recent_sys, prior_sys = split_recent_prior(bp_rows, "Systolic")
    if recent_sys:
        recent_avg = mean(recent_sys)
        prior_avg = mean(prior_sys) if prior_sys else None
        if recent_avg >= 140 or (prior_avg is not None and recent_avg >= prior_avg + 10):
            insights.append(
                {
                    "title": "Blood pressure pattern",
                    "body": "Your blood pressure has been higher than usual in the most recent saved readings. Consider clinical review if this continues or if symptoms develop.",
                    "detail": "blood_pressure",
                }
            )

    glucose_rows = build_vital_series(vitals, "blood_glucose")
    recent_glucose, prior_glucose = split_recent_prior(glucose_rows, "value")
    if recent_glucose:
        recent_avg = mean(recent_glucose)
        prior_avg = mean(prior_glucose) if prior_glucose else None
        if recent_avg >= 7.0 or (prior_avg is not None and recent_avg >= prior_avg + 1.0):
            insights.append(
                {
                    "title": "Glucose pattern",
                    "body": "Recent saved glucose readings are running higher than earlier entries. Review the details and consider clinical advice if this persists.",
                    "detail": "blood_glucose",
                }
            )

    weight_rows = build_vital_series(vitals, "weight")
    if len(weight_rows) >= 2:
        change = weight_rows[-1]["value"] - weight_rows[0]["value"]
        if abs(change) >= 2:
            direction = "increased" if change > 0 else "decreased"
            insights.append(
                {
                    "title": "Weight change",
                    "body": f"Saved weight readings have {direction} by about {abs(change):.1f} kg across the recorded period.",
                    "detail": "weight",
                }
            )

    egfr_rows = build_vital_series(vitals, "egfr")
    recent_egfr, prior_egfr = split_recent_prior(egfr_rows, "value")
    if recent_egfr:
        recent_avg = mean(recent_egfr)
        prior_avg = mean(prior_egfr) if prior_egfr else None
        if recent_avg < 60 or (prior_avg is not None and recent_avg <= prior_avg - 5):
            insights.append(
                {
                    "title": "eGFR pattern",
                    "body": "Recent saved eGFR readings are lower than earlier entries or below the usual review threshold. Consider clinical review in context.",
                    "detail": "egfr",
                }
            )

    symptom_names = unique_nonempty([entry.get("symptom", "") for entry in symptom_logs])
    for symptom_name in symptom_names[:4]:
        rows = build_symptom_series(symptom_logs, symptom_name)
        recent, prior = split_recent_prior(rows, "severity")
        if recent:
            recent_avg = mean(recent)
            prior_avg = mean(prior) if prior else None
            if recent_avg >= 7 or (prior_avg is not None and recent_avg >= prior_avg + 2):
                insights.append(
                    {
                        "title": f"{symptom_name} severity",
                        "body": f"Recent saved {symptom_name.lower()} severity is higher than usual. Use the details view to inspect the pattern.",
                        "detail": f"symptom:{symptom_name}",
                    }
                )

    return insights[:5]


def build_key_risks(
    latest_triage: dict,
    allergies: list[dict],
    symptom_logs: list[dict],
    trend_insights: list[dict],
) -> list[str]:
    risks = []
    if latest_triage:
        urgency = clean_text(latest_triage.get("urgency_level"), "Routine")
        next_step = clean_text(latest_triage.get("next_step"), "Self-care")
        risks.append(f"Latest triage: {urgency} - {next_step}")

    severe_allergies = [
        allergy.get("name", "")
        for allergy in allergies
        if allergy.get("severity") == "severe"
    ]
    if severe_allergies:
        risks.append("Severe allergies recorded: " + ", ".join(unique_nonempty(severe_allergies)[:3]))

    high_symptoms = [
        entry.get("symptom", "")
        for entry in symptom_logs
        if int(entry.get("severity", 0)) >= 7
    ]
    if high_symptoms:
        risks.append("High symptom severity logged: " + ", ".join(unique_nonempty(high_symptoms)[:3]))

    risks.extend(insight["title"] for insight in trend_insights[:2])
    return unique_nonempty(risks)


def build_timeline_events(
    uploads: list[dict],
    document_summaries: list[dict],
    symptom_logs: list[dict],
    medications: list[dict],
    allergies: list[dict],
    vitals: list[dict],
    triage_summaries: list[dict],
    audit_records: list[dict],
) -> list[dict]:
    events = []

    for upload in uploads:
        events.append(
            {
                "when": upload.get("uploaded_at", ""),
                "type": "Uploaded document",
                "title": clean_text(upload.get("file"), "Document uploaded"),
                "detail": "Summary available" if upload.get("summary_available") else "Saved to account",
            }
        )

    for summary in document_summaries:
        events.append(
            {
                "when": summary.get("updated_at", ""),
                "type": "Uploaded document",
                "title": clean_text(summary.get("file"), "Document indexed"),
                "detail": "Document summary refreshed",
            }
        )

    for entry in symptom_logs:
        severity = entry.get("severity", 0)
        events.append(
            {
                "when": entry.get("logged_for") or entry.get("created_at", ""),
                "type": "New symptom logged",
                "title": clean_text(entry.get("symptom"), "Symptom"),
                "detail": f"Severity {severity}/10" + (f" - {entry.get('triggers')}" if entry.get("triggers") else ""),
            }
        )

    for medication in medications:
        pieces = [
            medication.get("dose", ""),
            medication.get("schedule", ""),
            medication.get("reason", ""),
        ]
        events.append(
            {
                "when": medication.get("updated_at") or medication.get("created_at", ""),
                "type": "Medication added/changed",
                "title": clean_text(medication.get("name"), "Medication"),
                "detail": " - ".join(unique_nonempty(pieces)) or "Medication list updated",
            }
        )

    for allergy in allergies:
        detail_parts = [
            allergy.get("severity", ""),
            allergy.get("reaction", ""),
            allergy.get("allergy_type", ""),
        ]
        events.append(
            {
                "when": allergy.get("created_at", ""),
                "type": "Allergy recorded",
                "title": clean_text(allergy.get("name"), "Allergy"),
                "detail": " - ".join(unique_nonempty(detail_parts)) or "Allergy profile updated",
            }
        )

    for entry in vitals:
        events.append(
            {
                "when": entry.get("recorded_on") or entry.get("created_at", ""),
                "type": "Vitals/labs changed",
                "title": vital_label(entry.get("type", "")),
                "detail": format_vital(entry),
            }
        )

    for summary in triage_summaries:
        urgency = clean_text(summary.get("urgency_level"), "Routine")
        next_step = clean_text(summary.get("next_step"), "Self-care")
        events.append(
            {
                "when": summary.get("created_at", ""),
                "type": "Triage summary saved",
                "title": urgency,
                "detail": next_step,
            }
        )

    for record in audit_records:
        if record.get("event") != "summary_generated":
            continue
        events.append(
            {
                "when": record.get("time", ""),
                "type": "Health summary generated",
                "title": clean_text(record.get("details"), "Summary generated"),
                "detail": "Downloadable summary prepared from saved account data",
            }
        )

    events.sort(
        key=lambda event: parse_datetime(event.get("when", "")) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return events


def render_event(event: dict) -> None:
    st.markdown(
        f"""
        <div class="timeline-event">
            <div class="timeline-event-date">{html.escape(display_date(event.get("when", "")))}</div>
            <div class="timeline-event-body">
                <span>{html.escape(event.get("type", "Event"))}</span>
                <strong>{html.escape(event.get("title", "Untitled event"))}</strong>
                <p>{html.escape(event.get("detail", ""))}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_trend_detail(detail_key: str, vitals: list[dict], symptom_logs: list[dict]) -> None:
    if detail_key.startswith("symptom:"):
        symptom_name = detail_key.split(":", 1)[1]
        rows = build_symptom_series(symptom_logs, symptom_name)
        if rows:
            st.line_chart(rows, x="date", y="severity")
        return

    rows = build_vital_series(vitals, detail_key)
    if not rows:
        st.caption("No chartable saved readings yet.")
        return

    if detail_key == "blood_pressure":
        st.line_chart(rows, x="date", y=["Systolic", "Diastolic"])
    else:
        st.line_chart(rows, x="date", y="value")


def render_health_summary(
    profile: dict,
    uploads: list[dict],
    symptom_logs: list[dict],
    medications: list[dict],
    allergies: list[dict],
    vitals: list[dict],
    latest_triage: dict,
    memory: dict,
    trend_insights: list[dict],
) -> None:
    condition_context = unique_nonempty(
        [medication.get("reason", "") for medication in medications]
        + [latest_triage.get("pathway_label", "") if latest_triage else ""]
    )
    recent_vitals = [
        f"{vital_label(entry.get('type', ''))}: {format_vital(entry)} ({display_date(entry.get('recorded_on') or entry.get('created_at', ''))})"
        for entry in vitals[:5]
    ]
    risks = build_key_risks(latest_triage, allergies, symptom_logs, trend_insights)

    card_cols = st.columns(3, gap="medium")
    with card_cols[0]:
        render_summary_card(
            "Current conditions",
            render_list(condition_context, "No current condition context has been recorded yet."),
            "Derived from saved medication reasons and triage context.",
        )
    with card_cols[1]:
        render_summary_card(
            "Medications",
            render_list(
                [
                    " - ".join(unique_nonempty([m.get("name", ""), m.get("dose", ""), m.get("schedule", "")]))
                    for m in medications
                ],
                "No medications saved yet.",
            ),
            f"{len(medications)} saved",
        )
    with card_cols[2]:
        render_summary_card(
            "Allergies",
            render_list(
                [
                    " - ".join(unique_nonempty([a.get("name", ""), a.get("severity", ""), a.get("reaction", "")]))
                    for a in allergies
                ],
                "No allergies recorded.",
            ),
            f"{len(allergies)} saved",
        )

    lower_cols = st.columns([1, 1], gap="medium")
    with lower_cols[0]:
        render_summary_card(
            "Recent vitals/labs",
            render_list(recent_vitals, "No vitals or lab readings saved yet."),
            f"{len(vitals)} readings",
        )
    with lower_cols[1]:
        render_summary_card(
            "Key risks",
            render_list(risks, "No elevated risk pattern detected from saved timeline data yet."),
            "This is not a diagnosis.",
        )

    memory_text = clean_text(memory.get("summary", ""))
    if memory_text:
        st.markdown("### Longitudinal memory")
        st.info(memory_text)

    if uploads or symptom_logs:
        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("Documents", len(uploads))
        metric_cols[1].metric("Symptoms", len(symptom_logs))
        metric_cols[2].metric("Vitals/labs", len(vitals))
        metric_cols[3].metric("Latest role", clean_text(profile.get("clinical_role") or profile.get("role"), "Individual"))


def render_timeline(events: list[dict]) -> None:
    if not events:
        st.info("No timeline events yet. Upload documents, log symptoms, save medications, or record vitals to build this view.")
        return

    st.markdown("### Major events")
    for event in events[:12]:
        render_event(event)

    if len(events) > 12:
        with st.expander(f"View {len(events) - 12} older event(s)", expanded=False):
            for event in events[12:]:
                render_event(event)


def render_trends(vitals: list[dict], symptom_logs: list[dict], trend_insights: list[dict]) -> None:
    if trend_insights:
        for index, insight in enumerate(trend_insights):
            st.markdown(
                f"""
                <div class="timeline-insight-card">
                    <span>Pattern found</span>
                    <h3>{html.escape(insight["title"])}</h3>
                    <p>{html.escape(insight["body"])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("View details", expanded=False):
                render_trend_detail(insight["detail"], vitals, symptom_logs)
    else:
        st.info("No trend warnings yet. Add multiple readings over time to unlock pattern cards.")

    st.markdown("### Trend library")
    for vital_type in TREND_ORDER:
        rows = build_vital_series(vitals, vital_type)
        if not rows:
            continue
        with st.expander(vital_label(vital_type), expanded=False):
            render_trend_detail(vital_type, vitals, symptom_logs)

    symptom_names = unique_nonempty([entry.get("symptom", "") for entry in symptom_logs])
    for symptom_name in symptom_names[:6]:
        rows = build_symptom_series(symptom_logs, symptom_name)
        if len(rows) < 2:
            continue
        with st.expander(f"{symptom_name} severity", expanded=False):
            render_trend_detail(f"symptom:{symptom_name}", vitals, symptom_logs)


current_user = st.session_state.get("current_user")
if not current_user:
    st.warning("Please sign in to view your health timeline.")
    st.session_state.auth_panel = "Sign in"
    st.switch_page("pages/1_Landing.py")

profile = UserStore.get_user_profile(current_user)
uploads = UserStore.get_uploads(current_user)
document_summaries = UserStore.get_document_summaries(current_user)
symptom_logs = UserStore.get_symptom_logs(current_user, limit=None)
medications = UserStore.get_medications(current_user)
allergies = UserStore.get_allergies(current_user)
vitals = UserStore.get_vitals(current_user, limit=None)
triage_summaries = UserStore.get_triage_summaries(current_user, limit=None)
audit_records = UserStore.get_audit(current_user, limit=None)
latest_triage = triage_summaries[0] if triage_summaries else {}
memory = UserStore.get_longitudinal_memory(current_user)
trend_insights = build_trend_insights(vitals, symptom_logs)
timeline_events = build_timeline_events(
    uploads=uploads,
    document_summaries=document_summaries,
    symptom_logs=symptom_logs,
    medications=medications,
    allergies=allergies,
    vitals=vitals,
    triage_summaries=triage_summaries,
    audit_records=audit_records,
)

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-profile">
            <div class="feature-eyebrow">Timeline</div>
            <h2>{html.escape(clean_text(profile.get("display_name"), current_user))}</h2>
            <span class="clinical-role-badge">{html.escape(clean_text(profile.get("clinical_role") or profile.get("role"), "Patient / Individual"))}</span>
            <p>{html.escape(clean_text(profile.get("care_context"), "Personal health guidance"))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Workspace", use_container_width=True):
        st.switch_page("pages/2_Workspace.py")

    if st.button("Open chat", use_container_width=True):
        st.switch_page("pages/2_Chatbot.py")

    if st.button("Find clinical trials", use_container_width=True):
        st.switch_page("pages/4_Find_Clinical_Trials.py")

    if st.button("Sign out", use_container_width=True):
        st.session_state.current_user = None
        st.session_state.history_user = None
        st.session_state.chat_history = []
        st.session_state.auth_panel = "Sign in"
        st.switch_page("pages/1_Landing.py")

    st.caption(f"Support: {SUPPORT_EMAIL}")

st.markdown(
    f"""
    <div class="workspace-hero timeline-hero">
        <div class="feature-eyebrow">{PRODUCT_NAME}</div>
        <h1>Health Timeline</h1>
        <p>
            A clean view of saved health context, major events, and useful trends. The chat stays focused on conversation; this page keeps the longitudinal picture.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_metrics = st.columns(5, gap="small")
top_metrics[0].metric("Documents", len(uploads))
top_metrics[1].metric("Symptoms", len(symptom_logs))
top_metrics[2].metric("Medications", len(medications))
top_metrics[3].metric("Allergies", len(allergies))
top_metrics[4].metric("Vitals/labs", len(vitals))

st.session_state.setdefault("timeline_view", "Health Summary")
view = st.radio(
    "Timeline view",
    ["Health Summary", "Timeline", "Trends"],
    key="timeline_view",
    horizontal=True,
    label_visibility="collapsed",
)

if view == "Health Summary":
    render_health_summary(
        profile=profile,
        uploads=uploads,
        symptom_logs=symptom_logs,
        medications=medications,
        allergies=allergies,
        vitals=vitals,
        latest_triage=latest_triage,
        memory=memory,
        trend_insights=trend_insights,
    )
elif view == "Timeline":
    render_timeline(timeline_events)
else:
    render_trends(vitals, symptom_logs, trend_insights)
