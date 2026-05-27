import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from app_ui.theme import inject_custom_css
from backend.product_config import PRODUCT_NAME, SUPPORT_EMAIL
from backend.role_router import RoleRouter
from backend.user_store import UserStore

load_dotenv()

# Only genuine abbreviations that .title() can't produce correctly.
# Everything else is auto-formatted from the snake_case key.
_ABBREV_OVERRIDES: dict[str, str] = {
    "bmi": "BMI",
    "hba1c": "HbA1c",
    "egfr": "eGFR",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "ggt": "GGT",
    "ldh": "LDH",
    "crp": "CRP",
    "esr": "ESR",
    "tsh": "TSH",
    "inr": "INR",
    "psa": "PSA",
    "bnp": "BNP",
    "aptt": "APTT",
    "nt_probnp": "NT-proBNP",
    "b12": "Vitamin B12",
    "free_t4": "Free T4",
    "free_t3": "Free T3",
    "d_dimer": "D-Dimer",
}


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
    """Human-readable label for a vital/lab type key.

    Known abbreviations (e.g. HbA1c, eGFR) are overridden explicitly.
    Everything else is auto-formatted from its snake_case key so any
    new type extracted from a document works without code changes.
    """
    key = (vital_type or "").strip().lower()
    if key in _ABBREV_OVERRIDES:
        return _ABBREV_OVERRIDES[key]
    return key.replace("_", " ").title()


def format_vital(entry: dict) -> str:
    pieces = [clean_text(entry.get("value"))]
    if entry.get("unit"):
        pieces.append(clean_text(entry.get("unit")))
    return " ".join(piece for piece in pieces if piece).strip() or "Value not recorded"


def render_list(items: list[str], empty: str, cap: int = 6) -> str:
    cleaned = unique_nonempty(items)
    if not cleaned:
        return f"<p>{html.escape(empty)}</p>"
    visible = cleaned[:cap]
    overflow = len(cleaned) - cap
    html_items = "".join(f"<li>{html.escape(item)}</li>" for item in visible)
    suffix = f"<li><em>… and {overflow} more</em></li>" if overflow > 0 else ""
    return f"<ul class='summary-list'>{html_items}{suffix}</ul>"


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


def profile_role_key(profile: dict) -> str:
    return RoleRouter().resolve(
        profile.get("clinical_role") or profile.get("role", "")
    ).role_key


def health_summary_terms(role_key: str) -> dict[str, str]:
    if role_key in ("patient", "caregiver"):
        return {
            "current": "Current health snapshot",
            "recent": "What has changed recently",
            "medications": "Medicines",
            "allergies": "Allergies and reactions",
            "readings": "Recent readings",
            "history": "Past health history",
            "risks": "What needs attention",
            "latest_triage": "Latest guidance",
            "condition": "Health issue",
            "symptom": "Symptom",
            "reading": "Reading",
        }
    if role_key == "nurse":
        return {
            "current": "Current handover priorities",
            "recent": "Active symptoms and concerns",
            "medications": "Current medications",
            "allergies": "Allergies and safety alerts",
            "readings": "Latest observations and results",
            "history": "Relevant background",
            "risks": "Escalation priorities",
            "latest_triage": "Latest handover priority",
            "condition": "Active problem",
            "symptom": "Current symptom concern",
            "reading": "Observation/result",
        }
    if role_key == "midwife":
        return {
            "current": "Current maternity snapshot",
            "recent": "Active maternity concerns",
            "medications": "Medications and supplements",
            "allergies": "Allergies and contraindications",
            "readings": "Latest antenatal observations",
            "history": "Relevant obstetric and medical history",
            "risks": "Maternity escalation priorities",
            "latest_triage": "Latest maternity priority",
            "condition": "Maternity/background issue",
            "symptom": "Current symptom concern",
            "reading": "Antenatal observation/result",
        }
    if role_key == "physiotherapist":
        return {
            "current": "Current MSK and functional snapshot",
            "recent": "Current presentation",
            "medications": "Current medications",
            "allergies": "Allergies and contraindications",
            "readings": "Latest functional measures",
            "history": "Relevant medical and injury history",
            "risks": "Red flags and rehab priorities",
            "latest_triage": "Latest rehab priority",
            "condition": "Active MSK/functional issue",
            "symptom": "Current presentation",
            "reading": "Functional measure/result",
        }
    return {
        "current": "Current clinical snapshot",
        "recent": "Active concerns",
        "medications": "Current medication list",
        "allergies": "Allergies and contraindications",
        "readings": "Latest observations and results",
        "history": "Relevant past medical history",
        "risks": "Clinical priorities",
        "latest_triage": "Latest clinical priority",
        "condition": "Active problem",
        "symptom": "Presenting concern",
        "reading": "Observation/result",
    }


def memory_history_lines(memory_summary: str, cap: int = 4) -> list[str]:
    lines = []
    current_heading = ""
    for raw_line in (memory_summary or "").splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if line.endswith(":"):
            current_heading = line[:-1].strip().lower()
            continue
        if line.lower() in {"none", "none noted", "not recorded"}:
            continue
        if current_heading in {"patient summary", "conditions and history", "investigations or notable results"}:
            lines.append(line)
        if len(lines) >= cap:
            break
    return unique_nonempty(lines)


def build_previous_history_summary(
    conditions: list[dict],
    uploads: list[dict],
    memory: dict,
    role_key: str,
) -> list[str]:
    terms = health_summary_terms(role_key)
    lines = []
    for condition in conditions:
        status = clean_text(condition.get("status"), "unknown").lower()
        if status == "active":
            continue
        name = clean_text(condition.get("name"))
        if not name:
            continue
        recorded = display_date(condition.get("recorded_on") or condition.get("created_at", ""))
        detail = f"{name} - {status}"
        if recorded and recorded != "Date not recorded":
            detail += f" ({recorded})"
        lines.append(detail)
    lines.extend(memory_history_lines(memory.get("summary", "")))
    if uploads:
        lines.append(f"{len(uploads)} uploaded record{'s' if len(uploads) != 1 else ''} used in the account history.")
    if not lines:
        return []
    return [f"{terms['history']}: {item}" for item in unique_nonempty(lines)[:6]]


def get_vital_unit(vitals: list[dict], vital_type: str) -> str:
    """Returns the most common unit string for a vital type, or empty string."""
    units = [
        (entry.get("unit") or "").strip()
        for entry in vitals
        if (entry.get("type") or "").strip().lower() == vital_type
        and (entry.get("unit") or "").strip()
    ]
    if not units:
        return ""
    return max(set(units), key=units.count)


def get_unique_vital_types_in_data(vitals: list[dict]) -> list[str]:
    """Returns all vital types that have at least one chartable numeric reading, sorted."""
    chartable: set[str] = set()
    for entry in vitals:
        vtype = (entry.get("type") or "").strip().lower()
        if not vtype:
            continue
        if vtype == "blood_pressure":
            sys_val, _ = parse_blood_pressure(entry.get("value", ""))
            if sys_val is not None:
                chartable.add(vtype)
        else:
            if first_number(entry.get("value", "")) is not None:
                chartable.add(vtype)
    return sorted(chartable)


@st.cache_data(ttl=3600, show_spinner=False)
def get_llm_vital_priority(
    vital_types: tuple[str, ...],
    condition_names: tuple[str, ...],
    medication_names: tuple[str, ...],
) -> list[str]:
    """
    Ask the LLM to rank the available vital/lab types from most to least clinically
    important for this specific patient, given their conditions and medications.
    Falls back to alphabetical order if the call fails.
    """
    if not vital_types:
        return []
    if len(vital_types) == 1:
        return list(vital_types)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return list(vital_types)
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a clinical prioritisation assistant. "
                        "Given a patient's known conditions, current medications, and a list of "
                        "available vital/lab result type keys, rank the types from most to least "
                        "clinically important for monitoring this specific patient. "
                        "Return JSON with one key: 'priority_order' — an array of the exact type "
                        "keys in ranked order. Include every provided key exactly once."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Known conditions: {', '.join(condition_names) if condition_names else 'none recorded'}\n"
                        f"Current medications: {', '.join(medication_names) if medication_names else 'none recorded'}\n"
                        f"Available vital/lab type keys: {', '.join(vital_types)}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=400,
        )
        payload = json.loads(response.choices[0].message.content.strip())
        ranked = payload.get("priority_order", [])
        provided = set(vital_types)
        ordered = [str(t).strip() for t in ranked if str(t).strip() in provided]
        # Append any types the LLM missed
        ordered += [t for t in vital_types if t not in ordered]
        return ordered
    except Exception as exc:
        print(f"LLM vital priority failed: {exc}")
        return list(vital_types)


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
            trend_note = f" (up from ~{prior_avg:.0f} mmHg)" if prior_avg is not None else ""
            insights.append(
                {
                    "title": "Blood pressure pattern",
                    "body": f"Recent systolic average {recent_avg:.0f} mmHg{trend_note} — at or above 140. Consider clinical review if this continues or if symptoms develop.",
                    "detail": "blood_pressure",
                }
            )

    glucose_rows = build_vital_series(vitals, "blood_glucose")
    recent_glucose, prior_glucose = split_recent_prior(glucose_rows, "value")
    if recent_glucose:
        recent_avg = mean(recent_glucose)
        prior_avg = mean(prior_glucose) if prior_glucose else None
        if recent_avg >= 7.0 or (prior_avg is not None and recent_avg >= prior_avg + 1.0):
            trend_note = f" (up from ~{prior_avg:.1f})" if prior_avg is not None else ""
            insights.append(
                {
                    "title": "Glucose pattern",
                    "body": f"Recent blood glucose average {recent_avg:.1f} mmol/L{trend_note} — at or above the post-meal threshold of 7.0. Review details and consider clinical advice if this persists.",
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
                    "body": f"Weight has {direction} by {abs(change):.1f} kg across the recorded period ({weight_rows[0]['date']} → {weight_rows[-1]['date']}).",
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
                    "body": f"Recent eGFR average {recent_avg:.0f} — below the 60 review threshold or declining vs earlier readings. Consider clinical review in context.",
                    "detail": "egfr",
                }
            )

    hba1c_rows = build_vital_series(vitals, "hba1c")
    recent_hba1c, prior_hba1c = split_recent_prior(hba1c_rows, "value")
    if recent_hba1c:
        recent_avg = mean(recent_hba1c)
        prior_avg = mean(prior_hba1c) if prior_hba1c else None
        if recent_avg >= 48 or (prior_avg is not None and recent_avg >= prior_avg + 5):
            insights.append(
                {
                    "title": "HbA1c pattern",
                    "body": f"Recent HbA1c average {recent_avg:.0f} mmol/mol — at or above the diabetes diagnostic threshold (≥48), or rising vs earlier values. Review glycaemic management.",
                    "detail": "hba1c",
                }
            )

    chol_rows = build_vital_series(vitals, "cholesterol_total")
    recent_chol, prior_chol = split_recent_prior(chol_rows, "value")
    if recent_chol:
        recent_avg = mean(recent_chol)
        prior_avg = mean(prior_chol) if prior_chol else None
        if recent_avg >= 5.0 or (prior_avg is not None and recent_avg >= prior_avg + 0.5):
            insights.append(
                {
                    "title": "Cholesterol pattern",
                    "body": f"Recent total cholesterol average {recent_avg:.1f} mmol/L — at or above 5.0, or rising. Review cardiovascular risk in context.",
                    "detail": "cholesterol_total",
                }
            )

    hb_rows = build_vital_series(vitals, "haemoglobin")
    recent_hb, prior_hb = split_recent_prior(hb_rows, "value")
    if recent_hb:
        recent_avg = mean(recent_hb)
        prior_avg = mean(prior_hb) if prior_hb else None
        if recent_avg < 120 or (prior_avg is not None and recent_avg <= prior_avg - 10):
            insights.append(
                {
                    "title": "Haemoglobin pattern",
                    "body": f"Recent haemoglobin average {recent_avg:.0f} g/L — below 120 or falling vs earlier readings. Consider anaemia workup in context.",
                    "detail": "haemoglobin",
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
    conditions: list[dict],
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

    for condition in conditions:
        detail_parts = [
            condition.get("status", ""),
            condition.get("recorded_on", ""),
        ]
        events.append(
            {
                "when": condition.get("recorded_on") or condition.get("created_at", ""),
                "type": "Condition recorded",
                "title": clean_text(condition.get("name"), "Condition"),
                "detail": " - ".join(unique_nonempty(detail_parts)) or "Condition history updated",
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
    conditions: list[dict],
    vitals: list[dict],
    latest_triage: dict,
    memory: dict,
    trend_insights: list[dict],
) -> None:
    role_key = profile_role_key(profile)
    terms = health_summary_terms(role_key)

    active_conditions = [
        condition for condition in conditions
        if clean_text(condition.get("status"), "unknown").lower() == "active"
    ]
    condition_context = unique_nonempty(
        [condition.get("name", "") for condition in active_conditions]
    )

    recent_symptoms = []
    for entry in sorted(
        symptom_logs,
        key=lambda item: parse_datetime(item.get("logged_for") or item.get("created_at", ""))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )[:3]:
        symptom = clean_text(entry.get("symptom"))
        if not symptom:
            continue
        symptom_line = f"{terms['symptom']}: {symptom}"
        if entry.get("severity") not in ("", None):
            symptom_line += f" - severity {entry.get('severity')}/10"
        symptom_line += f" ({display_date(entry.get('logged_for') or entry.get('created_at', ''))})"
        recent_symptoms.append(symptom_line)

    # Show the latest reading for each unique vital/lab type (not just first 5 raw rows)
    latest_by_type: dict[str, dict] = {}
    for entry in vitals:
        vtype = (entry.get("type") or "").strip().lower()
        if not vtype:
            continue
        existing = latest_by_type.get(vtype)
        entry_dt = parse_datetime(entry.get("recorded_on") or entry.get("created_at", ""))
        existing_dt = (
            parse_datetime(existing.get("recorded_on") or existing.get("created_at", ""))
            if existing else None
        )
        if existing is None or (entry_dt or datetime.min.replace(tzinfo=timezone.utc)) > (
            existing_dt or datetime.min.replace(tzinfo=timezone.utc)
        ):
            latest_by_type[vtype] = entry
    latest_vital_rows = sorted(
        latest_by_type.items(),
        key=lambda item: parse_datetime(item[1].get("recorded_on") or item[1].get("created_at", ""))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    recent_vitals = [
        f"{terms['reading']}: {vital_label(vtype)} {format_vital(entry)} ({display_date(entry.get('recorded_on') or entry.get('created_at', ''))})"
        for vtype, entry in latest_vital_rows
    ]
    risks = build_key_risks(latest_triage, allergies, symptom_logs, trend_insights)
    previous_history = build_previous_history_summary(conditions, uploads, memory, role_key)

    current_snapshot = []
    if latest_triage:
        triage_line = " - ".join(
            unique_nonempty([
                latest_triage.get("urgency_level", ""),
                latest_triage.get("next_step", ""),
            ])
        )
        if triage_line:
            current_snapshot.append(f"{terms['latest_triage']}: {triage_line}")
        if latest_triage.get("pathway_label"):
            pathway_label = "Care topic" if role_key in ("patient", "caregiver") else "Pathway"
            current_snapshot.append(f"{pathway_label}: {latest_triage['pathway_label']}")
    current_snapshot.extend(
        [f"{terms['condition']}: {item}" for item in condition_context[:4]]
    )
    current_snapshot.extend(recent_symptoms[:2])
    if medications:
        current_snapshot.append(f"{terms['medications']}: {len(medications)} recorded")
    if recent_vitals:
        current_snapshot.append(recent_vitals[0])

    card_cols = st.columns(3, gap="medium")
    with card_cols[0]:
        render_summary_card(
            terms["current"],
            render_list(current_snapshot, "No current saved health data is available yet."),
            f"{len(active_conditions)} active condition{'s' if len(active_conditions) != 1 else ''}",
        )
    with card_cols[1]:
        render_summary_card(
            terms["readings"],
            render_list(recent_vitals, "No measurements or lab readings saved yet.", cap=12),
            f"{len(latest_by_type)} latest result{'s' if len(latest_by_type) != 1 else ''}",
        )
    with card_cols[2]:
        render_summary_card(
            terms["risks"],
            render_list(risks, "No elevated risk pattern detected from saved timeline data yet."),
            "Review alongside clinical judgement." if role_key not in ("patient", "caregiver") else "This is not a diagnosis.",
        )

    lower_cols = st.columns(3, gap="medium")
    with lower_cols[0]:
        render_summary_card(
            terms["medications"],
            render_list(
                [
                    " - ".join(unique_nonempty([m.get("name", ""), m.get("dose", ""), m.get("schedule", "")]))
                    for m in medications
                ],
                "No medications saved yet.",
            ),
            f"{len(medications)} saved",
        )
    with lower_cols[1]:
        render_summary_card(
            terms["allergies"],
            render_list(
                [
                    " - ".join(unique_nonempty([a.get("name", ""), a.get("severity", ""), a.get("reaction", "")]))
                    for a in allergies
                ],
                "No allergies recorded.",
            ),
            f"{len(allergies)} saved",
        )
    with lower_cols[2]:
        render_summary_card(
            terms["history"],
            render_list(previous_history, "No previous health history has been saved yet."),
            f"{len(uploads)} uploaded record{'s' if len(uploads) != 1 else ''}",
        )

    memory_text = clean_text(memory.get("summary", ""))
    if memory_text:
        memory_heading = "Saved history notes" if role_key in ("patient", "caregiver") else "Longitudinal clinical memory"
        st.markdown(f"### {memory_heading}")
        st.info(memory_text)

    if uploads or symptom_logs:
        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("Documents", len(uploads))
        metric_cols[1].metric("Symptoms", len(symptom_logs))
        metric_cols[2].metric("Vitals/labs", len(vitals))
        metric_cols[3].metric("Latest role", RoleRouter().resolve(profile.get("clinical_role") or profile.get("role", "")).display_label)


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


def render_trends(
    vitals: list[dict],
    symptom_logs: list[dict],
    trend_insights: list[dict],
    conditions: list[dict] | None = None,
    medications: list[dict] | None = None,
) -> None:
    if trend_insights:
        for insight in trend_insights:
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
    all_vital_types = get_unique_vital_types_in_data(vitals)
    if all_vital_types:
        # Build tuples for the cached LLM call
        condition_names = tuple(
            str(c.get("name", "")).strip()
            for c in (conditions or [])
            if str(c.get("name", "")).strip()
        )
        medication_names = tuple(
            str(m.get("name", "")).strip()
            for m in (medications or [])
            if str(m.get("name", "")).strip()
        )
        with st.spinner("Ordering results by clinical relevance…"):
            priority_types = get_llm_vital_priority(
                tuple(all_vital_types), condition_names, medication_names
            )

        for vital_type in priority_types:
            rows = build_vital_series(vitals, vital_type)
            if not rows:
                continue
            unit = get_vital_unit(vitals, vital_type)
            label = vital_label(vital_type)
            expander_label = f"{label} ({unit})" if unit else label
            expander_label += f"  ·  {len(rows)} reading{'s' if len(rows) != 1 else ''}"
            with st.expander(expander_label, expanded=False):
                render_trend_detail(vital_type, vitals, symptom_logs)
    else:
        st.caption("No chartable vitals or lab results saved yet. Upload a clinical document or record readings manually.")

    symptom_names = unique_nonempty([entry.get("symptom", "") for entry in symptom_logs])
    for symptom_name in symptom_names[:6]:
        rows = build_symptom_series(symptom_logs, symptom_name)
        if len(rows) < 2:
            continue
        with st.expander(f"{symptom_name} — severity over time  ·  {len(rows)} readings", expanded=False):
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
conditions = UserStore.get_conditions(current_user)
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
    conditions=conditions,
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

top_metrics = st.columns(6, gap="small")
top_metrics[0].metric("Documents", len(uploads))
top_metrics[1].metric("Symptoms", len(symptom_logs))
top_metrics[2].metric("Conditions", len(conditions))
top_metrics[3].metric("Medications", len(medications))
top_metrics[4].metric("Allergies", len(allergies))
top_metrics[5].metric("Vitals/labs", len(vitals))

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
        conditions=conditions,
        vitals=vitals,
        latest_triage=latest_triage,
        memory=memory,
        trend_insights=trend_insights,
    )
elif view == "Timeline":
    render_timeline(timeline_events)
else:
    render_trends(vitals, symptom_logs, trend_insights, conditions=conditions, medications=medications)
