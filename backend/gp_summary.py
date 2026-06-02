from __future__ import annotations

from datetime import datetime, timezone
import re
from textwrap import wrap
from typing import Dict, Iterable, List, Optional

import fitz

from backend.symptom_tracker import build_recent_symptom_lines


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN_X = 44
MARGIN_Y = 48
BODY_FONT_SIZE = 10
LINE_HEIGHT = 13
SECTION_GAP = 10
CONTENT_START_Y = MARGIN_Y + 86
CONTENT_END_Y = PAGE_HEIGHT - 56
HEADER_BOX_HEIGHT = 56



def _strip_markdown(value: str) -> str:
    """Remove markdown formatting so raw text is safe to insert into a PDF."""
    text = (value or "").strip()
    # Replace inline patterns preserving the inner text, strip block patterns
    text = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]*)_{1,2}", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\s*\[(?:auto[- ]?extracted|extracted|source|record used)[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"^[-*+]\s+", "", text)
    return " ".join(text.split()).strip()


def _normalize_text(value: str) -> str:
    return _strip_markdown(value)


def _is_empty_placeholder(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()
    return normalized in {"none", "none noted", "not available", "not recorded"}


def _clean_lines(lines: Iterable[str], max_lines: int) -> List[str]:
    cleaned = []
    seen = set()
    for line in lines:
        text = _normalize_text(line)
        if not text or _is_empty_placeholder(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= max_lines:
            break
    return cleaned


def _is_question_like(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return False
    starters = (
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
        "can ",
        "could ",
        "should ",
        "would ",
        "do ",
        "does ",
        "did ",
        "is ",
        "are ",
    )
    return normalized.endswith("?") or normalized.startswith(starters)


def _is_personal_concern(text: str) -> bool:
    normalized = f" {re.sub(r'[^a-z0-9]+', ' ', (text or '').lower())} "
    personal_markers = (
        " i ",
        " i have ",
        " i am ",
        " i feel ",
        " i felt ",
        " ive ",
        " i've ",
        " my ",
        " me ",
        " experiencing ",
        " feeling ",
    )
    return any(marker in normalized for marker in personal_markers)


def _is_low_value_chat_prompt(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not normalized:
        return True
    if _is_personal_concern(normalized):
        return False
    generic_starts = (
        "what symptoms",
        "what does the evidence",
        "summarize ",
        "summarise ",
        "explain ",
        "tell me ",
        "can you ",
        "could you ",
        "should i ",
        "what should ",
    )
    return _is_question_like(normalized) or normalized.startswith(generic_starts)


def _is_summary_noise_line(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not normalized:
        return True
    account_prefixes = (
        "age",
        "biological sex",
        "display name",
        "role",
        "username",
        "email",
        "support",
        "patient name",
        "record used",
        "source record",
        "uploaded document",
    )
    if any(normalized.startswith(prefix) for prefix in account_prefixes):
        return True
    if "auto extracted from" in normalized or "extracted from" in normalized:
        return True
    if _is_question_like(text) and not _is_personal_concern(text):
        return True
    return False


def _clean_health_lines(lines: Iterable[str], max_lines: int) -> List[str]:
    return _clean_lines(
        [line for line in lines if not _is_summary_noise_line(_normalize_text(line))],
        max_lines=max_lines,
    )


def _wrap_lines(lines: Iterable[str], width: int = 82) -> List[str]:
    wrapped = []
    for line in lines:
        chunks = wrap(line, width=width) or [line]
        wrapped.extend(chunks)
    return wrapped


def _parse_memory_sections(longitudinal_memory: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_key = "unstructured"

    for raw_line in (longitudinal_memory or "").splitlines():
        line = _normalize_text(raw_line)
        if not line:
            continue
        if line.endswith(":"):
            current_key = line[:-1].strip().lower()
            sections.setdefault(current_key, [])
            continue
        if _is_empty_placeholder(line):
            continue
        sections.setdefault(current_key, []).append(line)

    return sections


def _section_lines(
    sections: Dict[str, List[str]],
    keys: Iterable[str],
    max_lines: int,
) -> List[str]:
    candidates: List[str] = []
    for key in keys:
        candidates.extend(sections.get(key.lower(), []))
    return _clean_health_lines(candidates, max_lines)


def _memory_snapshot_lines(longitudinal_memory: str, max_lines: int = 5) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    snapshot = _section_lines(
        sections,
        ["patient summary", "conditions and history"],
        max_lines=max_lines,
    )
    if snapshot:
        return snapshot
    return _clean_health_lines(sections.get("unstructured", []), max_lines)


def _memory_active_concern_lines(longitudinal_memory: str, max_lines: int = 6) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    concern_lines = _section_lines(
        sections,
        ["recent symptoms or active concerns"],
        max_lines=max_lines,
    )
    if concern_lines:
        return concern_lines
    return _clean_health_lines(sections.get("unstructured", []), max_lines)


def _build_recorded_medication_line(medication: Dict) -> str:
    name = _normalize_text(medication.get("name", ""))
    if not name:
        return ""
    parts = [name]
    dose = _normalize_text(medication.get("dose", ""))
    schedule = _normalize_text(medication.get("schedule", ""))
    reason = _normalize_text(medication.get("reason", ""))
    if dose:
        parts.append(dose)
    if schedule:
        parts.append(schedule)
    if reason:
        parts.append(f"for {reason}")
    return " - ".join(parts)


def _medication_lines(
    medications: Iterable[Dict],
    longitudinal_memory: str = "",
    role_key: str = "patient",
    max_lines: int = 8,
) -> List[str]:
    words = _role_words(role_key)
    lines = []
    for med in medications:
        line = _build_recorded_medication_line(med)
        if line:
            updated = _date_label(med.get("updated_at") or med.get("created_at", ""))
            if updated:
                line = f"{words['medication']}: {line} (updated {updated})"
            else:
                line = f"{words['medication']}: {line}"
            lines.append(line)

    sections = _parse_memory_sections(longitudinal_memory)
    memory_lines = _section_lines(
        sections,
        ["current treatments and medicines"],
        max_lines=max_lines,
    )
    if memory_lines:
        label = "From previous notes" if _is_lay_role(role_key) else "From longitudinal record"
        lines.extend(f"{label}: {line}" for line in memory_lines)

    return _clean_lines(lines, max_lines=max_lines)


def _upload_lines(uploads: Iterable[Dict], max_lines: int = 5) -> List[str]:
    return _clean_lines(
        [item.get("file", "Uploaded document") for item in uploads],
        max_lines=max_lines,
    )


def _symptom_lines(
    symptom_logs: List[Dict],
    longitudinal_memory: str,
    role_key: str = "patient",
    max_lines: int = 6,
) -> List[str]:
    words = _role_words(role_key)
    tracked_lines = []
    for entry in sorted(
        symptom_logs or [],
        key=lambda item: _entry_datetime(item, "logged_for", "created_at"),
        reverse=True,
    ):
        symptom = _normalize_text(entry.get("symptom", ""))
        if not symptom:
            continue
        logged_for = _date_label(entry.get("logged_for") or entry.get("created_at", ""))
        severity = entry.get("severity", "")
        parts = [symptom]
        if severity != "":
            parts.append(f"severity {severity}/10")
        if entry.get("triggers"):
            parts.append(f"trigger: {_normalize_text(entry.get('triggers', ''))}")
        if entry.get("notes"):
            parts.append(_normalize_text(entry.get("notes", "")))
        line = f"{words['symptom']}: " + " - ".join(parts)
        if logged_for:
            line += f" ({logged_for})"
        tracked_lines.append(line)

    tracked = _clean_lines(tracked_lines, max_lines=max_lines)
    if tracked:
        return tracked
    return _memory_active_concern_lines(longitudinal_memory, max_lines=max_lines)


def _parse_ts(timestamp_str: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat((timestamp_str or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _parse_date_value(value: str) -> Optional[datetime]:
    parsed = _parse_ts(value)
    if parsed:
        return parsed
    try:
        parsed_date = datetime.strptime(str(value or "")[:10], "%Y-%m-%d").date()
        return datetime.combine(parsed_date, datetime.min.time(), tzinfo=timezone.utc)
    except Exception:
        return None


def _date_label(value: str) -> str:
    parsed = _parse_date_value(value)
    if parsed:
        return parsed.strftime("%d %b %Y")
    return _normalize_text(value)


def _entry_datetime(entry: Dict, *fields: str) -> datetime:
    for field in fields:
        parsed = _parse_date_value(str(entry.get(field, "")))
        if parsed:
            return parsed
    return datetime.min.replace(tzinfo=timezone.utc)


def _canonical_role_key(role_key: str) -> str:
    key = (role_key or "").strip().lower()
    aliases = {
        "patient": "patient",
        "patient / individual": "patient",
        "individual": "patient",
        "caregiver": "caregiver",
        "doctor": "doctor",
        "doctor / physician": "doctor",
        "physician": "doctor",
        "other clinician": "doctor",
        "clinician": "doctor",
        "clinician / care team": "doctor",
        "nurse": "nurse",
        "midwife": "midwife",
        "physiotherapist": "physiotherapist",
        "physio": "physiotherapist",
    }
    return aliases.get(key, "patient")


def _is_lay_role(role_key: str) -> bool:
    return _canonical_role_key(role_key) in {"patient", "caregiver"}


def _role_words(role_key: str) -> Dict[str, str]:
    role = _canonical_role_key(role_key)
    if role in {"patient", "caregiver"}:
        return {
            "latest_triage": "Latest guidance",
            "condition": "Health issue",
            "active_condition": "Current health issue",
            "past_condition": "Previous health issue",
            "symptom": "Recent symptom",
            "medication": "Medicine",
            "reading": "Reading",
            "previous": "Previous",
            "plan": "Suggested next step",
            "monitor": "What to watch",
            "escalation": "Get urgent help if",
        }
    if role == "nurse":
        return {
            "latest_triage": "Latest handover priority",
            "condition": "Background condition",
            "active_condition": "Active problem",
            "past_condition": "Past history",
            "symptom": "Current symptom concern",
            "medication": "Current medication",
            "reading": "Observation/result",
            "previous": "Previous",
            "plan": "Nursing/clinical plan",
            "monitor": "Monitoring",
            "escalation": "Escalation criteria",
        }
    if role == "midwife":
        return {
            "latest_triage": "Latest maternity priority",
            "condition": "Maternity/background issue",
            "active_condition": "Active maternity concern",
            "past_condition": "Relevant history",
            "symptom": "Current symptom concern",
            "medication": "Medication or supplement",
            "reading": "Antenatal observation/result",
            "previous": "Previous",
            "plan": "Maternity plan",
            "monitor": "Monitoring",
            "escalation": "Escalation criteria",
        }
    if role == "physiotherapist":
        return {
            "latest_triage": "Latest rehab priority",
            "condition": "Relevant condition",
            "active_condition": "Active MSK/functional issue",
            "past_condition": "Relevant history",
            "symptom": "Current presentation",
            "medication": "Current medication",
            "reading": "Functional measure/result",
            "previous": "Previous",
            "plan": "Rehab plan",
            "monitor": "Monitoring",
            "escalation": "Red flags/escalation",
        }
    return {
        "latest_triage": "Latest clinical priority",
        "condition": "Condition",
        "active_condition": "Active problem",
        "past_condition": "Past medical history",
        "symptom": "Presenting concern",
        "medication": "Current medication",
        "reading": "Observation/result",
        "previous": "Previous",
        "plan": "Plan",
        "monitor": "Monitoring",
        "escalation": "Escalation criteria",
    }


def _latest_consultation_lines(
    chat_history: List[Dict],
    triage_summaries: List[Dict],
    role_key: str = "patient",
    max_complaints: int = 2,
) -> List[str]:
    words = _role_words(role_key)
    lay_role = _is_lay_role(role_key)

    # Most recent triage is the anchor for the latest consultation
    sorted_triages = sorted(
        (t for t in (triage_summaries or []) if _parse_ts(t.get("created_at", ""))),
        key=lambda t: _parse_ts(t.get("created_at", "")),
        reverse=True,
    )
    latest_triage = sorted_triages[0] if sorted_triages else {}

    # Determine the consultation date from the triage, falling back to the
    # most recent user message if no triage exists yet
    consultation_date = None
    if latest_triage:
        consultation_date = (_parse_ts(latest_triage.get("created_at", "")) or datetime.now(timezone.utc)).date()
    else:
        for msg in reversed(chat_history or []):
            if (msg.get("role") or msg.get("type")) == "user":
                ts = _parse_ts(msg.get("timestamp", ""))
                if ts:
                    consultation_date = ts.date()
                    break

    if not latest_triage and not consultation_date:
        return []

    # Collect the real presenting concern, but skip generic education prompts
    # so the summary does not repeat unrelated chat questions.
    complaints: List[str] = []
    triage_question = _normalize_text(latest_triage.get("question", "")) if latest_triage else ""
    if triage_question and not _is_low_value_chat_prompt(triage_question):
        complaints.append(triage_question)

    if consultation_date:
        for msg in (chat_history or []):
            if (msg.get("role") or msg.get("type")) != "user":
                continue
            ts = _parse_ts(msg.get("timestamp", ""))
            if not ts or ts.date() != consultation_date:
                continue
            content = _normalize_text(msg.get("content", ""))
            duplicate = any(content.lower() == existing.lower() for existing in complaints)
            if content and not duplicate and not _is_low_value_chat_prompt(content):
                complaints.append(content)

    date_label = consultation_date.strftime("%d %b %Y") if consultation_date else "Date unknown"
    lines: List[str] = [f"Date: {date_label}"]

    if not lay_role:
        for i, complaint in enumerate(complaints[:max_complaints]):
            truncated = complaint[:200] + "..." if len(complaint) > 200 else complaint
            prefix = "Presenting complaint:" if i == 0 else "Also reported:"
            lines.append(f"{prefix} {truncated}")

    if latest_triage:
        if latest_triage.get("pathway_label") and not lay_role:
            lines.append(f"Pathway: {latest_triage['pathway_label']}")
        if latest_triage.get("decision_summary"):
            summary = latest_triage["decision_summary"]
            low_value_summary = "no specific high-acuity presentation matched" in summary.lower()
            if not lay_role or not low_value_summary:
                label = "Summary" if lay_role else "Assessment"
                lines.append(f"{label}: {summary[:250] + '...' if len(summary) > 250 else summary}")
        if latest_triage.get("urgency_level"):
            lines.append(f"Urgency: {latest_triage['urgency_level']}")
        if latest_triage.get("next_step"):
            lines.append(f"{words['plan']}: {latest_triage['next_step']}")
        monitor = latest_triage.get("what_to_monitor", [])[:3]
        if monitor:
            lines.append(f"{words['monitor']}: " + "; ".join(monitor))
        actions = latest_triage.get("immediate_actions", [])[:2]
        if actions:
            label = "Do now" if lay_role else "Immediate actions"
            lines.append(label + ": " + "; ".join(actions))
        escalations = latest_triage.get("escalation_triggers", [])[:2]
        if escalations:
            lines.append(f"{words['escalation']}: " + "; ".join(escalations))

    return lines


def _additional_context_lines(longitudinal_memory: str, max_lines: int = 6) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    return _section_lines(
        sections,
        ["investigations or notable results", "care plan and follow-up"],
        max_lines=max_lines,
    )


# ── Role-specific summary configuration ───────────────────────────────────────
# Each entry: title (PDF header), footer (PDF footer), sections (ordered list of
# (heading, section_key) pairs). section_key maps to a data builder in build_summary_pdf.
_ROLE_SUMMARY_CONFIGS: Dict[str, Dict] = {
    "patient": {
        "title": "Personal Health Summary",
        "footer": "Your personal health record. Share with your healthcare team as needed.",
        "sections": [
            ("Current Health Snapshot", "current_snapshot"),
            ("What Needs Attention Now", "active_concerns"),
            ("Recent Readings", "vitals"),
            ("My Medicines", "medications"),
            ("Allergies and Reactions", "allergies"),
            ("Past Health History", "previous_history"),
            ("Latest Care Advice", "consultation"),
        ],
    },
    "caregiver": {
        "title": "Personal Health Summary",
        "footer": "Personal health record for the person in your care. Share with their healthcare team as needed.",
        "sections": [
            ("Current Care Snapshot", "current_snapshot"),
            ("Recent Concerns", "active_concerns"),
            ("Recent Readings", "vitals"),
            ("Medicines", "medications"),
            ("Allergies and Reactions", "allergies"),
            ("Previous Health History", "previous_history"),
            ("Latest Care Advice", "consultation"),
        ],
    },
    "doctor": {
        "title": "GP Summary",
        "footer": "Prepared for GP or clinical handover. Medications shown are only those explicitly recorded by the patient.",
        "sections": [
            ("Current Clinical Snapshot", "current_snapshot"),
            ("Latest Consultation and Plan", "consultation"),
            ("Relevant Past Medical History", "previous_history"),
            ("Current Medication List", "medications"),
            ("Allergies and Contraindications", "allergies"),
            ("Latest Observations and Results", "vitals"),
            ("Investigations and Follow-Up", "investigations"),
            ("Supporting Records", "uploads"),
        ],
    },
    "nurse": {
        "title": "Nursing Handover Note",
        "footer": "Nursing documentation for handover. Verify all medications and observations at handover.",
        "sections": [
            ("Current Handover Priorities", "current_snapshot"),
            ("Latest Nursing/Clinical Plan", "consultation"),
            ("Relevant Background", "previous_history"),
            ("Current Medications", "medications"),
            ("Allergies and Safety Alerts", "allergies"),
            ("Latest Observations and Results", "vitals"),
            ("Care Plan and Follow-Up", "investigations"),
            ("Supporting Records", "uploads"),
        ],
    },
    "midwife": {
        "title": "Maternity Care Summary",
        "footer": "Prepared for midwifery handover or antenatal review.",
        "sections": [
            ("Current Maternity Snapshot", "current_snapshot"),
            ("Latest Maternity Review and Plan", "consultation"),
            ("Relevant Obstetric and Medical History", "previous_history"),
            ("Medications and Supplements", "medications"),
            ("Allergies and Contraindications", "allergies"),
            ("Latest Antenatal Observations and Results", "vitals"),
            ("Care Plan and Follow-Up", "investigations"),
            ("Supporting Records", "uploads"),
        ],
    },
    "physiotherapist": {
        "title": "Physiotherapy Assessment Summary",
        "footer": "Prepared for physiotherapy handover or inter-professional communication.",
        "sections": [
            ("Current MSK and Functional Snapshot", "current_snapshot"),
            ("Latest Assessment and Rehab Plan", "consultation"),
            ("Relevant Medical and Injury History", "previous_history"),
            ("Current Medications", "medications"),
            ("Allergies and Contraindications", "allergies"),
            ("Latest Functional Measures and Results", "vitals"),
            ("Treatment Plan and Goals", "investigations"),
            ("Supporting Records", "uploads"),
        ],
    },
}

_DEFAULT_SUMMARY_CONFIG = _ROLE_SUMMARY_CONFIGS["patient"]


def _get_summary_config(role_key: str) -> Dict:
    return _ROLE_SUMMARY_CONFIGS.get(_canonical_role_key(role_key), _DEFAULT_SUMMARY_CONFIG)


def _draw_page_frame(
    page: fitz.Page,
    display_name: str,
    exported_at: str,
    page_number: int,
    doc_title: str = "Health Summary",
    footer_text: str = "",
) -> None:
    page.draw_rect(
        fitz.Rect(MARGIN_X, MARGIN_Y, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - MARGIN_Y),
        color=(0.11, 0.23, 0.28),
        width=0.7,
    )
    page.draw_rect(
        fitz.Rect(MARGIN_X, MARGIN_Y, PAGE_WIDTH - MARGIN_X, MARGIN_Y + HEADER_BOX_HEIGHT),
        color=(0.09, 0.23, 0.28),
        fill=(0.09, 0.23, 0.28),
    )

    page_title = doc_title if page_number == 1 else f"{doc_title} (cont.)"
    page.insert_text(
        (MARGIN_X + 14, MARGIN_Y + 24),
        page_title,
        fontname="helv",
        fontsize=18,
        color=(0.98, 0.99, 0.99),
    )
    page.insert_text(
        (MARGIN_X + 14, MARGIN_Y + 42),
        f"{display_name} | Generated {exported_at}",
        fontname="helv",
        fontsize=10,
        color=(0.92, 0.96, 0.96),
    )
    page.insert_text(
        (MARGIN_X + 14, PAGE_HEIGHT - 28),
        footer_text or "",
        fontname="helv",
        fontsize=8,
        color=(0.35, 0.43, 0.47),
    )


def _new_page(
    doc: fitz.Document,
    display_name: str,
    exported_at: str,
    page_number: int,
    doc_title: str = "Health Summary",
    footer_text: str = "",
) -> fitz.Page:
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_page_frame(page, display_name, exported_at, page_number, doc_title, footer_text)
    return page


def _ensure_space(
    doc: fitz.Document,
    page: fitz.Page,
    y: int,
    lines_needed: int,
    display_name: str,
    exported_at: str,
    page_number: int,
    doc_title: str = "Health Summary",
    footer_text: str = "",
) -> tuple[fitz.Page, int, int]:
    estimated_height = 16 + (max(lines_needed, 1) * LINE_HEIGHT) + SECTION_GAP
    if y + estimated_height <= CONTENT_END_Y:
        return page, y, page_number
    page_number += 1
    page = _new_page(doc, display_name, exported_at, page_number, doc_title, footer_text)
    return page, CONTENT_START_Y, page_number


def _allergy_lines(allergies: Iterable[Dict], max_lines: int = 8) -> List[str]:
    lines = []
    for allergy in allergies:
        name = _normalize_text(allergy.get("name", ""))
        if not name:
            continue
        parts = [name]
        reaction = _normalize_text(allergy.get("reaction", ""))
        severity = _normalize_text(allergy.get("severity", ""))
        confirmed = allergy.get("confirmed", True)
        if reaction:
            parts.append(reaction)
        if severity and severity != "unknown":
            parts.append(severity)
        if not confirmed:
            parts.append("suspected")
        lines.append(" - ".join(parts))
    return _clean_lines(lines, max_lines)


def _condition_lines(
    conditions: Iterable[Dict],
    role_key: str = "patient",
    include_statuses: Optional[set[str]] = None,
    max_lines: int = 8,
) -> List[str]:
    words = _role_words(role_key)
    lines = []
    for condition in sorted(
        conditions or [],
        key=lambda item: (
            item.get("status", "unknown") != "active",
            _entry_datetime(item, "recorded_on", "updated_at", "created_at"),
        ),
        reverse=False,
    ):
        name = _normalize_text(condition.get("name", ""))
        if not name:
            continue
        status = _normalize_text(condition.get("status", ""))
        if include_statuses is not None and (status or "unknown").lower() not in include_statuses:
            continue
        recorded_on = _normalize_text(condition.get("recorded_on", ""))
        if status == "active":
            prefix = words["active_condition"]
        elif status in {"past", "resolved"}:
            prefix = words["past_condition"]
        else:
            prefix = words["condition"]
        parts = [f"{prefix}: {name}"]
        if status and status != "unknown":
            parts.append(status)
        if recorded_on:
            parts.append(_date_label(recorded_on))
        if condition.get("notes"):
            parts.append(_normalize_text(condition.get("notes", "")))
        lines.append(" - ".join(parts))
    return _clean_lines(lines, max_lines)


def _vital_type_label(vital_type: str, role_key: str = "patient") -> str:
    clinical_labels = {
        "blood_pressure": "BP",
        "heart_rate": "HR",
        "weight": "Weight",
        "height": "Height",
        "bmi": "BMI",
        "blood_glucose": "Glucose",
        "temperature": "Temp",
        "oxygen_saturation": "SpO2",
        "respiratory_rate": "Resp Rate",
        "peak_flow": "Peak Flow",
        "hba1c": "HbA1c",
        "egfr": "eGFR",
        "creatinine": "Creatinine",
        "fasting_glucose": "Fasting glucose",
        "potassium": "Potassium",
        "whitebloodcells": "WBC",
        "white_blood_cells": "WBC",
        "haemoglobin": "Hb",
        "hemoglobin": "Hb",
        "cholesterol": "Cholesterol",
    }
    lay_labels = {
        "blood_pressure": "Blood pressure",
        "heart_rate": "Heart rate",
        "weight": "Weight",
        "height": "Height",
        "bmi": "BMI",
        "blood_glucose": "Blood glucose",
        "temperature": "Temperature",
        "oxygen_saturation": "Oxygen level",
        "respiratory_rate": "Breathing rate",
        "peak_flow": "Peak flow",
        "hba1c": "HbA1c",
        "egfr": "Kidney function (eGFR)",
        "creatinine": "Creatinine",
        "fasting_glucose": "Fasting glucose",
        "potassium": "Potassium",
        "whitebloodcells": "White blood cells",
        "white_blood_cells": "White blood cells",
        "haemoglobin": "Haemoglobin",
        "hemoglobin": "Haemoglobin",
        "cholesterol": "Cholesterol",
    }
    key = _normalize_text(vital_type).lower()
    labels = lay_labels if _is_lay_role(role_key) else clinical_labels
    return labels.get(key, key.replace("_", " ").title() if key else "Reading")


def _format_vital_value(entry: Dict) -> str:
    value = _normalize_text(entry.get("value", ""))
    unit = _normalize_text(entry.get("unit", ""))
    if not value:
        return ""
    return f"{value}{' ' + unit if unit else ''}"


def _latest_by_type(vitals: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for entry in vitals or []:
        vtype = _normalize_text(entry.get("type", "")).lower()
        value = _normalize_text(entry.get("value", ""))
        if not vtype or not value:
            continue
        grouped.setdefault(vtype, []).append(entry)
    for vtype, rows in grouped.items():
        rows.sort(
            key=lambda item: _entry_datetime(item, "recorded_on", "created_at"),
            reverse=True,
        )
    return grouped


def _vitals_lines(
    vitals: Iterable[Dict],
    role_key: str = "patient",
    max_lines: int = 8,
) -> List[str]:
    words = _role_words(role_key)
    lines = []
    grouped = _latest_by_type(vitals)
    for vtype, rows in sorted(
        grouped.items(),
        key=lambda item: _entry_datetime(item[1][0], "recorded_on", "created_at"),
        reverse=True,
    ):
        latest = rows[0]
        latest_value = _format_vital_value(latest)
        latest_date = _date_label(latest.get("recorded_on") or latest.get("created_at", ""))
        label = _vital_type_label(vtype, role_key)
        line = f"{words['reading']}: {label} {latest_value}"
        if latest_date:
            line += f" ({latest_date})"

        previous = next(
            (
                row for row in rows[1:]
                if _format_vital_value(row).lower() != latest_value.lower()
            ),
            rows[1] if len(rows) > 1 else None,
        )
        if previous:
            previous_value = _format_vital_value(previous)
            previous_date = _date_label(previous.get("recorded_on") or previous.get("created_at", ""))
            previous_note = f"{words['previous']}: {previous_value}"
            if previous_date:
                previous_note += f" on {previous_date}"
            line += f"; {previous_note}"

        lines.append(line)
    return _clean_lines(lines, max_lines)


def _recent_reading_count(vitals: Iterable[Dict]) -> int:
    return len(_latest_by_type(vitals))


def _current_snapshot_lines(
    role_key: str,
    conditions: Iterable[Dict],
    symptom_logs: List[Dict],
    medications: Iterable[Dict],
    allergies: Iterable[Dict],
    vitals: Iterable[Dict],
    triage_summaries: List[Dict],
    longitudinal_memory: str,
    max_lines: int = 8,
) -> List[str]:
    words = _role_words(role_key)
    lay_role = _is_lay_role(role_key)
    lines: List[str] = []

    sorted_triages = sorted(
        triage_summaries or [],
        key=lambda item: _entry_datetime(item, "created_at"),
        reverse=True,
    )
    latest_triage = sorted_triages[0] if sorted_triages else {}
    if latest_triage:
        urgency = _normalize_text(latest_triage.get("urgency_level", ""))
        next_step = _normalize_text(latest_triage.get("next_step", ""))
        triage_date = _date_label(latest_triage.get("created_at", ""))
        triage_bits = [bit for bit in [urgency, next_step] if bit]
        if triage_bits:
            line = f"{words['latest_triage']}: " + " - ".join(triage_bits)
            if triage_date:
                line += f" ({triage_date})"
            lines.append(line)

    active_conditions = _condition_lines(
        conditions,
        role_key=role_key,
        include_statuses={"active"},
        max_lines=3,
    )
    lines.extend(active_conditions)

    latest_symptoms = _symptom_lines(
        symptom_logs,
        longitudinal_memory,
        role_key=role_key,
        max_lines=2,
    )
    if not lay_role or len(lines) < 2:
        lines.extend(latest_symptoms[:2])

    meds = list(medications or [])
    if meds:
        label = "Medicine count" if lay_role else "Current medication count"
        lines.append(f"{label}: {len(meds)} recorded")

    severe_allergies = [
        _normalize_text(item.get("name", ""))
        for item in allergies or []
        if _normalize_text(item.get("severity", "")).lower() == "severe"
    ]
    if severe_allergies:
        label = "Severe allergy" if lay_role else "Severe allergy/contraindication"
        lines.append(f"{label}: {', '.join(_clean_lines(severe_allergies, 3))}")

    if lay_role:
        reading_count = _recent_reading_count(vitals)
        if reading_count:
            lines.append(f"Recent results available: {reading_count} measurement type{'s' if reading_count != 1 else ''} recorded")
    else:
        latest_reading_lines = _vitals_lines(vitals, role_key=role_key, max_lines=4)
        lines.extend(latest_reading_lines[:4])

    return _clean_lines(lines, max_lines=max_lines)


def _previous_history_lines(
    role_key: str,
    conditions: Iterable[Dict],
    longitudinal_memory: str,
    uploads: Iterable[Dict],
    max_lines: int = 8,
) -> List[str]:
    lay_role = _is_lay_role(role_key)
    lines: List[str] = []

    past_conditions = _condition_lines(
        conditions,
        role_key=role_key,
        include_statuses={"past", "resolved", "unknown"},
        max_lines=4,
    )
    lines.extend(past_conditions)

    memory_lines = _memory_snapshot_lines(longitudinal_memory, max_lines=5)
    if memory_lines:
        label = "Earlier history" if lay_role else "Longitudinal history"
        lines.extend(f"{label}: {line}" for line in memory_lines)

    return _clean_lines(lines, max_lines=max_lines)


_SECTION_EMPTY_FALLBACKS: Dict[str, str] = {
    "current_snapshot": "No current saved health data is available yet.",
    "previous_history": "No previous health history has been saved yet.",
    "memory_snapshot": "No previous health history has been saved yet.",
    "active_concerns": "No recent symptoms or active concerns recorded yet.",
    "medications":     "No medications have been recorded.",
    "allergies":       "No allergies recorded.",
    "vitals":          "No measurements or lab results recorded.",
    "consultation":    "No consultation has been recorded yet.",
    "investigations":  "No investigations or follow-up plan noted.",
    "uploads":         "No uploaded records saved yet.",
}


def build_summary_pdf(
    user_profile: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    uploads: List[Dict],
    longitudinal_memory: str,
    role_key: str = "patient",
    triage_summaries: Optional[List[Dict]] = None,
    recent_chats: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
) -> bytes:
    canonical_role = _canonical_role_key(role_key)
    config = _get_summary_config(canonical_role)
    doc_title = config["title"]
    footer_text = config["footer"]

    doc = fitz.open()
    display_name = user_profile.get("display_name") or "Patient"
    exported_at = datetime.now(timezone.utc).strftime("%d %b %Y")
    memory_snapshot_lines = _clean_lines(
        _condition_lines(conditions or [], role_key=canonical_role)
        + _memory_snapshot_lines(longitudinal_memory),
        max_lines=8,
    )

    # Build the data for each section key once
    section_data: Dict[str, List[str]] = {
        "current_snapshot": _wrap_lines(
            _current_snapshot_lines(
                canonical_role,
                conditions or [],
                symptom_logs,
                medications,
                allergies or [],
                vitals or [],
                triage_summaries or [],
                longitudinal_memory,
            )
        ),
        "previous_history": _wrap_lines(
            _previous_history_lines(
                canonical_role,
                conditions or [],
                longitudinal_memory,
                uploads,
            )
        ),
        "memory_snapshot": _wrap_lines(
            memory_snapshot_lines
        ),
        "active_concerns": _wrap_lines(
            _symptom_lines(symptom_logs, longitudinal_memory, role_key=canonical_role)
        ),
        "medications": _wrap_lines(
            _medication_lines(medications, longitudinal_memory, role_key=canonical_role)
        ),
        "allergies": _wrap_lines(
            _allergy_lines(allergies or [])
        ),
        "vitals": _wrap_lines(
            _vitals_lines(vitals or [], role_key=canonical_role)
        ),
        "consultation": _wrap_lines(
            _latest_consultation_lines(
                recent_chats or [],
                triage_summaries or [],
                role_key=canonical_role,
            )
        ),
        "investigations": _wrap_lines(
            _additional_context_lines(longitudinal_memory)
        ),
        "uploads": _wrap_lines(
            _upload_lines(uploads)
        ),
    }

    # Render sections in the role-specific order with role-specific headings
    sections = [
        (
            heading,
            section_data[key] or [_SECTION_EMPTY_FALLBACKS.get(key, "No information available.")],
        )
        for heading, key in config["sections"]
    ]

    page_number = 1
    page = _new_page(doc, display_name, exported_at, page_number, doc_title, footer_text)
    y = CONTENT_START_Y

    for heading, lines in sections:
        visible_lines = lines or ["No information available."]
        page, y, page_number = _ensure_space(
            doc, page, y, len(visible_lines) + 1,
            display_name, exported_at, page_number, doc_title, footer_text,
        )
        page.insert_text(
            (MARGIN_X + 14, y),
            heading,
            fontname="helv",
            fontsize=11,
            color=(0.09, 0.23, 0.28),
        )
        y += 16
        for line in visible_lines:
            page, y, page_number = _ensure_space(
                doc, page, y, 1,
                display_name, exported_at, page_number, doc_title, footer_text,
            )
            page.insert_text(
                (MARGIN_X + 18, y),
                line,
                fontname="helv",
                fontsize=BODY_FONT_SIZE,
                color=(0.07, 0.14, 0.18),
            )
            y += LINE_HEIGHT
        y += SECTION_GAP

    return doc.tobytes()


# Keep old name as a shim so any other callers don't break
def build_gp_summary_pdf(
    user_profile: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    uploads: List[Dict],
    longitudinal_memory: str,
    triage_summaries: Optional[List[Dict]] = None,
    recent_chats: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    role_key: str = "doctor",
) -> bytes:
    return build_summary_pdf(
        user_profile=user_profile,
        symptom_logs=symptom_logs,
        medications=medications,
        uploads=uploads,
        longitudinal_memory=longitudinal_memory,
        role_key=role_key,
        triage_summaries=triage_summaries,
        recent_chats=recent_chats,
        allergies=allergies,
        conditions=conditions,
        vitals=vitals,
    )
