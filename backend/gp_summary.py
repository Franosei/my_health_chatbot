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
    return _clean_lines(candidates, max_lines)


def _memory_snapshot_lines(longitudinal_memory: str, max_lines: int = 5) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    snapshot = _section_lines(
        sections,
        ["patient summary", "conditions and history"],
        max_lines=max_lines,
    )
    if snapshot:
        return snapshot
    return _clean_lines(sections.get("unstructured", []), max_lines)


def _memory_active_concern_lines(longitudinal_memory: str, max_lines: int = 6) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    concern_lines = _section_lines(
        sections,
        ["recent symptoms or active concerns", "open questions or uncertainties"],
        max_lines=max_lines,
    )
    if concern_lines:
        return concern_lines
    return _clean_lines(sections.get("unstructured", []), max_lines)


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
    max_lines: int = 8,
) -> List[str]:
    return _clean_lines(
        [_build_recorded_medication_line(med) for med in medications],
        max_lines=max_lines,
    )


def _upload_lines(uploads: Iterable[Dict], max_lines: int = 5) -> List[str]:
    return _clean_lines(
        [item.get("file", "Uploaded document") for item in uploads],
        max_lines=max_lines,
    )


def _symptom_lines(symptom_logs: List[Dict], longitudinal_memory: str, max_lines: int = 6) -> List[str]:
    tracked = _clean_lines(build_recent_symptom_lines(symptom_logs, limit=max_lines), max_lines=max_lines)
    if tracked:
        return tracked
    return _memory_active_concern_lines(longitudinal_memory, max_lines=max_lines)


def _parse_ts(timestamp_str: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat((timestamp_str or "").replace("Z", "+00:00"))
    except Exception:
        return None


def _latest_consultation_lines(
    chat_history: List[Dict],
    triage_summaries: List[Dict],
    max_complaints: int = 5,
) -> List[str]:
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

    # Collect user messages from the same calendar date as the consultation
    complaints: List[str] = []
    if consultation_date:
        for msg in (chat_history or []):
            if (msg.get("role") or msg.get("type")) != "user":
                continue
            ts = _parse_ts(msg.get("timestamp", ""))
            if not ts or ts.date() != consultation_date:
                continue
            content = _normalize_text(msg.get("content", ""))
            if content and not _is_empty_placeholder(content):
                complaints.append(content)

    date_label = consultation_date.strftime("%d %b %Y") if consultation_date else "Date unknown"
    lines: List[str] = [f"Date: {date_label}"]

    for i, complaint in enumerate(complaints[:max_complaints]):
        truncated = complaint[:200] + "..." if len(complaint) > 200 else complaint
        prefix = "Presenting Complaint:" if i == 0 else "Also Reported:"
        lines.append(f"{prefix} {truncated}")

    if latest_triage:
        if latest_triage.get("pathway_label"):
            lines.append(f"Pathway: {latest_triage['pathway_label']}")
        if latest_triage.get("decision_summary"):
            summary = latest_triage["decision_summary"]
            lines.append(f"Assessment: {summary[:250] + '...' if len(summary) > 250 else summary}")
        if latest_triage.get("urgency_level"):
            lines.append(f"Urgency: {latest_triage['urgency_level']}")
        if latest_triage.get("next_step"):
            lines.append(f"Plan: {latest_triage['next_step']}")
        monitor = latest_triage.get("what_to_monitor", [])[:3]
        if monitor:
            lines.append("Monitor: " + "; ".join(monitor))
        actions = latest_triage.get("immediate_actions", [])[:2]
        if actions:
            lines.append("Immediate Actions: " + "; ".join(actions))
        escalations = latest_triage.get("escalation_triggers", [])[:2]
        if escalations:
            lines.append("Escalation Triggers: " + "; ".join(escalations))

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
            ("My Health History",      "memory_snapshot"),
            ("Current Concerns",       "active_concerns"),
            ("My Medications",         "medications"),
            ("Allergies and Reactions","allergies"),
            ("Recent Readings",        "vitals"),
            ("My Uploaded Records",    "uploads"),
        ],
    },
    "caregiver": {
        "title": "Personal Health Summary",
        "footer": "Personal health record for the person in your care. Share with their healthcare team as needed.",
        "sections": [
            ("Health History",         "memory_snapshot"),
            ("Current Concerns",       "active_concerns"),
            ("Medications",            "medications"),
            ("Allergies and Reactions","allergies"),
            ("Recent Readings",        "vitals"),
            ("Uploaded Records",       "uploads"),
        ],
    },
    "doctor": {
        "title": "GP Summary",
        "footer": "Prepared for GP or clinical handover. Medications shown are only those explicitly recorded by the patient.",
        "sections": [
            ("Previous Visit Summary",          "memory_snapshot"),
            ("Active Concerns",                 "active_concerns"),
            ("Medications",                     "medications"),
            ("Allergies and Contraindications", "allergies"),
            ("Recorded Vitals",                 "vitals"),
            ("Latest Consultation Note",        "consultation"),
            ("Investigations and Follow-Up",    "investigations"),
            ("Supporting Records",              "uploads"),
        ],
    },
    "nurse": {
        "title": "Nursing Handover Note",
        "footer": "Nursing documentation for handover. Verify all medications and observations at handover.",
        "sections": [
            ("Patient Overview",              "memory_snapshot"),
            ("Active Concerns and Symptoms",  "active_concerns"),
            ("Current Medications",           "medications"),
            ("Allergies",                     "allergies"),
            ("Observations",                  "vitals"),
            ("Consultation Summary",          "consultation"),
            ("Care Plan and Follow-Up",       "investigations"),
            ("Supporting Records",            "uploads"),
        ],
    },
    "midwife": {
        "title": "Maternity Care Summary",
        "footer": "Prepared for midwifery handover or antenatal review.",
        "sections": [
            ("Patient Overview",                      "memory_snapshot"),
            ("Active Concerns",                       "active_concerns"),
            ("Medications and Supplements",           "medications"),
            ("Allergies and Contraindications",       "allergies"),
            ("Antenatal Observations",                "vitals"),
            ("Latest Consultation Note",              "consultation"),
            ("Care Plan and Follow-Up",               "investigations"),
            ("Supporting Records",                    "uploads"),
        ],
    },
    "physiotherapist": {
        "title": "Physiotherapy Assessment Summary",
        "footer": "Prepared for physiotherapy handover or inter-professional communication.",
        "sections": [
            ("Patient Overview",              "memory_snapshot"),
            ("Presenting Complaints",         "active_concerns"),
            ("Current Medications",           "medications"),
            ("Allergies",                     "allergies"),
            ("Functional Measures",           "vitals"),
            ("Latest Assessment Note",        "consultation"),
            ("Treatment Plan and Goals",      "investigations"),
            ("Supporting Records",            "uploads"),
        ],
    },
}

_DEFAULT_SUMMARY_CONFIG = _ROLE_SUMMARY_CONFIGS["patient"]


def _get_summary_config(role_key: str) -> Dict:
    return _ROLE_SUMMARY_CONFIGS.get((role_key or "").strip().lower(), _DEFAULT_SUMMARY_CONFIG)


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


def _vitals_lines(vitals: Iterable[Dict], max_lines: int = 8) -> List[str]:
    _TYPE_LABELS = {
        "blood_pressure": "BP",
        "heart_rate": "HR",
        "weight": "Weight",
        "blood_glucose": "Glucose",
        "temperature": "Temp",
        "oxygen_saturation": "SpO2",
        "peak_flow": "Peak Flow",
        "hba1c": "HbA1c",
        "egfr": "eGFR",
        "creatinine": "Creatinine",
    }
    lines = []
    for entry in vitals:
        vtype = _normalize_text(entry.get("type", ""))
        value = _normalize_text(entry.get("value", ""))
        unit = _normalize_text(entry.get("unit", ""))
        recorded_on = _normalize_text(entry.get("recorded_on", ""))
        if not vtype or not value:
            continue
        label = _TYPE_LABELS.get(vtype, vtype)
        line = f"{label}: {value}{' ' + unit if unit else ''}"
        if recorded_on:
            line += f" ({recorded_on})"
        lines.append(line)
    return _clean_lines(lines, max_lines)


_SECTION_EMPTY_FALLBACKS: Dict[str, str] = {
    "memory_snapshot": "No previous visit summary available yet.",
    "active_concerns": "No recent symptoms or active concerns recorded yet.",
    "medications":     "No medications have been recorded.",
    "allergies":       "No allergies recorded.",
    "vitals":          "No readings recorded.",
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
    vitals: Optional[List[Dict]] = None,
) -> bytes:
    config = _get_summary_config(role_key)
    doc_title = config["title"]
    footer_text = config["footer"]

    doc = fitz.open()
    display_name = user_profile.get("display_name") or "Patient"
    exported_at = datetime.now(timezone.utc).strftime("%d %b %Y")

    # Build the data for each section key once
    section_data: Dict[str, List[str]] = {
        "memory_snapshot": _wrap_lines(
            _memory_snapshot_lines(longitudinal_memory)
        ),
        "active_concerns": _wrap_lines(
            _symptom_lines(symptom_logs, longitudinal_memory)
        ),
        "medications": _wrap_lines(
            _medication_lines(medications)
        ),
        "allergies": _wrap_lines(
            _allergy_lines(allergies or [])
        ),
        "vitals": _wrap_lines(
            _vitals_lines(vitals or [])
        ),
        "consultation": _wrap_lines(
            _latest_consultation_lines(recent_chats or [], triage_summaries or [])
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
        vitals=vitals,
    )
