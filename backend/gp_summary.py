from __future__ import annotations

from datetime import datetime, timezone
import re
from textwrap import wrap
from typing import Dict, Iterable, List

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

_MEDICATION_CONTEXT_PATTERN = re.compile(
    r"\b(on|taking|takes|take|prescribed|started|using|uses|medication|medications|drug|drugs|tablet|tablets|capsule|capsules)\b",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


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
    return "Recorded: " + " - ".join(parts)


def _infer_medication_lines(
    medications: Iterable[Dict],
    longitudinal_memory: str,
    max_lines: int = 6,
) -> List[str]:
    recorded_names = {
        _normalize_text(item.get("name", "")).lower()
        for item in medications
        if _normalize_text(item.get("name", ""))
    }
    sections = _parse_memory_sections(longitudinal_memory)
    candidates = (
        sections.get("current treatments and medicines", [])
        + sections.get("patient summary", [])
        + sections.get("conditions and history", [])
        + sections.get("unstructured", [])
    )

    inferred = []
    for line in _clean_lines(candidates, max_lines * 2):
        lower = line.lower()
        if not _MEDICATION_CONTEXT_PATTERN.search(lower):
            continue
        if recorded_names and any(name in lower for name in recorded_names):
            continue
        inferred.append("Mentioned in summary: " + line)
        if len(inferred) >= max_lines:
            break
    return inferred


def _medication_lines(
    medications: Iterable[Dict],
    longitudinal_memory: str,
    max_lines: int = 8,
) -> List[str]:
    recorded = _clean_lines(
        [_build_recorded_medication_line(medication) for medication in medications],
        max_lines=max_lines,
    )
    inferred = _infer_medication_lines(medications, longitudinal_memory, max_lines=max_lines)
    combined = _clean_lines([*recorded, *inferred], max_lines=max_lines)
    return combined


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


def _triage_lines(latest_triage: Dict, max_lines: int = 7) -> List[str]:
    candidates = [
        f"Pathway: {latest_triage.get('pathway_label', '')}" if latest_triage.get("pathway_label") else "",
        f"Clinical summary: {latest_triage.get('decision_summary', '')}" if latest_triage.get("decision_summary") else "",
        f"Urgency: {latest_triage.get('urgency_level', 'Not available')}",
        f"Suggested next step: {latest_triage.get('next_step', 'Not available')}",
        "Immediate actions: " + "; ".join(latest_triage.get("immediate_actions", [])[:2])
        if latest_triage.get("immediate_actions")
        else "",
        "Monitor: " + ", ".join(latest_triage.get("what_to_monitor", [])[:3])
        if latest_triage.get("what_to_monitor")
        else "",
        latest_triage.get("rationale", ""),
    ]
    return _clean_lines(candidates, max_lines=max_lines)


def _additional_context_lines(longitudinal_memory: str, max_lines: int = 6) -> List[str]:
    sections = _parse_memory_sections(longitudinal_memory)
    return _section_lines(
        sections,
        ["investigations or notable results", "care plan and follow-up"],
        max_lines=max_lines,
    )


def _draw_page_frame(page: fitz.Page, display_name: str, exported_at: str, page_number: int) -> None:
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

    page_title = "GP Summary" if page_number == 1 else "GP Summary (cont.)"
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
        "Prepared for sharing with a GP. Medication lines combine the saved medication list and medicine mentions found in the clinical summary.",
        fontname="helv",
        fontsize=8,
        color=(0.35, 0.43, 0.47),
    )


def _new_page(doc: fitz.Document, display_name: str, exported_at: str, page_number: int) -> fitz.Page:
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    _draw_page_frame(page, display_name, exported_at, page_number)
    return page


def _ensure_space(
    doc: fitz.Document,
    page: fitz.Page,
    y: int,
    lines_needed: int,
    display_name: str,
    exported_at: str,
    page_number: int,
) -> tuple[fitz.Page, int, int]:
    estimated_height = 16 + (max(lines_needed, 1) * LINE_HEIGHT) + SECTION_GAP
    if y + estimated_height <= CONTENT_END_Y:
        return page, y, page_number
    page_number += 1
    page = _new_page(doc, display_name, exported_at, page_number)
    return page, CONTENT_START_Y, page_number


def build_gp_summary_pdf(
    user_profile: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    uploads: List[Dict],
    longitudinal_memory: str,
    latest_triage: Dict,
) -> bytes:
    doc = fitz.open()

    display_name = user_profile.get("display_name") or "Patient"
    exported_at = datetime.now(timezone.utc).strftime("%d %b %Y")

    sections = [
        (
            "Clinical Snapshot",
            _wrap_lines(
                _memory_snapshot_lines(longitudinal_memory)
                or ["No high-level clinical snapshot is available yet."]
            ),
        ),
        (
            "Active Concerns",
            _wrap_lines(
                _symptom_lines(symptom_logs, longitudinal_memory)
                or ["No recent symptoms or active concerns recorded yet."]
            ),
        ),
        (
            "Medications",
            _wrap_lines(
                _medication_lines(medications, longitudinal_memory)
                or ["No medication details were found in the saved medication list or current summary."]
            ),
        ),
        (
            "Latest Triage And Safety Netting",
            _wrap_lines(
                _triage_lines(latest_triage)
                or ["No triage summary has been saved yet."]
            ),
        ),
        (
            "Investigations And Follow-Up",
            _wrap_lines(
                _additional_context_lines(longitudinal_memory)
                or ["No investigations or follow-up plan noted."]
            ),
        ),
        (
            "Supporting Records",
            _wrap_lines(
                _upload_lines(uploads)
                or ["No uploaded records saved yet."]
            ),
        ),
    ]

    page_number = 1
    page = _new_page(doc, display_name, exported_at, page_number)
    y = CONTENT_START_Y

    for heading, lines in sections:
        visible_lines = lines or ["No information available."]
        page, y, page_number = _ensure_space(
            doc,
            page,
            y,
            len(visible_lines) + 1,
            display_name,
            exported_at,
            page_number,
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
                doc,
                page,
                y,
                1,
                display_name,
                exported_at,
                page_number,
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
