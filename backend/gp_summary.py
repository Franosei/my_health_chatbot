from __future__ import annotations

from datetime import datetime, timezone
from textwrap import wrap
from typing import Dict, Iterable, List

import fitz

from backend.symptom_tracker import build_recent_symptom_lines


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN_X = 44
MARGIN_Y = 48
TEXT_WIDTH = PAGE_WIDTH - (MARGIN_X * 2)
BODY_FONT_SIZE = 10
LINE_HEIGHT = 13


def _clean_lines(lines: Iterable[str], max_lines: int) -> List[str]:
    cleaned = []
    for line in lines:
        text = " ".join((line or "").split()).strip()
        if text:
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


def _memory_lines(longitudinal_memory: str, max_lines: int = 8) -> List[str]:
    candidates = []
    for raw_line in (longitudinal_memory or "").splitlines():
        cleaned = raw_line.strip()
        if not cleaned:
            continue
        if cleaned.endswith(":"):
            continue
        if cleaned.lower() == "none noted":
            continue
        candidates.append(cleaned)
    return _clean_lines(candidates, max_lines)


def _medication_lines(medications: Iterable[Dict], max_lines: int = 6) -> List[str]:
    lines = []
    for medication in medications:
        name = (medication.get("name") or "").strip()
        if not name:
            continue
        parts = [name]
        if medication.get("dose"):
            parts.append(medication["dose"])
        if medication.get("schedule"):
            parts.append(medication["schedule"])
        if medication.get("reason"):
            parts.append(f"for {medication['reason']}")
        lines.append(" - ".join(parts))
    return _clean_lines(lines, max_lines)


def _upload_lines(uploads: Iterable[Dict], max_lines: int = 5) -> List[str]:
    return _clean_lines(
        [item.get("file", "Uploaded document") for item in uploads],
        max_lines=max_lines,
    )


def build_gp_summary_pdf(
    user_profile: Dict,
    symptom_logs: List[Dict],
    medications: List[Dict],
    uploads: List[Dict],
    longitudinal_memory: str,
    latest_triage: Dict,
) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    page.draw_rect(
        fitz.Rect(MARGIN_X, MARGIN_Y, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - MARGIN_Y),
        color=(0.11, 0.23, 0.28),
        width=0.7,
    )
    page.draw_rect(
        fitz.Rect(MARGIN_X, MARGIN_Y, PAGE_WIDTH - MARGIN_X, MARGIN_Y + 56),
        color=(0.09, 0.23, 0.28),
        fill=(0.09, 0.23, 0.28),
    )

    display_name = user_profile.get("display_name") or "Patient"
    exported_at = datetime.now(timezone.utc).strftime("%d %b %Y")
    page.insert_text(
        (MARGIN_X + 14, MARGIN_Y + 24),
        "GP Summary",
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

    sections = [
        (
            "Symptoms",
            _wrap_lines(
                _clean_lines(
                    build_recent_symptom_lines(symptom_logs, limit=5)
                    or ["No symptom tracker entries saved yet."],
                    max_lines=5,
                )
            ),
        ),
        (
            "Medications",
            _wrap_lines(_medication_lines(medications) or ["No medication list saved yet."]),
        ),
        (
            "Uploaded Documents",
            _wrap_lines(_upload_lines(uploads) or ["No uploaded records saved yet."]),
        ),
        (
            "AI Summary",
            _wrap_lines(
                _memory_lines(longitudinal_memory)
                or ["No longitudinal AI summary is available yet."]
            ),
        ),
        (
            "Latest Triage",
            _wrap_lines(
                _clean_lines(
                    [
                        f"Urgency: {latest_triage.get('urgency_level', 'Not available')}",
                        f"Suggested next step: {latest_triage.get('next_step', 'Not available')}",
                        "Monitor: " + ", ".join(latest_triage.get("what_to_monitor", [])[:3])
                        if latest_triage.get("what_to_monitor")
                        else "Monitor: Not available",
                        latest_triage.get("rationale", ""),
                    ],
                    max_lines=4,
                )
                or ["No triage summary has been saved yet."]
            ),
        ),
    ]

    y = MARGIN_Y + 80
    for heading, lines in sections:
        if y > PAGE_HEIGHT - 84:
            break
        page.insert_text((MARGIN_X + 14, y), heading, fontname="helv", fontsize=11, color=(0.09, 0.23, 0.28))
        y += 16
        for line in lines:
            if y > PAGE_HEIGHT - 68:
                break
            page.insert_text(
                (MARGIN_X + 18, y),
                line,
                fontname="helv",
                fontsize=BODY_FONT_SIZE,
                color=(0.07, 0.14, 0.18),
            )
            y += LINE_HEIGHT
        y += 10

    page.insert_text(
        (MARGIN_X + 14, PAGE_HEIGHT - 28),
        "Prepared for sharing with a GP. Review for accuracy before use.",
        fontname="helv",
        fontsize=8,
        color=(0.35, 0.43, 0.47),
    )

    return doc.tobytes()
