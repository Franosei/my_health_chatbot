from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime
from statistics import mean
from typing import Dict, Iterable, List


def _parse_date(value: str) -> date | None:
    cleaned = (value or "").strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned)
    except ValueError:
        return None


def _split_triggers(raw: str) -> List[str]:
    tokens = []
    for chunk in (raw or "").replace(";", ",").split(","):
        cleaned = chunk.strip()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _trend_label(severities: List[int]) -> str:
    if len(severities) < 2:
        return "stable"
    if severities[-1] >= severities[0] + 2:
        return "worsening"
    if severities[-1] <= severities[0] - 2:
        return "improving"
    return "stable"


def build_symptom_pattern_summary(symptom_logs: Iterable[Dict], max_items: int = 4) -> str:
    logs = [entry for entry in symptom_logs if (entry.get("symptom") or "").strip()]
    if not logs:
        return ""

    grouped: dict[str, list[Dict]] = defaultdict(list)
    for entry in logs:
        grouped[entry["symptom"].strip().lower()].append(entry)

    ranked_groups = sorted(
        grouped.values(),
        key=lambda items: (
            len(items),
            max((_parse_date(item.get("logged_for", "")) or date.min) for item in items),
        ),
        reverse=True,
    )

    lines = ["Tracked symptom patterns:"]
    for entries in ranked_groups[:max_items]:
        symptom_name = entries[0].get("symptom", "Symptom").strip()
        sorted_entries = sorted(
            entries,
            key=lambda item: (
                _parse_date(item.get("logged_for", "")) or date.min,
                item.get("created_at", ""),
            ),
        )
        dates = [_parse_date(item.get("logged_for", "")) for item in sorted_entries]
        valid_dates = [item for item in dates if item]
        severities = [int(item.get("severity", 0) or 0) for item in sorted_entries]
        severity_values = [value for value in severities if value > 0]
        triggers = Counter(
            trigger.lower()
            for item in sorted_entries
            for trigger in _split_triggers(item.get("triggers", ""))
        )

        parts = [f"{symptom_name} logged {len(sorted_entries)} time(s)"]
        if valid_dates:
            start_date = min(valid_dates).isoformat()
            end_date = max(valid_dates).isoformat()
            parts.append(f"between {start_date} and {end_date}")
        if severity_values:
            parts.append(
                f"severity average {round(mean(severity_values), 1)}/10"
                f" (range {min(severity_values)}-{max(severity_values)})"
            )
            parts.append(f"trend { _trend_label(severity_values) }")
        common_triggers = [name for name, _ in triggers.most_common(2)]
        if common_triggers:
            parts.append("common triggers: " + ", ".join(common_triggers))

        latest_note = (sorted_entries[-1].get("notes") or "").strip()
        if latest_note:
            parts.append(f"latest note: {latest_note[:100]}")

        lines.append("- " + "; ".join(parts))

    return "\n".join(lines)


def build_recent_symptom_lines(symptom_logs: Iterable[Dict], limit: int = 6) -> List[str]:
    logs = [entry for entry in symptom_logs if (entry.get("symptom") or "").strip()]
    if not logs:
        return []

    sorted_logs = sorted(
        logs,
        key=lambda item: (
            _parse_date(item.get("logged_for", "")) or date.min,
            item.get("created_at", ""),
        ),
        reverse=True,
    )
    lines = []
    for entry in sorted_logs[:limit]:
        pieces = [entry.get("logged_for", "Unknown date"), entry.get("symptom", "Symptom")]
        severity = int(entry.get("severity", 0) or 0)
        if severity:
            pieces.append(f"{severity}/10")
        triggers = (entry.get("triggers") or "").strip()
        if triggers:
            pieces.append(f"triggered by {triggers}")
        note = (entry.get("notes") or "").strip()
        if note:
            pieces.append(note[:90])
        lines.append(" - ".join(piece for piece in pieces if piece))
    return lines
