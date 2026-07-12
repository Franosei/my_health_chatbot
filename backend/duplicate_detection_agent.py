"""
Duplicate detection agent -- catches near-duplicate document uploads that
exact content hashing can't: the same clinical letter re-exported, rescanned,
or re-downloaded with different bytes but the same substantive content.

Byte-identical duplicates are already caught for free by a content hash
check before this agent ever runs (see backend/api.py). This agent only
handles the harder, genuinely judgment-requiring case of semantic sameness --
telling "the same GP letter, re-exported" apart from "two different blood
test results that happen to look structurally similar" is not something a
hash or a simple text-similarity score can do reliably.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional


_MAX_INPUT_CHARS = 4000
_MAX_CANDIDATES = 20


class DuplicateDetectionAgent:
    """Decides whether a newly uploaded document is substantively the same as one already on file."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def check(
        self,
        new_text: str,
        new_filename: str,
        existing_summaries: List[Dict],
    ) -> Optional[Dict]:
        """
        existing_summaries: [{"file": ..., "summary": ...}, ...] for this user.

        Returns None when this is not a duplicate, or on any error -- fails
        open so a model hiccup never blocks a genuine new upload. Otherwise:
        {"matches_file": str, "confidence": "high", "reason": str}
        """
        if not existing_summaries or not (new_text or "").strip():
            return None

        candidates = "\n\n".join(
            f"- File: {item.get('file', 'unknown')}\n  Summary: {(item.get('summary') or '')[:600]}"
            for item in existing_summaries[:_MAX_CANDIDATES]
        )

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.AUX_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You detect duplicate document uploads in a patient's health record. "
                            "A duplicate means the SAME underlying document (same encounter, same "
                            "letter, same report, same date, same specific content) uploaded again "
                            "under a different file -- a rescan, re-export, or re-download of the "
                            "exact same document -- not merely a similar type of document. Two "
                            "different blood test results from different dates are NOT duplicates, "
                            "even though they look alike structurally.\n\n"
                            "This is a strict, 100%-certainty check: only flag as duplicate when the "
                            "substantive content is effectively identical to one of the previous "
                            "documents -- same facts, same values, same dates, same author. If there "
                            "is any real chance it is a different document, a follow-up letter, an "
                            "updated version, or covers a different date/encounter, it is NOT a "
                            "duplicate. When unsure, say it is not a duplicate -- rejecting a "
                            "genuinely new document is worse than missing an exact duplicate.\n"
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Newly uploaded file: {new_filename}\n"
                            f"New document text (excerpt):\n{new_text[:_MAX_INPUT_CHARS]}\n\n"
                            f"Previously uploaded documents on this account:\n{candidates}\n\n"
                            "Return JSON with exactly these keys:\n"
                            "{\n"
                            '  "is_duplicate": boolean,\n'
                            '  "matches_file": string,\n'
                            '  "confidence": "high" | "medium" | "low",\n'
                            '  "reason": string\n'
                            "}"
                        ),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=300,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
        except Exception as exc:
            print(f"[DuplicateDetectionAgent] check failed, treating as not-duplicate: {exc}")
            return None

        if not parsed.get("is_duplicate") or parsed.get("confidence") != "high":
            return None

        matches_file = str(parsed.get("matches_file") or "").strip()
        if not matches_file:
            return None

        return {
            "matches_file": matches_file,
            "confidence": "high",
            "reason": str(parsed.get("reason") or "").strip(),
        }
