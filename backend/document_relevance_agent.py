"""
Document relevance agent -- rejects uploads that are not legitimate personal
health/clinical documents, the same way ImageAnalysisAgent rejects
non-medical images. Runs once per new upload, on the extracted PDF text,
before any anonymization, summarization, or extraction work is spent on it.
"""
from __future__ import annotations

import json
from typing import Dict


_MAX_INPUT_CHARS = 4000


class DocumentRelevanceAgent:
    """Screens uploaded documents for health/clinical relevance via LLM."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def check(self, text: str, filename: str) -> Dict:
        """
        Returns {"is_relevant": bool, "document_type": str, "reason": str}.
        Fails open (is_relevant=True) on any error or empty text -- an agent
        hiccup or an unreadable scan must never block a legitimate upload.
        """
        if not text or not text.strip():
            return {"is_relevant": True, "document_type": "unknown", "reason": ""}

        sample = text[:_MAX_INPUT_CHARS]

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.AUX_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You screen document uploads for a personal health record app.\n\n"
                            "Accept anything that is plausibly a personal health, medical, or "
                            "clinical document: GP/hospital letters, test/lab results, prescriptions, "
                            "discharge summaries, referral letters, vaccination records, imaging "
                            "reports, therapy/physio notes, insurance or appointment correspondence "
                            "about the person's own care, or personal notes about symptoms/health.\n\n"
                            "Reject only documents that are clearly NOT health-related and have "
                            "nothing to do with the person's care -- for example: invoices/receipts "
                            "for unrelated purchases, marketing/spam, unrelated contracts, recipes, "
                            "unrelated news articles or ebooks, or a document about an entirely "
                            "different, unrelated person or business.\n\n"
                            "When genuinely unsure, ACCEPT -- a missed rejection is far less harmful "
                            "than blocking someone's real medical document.\n"
                            "Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Filename: {filename}\n"
                            f"Document text (excerpt):\n{sample}\n\n"
                            "Return JSON with exactly these keys:\n"
                            "{\n"
                            '  "is_relevant": boolean,\n'
                            '  "document_type": string,\n'
                            '  "reason": string\n'
                            "}"
                        ),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=250,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
        except Exception as exc:
            print(f"[DocumentRelevanceAgent] check failed, accepting by default: {exc}")
            return {"is_relevant": True, "document_type": "unknown", "reason": ""}

        return {
            "is_relevant": bool(parsed.get("is_relevant", True)),
            "document_type": str(parsed.get("document_type") or "").strip(),
            "reason": str(parsed.get("reason") or "").strip(),
        }
