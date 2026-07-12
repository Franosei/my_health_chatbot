"""
Anonymization agent -- LLM-based PII entity detection for uploaded documents.

Regex catches structured PII (dates, phone numbers, emails, addresses, ID
numbers) cheaply and deterministically -- see backend/anonymizer.py. It
cannot reliably catch freeform person/location/organization names. That was
spaCy NER's job, but spaCy currently fails to import under Python 3.14 (see
requirements.txt), so unlabeled names slip through regex entirely.

This agent identifies PII entities as structured JSON and never rewrites the
document itself -- the caller does the actual find-and-replace. Keeping
identification and redaction separate means a model mistake can at worst miss
an entity, not silently alter clinical content elsewhere in the document.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List


_MAX_INPUT_CHARS = 8000


class AnonymizationAgent:
    """Finds person/location/organization names in document text via LLM."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def find_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Returns {"person_names": [...], "locations": [...], "organizations": [...]}
        as exact substrings found in `text`, so they can be located with a
        literal search. Fails open (empty lists) on any error -- a model
        hiccup must never block a document upload.
        """
        if not text or not text.strip():
            return {"person_names": [], "locations": [], "organizations": []}

        sample = text[:_MAX_INPUT_CHARS]

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.AUX_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a PII-detection assistant for a clinical document redaction "
                            "pipeline. You do not summarize, diagnose, or comment on the document. "
                            "Your only job is to find personally identifying names in the text.\n\n"
                            "Find every occurrence of:\n"
                            "- person_names: full or partial names of any human being referenced "
                            "(patients, clinicians, family members, witnesses) -- not job titles or "
                            "generic roles alone.\n"
                            "- locations: specific places that could identify where someone lives or "
                            "was treated (street addresses, named towns/cities, named hospitals, "
                            "named GP surgeries, postcodes).\n"
                            "- organizations: named companies, employers, insurers, or institutions.\n\n"
                            "Rules:\n"
                            "- Return each match EXACTLY as it appears in the source text (same "
                            "spelling, capitalization, spacing) so it can be found with a literal "
                            "text search. Do not paraphrase or normalize.\n"
                            "- Do not invent entities that are not in the text.\n"
                            "- Do not include generic medical terms, drug names, or condition names.\n"
                            "- Return only valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Document text:\n{sample}\n\n"
                            "Return JSON with exactly these keys:\n"
                            "{\n"
                            '  "person_names": string[],\n'
                            '  "locations": string[],\n'
                            '  "organizations": string[]\n'
                            "}"
                        ),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=700,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
        except Exception as exc:
            print(f"[AnonymizationAgent] entity detection failed, continuing without it: {exc}")
            return {"person_names": [], "locations": [], "organizations": []}

        return {
            "person_names": self._clean_list(parsed.get("person_names")),
            "locations": self._clean_list(parsed.get("locations")),
            "organizations": self._clean_list(parsed.get("organizations")),
        }

    def redact(self, text: str, entities: Dict[str, List[str]]) -> str:
        """Deterministically replaces each found entity with its redaction tag."""
        redacted = text
        tag_by_field = {
            "person_names": "REDACTED_NAME",
            "locations": "REDACTED_LOCATION",
            "organizations": "REDACTED_ORGANIZATION",
        }
        for field, tag in tag_by_field.items():
            for entity in entities.get(field, []):
                if not entity or len(entity) < 2:
                    continue
                pattern = re.compile(r"\b" + re.escape(entity) + r"\b")
                redacted = pattern.sub(f"[{tag}]", redacted)
        return redacted

    def anonymize(self, text: str) -> str:
        """Convenience: detect + redact in one call."""
        entities = self.find_entities(text)
        return self.redact(text, entities)

    @staticmethod
    def _clean_list(value) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text or len(text) > 120:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned
