from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, Iterable, List

import requests


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
LOOKUP_FIELDS = (
    "openfda.generic_name",
    "openfda.brand_name",
    "openfda.substance_name",
)
INTERACTION_FIELDS = (
    ("drug_interactions", "Drug interactions"),
    ("drug_interactions_table", "Interaction table"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and cautions"),
)
HIGH_RISK_MARKERS = (
    "contraindicat",
    "avoid concomitant",
    "avoid use",
    "major interaction",
    "life-threatening",
    "serious bleeding",
    "fatal",
    "do not use",
)
MONITOR_MARKERS = (
    "monitor",
    "dose adjustment",
    "increase",
    "decrease",
    "increased risk",
    "reduced effect",
    "bleeding risk",
    "toxicity",
)


def _clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _severity_rank(value: str) -> int:
    return {"high": 3, "monitor": 2, "mentioned": 1}.get(value, 0)


class MedicationInteractionChecker:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self._resolution_cache: Dict[str, Dict | None] = {}

    def resolve_medication(self, medication_name: str) -> Dict | None:
        key = (medication_name or "").strip().lower()
        if not key:
            return None
        if key in self._resolution_cache:
            return self._resolution_cache[key]

        safe_name = medication_name.replace('"', "").strip()
        query_parts = [f'{field}:"{safe_name}"' for field in LOOKUP_FIELDS]
        params = {
            "search": " OR ".join(query_parts),
            "limit": "1",
        }
        try:
            response = self.session.get(OPENFDA_LABEL_URL, params=params, timeout=self.timeout)
            if response.status_code == 404:
                self._resolution_cache[key] = None
                return None
            response.raise_for_status()
            payload = response.json()
        except Exception:
            self._resolution_cache[key] = None
            return None

        results = payload.get("results", [])
        if not results:
            self._resolution_cache[key] = None
            return None

        result = results[0]
        openfda = result.get("openfda", {})
        aliases = {
            medication_name.strip(),
            *(openfda.get("generic_name") or []),
            *(openfda.get("brand_name") or []),
            *(openfda.get("substance_name") or []),
        }
        aliases = {alias.strip() for alias in aliases if alias and alias.strip()}

        sections = []
        for field_name, label in INTERACTION_FIELDS:
            values = result.get(field_name, [])
            if isinstance(values, str):
                values = [values]
            cleaned = " ".join(_clean_text(value) for value in values if _clean_text(value))
            if cleaned:
                sections.append(
                    {
                        "field": field_name,
                        "label": label,
                        "text": cleaned,
                    }
                )

        resolved = {
            "query_name": medication_name.strip(),
            "canonical_name": (
                (openfda.get("generic_name") or [])
                or (openfda.get("brand_name") or [])
                or [medication_name.strip()]
            )[0],
            "aliases": sorted(aliases),
            "sections": sections,
            "api_url": response.url,
            "effective_time": result.get("effective_time", ""),
        }
        self._resolution_cache[key] = resolved
        return resolved

    def check_interactions(self, medication_names: Iterable[str]) -> Dict:
        unique_names = []
        for name in medication_names:
            cleaned = (name or "").strip()
            if cleaned and cleaned.lower() not in {item.lower() for item in unique_names}:
                unique_names.append(cleaned)

        resolved = []
        unresolved = []
        for name in unique_names:
            resolved_medication = self.resolve_medication(name)
            if resolved_medication:
                resolved.append(resolved_medication)
            else:
                unresolved.append(name)

        alerts = []
        for left, right in combinations(resolved, 2):
            alert = self._build_pair_alert(left, right)
            if alert:
                alerts.append(alert)

        alerts.sort(key=lambda item: (_severity_rank(item.get("severity", "")), item.get("pair", "")), reverse=True)
        return {
            "resolved_medications": resolved,
            "unresolved_medications": unresolved,
            "alerts": alerts,
        }

    def _build_pair_alert(self, left: Dict, right: Dict) -> Dict | None:
        evidence_matches = []
        for source, target in ((left, right), (right, left)):
            match = self._find_match(source, target)
            if match:
                evidence_matches.append(match)

        if not evidence_matches:
            return None

        evidence_matches.sort(key=lambda item: _severity_rank(item["severity"]), reverse=True)
        top_match = evidence_matches[0]
        return {
            "pair": f"{left['canonical_name']} + {right['canonical_name']}",
            "severity": top_match["severity"],
            "summary": top_match["summary"],
            "evidence": evidence_matches,
        }

    def _find_match(self, source: Dict, target: Dict) -> Dict | None:
        target_aliases = sorted(
            {alias for alias in target.get("aliases", []) if len(alias) >= 3},
            key=len,
            reverse=True,
        )
        for section in source.get("sections", []):
            section_text = section.get("text", "")
            for alias in target_aliases:
                pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
                match = pattern.search(section_text)
                if not match:
                    continue
                excerpt = self._extract_excerpt(section_text, match.start(), match.end())
                severity = self._classify_excerpt(excerpt)
                summary = (
                    f"{source['canonical_name']} label mentions {target['canonical_name']} in "
                    f"{section['label'].lower()}: {excerpt}"
                )
                return {
                    "source_medication": source["canonical_name"],
                    "target_medication": target["canonical_name"],
                    "section": section["label"],
                    "severity": severity,
                    "summary": summary,
                    "source_url": source.get("api_url", ""),
                    "effective_time": source.get("effective_time", ""),
                }
        return None

    @staticmethod
    def _extract_excerpt(text: str, start: int, end: int, window: int = 180) -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        excerpt = text[left:right].strip()
        return excerpt[:320].strip()

    @staticmethod
    def _classify_excerpt(excerpt: str) -> str:
        lowered = excerpt.lower()
        if any(marker in lowered for marker in HIGH_RISK_MARKERS):
            return "high"
        if any(marker in lowered for marker in MONITOR_MARKERS):
            return "monitor"
        return "mentioned"
