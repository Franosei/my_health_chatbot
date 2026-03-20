"""
Intent and risk classification for incoming clinical questions.
Combines a fast regex pre-screen with an LLM-based structured classifier.
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class IntentClassification:
    intent_category: str = "general_info"
    # symptom_triage | medication_query | chronic_condition | maternity |
    # msk | mental_health | general_info | crisis | administrative
    risk_level: str = "routine"
    # routine | elevated | urgent | crisis
    vulnerable_flags: List[str] = field(default_factory=list)
    # pregnancy | paediatric | elderly | renal_impairment | immunocompromised
    escalation_required: bool = False
    escalation_reason: str = ""
    crisis_detected: bool = False
    pathway_hint: str = "general_triage"
    # general_triage | maternity | msk | medications | chronic_conditions
    confidence: float = 0.8


# ── Fast regex crisis patterns ─────────────────────────────────────────────────
_CRISIS_PATTERNS: List[re.Pattern] = [
    # Cardiac / respiratory arrest
    re.compile(
        r"(chest\s*pain.{0,30}(shortness?|difficult|breath)|"
        r"not\s*breath|stopped\s*breath|cardiac\s*arrest|heart\s*attack)",
        re.IGNORECASE,
    ),
    # Stroke
    re.compile(
        r"(face\s*drooping|arm\s*weak|slurred?\s*speech|sudden\s*(vision|headache|weakness)|"
        r"fast\s*test|FAST\s*test|stroke\s*symptoms?)",
        re.IGNORECASE,
    ),
    # Anaphylaxis
    re.compile(
        r"(anaphyla|severe\s*allergic|throat\s*(closing|swelling)|"
        r"epipen|epinephrine\s*now|can\s*t\s*breathe)",
        re.IGNORECASE,
    ),
    # Obstetric emergencies
    re.compile(
        r"(heavy\s*bleed.{0,20}pregnan|eclampsia|cord\s*prolapse|"
        r"placental?\s*abruption|baby\s*not\s*moving.{0,10}hours?)",
        re.IGNORECASE,
    ),
    # Major trauma / overdose
    re.compile(
        r"(overdosed?|taken\s*too\s*many\s*(pills|tablets)|"
        r"unconscious|unresponsive|not\s*waking)",
        re.IGNORECASE,
    ),
    # Meningitis
    re.compile(
        r"(meningitis|non.?blanching\s*rash|glass\s*test|"
        r"stiff\s*neck.{0,20}(fever|rash))",
        re.IGNORECASE,
    ),
]

# ── Intent → pathway mapping ───────────────────────────────────────────────────
_INTENT_TO_PATHWAY: dict[str, str] = {
    "symptom_triage": "general_triage",
    "medication_query": "medications",
    "chronic_condition": "chronic_conditions",
    "maternity": "maternity",
    "msk": "msk",
    "mental_health": "general_triage",
    "general_info": "general_triage",
    "crisis": "general_triage",
    "administrative": "general_triage",
}


class IntentRiskClassifier:
    """
    Two-stage classifier:
    1. Fast regex crisis pre-screen (no LLM latency)
    2. LLM structured classification for intent + risk
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        import openai
        self.client = openai.OpenAI(api_key=api_key)

    def classify(
        self,
        question: str,
        user_profile: Optional[dict] = None,
        role_key: str = "patient",
    ) -> IntentClassification:
        """
        Full classification pipeline. Run this concurrently with query expansion
        inside the orchestrator to minimise latency.
        """
        # Stage 1: Instant crisis check
        if self._crisis_prescreen(question):
            return IntentClassification(
                intent_category="crisis",
                risk_level="crisis",
                escalation_required=True,
                escalation_reason="Potential emergency symptoms detected — please seek immediate help.",
                crisis_detected=True,
                pathway_hint="general_triage",
                confidence=0.95,
            )

        # Stage 2: LLM classification
        try:
            return self._llm_classify(question, user_profile or {}, role_key)
        except Exception as exc:
            print(f"IntentRiskClassifier LLM call failed, using safe defaults: {exc}")
            return self._safe_default(question)

    def _crisis_prescreen(self, question: str) -> bool:
        """Fast regex check — runs synchronously before any LLM call."""
        text = (question or "").strip()
        return any(pattern.search(text) for pattern in _CRISIS_PATTERNS)

    def _llm_classify(
        self, question: str, user_profile: dict, role_key: str
    ) -> IntentClassification:
        role_hint = f"The user's clinical role is: {role_key}." if role_key else ""
        pregnancy_hint = ""
        if "pregnan" in question.lower() or role_key == "midwife":
            pregnancy_hint = " Note: pregnancy context may be present — apply maternity flags carefully."

        prompt = (
            "You are a clinical intent classifier for a health information system.\n"
            f"{role_hint}{pregnancy_hint}\n\n"
            "Classify the following health question and return a JSON object with these exact keys:\n"
            "- intent_category: one of [symptom_triage, medication_query, chronic_condition, "
            "maternity, msk, mental_health, general_info, crisis, administrative]\n"
            "- risk_level: one of [routine, elevated, urgent, crisis]\n"
            "- vulnerable_flags: array of applicable flags from "
            "[pregnancy, paediatric, elderly, renal_impairment, immunocompromised, postpartum, newborn]\n"
            "- escalation_required: boolean — true if the question suggests urgent clinical need\n"
            "- escalation_reason: short string (≤60 chars) explaining why escalation is needed, "
            "or empty string if not required\n"
            "- pathway_hint: one of [general_triage, maternity, msk, medications, chronic_conditions]\n"
            "- confidence: float 0.0–1.0\n\n"
            f"Question: {question}\n\n"
            "Return only valid JSON, no other text."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)

        return IntentClassification(
            intent_category=data.get("intent_category", "general_info"),
            risk_level=data.get("risk_level", "routine"),
            vulnerable_flags=data.get("vulnerable_flags", []),
            escalation_required=bool(data.get("escalation_required", False)),
            escalation_reason=data.get("escalation_reason", ""),
            crisis_detected=data.get("risk_level", "") == "crisis",
            pathway_hint=_INTENT_TO_PATHWAY.get(
                data.get("pathway_hint", "general_triage"), "general_triage"
            ),
            confidence=float(data.get("confidence", 0.8)),
        )

    @staticmethod
    def _safe_default(question: str) -> IntentClassification:
        """Fallback when LLM classification fails — conservative safe defaults."""
        lower = question.lower()
        if any(word in lower for word in ("pregnan", "trimester", "antenatal", "postnatal", "labour")):
            return IntentClassification(
                intent_category="maternity",
                risk_level="elevated",
                vulnerable_flags=["pregnancy"],
                escalation_required=False,
                pathway_hint="maternity",
                confidence=0.5,
            )
        if any(word in lower for word in ("pain", "ache", "muscle", "joint", "back", "neck", "physio")):
            return IntentClassification(
                intent_category="msk",
                risk_level="routine",
                pathway_hint="msk",
                confidence=0.5,
            )
        if any(word in lower for word in ("medication", "drug", "tablet", "dose", "prescription")):
            return IntentClassification(
                intent_category="medication_query",
                risk_level="routine",
                pathway_hint="medications",
                confidence=0.5,
            )
        return IntentClassification(confidence=0.4)
