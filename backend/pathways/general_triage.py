"""Pathway: general symptom triage and health information."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.intent_risk_classifier import IntentClassification
    from backend.role_router import RoleConfig


@dataclass
class PathwayContext:
    pathway_name: str
    additional_search_terms: List[str] = field(default_factory=list)
    safety_rules: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    escalation_signals: List[str] = field(default_factory=list)
    evidence_tier_override: Optional[int] = None


def get_pathway_context(
    intent: "IntentClassification",
    role_config: "RoleConfig",
) -> PathwayContext:
    return PathwayContext(
        pathway_name="general_triage",
        additional_search_terms=[
            "NICE CKS clinical knowledge summary",
            "NHS clinical guidelines",
            "NICE evidence-based guideline",
        ],
        safety_rules=[
            "Always check for red flag symptoms before educational content: unexplained weight loss, "
            "haemoptysis, dysphagia, postmenopausal bleeding, rectal bleeding, severe headache.",
            "For fever: include serious bacterial infection red flags (petechial rash, neck stiffness, "
            "photophobia, purpuric rash).",
            "Sepsis red flags must always be escalated: confusion, tachycardia, hypotension, "
            "reduced urine output, mottled skin.",
            "For chest symptoms: always exclude ACS, PE, and pneumothorax before reassurance.",
            "Neurological sudden-onset symptoms (FAST) require immediate 999 escalation.",
        ],
        preferred_sources=["NICE", "NHS", "NICE CKS", "SIGN"],
        escalation_signals=[
            "red flag symptoms", "unexplained weight loss", "haemoptysis",
            "meningitis symptoms", "sepsis signs", "chest pain exertional",
            "stroke symptoms", "sudden neurological deficit",
        ],
        evidence_tier_override=1,  # Prefer formal guidance for general triage
    )
