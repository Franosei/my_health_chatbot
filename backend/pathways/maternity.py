"""Pathway: maternity, antenatal, postnatal, and newborn care."""
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
        pathway_name="maternity",
        additional_search_terms=[
            "NICE antenatal care guideline NG201",
            "RCOG guideline obstetric",
            "maternity NICE postnatal care",
            "NICE intrapartum care guideline",
        ],
        safety_rules=[
            "Use pregnancy-, postpartum-, or newborn-specific evidence only when that context is established.",
            "For medicine questions, verify the exact product and pregnancy or breastfeeding applicability.",
            "Include only presentation-specific obstetric or neonatal warning signs that change disposition.",
            "Use the lowest safe fact-supported disposition; uncertainty alone is not an emergency trigger.",
        ],
        preferred_sources=["RCOG", "NICE", "NHS", "NICE CKS"],
        escalation_signals=[
            "reduced fetal movements", "vaginal bleeding pregnancy", "pre-eclampsia",
            "eclampsia", "cord prolapse", "placental abruption", "premature rupture membranes",
            "postpartum haemorrhage", "postpartum psychosis", "neonatal jaundice severe",
            "newborn respiratory distress",
        ],
        evidence_tier_override=1,  # Always prefer formal guidance for maternity
    )
