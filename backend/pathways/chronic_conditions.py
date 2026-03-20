"""Pathway: chronic disease management and long-term condition education."""
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
        pathway_name="chronic_conditions",
        additional_search_terms=[
            "NICE long-term condition management",
            "chronic disease self-management evidence",
            "NICE QOF quality outcomes framework",
            "comorbidity management guidelines",
        ],
        safety_rules=[
            "Reference NICE disease-specific guidelines (e.g. NG28 for type 2 diabetes, "
            "NG80 for COPD, NG106 for heart failure).",
            "For multi-morbidity: reference NICE NG56 multimorbidity guidelines.",
            "Flag decompensation red flags (e.g. acute HF exacerbation, DKA, severe COPD exacerbation).",
            "Include self-management goals and patient education points.",
            "For elderly users with polypharmacy: apply STOPP/START criteria context.",
        ],
        preferred_sources=["NICE", "NICE CKS", "NHS", "SIGN", "BTS"],
        escalation_signals=[
            "acute exacerbation COPD", "DKA diabetic ketoacidosis", "acute heart failure",
            "hypertensive crisis", "acute kidney injury chronic kidney disease",
            "chest pain chronic cardiac patient", "HbA1c dangerously elevated",
        ],
        evidence_tier_override=None,
    )
