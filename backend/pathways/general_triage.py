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
            "current official clinical guideline",
            "systematic review clinical guidance",
        ],
        safety_rules=[
            "Use only warning signs connected to the supplied presentation and capable of changing disposition.",
            "Choose the lowest safe disposition supported by facts; do not escalate for missing information alone.",
            "Give a direct next action and timeframe before optional educational detail.",
        ],
        preferred_sources=[],
        escalation_signals=[],
        evidence_tier_override=1,  # Prefer formal guidance for general triage
    )
