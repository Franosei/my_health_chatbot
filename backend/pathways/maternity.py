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
            "All medication advice in pregnancy must reference NICE or RCOG guidelines. "
            "No off-label recommendation without explicit source backing.",
            "Always list obstetric red flags requiring immediate 999: "
            "heavy vaginal bleeding, reduced fetal movements (>28 weeks), severe headache with visual disturbance, "
            "upper abdominal pain, rupture of membranes before 37 weeks, cord prolapse, "
            "signs of pre-eclampsia (hypertension, proteinuria, oedema).",
            "Pre-eclampsia and eclampsia: always escalate to emergency services immediately.",
            "For postnatal period: include postpartum haemorrhage red flags, thromboembolism risk, "
            "signs of postpartum psychosis (rapid behaviour change, confusion, hallucinations).",
            "For newborn questions: include neonatal red flags (poor feeding, jaundice, temperature instability, "
            "respiratory distress, hypotonia).",
            "Apply highest safety threshold — when in doubt, escalate.",
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
