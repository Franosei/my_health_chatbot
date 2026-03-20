"""Pathway: medication queries, drug interactions, and pharmacology."""
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
    clinical_role = role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist")

    search_terms = [
        "BNF drug interaction guideline",
        "NICE drug prescribing guideline",
        "MHRA drug safety alert",
    ]
    if not clinical_role:
        search_terms.append("NHS patient medication information")

    safety_rules = [
        "Always reference BNF or MHRA for dosage, interaction, and contraindication information.",
        "Never advise stopping a prescribed medication without directing to the prescribing clinician.",
        "Drug interactions: flag any clinically significant interactions (narrow therapeutic index drugs, "
        "warfarin, lithium, digoxin, methotrexate, insulin, anticoagulants).",
        "For renal/hepatic impairment: always note dose adjustment requirements.",
        "For pregnancy: reference UKTERIS or BNF pregnancy appendix; "
        "never advise teratogenic drugs without explicit safety statement.",
        "MHRA black triangle drugs: flag as under additional monitoring.",
    ]
    if not clinical_role:
        safety_rules.append(
            "Patient-facing: recommend pharmacist or GP review for all dosage queries. "
            "Do not provide specific mg dosage for prescription-only medicines."
        )
    else:
        safety_rules.append(
            "Clinician-facing: include therapeutic range, monitoring parameters, "
            "and evidence class for prescribing decisions."
        )

    return PathwayContext(
        pathway_name="medications",
        additional_search_terms=search_terms,
        safety_rules=safety_rules,
        preferred_sources=["BNF", "BNFC", "MHRA", "NICE", "NHS"],
        escalation_signals=[
            "narrow therapeutic index", "warfarin toxicity", "lithium toxicity",
            "digoxin toxicity", "methotrexate toxicity", "anaphylaxis drug reaction",
            "Stevens-Johnson syndrome", "agranulocytosis drug-induced",
            "serotonin syndrome", "neuroleptic malignant syndrome",
        ],
        evidence_tier_override=None,
    )
