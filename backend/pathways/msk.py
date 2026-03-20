"""Pathway: musculoskeletal and rehabilitation (physiotherapy focus)."""
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
        pathway_name="msk",
        additional_search_terms=[
            "NICE musculoskeletal guideline",
            "NICE low back pain guideline NG59",
            "physiotherapy rehabilitation evidence",
            "NICE neck pain osteoarthritis guideline",
        ],
        safety_rules=[
            "Always screen for non-mechanical red flags before recommending exercise or movement: "
            "unexplained night pain, rest pain, fever, unexplained weight loss, history of cancer, "
            "morning stiffness >45 minutes, bilateral symptoms, bowel/bladder dysfunction.",
            "Cauda equina syndrome: always escalate immediately — bilateral leg weakness, "
            "saddle anaesthesia, bladder/bowel dysfunction with back pain = 999 emergency.",
            "Neurovascular compromise: check pulses, capillary refill, sensation, and power "
            "for acute limb injuries before movement recommendations.",
            "Fracture red flags: bone pain after trauma, point tenderness, inability to weight-bear.",
            "For inflammatory arthropathy (RA, PsA, AS): reference NICE biologics/DMARD pathways. "
            "Do not delay referral to rheumatology.",
            "Exercise and load management: reference NICE NG59 for back pain, "
            "NICE NG226 for osteoarthritis.",
        ],
        preferred_sources=["NICE", "NICE CKS", "NHS", "SIGN", "BJSM"],
        escalation_signals=[
            "cauda equina", "cord compression", "bilateral leg weakness",
            "saddle anaesthesia", "bowel bladder dysfunction back pain",
            "non-blanching rash", "acute limb ischaemia",
            "unexplained bone pain weight loss", "cancer red flag MSK",
        ],
        evidence_tier_override=None,
    )
