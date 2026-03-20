"""
Structured audit models for clinical governance, policy tracking, and evidence traceability.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PolicyGateRecord:
    gate_name: str
    applied: bool
    reason: str
    timestamp: str = field(default_factory=_utc_now)

    def as_dict(self) -> Dict:
        return {
            "gate_name": self.gate_name,
            "applied": self.applied,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class ClinicalAuditTrace:
    # Core identifiers
    trace_id: str
    created_at: str
    question: str
    answer_preview: str

    # Retrieval metadata
    sources: List[Dict] = field(default_factory=list)
    retrieval_mode: str = "live_multi_source"
    expanded_queries: List[str] = field(default_factory=list)
    model: str = ""

    # Clinical governance fields (new)
    role_key: str = "patient"
    intent_category: str = ""
    risk_level: str = "routine"
    escalation_triggered: bool = False
    crisis_detected: bool = False
    evidence_tiers_present: List[int] = field(default_factory=list)
    pathway_used: str = ""
    vulnerable_flags: List[str] = field(default_factory=list)

    # Safety / moderation fields
    moderation_category: str = ""
    moderation_details: Dict = field(default_factory=dict)
    policy_gates_applied: List[Dict] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "question": self.question,
            "answer_preview": self.answer_preview,
            "sources": self.sources,
            "retrieval_mode": self.retrieval_mode,
            "expanded_queries": self.expanded_queries,
            "model": self.model,
            "role_key": self.role_key,
            "intent_category": self.intent_category,
            "risk_level": self.risk_level,
            "escalation_triggered": self.escalation_triggered,
            "crisis_detected": self.crisis_detected,
            "evidence_tiers_present": self.evidence_tiers_present,
            "pathway_used": self.pathway_used,
            "vulnerable_flags": self.vulnerable_flags,
            "moderation_category": self.moderation_category,
            "moderation_details": self.moderation_details,
            "policy_gates_applied": self.policy_gates_applied,
        }
