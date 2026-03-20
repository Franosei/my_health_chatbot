"""
Evidence tiering and role-aware ranking.
Assigns evidence tiers to sources and re-ranks them combining semantic
similarity with source authority and role preferences.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from backend.role_router import RoleConfig
from backend.intent_risk_classifier import IntentClassification
from backend.response_templates import build_tier_badge, get_tier_description

if TYPE_CHECKING:
    from backend.memory_store import MemoryStore


# ── Tier assignment constants ──────────────────────────────────────────────────

# Tier 1: Formal guidance providers
_TIER1_PROVIDERS = {
    "nhs", "nice", "mhra", "sign", "bnf", "nice cks", "rcog",
    "phe", "public health england", "public health wales",
    "uk health security agency", "ukhsa", "gov.uk",
}

# Tier 2 signals in article titles / journals
_TIER2_TITLE_PATTERNS = [
    re.compile(r"\b(systematic review|meta.?analysis|cochrane|scoping review)\b", re.IGNORECASE),
]
_TIER2_JOURNALS = {
    "lancet", "bmj", "new england journal of medicine", "nejm",
    "jama", "annals of internal medicine", "plos medicine",
    "british medical journal", "nature medicine",
}


@dataclass
class TieredSource:
    """A source dict enriched with evidence tier and ranking metadata."""
    # Core keys (must match existing source dict structure used by UI)
    source_id: str = ""
    title: str = ""
    journal: str = ""
    year: str = ""
    authors: str = ""
    section: str = ""
    url: str = ""
    snippet: str = ""
    detail_snippet: str = ""
    source_type: str = ""
    provider: str = ""
    pmcid: str = ""
    query: str = ""
    similarity: float = 0.0
    relevance: float = 0.0

    # New tier fields
    evidence_tier: int = 3
    tier_label: str = ""
    tier_description: str = ""
    tier_badge: str = ""
    role_boost: float = 0.0
    combined_score: float = 0.0

    @classmethod
    def from_dict(cls, source: Dict) -> "TieredSource":
        ts = cls(
            source_id=source.get("source_id", ""),
            title=source.get("title", ""),
            journal=source.get("journal", ""),
            year=source.get("year", ""),
            authors=source.get("authors", ""),
            section=source.get("section", ""),
            url=source.get("url", ""),
            snippet=source.get("snippet", ""),
            detail_snippet=source.get("detail_snippet", source.get("snippet", "")),
            source_type=source.get("source_type", ""),
            provider=source.get("provider", ""),
            pmcid=source.get("pmcid", ""),
            query=source.get("query", ""),
            similarity=float(source.get("similarity", source.get("relevance", 0.0))),
            relevance=float(source.get("relevance", source.get("similarity", 0.0))),
        )
        return ts

    def as_dict(self) -> Dict:
        """Returns a dict fully compatible with the existing UI source rendering."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "journal": self.journal,
            "year": self.year,
            "authors": self.authors,
            "section": self.section,
            "url": self.url,
            "snippet": self.snippet,
            "detail_snippet": self.detail_snippet,
            "source_type": self.source_type,
            "provider": self.provider,
            "pmcid": self.pmcid,
            "query": self.query,
            "similarity": self.similarity,
            "relevance": self.combined_score or self.relevance,
            # New fields
            "evidence_tier": self.evidence_tier,
            "tier_label": self.tier_label,
            "tier_description": self.tier_description,
            "tier_badge": self.tier_badge,
            "role_boost": self.role_boost,
        }


class EvidenceRanker:
    """
    Assigns evidence tiers and re-ranks sources using semantic similarity
    combined with source authority and role-preference boosts.
    """

    def rank_and_tier(
        self,
        sources: List[Dict],
        question: str,
        role_config: RoleConfig,
        intent: IntentClassification,
        memory_store: "MemoryStore",
        top_k: int = 6,
    ) -> List[Dict]:
        """
        Returns a list of source dicts (as_dict format) sorted by combined_score descending.
        Assigns evidence_tier and tier_label to each source.
        """
        if not sources:
            return []

        # Embed the question once
        try:
            query_vector = memory_store._embed_text(question)
            source_texts = [self._source_text(s) for s in sources]
            source_vectors = memory_store._embed_texts(source_texts)
            semantic_scores = [
                float(np.dot(query_vector, sv)) for sv in source_vectors
            ]
        except Exception as exc:
            print(f"EvidenceRanker embedding failed, using fallback scores: {exc}")
            semantic_scores = [s.get("relevance", s.get("similarity", 0.5)) for s in sources]

        tiered: List[TieredSource] = []
        for source, sem_score in zip(sources, semantic_scores):
            ts = TieredSource.from_dict(source)
            ts.evidence_tier = self._assign_tier(source)
            ts.tier_label = _TIER1_PROVIDERS and f"Tier {ts.evidence_tier}" or ""
            ts.tier_label = f"Tier {ts.evidence_tier}"
            ts.tier_description = get_tier_description(ts.evidence_tier)
            ts.tier_badge = build_tier_badge(ts.evidence_tier)
            ts.role_boost = self._compute_role_boost(ts.evidence_tier, role_config)
            ts.relevance = round(sem_score, 3)

            # Authority weight: Tier1 = 1.0, Tier2 = 0.85, Tier3 = 0.70
            authority_weight = {1: 1.0, 2: 0.85, 3: 0.70}.get(ts.evidence_tier, 0.70)
            ts.combined_score = round(
                sem_score * authority_weight + ts.role_boost * 0.15, 4
            )
            tiered.append(ts)

        # Sort by combined_score descending, then apply top_k
        tiered.sort(key=lambda t: t.combined_score, reverse=True)
        ranked = tiered[:top_k]

        # Re-assign sequential source IDs
        result = []
        for idx, ts in enumerate(ranked, start=1):
            ts.source_id = f"S{idx}"
            result.append(ts.as_dict())

        return result

    def _assign_tier(self, source: Dict) -> int:
        """Assign Tier 1, 2, or 3 based on source metadata."""
        source_type = source.get("source_type", "")
        provider = (source.get("provider", "") or "").lower()
        title = (source.get("title", "") or "").lower()
        journal = (source.get("journal", "") or "").lower()

        # Tier 1: formal guidance
        if source_type == "official_guidance":
            return 1
        if any(p in provider for p in _TIER1_PROVIDERS):
            return 1

        # Tier 2: reviews and high-impact journals
        for pattern in _TIER2_TITLE_PATTERNS:
            if pattern.search(title):
                return 2
        if any(j in journal for j in _TIER2_JOURNALS):
            return 2

        # Check publication_type metadata if present
        pub_type = (source.get("publication_type", "") or "").lower()
        if any(t in pub_type for t in ("review", "meta-analysis", "systematic")):
            return 2

        # Tier 3: primary research (default for PubMed)
        return 3

    @staticmethod
    def _compute_role_boost(tier: int, role_config: RoleConfig) -> float:
        """
        Returns a boost 0.0–1.0 based on how preferred this tier is for the role.
        Preferred tiers are listed first in role_config.preferred_evidence_tiers.
        """
        preferred = role_config.preferred_evidence_tiers
        if tier not in preferred:
            return 0.0
        # Position in preferred list → higher position = lower boost
        pos = preferred.index(tier)
        return max(0.0, 1.0 - pos * 0.2)

    @staticmethod
    def _source_text(source: Dict) -> str:
        """Build a text representation of a source for embedding."""
        parts = [
            source.get("title", ""),
            source.get("section", ""),
            source.get("detail_snippet", ""),
            source.get("snippet", ""),
            source.get("query", ""),
        ]
        return " ".join(p for p in parts if p) or source.get("title", "Retrieved source")

    @staticmethod
    def get_tiers_present(sources: List[Dict]) -> List[int]:
        """Returns a sorted list of unique evidence tiers present in the source list."""
        tiers = {s.get("evidence_tier", 3) for s in sources if s.get("evidence_tier")}
        return sorted(tiers)
