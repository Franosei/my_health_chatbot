"""
ClinicalOrchestrator: central workflow engine for Dr. Charlotte.
Coordinates role detection, risk classification, tiered evidence retrieval,
policy gating, and response assembly.

Replaces the internals of RAGEngine._prepare_answer_bundle() while keeping
the public interface of RAGEngine fully backward-compatible.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from backend.audit_models import ClinicalAuditTrace
from backend.evidence_ranker import EvidenceRanker
from backend.intent_risk_classifier import IntentClassification, IntentRiskClassifier
from backend.policy_engine import PolicyEngine, PolicyDecision
from backend.response_templates import CRISIS_RESPONSE
from backend.role_router import RoleConfig, RoleRouter
from backend.utils import build_excerpt

if TYPE_CHECKING:
    from backend.memory_store import MemoryStore
    from backend.moderation_ml import ModerationEnsemble
    from backend.official_guidance import OfficialGuidanceEngine
    from backend.pubmed_search import PubMedCentralSearcher
    from backend.query_expander import QueryExpander
    from backend.summarizer import LLMHelper


class ClinicalOrchestrator:
    """
    Central workflow engine. Called by RAGEngine._prepare_answer_bundle().
    Returns a bundle dict that is a superset of the original bundle structure.
    """

    def __init__(
        self,
        memory: "MemoryStore",
        pubmed: "PubMedCentralSearcher",
        official_guidance: "OfficialGuidanceEngine",
        llm: "LLMHelper",
        query_expander: "QueryExpander",
        moderation: "ModerationEnsemble",
    ) -> None:
        self.memory = memory
        self.pubmed = pubmed
        self.official_guidance = official_guidance
        self.llm = llm
        self.query_expander = query_expander
        self.moderation = moderation

        self.role_router = RoleRouter()
        self.intent_classifier = IntentRiskClassifier()
        self.policy_engine = PolicyEngine()
        self.evidence_ranker = EvidenceRanker()

    def prepare_bundle(
        self,
        question: str,
        user: Optional[str],
        user_profile: dict,
        longitudinal_memory_summary: str,
    ) -> Dict:
        """
        Full clinical orchestration pipeline.
        Returns a dict compatible with RAGEngine._finalize_answer_payload()
        plus new clinical governance keys.
        """
        normalized_user = (user or "").strip().lower() or None

        # ── Step 1: Role resolution (instant) ─────────────────────────────────
        clinical_role = user_profile.get("clinical_role") or user_profile.get("role", "")
        role_config = self.role_router.resolve(clinical_role)

        # ── Step 2: Crisis pre-screen (regex, instant) ─────────────────────────
        if self.intent_classifier._crisis_prescreen(question):
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # ── Step 3: Moderation ─────────────────────────────────────────────────
        blocked, category, safe_msg, details = self.moderation.decide(
            question, role_key=role_config.role_key
        )
        if blocked:
            return self._build_moderation_bundle(
                question, normalized_user, safe_msg, category, details, role_config
            )

        # ── Step 4: Concurrent — intent classification + query expansion ───────
        intent: IntentClassification
        expanded_queries: List[str]

        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(
                self.intent_classifier.classify,
                question,
                user_profile,
                role_config.role_key,
            )
            expand_future = executor.submit(self._build_search_queries, question)

            try:
                intent = intent_future.result()
            except Exception as exc:
                print(f"Intent classification failed: {exc}")
                intent = IntentClassification()

            try:
                expanded_queries = expand_future.result()
            except Exception as exc:
                print(f"Query expansion failed: {exc}")
                expanded_queries = [question]

        # ── Step 5: Policy gate ────────────────────────────────────────────────
        policy_decision = self.policy_engine.gate(intent, role_config, question)
        if policy_decision.action == "escalate_only" and policy_decision.crisis_response:
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # ── Step 6: Pathway context → augment search queries ──────────────────
        pathway_context = self._get_pathway_context(intent, role_config)
        search_queries = self._augment_queries_with_pathway(expanded_queries, pathway_context)

        # ── Step 7: Parallel retrieval ─────────────────────────────────────────
        official_sources: List[Dict] = []
        preferred = list(
            dict.fromkeys(pathway_context.preferred_sources or [])
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            official_future = executor.submit(
                self.official_guidance.search, search_queries, preferred or None
            )
            pubmed_future = executor.submit(
                self._retrieve_pubmed_for_queries, search_queries, normalized_user
            )
            try:
                official_sources = official_future.result()
            except Exception as exc:
                print(f"Official guidance search failed: {exc}")
            try:
                pubmed_future.result()
            except Exception as exc:
                print(f"PubMed retrieval failed: {exc}")

        # ── Step 8: Semantic search personal context + pubmed ─────────────────
        matches = self.memory.search(query=question, user=normalized_user)
        personal_context, pubmed_matches = self._split_matches(matches)
        pubmed_sources = self._build_source_briefings(pubmed_matches)

        # ── Step 9: Combine and de-duplicate ──────────────────────────────────
        raw_sources = self._combine_sources(pubmed_sources, official_sources)

        # ── Step 10: Evidence ranking with tiers ───────────────────────────────
        combined_sources = self.evidence_ranker.rank_and_tier(
            sources=raw_sources,
            question=question,
            role_config=role_config,
            intent=intent,
            memory_store=self.memory,
            top_k=6,
        )

        retrieval_mode = "live_multi_source" if combined_sources else "general_knowledge"

        # ── Step 11: Build role-aware context for LLM ─────────────────────────
        full_context = self._build_role_context(
            combined_sources=combined_sources,
            personal_context=personal_context,
            policy_decision=policy_decision,
            pathway_context=pathway_context,
            no_sources=not combined_sources,
        )

        return {
            "kind": "answer",
            # Existing keys (backward-compatible)
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": combined_sources,
            "personal_context": personal_context,
            "longitudinal_memory_summary": longitudinal_memory_summary,
            "expanded_queries": expanded_queries,
            "matches": matches,
            "retrieval_mode": retrieval_mode,
            "full_context": full_context,
            # New clinical governance keys
            "role_config": role_config,
            "intent": intent,
            "policy_decision": policy_decision,
            "pathway_context": pathway_context,
        }

    # ── Bundle builders ────────────────────────────────────────────────────────

    def _build_crisis_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
    ) -> Dict:
        """Returns a pre-built crisis response without LLM generation."""
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": CRISIS_RESPONSE,
                "answer_text": CRISIS_RESPONSE,
                "sources": [],
                "personal_context": [],
                "trace": {
                    "trace_id": f"trace-crisis",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": CRISIS_RESPONSE[:280],
                    "sources": [],
                    "retrieval_mode": "crisis_escalation",
                    "role_key": role_config.role_key,
                    "intent_category": "crisis",
                    "risk_level": "crisis",
                    "escalation_triggered": True,
                    "crisis_detected": True,
                },
            },
        }

    def _build_moderation_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        safe_msg: str,
        category: str,
        details: Dict,
        role_config: RoleConfig,
    ) -> Dict:
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": safe_msg,
                "answer_text": safe_msg,
                "sources": [],
                "personal_context": [],
                "trace": {
                    "trace_id": f"trace-mod",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": safe_msg[:280],
                    "sources": [],
                    "retrieval_mode": "moderation_block",
                    "moderation_category": category,
                    "moderation_details": details,
                    "role_key": role_config.role_key,
                },
            },
        }

    def _build_limited_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        personal_context: List[Dict],
        retrieval_mode: str,
        expanded_queries: List[str],
        role_config: RoleConfig,
        intent: IntentClassification,
        policy_decision: PolicyDecision,
    ) -> Dict:
        limited_answer = self._build_limited_evidence_response(personal_context, role_config)
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": limited_answer,
                "answer_text": limited_answer,
                "sources": [],
                "personal_context": personal_context,
                "trace": {
                    "trace_id": "trace-limited",
                    "created_at": _utc_now(),
                    "question": question,
                    "answer_preview": limited_answer[:280],
                    "sources": [],
                    "retrieval_mode": retrieval_mode,
                    "expanded_queries": expanded_queries,
                    "role_key": role_config.role_key,
                    "intent_category": intent.intent_category,
                    "risk_level": intent.risk_level,
                    "escalation_triggered": policy_decision.action != "allow",
                    "policy_gates_applied": policy_decision.gates_as_dicts(),
                },
            },
        }

    # ── Context builders ───────────────────────────────────────────────────────

    def _build_role_context(
        self,
        combined_sources: List[Dict],
        personal_context: List[Dict],
        policy_decision: PolicyDecision,
        pathway_context,
        no_sources: bool = False,
    ) -> str:
        parts = []

        # Personal context
        if personal_context:
            personal_lines = "\n".join(
                f"- {item['title']}: {item['snippet']}" for item in personal_context
            )
            parts.append(f"Personal context:\n{personal_lines}")

        # Policy notes for LLM
        if policy_decision.context_notes:
            notes = "\n".join(policy_decision.context_notes)
            parts.append(f"Clinical policy notes (must be followed):\n{notes}")

        # Pathway safety rules
        if pathway_context and pathway_context.safety_rules:
            rules = "\n".join(f"- {r}" for r in pathway_context.safety_rules)
            parts.append(f"Pathway safety rules:\n{rules}")

        # Evidence with tier labelling
        if combined_sources:
            evidence_parts = []
            for source in combined_sources:
                tier = source.get("evidence_tier", 3)
                tier_label = source.get("tier_label", f"Tier {tier}")
                snippet = source.get("detail_snippet") or source.get("snippet", "")
                evidence_parts.append(
                    f"[{tier_label}] {source.get('title', 'Source')}: {snippet}"
                )
            parts.append("Biomedical evidence (tiered by source authority):\n" + "\n\n".join(evidence_parts))
        elif no_sources:
            parts.append(
                "Note: No live evidence was retrieved for this query. "
                "Answer from your clinical training knowledge, clearly indicating this is general guidance "
                "and not based on retrieved literature. Advise the user to seek professional assessment where appropriate."
            )

        return "\n\n".join(parts)

    # ── Pathway routing ────────────────────────────────────────────────────────

    def _get_pathway_context(self, intent: IntentClassification, role_config: RoleConfig):
        """Load the appropriate pathway module based on intent."""
        hint = intent.pathway_hint or "general_triage"
        try:
            if hint == "maternity":
                from backend.pathways.maternity import get_pathway_context
            elif hint == "msk":
                from backend.pathways.msk import get_pathway_context
            elif hint == "medications":
                from backend.pathways.medications import get_pathway_context
            elif hint == "chronic_conditions":
                from backend.pathways.chronic_conditions import get_pathway_context
            else:
                from backend.pathways.general_triage import get_pathway_context
            return get_pathway_context(intent, role_config)
        except Exception as exc:
            print(f"Pathway load failed ({hint}): {exc}")
            from backend.pathways.general_triage import get_pathway_context
            return get_pathway_context(intent, role_config)

    # ── Query building ─────────────────────────────────────────────────────────

    def _build_search_queries(self, question: str) -> List[str]:
        queries = [question]
        try:
            queries.extend(self.query_expander.expand(question))
        except Exception as exc:
            print(f"Query expansion failed: {exc}")
        return list(dict.fromkeys(q for q in queries if q))[:3]

    def _augment_queries_with_pathway(
        self, queries: List[str], pathway_context
    ) -> List[str]:
        if not pathway_context or not pathway_context.additional_search_terms:
            return queries
        augmented = list(queries)
        # Add up to 2 pathway-specific terms to the query list
        for term in pathway_context.additional_search_terms[:2]:
            combined = f"{queries[0]} {term}"
            if combined not in augmented:
                augmented.append(combined)
        return augmented[:5]

    # ── Source processing (mirrors RAGEngine helpers) ──────────────────────────

    def _retrieve_pubmed_for_queries(
        self, queries: List[str], user: Optional[str]
    ) -> None:
        """Fetch PubMed articles and add to memory store."""
        pending_entries = []
        article_batches = []

        with ThreadPoolExecutor(max_workers=min(3, max(1, len(queries)))) as executor:
            query_futures = {
                executor.submit(self.pubmed.search_article_records, query, 2): query
                for query in queries
            }
            for future, query in query_futures.items():
                try:
                    article_batches.append((query, future.result()))
                except Exception as exc:
                    print(f"PubMed search failed for '{query}': {exc}")

        article_records = [
            (query, record)
            for query, records in article_batches
            for record in records
        ]

        with ThreadPoolExecutor(max_workers=min(6, max(1, len(article_records)))) as executor:
            section_futures = {
                executor.submit(self.pubmed.fetch_article_sections, record["pmcid"]): (query, record)
                for query, record in article_records
            }
            for future, (query, record) in section_futures.items():
                try:
                    sections = future.result()
                except Exception as exc:
                    print(f"PubMed section fetch failed: {exc}")
                    sections = {}

                best_section_name, best_section_text = self._select_best_pubmed_section(sections)
                if best_section_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:{best_section_name}"
                    pending_entries.append({
                        "text": best_section_text,
                        "metadata": {
                            "type": "pubmed",
                            "source_type": "pubmed_literature",
                            "pmcid": record["pmcid"],
                            "section": best_section_name,
                            "title": record.get("title", "Untitled article"),
                            "journal": record.get("journal", ""),
                            "year": record.get("year", ""),
                            "authors": record.get("authors", ""),
                            "url": record.get("url", ""),
                            "query": query,
                            "entry_key": entry_key,
                        },
                        "user": user,
                        "entry_key": entry_key,
                    })

                abstract_text = record.get("abstract", "")
                if abstract_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:abstract"
                    pending_entries.append({
                        "text": abstract_text,
                        "metadata": {
                            "type": "pubmed",
                            "source_type": "pubmed_literature",
                            "pmcid": record["pmcid"],
                            "section": "abstract",
                            "title": record.get("title", "Untitled article"),
                            "journal": record.get("journal", ""),
                            "year": record.get("year", ""),
                            "authors": record.get("authors", ""),
                            "url": record.get("url", ""),
                            "query": query,
                            "entry_key": entry_key,
                        },
                        "user": user,
                        "entry_key": entry_key,
                    })

        self.memory.add_entries(pending_entries)

    def _split_matches(
        self, matches: List[Tuple[Dict, float]]
    ) -> Tuple[List[Dict], List[Tuple[Dict, float]]]:
        personal_context = []
        pubmed_matches = []
        for entry, score in matches:
            metadata = entry.get("metadata", {})
            if metadata.get("type") == "user_summary":
                personal_context.append({
                    "title": metadata.get("title", metadata.get("source", "Uploaded document")),
                    "source": metadata.get("source", ""),
                    "snippet": build_excerpt(entry.get("text", "")),
                    "score": round(score, 3),
                })
            elif metadata.get("type") == "pubmed":
                pubmed_matches.append((entry, score))
        return personal_context[:2], pubmed_matches[:4]

    def _build_source_briefings(
        self, matches: List[Tuple[Dict, float]]
    ) -> List[Dict]:
        sources = []
        seen = set()
        for entry, score in matches:
            metadata = entry.get("metadata", {})
            key = (metadata.get("pmcid"), metadata.get("section"))
            if key in seen:
                continue
            seen.add(key)
            source_id = f"S{len(sources) + 1}"
            sources.append({
                "source_id": source_id,
                "pmcid": metadata.get("pmcid", ""),
                "title": metadata.get("title", "Untitled article"),
                "journal": metadata.get("journal", ""),
                "year": metadata.get("year", ""),
                "authors": metadata.get("authors", ""),
                "section": metadata.get("section", "retrieved text").replace("_", " ").title(),
                "url": metadata.get("url", ""),
                "query": metadata.get("query", ""),
                "similarity": round(score, 3),
                "snippet": build_excerpt(entry.get("text", "")),
                "detail_snippet": build_excerpt(entry.get("text", ""), max_chars=800),
                "source_type": metadata.get("source_type", "pubmed_literature"),
                "provider": "Europe PMC / PubMed Central",
            })
        return sources

    @staticmethod
    def _combine_sources(
        pubmed_sources: List[Dict], official_sources: List[Dict]
    ) -> List[Dict]:
        combined = []
        seen = set()
        for source in [*official_sources, *pubmed_sources]:
            key = source.get("url") or f"{source.get('title')}::{source.get('section')}"
            if key in seen:
                continue
            seen.add(key)
            combined.append(dict(source))
        for idx, source in enumerate(combined, start=1):
            source["source_id"] = f"S{idx}"
        return combined

    @staticmethod
    def _select_best_pubmed_section(sections: Dict[str, str]) -> Tuple[str, str]:
        for key in ("discussion", "conclusion", "introduction"):
            text = (sections.get(key) or "").strip()
            if text:
                return key, text
        return "", ""

    @staticmethod
    def _build_limited_evidence_response(
        personal_context: List[Dict], role_config: RoleConfig
    ) -> str:
        personal_note = ""
        if personal_context:
            personal_note = (
                "\n\n## Available Personal Context\n"
                + "\n".join(f"- {item['title']}: {item['snippet']}" for item in personal_context)
            )
        if role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist"):
            return (
                "## Evidence Retrieval\n"
                "Insufficient live evidence was retrieved for this query. "
                "Please consult current local guidelines, BNF, or NICE CKS directly.\n\n"
                "## Recommended Action\n"
                "Try a more specific query or consult the relevant NICE guideline directly."
                + personal_note
            )
        return (
            "## Clinical Takeaway\n"
            "I could not retrieve enough reliable live evidence for this question right now, "
            "so I do not want to overstate an answer.\n\n"
            "## Recommended Next Step\n"
            "Please try rephrasing the question, narrowing it to a condition, treatment, or population, "
            "or ask a clinician if you need a decision that affects immediate care."
            + personal_note
        )


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
