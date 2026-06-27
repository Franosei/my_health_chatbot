"""
ClinicalOrchestrator: central workflow engine for the health assistant.

Coordinates role detection, risk classification, safety policy gating,
and an LLM-driven agentic retrieval loop that decides which tools to
call before generating the final answer.

Architecture
------------
Deterministic safety layer (runs first, always):
  1. Role resolution
  2. Patient history context
  3. Crisis pre-screen (regex)
  4. Moderation
  5. Intent classification
  6. Policy gate (8 hard safety gates)
  7. Pathway context

Agentic retrieval layer (LLM drives this):
  8. AgenticRetrievalLoop: the model chooses which tools to call
     - search_nhs_guidance    NHS/NICE Tier 1 evidence
     - search_pubmed          PubMed Central Tier 2-3 evidence
     - check_drug_interactions openFDA drug label warnings
     - search_patient_documents patient uploaded records
     - search_clinical_trials ClinicalTrials.gov
  9. Fallback retrieval if agent returns nothing
  10. Evidence ranking (deterministic quality gate)
  11. Evidence dossier (anti-hallucination extraction)
  12. Context assembly for the final LLM answer
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from backend.clinical_decision_support import ClinicalDecision, ClinicalDecisionSupportEngine
from backend.evidence_ranker import EvidenceRanker
from backend.intent_risk_classifier import IntentClassification, IntentRiskClassifier
from backend.patient_history import PatientHistoryContext, build_patient_history_context
from backend.policy_engine import PolicyEngine, PolicyDecision
from backend.response_templates import CRISIS_RESPONSE
from backend.role_router import RoleConfig, RoleRouter
from backend.utils import build_excerpt

if TYPE_CHECKING:
    from backend.context_graph import ContextGraph
    from backend.memory_store import MemoryStore
    from backend.moderation_ml import ModerationEnsemble
    from backend.official_guidance import OfficialGuidanceEngine
    from backend.pubmed_search import PubMedCentralSearcher
    from backend.query_expander import QueryExpander
    from backend.summarizer import LLMHelper


# ---------------------------------------------------------------------------
# Agentic retrieval loop
# ---------------------------------------------------------------------------

class AgenticRetrievalLoop:
    """
    LLM-driven tool-calling retrieval agent.

    Given the clinical question and patient context the model decides which
    sources to fetch -- NHS guidance, PubMed, drug interactions, personal
    documents, clinical trials -- and in what order. It runs until it has
    enough evidence or exhausts its iteration budget.

    This replaces the hardcoded parallel retrieval pipeline with a
    model-driven workflow.
    """

    _TOOLS: List[Dict] = [
        {
            "type": "function",
            "function": {
                "name": "search_nhs_guidance",
                "description": (
                    "Search NHS and NICE official guidance for UK clinical guidelines, "
                    "treatment recommendations, prescribing information, and patient safety advice. "
                    "Call this first for any clinical, medication, or condition question. "
                    "Returns Tier 1 (highest-authority) evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Clinical search query, e.g. 'hypertension management in adults with CKD'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pubmed",
                "description": (
                    "Search PubMed Central for biomedical research literature including "
                    "clinical trials, systematic reviews, and research articles. "
                    "Use for specific conditions, treatment mechanisms, or when NHS guidance "
                    "needs supporting research evidence. "
                    "Returns Tier 2-3 evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Medical search query, e.g. 'metformin HbA1c type 2 diabetes systematic review'",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_drug_interactions",
                "description": (
                    "Look up openFDA drug label data for interaction warnings, contraindications, "
                    "side effects, and dosing information. "
                    "Use whenever the question involves medications or the patient's medication "
                    "list may interact with the topic being discussed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "medications": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Medication names to check, e.g. ['metformin', 'lisinopril']",
                        }
                    },
                    "required": ["medications"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_patient_documents",
                "description": (
                    "Search the patient's uploaded health documents and personal records. "
                    "Use when the question relates to their specific test results, uploaded "
                    "discharge letters, clinic letters, or personal health history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for in the patient's personal documents",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_clinical_trials",
                "description": (
                    "Search ClinicalTrials.gov for recruiting clinical trials. "
                    "Use only when the patient explicitly asks about trials, experimental "
                    "treatments, or eligibility for research studies. Call at most once."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Medical condition to find trials for",
                        },
                        "location": {
                            "type": "string",
                            "description": "Preferred trial location (default: United Kingdom)",
                        },
                    },
                    "required": ["condition"],
                },
            },
        },
    ]

    def __init__(
        self,
        llm: "LLMHelper",
        official_guidance: "OfficialGuidanceEngine",
        pubmed: "PubMedCentralSearcher",
        memory: "MemoryStore",
        user: Optional[str],
    ) -> None:
        self.llm = llm
        self.official_guidance = official_guidance
        self.pubmed = pubmed
        self.memory = memory
        self.user = user

    def run(
        self,
        question: str,
        patient_summary: str,
        role_key: str,
        pathway_hint: str,
        patient_medications: Optional[List[str]] = None,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Run the agentic retrieval loop.

        Returns a dict with:
          collected_sources   list of source dicts (NHS, PubMed, openFDA)
          personal_context    list of personal-document match dicts
          trial_results       list of clinical trial dicts
          tool_calls_made     audit log of every tool call made
        """
        pathway_guidance = {
            "maternity": (
                "Prioritise RCOG and NICE maternity guidelines. "
                "Check pregnancy contraindications if medications are mentioned."
            ),
            "msk": (
                "Prioritise NICE MSK guidelines and physiotherapy evidence. "
                "Search for the specific injury or condition plus rehabilitation."
            ),
            "medications": (
                "Always check drug interactions. "
                "Search NHS/BNF for prescribing guidance."
            ),
            "chronic_conditions": (
                "Prioritise NICE chronic disease guidelines. "
                "Focus on long-term management and patient-specific risks."
            ),
        }.get(pathway_hint, "Search NHS guidance first, then PubMed if more detail is needed.")

        med_hint = ""
        if patient_medications:
            med_hint = (
                f"\nPatient's current medications: {', '.join(patient_medications[:8])}. "
                "Consider checking drug interactions if relevant."
            )

        system_prompt = (
            "You are a clinical evidence retrieval agent for Dr. Charlotte, a UK health AI assistant.\n"
            "Your task: decide which tools to call to gather the right evidence BEFORE the answer is written.\n"
            "Do NOT answer the question yourself.\n\n"
            f"Clinical role: {role_key}\n"
            f"Pathway: {pathway_hint}\n"
            f"Retrieval strategy: {pathway_guidance}{med_hint}\n\n"
            "Rules:\n"
            "- Call search_nhs_guidance first for any clinical or medication question.\n"
            "- Call search_pubmed when you need research evidence or more detail.\n"
            "- Call check_drug_interactions if the question involves medications or interactions.\n"
            "- Call search_patient_documents if the question relates to the patient's own records.\n"
            "- Call search_clinical_trials ONLY if the question explicitly asks about trials.\n"
            "- Make at most 4 tool calls total. Stop as soon as you have sufficient evidence.\n"
            "- When you have finished gathering evidence, respond with the word DONE."
        )

        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Gather evidence for this question: {question}\n\n"
                    f"Patient context:\n{patient_summary}"
                ),
            },
        ]

        collected_sources: List[Dict] = []
        personal_context: List[Dict] = []
        trial_results: List[Dict] = []
        tool_calls_made: List[Dict] = []

        for iteration in range(max_iterations):
            try:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.AUX_MODEL,
                    messages=messages,
                    tools=self._TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_tokens=400,
                )
            except Exception as exc:
                print(f"[AgenticLoop] LLM call failed on iteration {iteration}: {exc}")
                break

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Build a serializable dict for the assistant turn
            assistant_entry: Dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_entry)

            # No tool calls: agent is done retrieving
            if not msg.tool_calls or finish_reason == "stop":
                break

            # Execute each tool call and collect results
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}

                tool_calls_made.append({
                    "tool": fn_name,
                    "args": args,
                    "iteration": iteration,
                })
                print(f"[AgenticLoop] {fn_name}({args})")

                result = self._execute_tool(fn_name, args)

                if "sources" in result:
                    collected_sources.extend(result["sources"])
                if "personal_matches" in result:
                    personal_context.extend(result["personal_matches"])
                if "trials" in result:
                    trial_results.extend(result["trials"])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.get("summary", "No results.")[:2000],
                })

        return {
            "collected_sources": collected_sources,
            "personal_context": personal_context,
            "trial_results": trial_results,
            "tool_calls_made": tool_calls_made,
        }

    # -- Tool implementations ------------------------------------------------

    def _execute_tool(self, name: str, args: Dict) -> Dict:
        try:
            if name == "search_nhs_guidance":
                return self._search_nhs(args.get("query", ""))
            if name == "search_pubmed":
                return self._search_pubmed(args.get("query", ""))
            if name == "check_drug_interactions":
                return self._check_drug_interactions(args.get("medications", []))
            if name == "search_patient_documents":
                return self._search_personal(args.get("query", ""))
            if name == "search_clinical_trials":
                return self._search_trials(
                    args.get("condition", ""),
                    args.get("location", "United Kingdom"),
                )
            return {"summary": f"Unknown tool: {name}"}
        except Exception as exc:
            print(f"[AgenticLoop] Tool {name} raised: {exc}")
            return {"summary": f"{name} error: {exc}", "sources": []}

    def _search_nhs(self, query: str) -> Dict:
        if not query:
            return {"summary": "No query provided.", "sources": []}
        sources = self.official_guidance.search([query], 1)
        return {
            "sources": sources,
            "summary": f"Found {len(sources)} NHS/NICE sources for '{query}'.",
        }

    def _search_pubmed(self, query: str) -> Dict:
        if not query:
            return {"summary": "No query provided.", "sources": []}
        records = self.pubmed.search_article_records(query, 2)
        sources: List[Dict] = []
        memory_entries: List[Dict] = []

        for record in records:
            pmcid = record.get("pmcid", "")
            try:
                sections = self.pubmed.fetch_article_sections(pmcid)
            except Exception:
                sections = {}

            # Pick the best section
            section_text = ""
            section_name = "abstract"
            for key in ("discussion", "conclusion", "introduction"):
                text = (sections.get(key) or "").strip()
                if text:
                    section_text = text
                    section_name = key
                    break

            if section_text:
                entry_key = f"{self.user or 'global'}:pmc:{pmcid}:{section_name}"
                sources.append({
                    "source_id": f"pmc-{pmcid}",
                    "title": record.get("title", "Untitled"),
                    "journal": record.get("journal", ""),
                    "year": record.get("year", ""),
                    "authors": record.get("authors", ""),
                    "url": record.get("url", ""),
                    "pmcid": pmcid,
                    "section": section_name,
                    "snippet": section_text[:300],
                    "detail_snippet": section_text[:800],
                    "source_type": "pubmed_literature",
                    "provider": "Europe PMC / PubMed Central",
                    "query": query,
                })
                memory_entries.append({
                    "text": section_text,
                    "metadata": {
                        "type": "pubmed",
                        "source_type": "pubmed_literature",
                        "pmcid": pmcid,
                        "section": section_name,
                        "title": record.get("title", "Untitled"),
                        "journal": record.get("journal", ""),
                        "year": record.get("year", ""),
                        "authors": record.get("authors", ""),
                        "url": record.get("url", ""),
                        "query": query,
                        "entry_key": entry_key,
                    },
                    "user": self.user,
                    "entry_key": entry_key,
                })

            # Also store the abstract
            abstract = record.get("abstract", "")
            if abstract:
                abs_key = f"{self.user or 'global'}:pmc:{pmcid}:abstract"
                sources.append({
                    "source_id": f"pmc-{pmcid}-abs",
                    "title": record.get("title", "Untitled"),
                    "journal": record.get("journal", ""),
                    "year": record.get("year", ""),
                    "authors": record.get("authors", ""),
                    "url": record.get("url", ""),
                    "pmcid": pmcid,
                    "section": "abstract",
                    "snippet": abstract[:300],
                    "detail_snippet": abstract[:800],
                    "source_type": "pubmed_literature",
                    "provider": "Europe PMC / PubMed Central",
                    "query": query,
                })
                memory_entries.append({
                    "text": abstract,
                    "metadata": {
                        "type": "pubmed",
                        "source_type": "pubmed_literature",
                        "pmcid": pmcid,
                        "section": "abstract",
                        "title": record.get("title", "Untitled"),
                        "journal": record.get("journal", ""),
                        "year": record.get("year", ""),
                        "authors": record.get("authors", ""),
                        "url": record.get("url", ""),
                        "query": query,
                        "entry_key": abs_key,
                    },
                    "user": self.user,
                    "entry_key": abs_key,
                })

        if memory_entries:
            try:
                self.memory.add_entries(memory_entries)
            except Exception as exc:
                print(f"[AgenticLoop] Memory add failed: {exc}")

        return {
            "sources": sources,
            "summary": f"Found {len(sources)} PubMed sources for '{query}'.",
        }

    def _check_drug_interactions(self, medications: List[str]) -> Dict:
        if not medications:
            return {"summary": "No medications provided.", "sources": []}
        try:
            from backend.medication_checker import MedicationChecker
            checker = MedicationChecker()
            result = checker.check_interactions(medications)
            alerts = result.get("alerts", [])
            sources: List[Dict] = []
            for alert in alerts[:4]:
                pair = alert.get("pair", "medication pair")
                summary_text = alert.get("summary", "")
                if summary_text:
                    sources.append({
                        "source_id": f"fda-{pair.replace(' ', '-')[:40]}",
                        "title": f"Drug interaction: {pair}",
                        "snippet": str(summary_text)[:300],
                        "detail_snippet": str(summary_text)[:800],
                        "source_type": "official_guidance",
                        "provider": "openFDA",
                        "url": "https://open.fda.gov/",
                        "query": f"drug interactions {' '.join(medications)}",
                    })
            msg = (
                f"Drug interaction check for {', '.join(medications)}: "
                f"{len(alerts)} alert(s), {len(result.get('resolved_medications', []))} resolved, "
                f"{len(result.get('unresolved_medications', []))} unresolved."
            )
            return {"sources": sources, "summary": msg}
        except Exception as exc:
            return {"summary": f"Drug interaction check failed: {exc}", "sources": []}

    def _search_personal(self, query: str) -> Dict:
        if not query or not self.user:
            return {"summary": "No query or user.", "personal_matches": []}
        matches = self.memory.search(query=query, user=self.user)
        personal: List[Dict] = []
        for entry, score in matches[:4]:
            meta = entry.get("metadata", {})
            if meta.get("type") == "user_summary":
                personal.append({
                    "title": meta.get("title", meta.get("source", "Uploaded document")),
                    "source": meta.get("source", ""),
                    "snippet": build_excerpt(entry.get("text", "")),
                    "score": round(score, 3),
                })
        return {
            "personal_matches": personal,
            "summary": f"Found {len(personal)} matches in patient documents.",
        }

    def _search_trials(self, condition: str, location: str = "United Kingdom") -> Dict:
        if not condition:
            return {"summary": "No condition provided.", "trials": []}
        try:
            from backend.clinical_trials import find_matching_trials
            minimal_profile = {"conditions": [{"name": condition}]}
            results = find_matching_trials(minimal_profile, location=location, max_results=5)
            return {
                "trials": results or [],
                "summary": f"Found {len(results or [])} trials for '{condition}' in {location}.",
            }
        except Exception as exc:
            return {"summary": f"Trial search failed: {exc}", "trials": []}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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
        self.decision_support = ClinicalDecisionSupportEngine()
        self.policy_engine = PolicyEngine()
        self.evidence_ranker = EvidenceRanker()

    def prepare_bundle(
        self,
        question: str,
        user: Optional[str],
        user_profile: dict,
        longitudinal_memory_summary: str,
        medications: Optional[List[Dict]] = None,
        triage_summaries: Optional[List[Dict]] = None,
        allergies: Optional[List[Dict]] = None,
        conditions: Optional[List[Dict]] = None,
        vitals: Optional[List[Dict]] = None,
        context_graph: Optional["ContextGraph"] = None,
    ) -> Dict:
        """
        Full clinical orchestration pipeline.
        Returns a dict compatible with RAGEngine._finalize_answer_payload()
        plus new clinical governance and agentic metadata keys.
        """
        normalized_user = (user or "").strip().lower() or None

        # -- Step 1: Role resolution (instant) --------------------------------
        clinical_role = user_profile.get("clinical_role") or user_profile.get("role", "")
        role_config = self.role_router.resolve(clinical_role)

        # -- Step 2: Patient history context ----------------------------------
        patient_history: PatientHistoryContext = build_patient_history_context(
            longitudinal_memory=longitudinal_memory_summary,
            medications=medications or [],
            triage_summaries=triage_summaries or [],
            user_profile=user_profile,
            allergies=allergies or [],
            conditions=conditions or [],
            vitals=vitals or [],
        )

        # -- Step 3: Crisis pre-screen (regex, instant, before any LLM call) --
        if self.intent_classifier._crisis_prescreen(question):
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # -- Step 4: Moderation -----------------------------------------------
        blocked, category, safe_msg, details = self.moderation.decide(
            question, role_key=role_config.role_key
        )
        if blocked:
            return self._build_moderation_bundle(
                question, normalized_user, safe_msg, category, details, role_config
            )

        # -- Step 5: Intent classification (needed for policy gate) -----------
        history_context = patient_history.as_prompt_block() if not patient_history.is_empty() else ""
        graph_hints: List[str] = list(context_graph.search_hints) if context_graph else []

        try:
            intent = self.intent_classifier.classify(
                question, user_profile, role_config.role_key, patient_history
            )
        except Exception as exc:
            print(f"[Orchestrator] Intent classification failed: {exc}")
            intent = IntentClassification()

        # -- Step 6: Policy gate (8 hard safety gates) ------------------------
        clinical_decision = self.decision_support.assess(question, intent, role_config)
        intent = self.decision_support.apply_to_intent(intent, clinical_decision)

        policy_decision = self.policy_engine.gate(intent, role_config, question, patient_history)
        if policy_decision.action == "escalate_only" and policy_decision.crisis_response:
            return self._build_crisis_bundle(question, normalized_user, role_config)

        # -- Step 7: Pathway context ------------------------------------------
        pathway_context = self._get_pathway_context(intent, role_config)

        # -- Step 8: Agentic retrieval loop -----------------------------------
        # Build a compact patient summary for the agent system prompt
        patient_summary = history_context or f"Role: {role_config.role_key}"
        if graph_hints:
            patient_summary += "\nRelevant health terms: " + ", ".join(graph_hints[:6])

        med_names: List[str] = [
            m.get("name", "") for m in (medications or []) if m.get("name")
        ]

        agent_loop = AgenticRetrievalLoop(
            llm=self.llm,
            official_guidance=self.official_guidance,
            pubmed=self.pubmed,
            memory=self.memory,
            user=normalized_user,
        )

        try:
            agent_result = agent_loop.run(
                question=question,
                patient_summary=patient_summary,
                role_key=role_config.role_key,
                pathway_hint=intent.pathway_hint or "general_triage",
                patient_medications=med_names,
            )
        except Exception as exc:
            print(f"[Orchestrator] Agentic loop failed, using fallback: {exc}")
            agent_result = {
                "collected_sources": [],
                "personal_context": [],
                "trial_results": [],
                "tool_calls_made": [],
            }

        collected_sources: List[Dict] = agent_result.get("collected_sources", [])
        personal_context: List[Dict] = agent_result.get("personal_context", [])
        tool_calls_made: List[Dict] = agent_result.get("tool_calls_made", [])

        # Derive expanded_queries from what the agent actually searched
        expanded_queries: List[str] = list(dict.fromkeys(
            tc["args"].get("query", tc["args"].get("condition", ""))
            for tc in tool_calls_made
            if tc.get("tool") in ("search_nhs_guidance", "search_pubmed", "search_clinical_trials")
            and tc.get("args", {}).get("query") or tc.get("args", {}).get("condition")
        )) or [question]

        # -- Step 9: Fallback retrieval if agent returned nothing -------------
        if not collected_sources:
            print("[Orchestrator] Agent found no sources -- falling back to direct retrieval.")
            fallback_queries = self._build_search_queries(question, history_context, graph_hints)
            search_queries = self._augment_queries_with_pathway(
                fallback_queries, pathway_context, clinical_decision
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                preferred = list(dict.fromkeys(pathway_context.preferred_sources or []))
                official_future = executor.submit(
                    self.official_guidance.search, search_queries, 1, preferred or None
                )
                pubmed_future = executor.submit(
                    self._retrieve_pubmed_for_queries, search_queries, normalized_user
                )
                try:
                    collected_sources = official_future.result()
                except Exception as exc:
                    print(f"[Orchestrator] Fallback NHS search failed: {exc}")
                try:
                    pubmed_future.result()
                except Exception as exc:
                    print(f"[Orchestrator] Fallback PubMed search failed: {exc}")

            # Semantic search for personal context in fallback path
            matches = self.memory.search(query=question, user=normalized_user)
            personal_context, pubmed_matches = self._split_matches(matches)
            collected_sources.extend(self._build_source_briefings(pubmed_matches))
            expanded_queries = fallback_queries

        # -- Step 10: Deduplicate and rank evidence ---------------------------
        raw_sources = self._deduplicate_sources(collected_sources)

        combined_sources, evidence_quality_report = self.evidence_ranker.rank_and_tier_with_report(
            sources=raw_sources,
            question=question,
            role_config=role_config,
            intent=intent,
            memory_store=self.memory,
            top_k=6,
            patient_history=patient_history,
            context_graph=context_graph,
        )

        if combined_sources:
            retrieval_mode = "agentic_multi_source" if tool_calls_made else "live_multi_source"
        elif raw_sources:
            retrieval_mode = "evidence_quality_filtered"
        else:
            retrieval_mode = "general_knowledge"

        # -- Step 11: Evidence dossier (anti-hallucination layer) -------------
        evidence_dossier = None
        if combined_sources:
            try:
                from backend.evidence_extractor import build_evidence_dossier
                evidence_dossier = build_evidence_dossier(
                    llm=self.llm,
                    sources=combined_sources,
                    question=question,
                    user_profile=user_profile,
                    patient_history_ctx=patient_history,
                    medications=medications or [],
                    conditions=conditions or [],
                )
            except Exception as exc:
                print(f"[Orchestrator] Evidence dossier build failed (non-fatal): {exc}")

        # -- Step 12: Build role-aware LLM context ----------------------------
        full_context = self._build_role_context(
            combined_sources=combined_sources,
            personal_context=personal_context,
            policy_decision=policy_decision,
            pathway_context=pathway_context,
            clinical_decision=clinical_decision,
            evidence_quality_report=evidence_quality_report,
            no_sources=not combined_sources,
            evidence_dossier=evidence_dossier,
        )

        return {
            "kind": "answer",
            # Backward-compatible keys
            "normalized_user": normalized_user,
            "user_profile": user_profile,
            "combined_sources": combined_sources,
            "personal_context": personal_context,
            "longitudinal_memory_summary": longitudinal_memory_summary,
            "expanded_queries": expanded_queries,
            "matches": [],
            "retrieval_mode": retrieval_mode,
            "full_context": full_context,
            "evidence_quality_report": evidence_quality_report,
            # Clinical governance
            "role_config": role_config,
            "intent": intent,
            "policy_decision": policy_decision,
            "pathway_context": pathway_context,
            "clinical_decision": clinical_decision,
            # Structured evidence (anti-hallucination layer)
            "evidence_dossier": evidence_dossier,
            # Agentic metadata (new)
            "agentic_tool_calls": tool_calls_made,
        }

    # -- Bundle builders ------------------------------------------------------

    def _build_crisis_bundle(
        self,
        question: str,
        normalized_user: Optional[str],
        role_config: RoleConfig,
    ) -> Dict:
        return {
            "kind": "final",
            "payload": {
                "answer_markdown": CRISIS_RESPONSE,
                "answer_text": CRISIS_RESPONSE,
                "sources": [],
                "personal_context": [],
                "trace": {
                    "trace_id": "trace-crisis",
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
                    "trace_id": "trace-mod",
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

    # -- Context builders -----------------------------------------------------

    def _build_role_context(
        self,
        combined_sources: List[Dict],
        personal_context: List[Dict],
        policy_decision: PolicyDecision,
        pathway_context,
        clinical_decision: ClinicalDecision,
        evidence_quality_report: Optional[Dict] = None,
        no_sources: bool = False,
        evidence_dossier=None,
    ) -> str:
        parts = []

        if personal_context:
            personal_lines = "\n".join(
                f"- {item['title']}: {item['snippet']}" for item in personal_context
            )
            parts.append(f"Personal context:\n{personal_lines}")

        if clinical_decision:
            decision_lines = [
                f"- Pathway: {clinical_decision.pathway_label}",
                f"- Urgency: {clinical_decision.urgency_level}",
                f"- Primary action: {clinical_decision.next_step}",
                f"- Summary: {clinical_decision.summary}",
            ]
            decision_lines.extend(
                f"- Immediate action: {item}"
                for item in clinical_decision.immediate_actions[:4]
            )
            decision_lines.extend(
                f"- Monitor now: {item}"
                for item in clinical_decision.monitoring_priorities[:3]
            )
            if clinical_decision.triggered_rules:
                decision_lines.extend(
                    f"- Rule hit: {item.finding}"
                    for item in clinical_decision.triggered_rules
                )
            parts.append(
                "Deterministic clinical decision support output (must not be contradicted):\n"
                + "\n".join(decision_lines)
            )

        if policy_decision.context_notes:
            notes = "\n".join(policy_decision.context_notes)
            parts.append(f"Clinical policy notes (must be followed):\n{notes}")

        if pathway_context and pathway_context.safety_rules:
            rules = "\n".join(f"- {r}" for r in pathway_context.safety_rules)
            parts.append(f"Pathway safety rules:\n{rules}")

        if evidence_quality_report:
            quality_lines = [
                f"- Overall status: {evidence_quality_report.get('overall_status', 'unknown')}",
                f"- Accepted sources: {evidence_quality_report.get('accepted_source_count', 0)}",
                f"- Excluded sources: {evidence_quality_report.get('excluded_source_count', 0)}",
            ]
            profile_facts = evidence_quality_report.get("profile_facts_checked") or []
            if profile_facts:
                quality_lines.append(
                    "- Profile facts checked: " + "; ".join(str(f) for f in profile_facts[:8])
                )
            status_counts = evidence_quality_report.get("status_counts") or {}
            if status_counts:
                counts_text = ", ".join(f"{k}={v}" for k, v in status_counts.items())
                quality_lines.append(f"- Source usability counts: {counts_text}")
            for item in (evidence_quality_report.get("excluded_sources") or [])[:3]:
                reasons = "; ".join(str(r) for r in item.get("reasons", [])[:2])
                quality_lines.append(
                    f"- Filtered out: {item.get('title', 'Source')} ({reasons})"
                )
            quality_lines.append(
                "- Binding rule: use patient_aligned sources for patient-specific guidance; "
                "use question_aligned or background_only sources only for general context."
            )
            parts.append("Evidence quality gate:\n" + "\n".join(quality_lines))

        if evidence_dossier and evidence_dossier.articles:
            parts.append(
                "Structured patient-aligned evidence dossier "
                "(extracted facts matched to this patient -- do not cite facts not present here):\n"
                + evidence_dossier.to_prompt_context()
            )
        elif combined_sources:
            evidence_parts = []
            for source in combined_sources:
                tier = source.get("evidence_tier", 3)
                tier_label = source.get("tier_label", f"Tier {tier}")
                snippet = source.get("detail_snippet") or source.get("snippet", "")
                quality_status = source.get("evidence_quality_status", "question_aligned")
                use_label = (
                    "patient-specific guidance"
                    if source.get("usable_for_patient_specific_guidance")
                    else "general/background context"
                )
                quality_notes = "; ".join(
                    str(r) for r in source.get("evidence_quality_reasons", [])[:2]
                )
                evidence_parts.append(
                    f"[{tier_label}] {source.get('title', 'Source')} "
                    f"(quality: {quality_status}; use: {use_label}): {snippet}"
                    + (f"\nQuality notes: {quality_notes}" if quality_notes else "")
                )
            parts.append(
                "Biomedical evidence (tiered by source authority):\n" + "\n\n".join(evidence_parts)
            )
        elif no_sources:
            filtered = (
                evidence_quality_report
                and evidence_quality_report.get("overall_status") == "no_sources_passed_quality_gate"
            )
            if filtered:
                parts.append(
                    "Note: Retrieved evidence was found but did not pass the evidence-quality gate. "
                    "Do not cite filtered sources. State that patient-aligned live evidence was not "
                    "available and keep guidance safety-oriented and grounded in the clinical pathway."
                )
            else:
                parts.append(
                    "Note: No live evidence was retrieved for this query. "
                    "Answer from general clinical knowledge, clearly indicating this is general guidance "
                    "and not based on retrieved literature. Still provide a clear disposition and "
                    "concrete next-step management plan where possible."
                )

        return "\n\n".join(parts)

    # -- Pathway routing ------------------------------------------------------

    def _get_pathway_context(self, intent: IntentClassification, role_config: RoleConfig):
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
            print(f"[Orchestrator] Pathway load failed ({hint}): {exc}")
            from backend.pathways.general_triage import get_pathway_context
            return get_pathway_context(intent, role_config)

    # -- Fallback query helpers -----------------------------------------------

    def _build_search_queries(
        self,
        question: str,
        patient_history_context: str = "",
        graph_hints: Optional[List[str]] = None,
    ) -> List[str]:
        queries = [question]
        try:
            if patient_history_context:
                queries.extend(
                    self.query_expander.expand_with_patient_context(question, patient_history_context)
                )
            else:
                queries.extend(self.query_expander.expand(question))
        except Exception as exc:
            print(f"[Orchestrator] Query expansion failed: {exc}")
        for hint in (graph_hints or []):
            if hint and hint not in queries:
                queries.append(hint)
        return list(dict.fromkeys(q for q in queries if q))[:5]

    def _augment_queries_with_pathway(
        self,
        queries: List[str],
        pathway_context,
        clinical_decision: Optional[ClinicalDecision] = None,
    ) -> List[str]:
        augmented = list(queries)
        if pathway_context and pathway_context.additional_search_terms:
            for term in pathway_context.additional_search_terms[:2]:
                combined = f"{queries[0]} {term}"
                if combined not in augmented:
                    augmented.append(combined)
        if clinical_decision:
            for term in clinical_decision.search_terms[:2]:
                if term not in augmented:
                    augmented.append(term)
        return augmented[:5]

    # -- Source processing helpers --------------------------------------------

    @staticmethod
    def _deduplicate_sources(sources: List[Dict]) -> List[Dict]:
        seen: set = set()
        deduped: List[Dict] = []
        for source in sources:
            key = (
                source.get("url")
                or source.get("pmcid")
                or f"{source.get('title', '')}::{source.get('section', '')}"
            )
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(dict(source))
        for idx, source in enumerate(deduped, start=1):
            source["source_id"] = f"S{idx}"
        return deduped

    def _retrieve_pubmed_for_queries(
        self, queries: List[str], user: Optional[str]
    ) -> None:
        """Fallback: fetch PubMed articles and add to memory store."""
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
                    print(f"[Orchestrator] PubMed search failed for '{query}': {exc}")

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
                    print(f"[Orchestrator] PubMed section fetch failed: {exc}")
                    sections = {}

                best_name, best_text = self._select_best_pubmed_section(sections)
                if best_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:{best_name}"
                    pending_entries.append({
                        "text": best_text,
                        "metadata": {
                            "type": "pubmed",
                            "source_type": "pubmed_literature",
                            "pmcid": record["pmcid"],
                            "section": best_name,
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

                abstract = record.get("abstract", "")
                if abstract:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:abstract"
                    pending_entries.append({
                        "text": abstract,
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
        seen: set = set()
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
                + "\n".join(
                    f"- {item['title']}: {item['snippet']}" for item in personal_context
                )
            )
        if role_config.role_key in ("doctor", "nurse", "midwife", "physiotherapist"):
            return (
                "## Evidence Retrieval\n"
                "Insufficient live evidence was retrieved for this query. "
                "Please consult current local guidelines, BNF, or NICE CKS directly.\n\n"
                "## Recommended Action\n"
                "Use the relevant local pathway or guideline now, or rephrase the query "
                "with the exact condition, drug, population, or decision point you need."
                + personal_note
            )
        return (
            "## Working Impression\n"
            "I could not retrieve enough reliable live evidence for this question right now.\n\n"
            "## What To Do Now\n"
            "Please narrow the question to a specific symptom, condition, treatment, or population, "
            "or contact a clinician directly if this affects a decision that needs to be made now."
            + personal_note
        )


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
