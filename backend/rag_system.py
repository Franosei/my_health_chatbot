from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from uuid import uuid4
import re

import numpy as np

from backend.anonymizer import DocumentAnonymizer
from backend.clinical_orchestrator import ClinicalOrchestrator
from backend.image_generator import ImageGenerator
from backend.memory_store import MemoryStore
from backend.video_generator import VideoGenerator
from backend.moderation_ml import ModerationEnsemble
from backend.official_guidance import OfficialGuidanceEngine
from backend.pubmed_search import PubMedCentralSearcher
from backend.query_expander import QueryExpander
from backend.summarizer import LLMHelper
from backend.user_store import UserStore
from backend.utils import build_excerpt, extract_text_from_pdf


class RAGEngine:
    """
    Retrieval-augmented engine that combines user context, PubMed evidence, and
    audit metadata for a professional clinical chat experience.
    """

    def __init__(self, embedding_dir: str = "data/uploads"):
        self.embedding_dir = Path(embedding_dir)
        self.query_expander = QueryExpander()
        self.memory = MemoryStore()
        self.pubmed = PubMedCentralSearcher()
        self.anonymizer = DocumentAnonymizer()
        self.llm = LLMHelper()
        self.moderation = ModerationEnsemble()
        self.official_guidance = OfficialGuidanceEngine()
        self._primed_users: set[str] = set()
        self._orchestrator = ClinicalOrchestrator(
            memory=self.memory,
            pubmed=self.pubmed,
            official_guidance=self.official_guidance,
            llm=self.llm,
            query_expander=self.query_expander,
            moderation=self.moderation,
        )
        self._image_generator = ImageGenerator()
        self._video_generator = VideoGenerator()

    def restore_user_context(self, user: Optional[str]) -> None:
        """
        Restores persisted user document summaries into memory once per session.
        """
        if not user:
            return

        normalized_user = user.strip().lower()
        if normalized_user in self._primed_users:
            return

        pending_entries = []
        for summary_record in UserStore.get_document_summaries(normalized_user):
            summary_text = summary_record.get("summary", "").strip()
            if not summary_text:
                continue

            filename = summary_record.get("file", "uploaded document")
            pending_entries.append(
                {
                    "text": summary_text,
                    "metadata": {
                    "type": "user_summary",
                    "source": filename,
                    "title": f"User-uploaded record: {filename}",
                    "section": "document summary",
                    "stored_path": summary_record.get("stored_path", ""),
                    "entry_key": f"{normalized_user}:upload:{filename}",
                },
                    "user": normalized_user,
                    "entry_key": f"{normalized_user}:upload:{filename}",
                }
            )

        self.memory.add_entries(pending_entries)
        self._primed_users.add(normalized_user)

    def ingest_documents(
        self,
        user: Optional[str] = None,
        file_paths: Optional[List[Path]] = None,
    ) -> List[Dict]:
        """
        Loads uploaded documents, anonymizes them, summarizes them, and persists
        a retrieval-friendly document summary per user.
        """
        normalized_user = user.strip().lower() if user else None
        documents = [Path(path) for path in (file_paths or self._default_document_paths(normalized_user))]
        indexed_documents = []
        known_uploads = {
            item.get("file")
            for item in UserStore.get_uploads(normalized_user)
        } if normalized_user else set()

        for path in documents:
            if not path.exists() or path.suffix.lower() != ".pdf":
                continue

            raw_text = extract_text_from_pdf(path)
            anonymized = self.anonymizer.anonymize(raw_text)
            summary = self.llm.summarize_user_health_record(anonymized)

            memory_key = f"{normalized_user or 'global'}:upload:{path.name}"
            self.memory.add_entry(
                text=summary,
                metadata={
                    "type": "user_summary",
                    "source": path.name,
                    "title": f"User-uploaded record: {path.name}",
                    "section": "document summary",
                    "stored_path": str(path),
                    "entry_key": memory_key,
                },
                user=normalized_user,
                entry_key=memory_key,
            )

            if normalized_user:
                if path.name not in known_uploads:
                    UserStore.add_upload(normalized_user, path.name, stored_path=str(path))
                    known_uploads.add(path.name)
                UserStore.save_document_summary(
                    normalized_user,
                    path.name,
                    summary,
                    stored_path=str(path),
                )

            indexed_documents.append(
                {
                    "file": path.name,
                    "stored_path": str(path),
                    "summary": summary,
                }
            )

        if normalized_user and indexed_documents:
            self.refresh_longitudinal_memory_from_documents(
                user=normalized_user,
                indexed_documents=indexed_documents,
            )
            self._primed_users.add(normalized_user)

        return indexed_documents

    def handle_user_question(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        stream: bool = False,
        user: Optional[str] = None,
    ) -> Dict:
        """
        Responds to a user query with a structured payload that includes answer markdown,
        clickable sources, personal context traceability, and audit metadata.
        """
        del stream
        bundle = self._prepare_answer_bundle(question=question, user=user)
        if bundle["kind"] == "final":
            return bundle["payload"]

        _pd = bundle.get("policy_decision")
        raw_answer = self.llm.answer_question(
            question=question,
            context=bundle["full_context"],
            chat_history=chat_history,
            stream=False,
            user_profile=bundle["user_profile"],
            source_briefings=bundle["combined_sources"],
            longitudinal_memory=bundle["longitudinal_memory_summary"],
            role_config=bundle.get("role_config"),
            escalation_banner=_pd.escalation_banner if _pd else "",
            policy_context_note="\n".join(_pd.context_notes) if _pd else "",
        )
        return self._finalize_answer_payload(question=question, raw_answer=raw_answer, bundle=bundle)

    def stream_user_question_events(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        user: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        Streams retrieval progress events and final answer tokens so the UI can
        show live search status followed by incremental generation.
        """
        yield {
            "type": "status",
            "message": "Searching live guidance, Europe PMC, and your saved context...",
        }

        # Detect if an illustration or video is needed early (fast regex, before retrieval)
        needs_illustration = self._image_generator.detect_illustration_need(question)
        needs_video = self._video_generator.detect_video_request(question)
        # Video takes priority over static illustration when both match
        if needs_video:
            needs_illustration = False

        bundle = self._prepare_answer_bundle(question=question, user=user)
        if bundle["kind"] == "final":
            yield {"type": "final", "payload": bundle["payload"]}
            return

        yield {
            "type": "status",
            "message": "Composing the answer from the retrieved evidence...",
        }
        streamed_chunks: List[str] = []
        policy_decision = bundle.get("policy_decision")
        for chunk in self.llm.answer_question(
            question=question,
            context=bundle["full_context"],
            chat_history=chat_history,
            stream=True,
            user_profile=bundle["user_profile"],
            source_briefings=bundle["combined_sources"],
            longitudinal_memory=bundle["longitudinal_memory_summary"],
            role_config=bundle.get("role_config"),
            escalation_banner=policy_decision.escalation_banner if policy_decision else "",
            policy_context_note="\n".join(policy_decision.context_notes) if policy_decision else "",
        ):
            streamed_chunks.append(chunk)
            yield {"type": "token", "delta": chunk}

        raw_answer = "".join(streamed_chunks).strip()

        # Generate illustration or video after streaming (non-blocking for tokens)
        illustration = None
        video_result = None
        video_rate_limit_msg = ""

        if needs_video and user:
            from backend.user_store import UserStore as _US
            last_video_at = _US.get_last_video_generated_at(user)
            rate = self._video_generator.check_rate_limit(last_video_at)
            if not rate.allowed:
                video_rate_limit_msg = rate.message
            else:
                yield {"type": "status", "message": "Generating Sora-2 video (this may take a moment)..."}
                try:
                    video_result = self._video_generator.generate_video(
                        question=question,
                        context_answer=raw_answer[:400],
                    )
                    if video_result:
                        _US.record_video_generated(user)
                except Exception as exc:
                    print(f"Video generation failed: {exc}")

        elif needs_illustration:
            yield {"type": "status", "message": "Generating illustration..."}
            try:
                illustration = self._image_generator.generate_illustration(
                    question=question,
                    context_answer=raw_answer[:400],
                )
            except Exception as exc:
                print(f"Illustration generation failed: {exc}")

        payload = self._finalize_answer_payload(
            question=question,
            raw_answer=raw_answer,
            bundle=bundle,
        )
        if illustration:
            payload["image_url"] = illustration.image_url
            payload["image_caption"] = illustration.caption
        if video_result:
            payload["video_url"] = video_result.video_url
            payload["video_caption"] = video_result.caption
        if video_rate_limit_msg:
            payload["video_rate_limit_msg"] = video_rate_limit_msg

        yield {"type": "final", "payload": payload}

    def _prepare_answer_bundle(self, question: str, user: Optional[str] = None) -> Dict:
        normalized_user = user.strip().lower() if user else None
        self.restore_user_context(normalized_user)
        longitudinal_memory = UserStore.get_longitudinal_memory(normalized_user) if normalized_user else {}
        longitudinal_memory_summary = (longitudinal_memory.get("summary") or "").strip()
        user_profile = UserStore.get_user_profile(normalized_user) if normalized_user else {}

        bundle = self._orchestrator.prepare_bundle(
            question=question,
            user=normalized_user,
            user_profile=user_profile,
            longitudinal_memory_summary=longitudinal_memory_summary,
        )
        return bundle

    def _build_moderation_payload(
        self,
        question: str,
        normalized_user: Optional[str],
        safe_msg: str,
        category: str,
        details: Dict,
    ) -> Dict:
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": safe_msg[:280],
            "sources": [],
            "retrieval_mode": "moderation_block",
            "moderation_category": category,
            "moderation_details": details,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": safe_msg,
            "answer_text": safe_msg,
            "sources": [],
            "personal_context": [],
            "trace": trace,
        }

    def _build_limited_payload(
        self,
        question: str,
        normalized_user: Optional[str],
        personal_context: List[Dict],
        retrieval_mode: str,
        expanded_queries: List[str],
    ) -> Dict:
        limited_answer = self._build_limited_evidence_response(personal_context)
        trace_id = f"trace-{uuid4().hex[:12]}"
        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": limited_answer[:280],
            "sources": [],
            "personal_context": personal_context,
            "retrieval_mode": retrieval_mode,
            "expanded_queries": expanded_queries,
            "model": self.llm.model,
        }
        if normalized_user:
            UserStore.save_interaction_trace(normalized_user, trace)
        return {
            "answer_markdown": limited_answer,
            "answer_text": limited_answer,
            "sources": [],
            "personal_context": personal_context,
            "trace": trace,
        }

    def _finalize_answer_payload(self, question: str, raw_answer: str, bundle: Dict) -> Dict:
        answer_markdown = self._link_citations(raw_answer, bundle["combined_sources"])

        # Prepend escalation banner to answer if policy triggered one
        policy_decision = bundle.get("policy_decision")
        if policy_decision and policy_decision.escalation_banner:
            answer_markdown = policy_decision.escalation_banner + answer_markdown

        # Append disclaimer
        if policy_decision and policy_decision.disclaimer:
            answer_markdown = answer_markdown + policy_decision.disclaimer

        # Append vulnerability notice near top if applicable
        if policy_decision and policy_decision.vulnerability_notice:
            answer_markdown = policy_decision.vulnerability_notice + answer_markdown

        trace_id = f"trace-{uuid4().hex[:12]}"
        intent = bundle.get("intent")
        role_config = bundle.get("role_config")

        from backend.evidence_ranker import EvidenceRanker
        tiers_present = EvidenceRanker.get_tiers_present(bundle["combined_sources"])

        trace = {
            "trace_id": trace_id,
            "created_at": self._utc_now(),
            "question": question,
            "answer_preview": raw_answer[:280],
            "sources": bundle["combined_sources"],
            "personal_context": bundle["personal_context"],
            "retrieval_mode": bundle["retrieval_mode"],
            "expanded_queries": bundle["expanded_queries"],
            "memory_match_count": len(bundle["matches"]),
            "model": self.llm.model,
            # Clinical governance fields
            "role_key": role_config.role_key if role_config else "patient",
            "intent_category": intent.intent_category if intent else "",
            "risk_level": intent.risk_level if intent else "routine",
            "escalation_triggered": bool(policy_decision and policy_decision.action != "allow"),
            "crisis_detected": intent.crisis_detected if intent else False,
            "evidence_tiers_present": tiers_present,
            "pathway_used": intent.pathway_hint if intent else "",
            "vulnerable_flags": intent.vulnerable_flags if intent else [],
            "policy_gates_applied": policy_decision.gates_as_dicts() if policy_decision else [],
        }
        if bundle["normalized_user"]:
            UserStore.save_interaction_trace(bundle["normalized_user"], trace)
        return {
            "answer_markdown": answer_markdown,
            "answer_text": raw_answer,
            "sources": bundle["combined_sources"],
            "personal_context": bundle["personal_context"],
            "longitudinal_memory": bundle["longitudinal_memory_summary"],
            "trace": trace,
        }

    def refresh_longitudinal_memory_from_turn(
        self,
        user: Optional[str],
        user_message: str,
        personal_context: Optional[List[Dict]] = None,
    ) -> str:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user:
            return ""

        new_information = self._build_longitudinal_memory_turn_input(
            user_message=user_message,
            personal_context=personal_context or [],
        )
        return self._refresh_longitudinal_memory(
            user=normalized_user,
            new_information=new_information,
            source_label="conversation",
        )

    def refresh_longitudinal_memory_from_documents(
        self,
        user: Optional[str],
        indexed_documents: List[Dict],
    ) -> str:
        normalized_user = user.strip().lower() if user else None
        if not normalized_user or not indexed_documents:
            return ""

        new_information = "\n\n".join(
            f"{item.get('file', 'Document')}:\n{item.get('summary', '').strip()}"
            for item in indexed_documents
            if item.get("summary", "").strip()
        )
        return self._refresh_longitudinal_memory(
            user=normalized_user,
            new_information=new_information,
            source_label="uploaded documents",
        )

    def _refresh_longitudinal_memory(
        self,
        user: str,
        new_information: str,
        source_label: str,
    ) -> str:
        cleaned_information = (new_information or "").strip()
        if not cleaned_information:
            return ""

        existing_memory = UserStore.get_longitudinal_memory(user)
        existing_summary = (existing_memory.get("summary") or "").strip()
        updated_summary = self.llm.refresh_longitudinal_memory(
            existing_memory=existing_summary,
            new_information=cleaned_information,
            user_profile=UserStore.get_user_profile(user),
            source_label=source_label,
        )
        normalized_summary = self._normalize_longitudinal_memory_summary(updated_summary)
        UserStore.save_longitudinal_memory(
            user,
            normalized_summary,
            source=source_label,
            metadata={
                "input_length": len(cleaned_information),
                "summary_length": len(normalized_summary),
            },
        )
        return normalized_summary

    def _default_document_paths(self, user: Optional[str]) -> List[Path]:
        if user:
            upload_dir = UserStore.get_upload_dir(user)
            return sorted(upload_dir.glob("*.pdf"))
        if not self.embedding_dir.exists():
            return []
        return sorted(self.embedding_dir.glob("*.pdf"))

    def _retrieve_pubmed_for_queries(self, queries: List[str], user: Optional[str]) -> None:
        pending_entries = []
        article_batches: List[Tuple[str, List[Dict[str, str]]]] = []

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

        article_records: List[Tuple[str, Dict[str, str]]] = []
        for query, records in article_batches:
            for record in records:
                article_records.append((query, record))

        with ThreadPoolExecutor(max_workers=min(6, max(1, len(article_records)))) as executor:
            section_futures = {
                executor.submit(self.pubmed.fetch_article_sections, record["pmcid"]): (query, record)
                for query, record in article_records
            }
            for future, payload in section_futures.items():
                query, record = payload
                try:
                    sections = future.result()
                except Exception as exc:
                    print(f"PubMed full-text fetch failed for {record.get('pmcid', '')}: {exc}")
                    sections = {}

                best_section_name, best_section_text = self._select_best_pubmed_section(sections)
                if best_section_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:{best_section_name}"
                    pending_entries.append(
                        {
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
                        }
                    )

                abstract_text = record.get("abstract", "")
                if abstract_text:
                    entry_key = f"{user or 'global'}:pmc:{record['pmcid']}:abstract"
                    pending_entries.append(
                        {
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
                        }
                    )

        self.memory.add_entries(pending_entries)

    def _split_matches(self, matches: List[Tuple[Dict, float]]) -> Tuple[List[Dict], List[Tuple[Dict, float]]]:
        personal_context = []
        pubmed_matches = []

        for entry, score in matches:
            metadata = entry.get("metadata", {})
            if metadata.get("type") == "user_summary":
                personal_context.append(
                    {
                        "title": metadata.get("title", metadata.get("source", "Uploaded document")),
                        "source": metadata.get("source", ""),
                        "snippet": build_excerpt(entry.get("text", "")),
                        "score": round(score, 3),
                    }
                )
            elif metadata.get("type") == "pubmed":
                pubmed_matches.append((entry, score))

        return personal_context[:2], pubmed_matches[:4]

    def _build_source_briefings(self, matches: List[Tuple[Dict, float]]) -> List[Dict]:
        sources = []
        seen = set()

        for entry, score in matches:
            metadata = entry.get("metadata", {})
            key = (metadata.get("pmcid"), metadata.get("section"))
            if key in seen:
                continue
            seen.add(key)
            source_id = f"S{len(sources) + 1}"
            sources.append(
                {
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
                }
            )

        return sources

    @staticmethod
    def _combine_sources(pubmed_sources: List[Dict], official_sources: List[Dict]) -> List[Dict]:
        combined = []
        seen = set()

        for source in [*official_sources, *pubmed_sources]:
            key = source.get("url") or f"{source.get('title')}::{source.get('section')}"
            if key in seen:
                continue
            seen.add(key)
            combined.append(dict(source))

        for index, source in enumerate(combined, start=1):
            source["source_id"] = f"S{index}"
        return combined

    def _rank_sources(self, question: str, sources: List[Dict], top_k: int = 6) -> List[Dict]:
        if not sources:
            return []

        query_vector = self.memory._embed_text(question)
        source_texts = [
            (
                " ".join(
                    part
                    for part in (
                        source.get("title", ""),
                        source.get("section", ""),
                        source.get("detail_snippet", ""),
                        source.get("snippet", ""),
                        source.get("query", ""),
                    )
                    if part
                )
                or source.get("title", "Retrieved source")
            )
            for source in sources
        ]
        source_vectors = self.memory._embed_texts(source_texts)
        scored_sources = []
        for source, source_vector in zip(sources, source_vectors):
            score = float(np.dot(query_vector, source_vector))
            payload = dict(source)
            payload["relevance"] = round(score, 3)
            scored_sources.append(payload)

        scored_sources.sort(key=lambda item: item.get("relevance", 0.0), reverse=True)
        ranked = scored_sources[:top_k]
        for index, source in enumerate(ranked, start=1):
            source["source_id"] = f"S{index}"
        return ranked

    def _build_search_queries(self, question: str) -> List[str]:
        queries = [question]
        try:
            queries.extend(self.query_expander.expand(question))
        except Exception as exc:
            print(f"Query expansion fallback: {exc}")
        return list(dict.fromkeys(query for query in queries if query))[:3]

    @staticmethod
    def _build_longitudinal_memory_turn_input(
        user_message: str,
        personal_context: List[Dict],
    ) -> str:
        parts = []
        cleaned_message = (user_message or "").strip()
        if cleaned_message:
            parts.append(f"Latest user message:\n{cleaned_message}")

        if personal_context:
            context_lines = [
                f"- {item.get('title', item.get('source', 'Context'))}: {item.get('snippet', '').strip()}"
                for item in personal_context
                if item.get("snippet", "").strip()
            ]
            if context_lines:
                parts.append("Relevant patient-specific context already on file:\n" + "\n".join(context_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _select_best_pubmed_section(sections: Dict[str, str]) -> Tuple[str, str]:
        for key in ("discussion", "conclusion", "introduction"):
            text = (sections.get(key) or "").strip()
            if text:
                return key, text
        return "", ""

    @staticmethod
    def _normalize_longitudinal_memory_summary(summary: str) -> str:
        cleaned = " ".join((summary or "").split()).strip()
        if cleaned.lower() == "no durable patient-specific memory recorded yet.":
            return ""
        return summary.strip()

    @staticmethod
    def _build_personal_context_text(personal_context: List[Dict]) -> str:
        if not personal_context:
            return ""

        return "\n".join(
            f"- {item['title']}: {item['snippet']}"
            for item in personal_context
        )

    @staticmethod
    def _link_citations(answer_text: str, sources: List[Dict]) -> str:
        source_map = {
            source["source_id"]: source.get("url")
            for source in sources
            if source.get("source_id")
        }

        def replace_match(match: re.Match) -> str:
            source_id = match.group(1)
            url = source_map.get(source_id)
            if not url:
                return f"`[{source_id}]`"
            return f"[{source_id}]({url})"

        linked_answer = re.sub(r"\[(S\d+)\]", replace_match, answer_text)
        if sources and linked_answer == answer_text:
            fallback_links = []
            for source in sources:
                source_id = source.get("source_id", "Source")
                url = source.get("url")
                if url:
                    fallback_links.append(f"[{source_id}]({url})")
                else:
                    fallback_links.append(f"`[{source_id}]`")
            linked_answer += "\n\n**Sources used:** " + ", ".join(fallback_links)
        return linked_answer

    @staticmethod
    def _build_limited_evidence_response(personal_context: List[Dict]) -> str:
        personal_note = ""
        if personal_context:
            personal_note = (
                "\n\n## Available Personal Context\n"
                + "\n".join(f"- {item['title']}: {item['snippet']}" for item in personal_context)
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

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()
