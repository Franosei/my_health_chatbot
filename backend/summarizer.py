from typing import Generator, Optional
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMHelper:
    """
    Wrapper around OpenAI's Chat Completions API for question answering and summarization.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key)

    def answer_question(
        self,
        question: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
        stream: bool = False,
        user_profile: Optional[dict] = None,
        source_briefings: Optional[list[dict]] = None,
    ) -> str | Generator[str, None, None]:
        """
        Creates a professional, domain-aware response using the supplied evidence dossier.
        The assistant is instructed to cite claims inline with source markers like [S1].
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Dr. Charlotte, a senior clinical information specialist supporting "
                    "individual health users, caregivers, and hospital or ambulatory teams. "
                    "You provide polished, evidence-grounded explanations without replacing a treating clinician.\n\n"
                    "Rules:\n"
                    "1. Use only the supplied evidence dossier and conversation context.\n"
                    "2. Use concise markdown with practical section headings.\n"
                    "3. Cite factual claims inline using the provided source markers like [S1] or [S1][S2].\n"
                    "4. If evidence is limited or conflicting, say so explicitly.\n"
                    "5. Do not diagnose or make unsafe medication changes.\n"
                    "6. Include escalation guidance if symptoms sound urgent.\n"
                    "7. Synthesize across sources rather than copying any single source.\n"
                    "8. For symptom triage, diagnosis, or initial workup questions, prioritize trusted official guidance "
                    "and use literature to add nuance or detail.\n"
                    "9. Keep the tone appropriate for a premium, client-facing digital health application."
                ),
            }
        ]

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Recent conversation:\n{self._render_chat_history(chat_history)}\n\n"
                    f"Evidence dossier:\n{self._render_evidence_dossier(source_briefings, context)}\n\n"
                    f"Current question:\n{question}\n\n"
                    "Write the answer using these headings whenever helpful:\n"
                    "## Clinical Takeaway\n"
                    "## What This Means In Practice\n"
                    "## Evidence Snapshot\n"
                    "## Recommended Next Step\n"
                    "## Safety Note\n\n"
                    "Every evidence-based statement should include one or more source markers.\n"
                    "Where multiple sources agree, synthesize them into one clearer statement with combined citations."
                ),
            }
        )

        return self._stream_response(messages) if stream else self._complete_response(messages)

    def summarize_user_health_record(self, record_text: str) -> str:
        """
        Summarizes an anonymized health document into a retrieval-friendly clinical overview.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are preparing an intake summary from an anonymized health document. "
                    "Capture diagnoses, therapies, abnormal findings, timelines, and care priorities "
                    "that would help a medical evidence system retrieve relevant literature."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Health document:\n{record_text}\n\n"
                    "Produce one short clinical paragraph followed by a compact plain-text list."
                ),
            },
        ]
        return self._complete_response(messages)

    def _complete_response(self, messages) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _stream_response(self, messages) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    @staticmethod
    def _render_chat_history(chat_history: Optional[list[dict]]) -> str:
        if not chat_history:
            return "No prior conversation."

        lines = []
        for message in chat_history[-6:]:
            role = message.get("role", "user").title()
            content = message.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No prior conversation."

    @staticmethod
    def _render_profile_summary(user_profile: Optional[dict]) -> str:
        if not user_profile:
            return "No additional user profile available."

        fragments = []
        for field in (
            "display_name",
            "role",
            "care_context",
            "organization",
            "follow_up_preferences",
        ):
            value = (user_profile.get(field) or "").strip()
            if value:
                fragments.append(f"{field.replace('_', ' ').title()}: {value}")
        return "\n".join(fragments) if fragments else "No additional user profile available."

    @staticmethod
    def _render_evidence_dossier(source_briefings: Optional[list[dict]], fallback_context: str) -> str:
        if source_briefings:
            blocks = []
            for source in source_briefings:
                blocks.append(
                    "\n".join(
                        [
                            f"[{source['source_id']}] {source.get('title', 'Untitled article')}",
                            f"Source type: {source.get('source_type', 'evidence source')}",
                            f"Provider: {source.get('provider', source.get('journal', 'Unknown provider'))}",
                            f"Journal: {source.get('journal', 'Unknown journal')}",
                            f"Year: {source.get('year', 'Unknown year')}",
                            f"Section: {source.get('section', 'Retrieved text')}",
                            f"Relevance: {source.get('relevance', source.get('similarity', 'n/a'))}",
                            f"Evidence: {source.get('detail_snippet', source.get('snippet', source.get('evidence', '')))}",
                        ]
                    )
                )
            return "\n\n".join(blocks)

        return fallback_context or "No biomedical evidence was retrieved."
