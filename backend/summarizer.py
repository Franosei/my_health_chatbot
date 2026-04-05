from typing import Generator, Optional, TYPE_CHECKING
import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.product_config import PRODUCT_NAME

if TYPE_CHECKING:
    from backend.role_router import RoleConfig

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
        longitudinal_memory: Optional[str] = None,
        role_config: Optional["RoleConfig"] = None,
        escalation_banner: str = "",
        policy_context_note: str = "",
    ) -> str | Generator[str, None, None]:
        """
        Creates a role-aware, evidence-grounded response using the supplied evidence dossier.
        The assistant is instructed to cite claims inline with source markers like [S1].
        """
        # Build role-specific system persona
        if role_config:
            from backend.response_templates import get_persona_block
            persona = get_persona_block(role_config.role_key)
        else:
            persona = (
                f"You are {PRODUCT_NAME}, a senior clinical information specialist supporting "
                "individual health users, caregivers, and hospital or ambulatory teams. "
                "You provide polished, evidence-grounded explanations without replacing a treating clinician."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{persona}\n\n"
                    "Rules:\n"
                    "1. Use only the supplied evidence dossier and conversation context.\n"
                    "2. Use concise markdown with the role-appropriate section headings provided.\n"
                    "3. Cite factual claims inline using the provided source markers like [S1] or [S1][S2].\n"
                    "4. If evidence is limited or conflicting, say so explicitly.\n"
                    "5. Do not state a definitive diagnosis — discuss possibilities and direct to appropriate care.\n"
                    "6. Always escalate emergency or urgent symptom patterns before educational content.\n"
                    "7. Synthesize across sources rather than copying any single source.\n"
                    "8. For symptom triage, prioritize Tier 1 (formal guidance) sources first, "
                    "then use Tier 2/3 to add nuance.\n"
                    "9. Use longitudinal patient memory when relevant, but do not override current evidence.\n"
                    "10. Label evidence confidence when sources conflict or are limited.\n"
                    "11. Keep the tone appropriate for a premium, clinical-grade health application.\n"
                    "12. Do NOT add a disclaimer, safety note, or 'this is not medical advice' footer — "
                    "one will be appended automatically."
                ),
            }
        ]

        # Build role-appropriate section headings
        if role_config:
            from backend.response_templates import get_section_headings_text
            headings_text = get_section_headings_text(role_config.role_key)
        else:
            headings_text = (
                "## Clinical Takeaway\n"
                "## What This Means In Practice\n"
                "## Evidence Snapshot\n"
                "## Recommended Next Step\n"
                "## Safety Note"
            )

        # Build policy context injection
        policy_block = ""
        if policy_context_note:
            policy_block = f"Clinical policy instructions (must be followed):\n{policy_context_note}\n\n"

        # Build escalation banner injection (pre-pended to answer)
        banner_instruction = ""
        if escalation_banner:
            banner_instruction = (
                f"IMPORTANT: Begin your response with this escalation notice verbatim:\n"
                f"{escalation_banner}\n\n"
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Longitudinal patient memory:\n{self._render_longitudinal_memory(longitudinal_memory)}\n\n"
                    f"Recent conversation:\n{self._render_chat_history(chat_history)}\n\n"
                    f"Evidence dossier:\n{self._render_evidence_dossier(source_briefings, context)}\n\n"
                    f"{policy_block}"
                    f"Current question:\n{question}\n\n"
                    f"{banner_instruction}"
                    f"Write the answer using these headings:\n{headings_text}\n\n"
                    "Every evidence-based statement should include one or more source markers.\n"
                    "Where multiple sources agree, synthesize them into one clearer statement with combined citations.\n"
                    "Label the evidence tier (Tier 1 / Tier 2 / Tier 3) inline when it helps the reader "
                    "assess the strength of the recommendation."
                ),
            }
        )

        return self._stream_response(messages) if stream else self._complete_response(messages)

    def refresh_longitudinal_memory(
        self,
        existing_memory: str,
        new_information: str,
        user_profile: Optional[dict] = None,
        source_label: str = "conversation",
    ) -> str:
        """
        Merges new patient-specific facts into a durable longitudinal memory summary.
        Generic education, hypotheticals, and unsupported assistant inferences should
        not be written into the memory.
        """
        cleaned_new_information = (new_information or "").strip()
        if not cleaned_new_information:
            return (existing_memory or "").strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You maintain a longitudinal patient memory for a health assistant. "
                    "Update the memory using only durable patient-specific facts that are explicitly stated "
                    "in the supplied new information or clearly present in provided record summaries.\n\n"
                    "Rules:\n"
                    "1. Keep existing confirmed facts unless the new information clearly supersedes them.\n"
                    "2. Do not add generic medical education, hypothetical examples, or assistant speculation.\n"
                    "3. If the new information is not about the specific patient, leave the memory unchanged.\n"
                    "4. Keep the output concise, de-duplicated, and clinically useful.\n"
                    "5. Use the exact headings below.\n"
                    "6. If a section has no reliable facts, write `None noted`.\n"
                    "7. If there is no durable patient-specific information at all, return the existing memory unchanged.\n"
                    "8. Never invent medications, diagnoses, allergies, dates, or test results."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Existing longitudinal memory:\n{self._render_longitudinal_memory(existing_memory)}\n\n"
                    f"New information source: {source_label}\n"
                    f"New information:\n{cleaned_new_information}\n\n"
                    "Return the refreshed longitudinal memory using exactly this structure:\n"
                    "Patient Summary:\n"
                    "Conditions and history:\n"
                    "Current treatments and medicines:\n"
                    "Recent symptoms or active concerns:\n"
                    "Investigations or notable results:\n"
                    "Risks, allergies, or safety flags:\n"
                    "Care plan and follow-up:\n"
                    "Open questions or uncertainties:\n"
                ),
            },
        ]
        return self._complete_response(messages)

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
    def _render_longitudinal_memory(longitudinal_memory: Optional[str]) -> str:
        cleaned = (longitudinal_memory or "").strip()
        return cleaned or "No durable patient-specific memory recorded yet."

    @staticmethod
    def _render_evidence_dossier(source_briefings: Optional[list[dict]], fallback_context: str) -> str:
        if source_briefings:
            blocks = []
            for source in source_briefings:
                tier_label = source.get("tier_label", "")
                tier_str = f" | {tier_label}" if tier_label else ""
                blocks.append(
                    "\n".join(
                        [
                            f"[{source['source_id']}] {source.get('title', 'Untitled article')}{tier_str}",
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
