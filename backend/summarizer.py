import json
import os
from typing import Generator, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from backend.product_config import PRODUCT_NAME
from backend.user_store import compute_current_age

if TYPE_CHECKING:
    from backend.role_router import RoleConfig

load_dotenv()


class LLMHelper:
    """
    Wrapper around OpenAI's Chat Completions API for question answering and summarization.
    """

    # gpt-4o for all answer generation — quality over cost for a health application.
    # gpt-4o-mini is used only for cheap auxiliary calls (triage JSON, extraction, etc.)
    ANSWER_MODEL = "gpt-4o"
    AUX_MODEL = "gpt-4o-mini"

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
        Uses gpt-4o for answer quality. Inline source citations like [S1] are mandatory.
        """
        if role_config:
            from backend.response_templates import get_persona_block
            persona = get_persona_block(role_config.role_key)
        else:
            persona = (
                f"You are {PRODUCT_NAME}, a safe and competent clinical information assistant supporting "
                "individual health users, caregivers, and healthcare teams. "
                "You provide decisive, evidence-grounded guidance with a clear next-step plan."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{persona}\n\n"
                    "CORE RULES:\n"
                    "1. Use only the supplied evidence dossier and conversation context.\n"
                    "2. Use concise markdown with the role-appropriate section headings provided.\n"
                    "3. Cite every factual claim inline with source markers [S1], [S1][S2], etc.\n"
                    "4. Do not state a definitive diagnosis. For clinicians, label impressions as provisional.\n"
                    "5. Always surface emergency or urgent patterns before educational content.\n"
                    "6. Synthesize across sources — do not copy any single source.\n"
                    "7. Prioritize Tier 1 (formal guidance) first, then Tier 2/3 for nuance.\n"
                    "8. CAUSAL REASONING: Before composing your answer, review the longitudinal patient "
                    "memory. Reason explicitly about how this patient's specific conditions, medications, "
                    "lab results, and vitals modify the risk, differential, or management of their question. "
                    "Name the connection out loud — do not silently ignore it.\n"
                    "9. Do NOT add a disclaimer footer — one is appended automatically.\n\n"
                    "SPECIFICITY REQUIREMENTS — these are mandatory:\n"
                    "- Quote the patient's actual recorded values where relevant. "
                    "Never write 'your blood pressure appears elevated' when you have the number; "
                    "write 'your last recorded BP of X/Y mmHg on [date] is Stage 2 hypertension'.\n"
                    "- Name every medication, condition, lab result, and vital sign by its actual name "
                    "from the patient record — never say 'your medication' or 'your condition'.\n"
                    "- Give concrete timeframes: not 'see a doctor soon' but 'book a GP appointment "
                    "within 2 working days' or 'seek same-day urgent review'.\n"
                    "- Give threshold values: not 'if it gets worse' but 'return if systolic BP exceeds "
                    "180 mmHg, O2 drops below 94%, or new chest pain develops'.\n"
                    "- Every monitoring point must have a measurable threshold, not just a description.\n"
                    "- For clinical users: include specific investigation targets, drug doses where the "
                    "evidence explicitly supports them, and escalation criteria.\n\n"
                    "FORBIDDEN — never write these vague phrases:\n"
                    "'consult a healthcare professional', 'seek medical advice if concerned', "
                    "'this varies from person to person', 'it is always best to speak to your doctor', "
                    "'you should discuss this with your GP', 'everyone is different'. "
                    "Replace every instance with a specific, actionable instruction."
                ),
            }
        ]

        if role_config:
            from backend.response_templates import get_section_headings_text
            headings_text = get_section_headings_text(role_config.role_key)
        else:
            headings_text = (
                "## Working Impression\n"
                "## What To Do Now\n"
                "## What To Monitor\n"
                "## Evidence Snapshot\n"
                "## Recommended Next Step"
            )

        policy_block = ""
        if policy_context_note:
            policy_block = f"Clinical policy instructions (must be followed):\n{policy_context_note}\n\n"

        banner_instruction = ""
        if escalation_banner:
            banner_instruction = (
                f"IMPORTANT: Begin your response with this escalation notice verbatim:\n"
                f"{escalation_banner}\n\n"
            )

        memory_text = self._render_longitudinal_memory(longitudinal_memory)
        has_patient_data = memory_text != "No durable patient-specific memory recorded yet."

        messages.append(
            {
                "role": "user",
                "content": (
                    f"User profile:\n{self._render_profile_summary(user_profile)}\n\n"
                    f"Longitudinal patient memory (use these specific values in your answer):\n{memory_text}\n\n"
                    f"Recent conversation:\n{self._render_chat_history(chat_history)}\n\n"
                    f"Evidence dossier:\n{self._render_evidence_dossier(source_briefings, context)}\n\n"
                    f"{policy_block}"
                    f"Current question:\n{question}\n\n"
                    f"{banner_instruction}"
                    f"Write the answer using these headings:\n{headings_text}\n\n"
                    + (
                        "MANDATORY: The patient's longitudinal memory contains specific lab values, vitals, "
                        "conditions, and medications. You MUST reference these by their actual numbers and names "
                        "in your answer — connect this patient's specific data to your guidance.\n\n"
                        if has_patient_data else ""
                    )
                    + "Every evidence-based statement must include source markers.\n"
                    "Where multiple sources agree, synthesize into one statement with combined citations.\n"
                    "Label evidence tier (Tier 1 / Tier 2 / Tier 3) when it helps assess recommendation strength.\n"
                    "Be decisive: give specific routes, thresholds, and timeframes throughout."
                ),
            }
        )

        return (
            self._stream_response(messages, model=self.ANSWER_MODEL)
            if stream
            else self._complete_response(messages, model=self.ANSWER_MODEL)
        )

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
                    "8. Never invent medications, diagnoses, allergies, dates, or test results.\n"
                    "9. Write in plain text only — no markdown, no asterisks, no bold, no bullet dashes, "
                    "no hyphens as list markers. Each fact should be a short plain sentence or phrase."
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

    def extract_medication_mentions(self, text: str) -> list[str]:
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return []

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only direct medication names from the user's message. "
                        "Do not infer diagnoses, drug classes, supplements, or vague categories. "
                        "Return a JSON object with one key: medications."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Message:\n{cleaned_text}\n\n"
                        "Return JSON in this shape only:\n"
                        '{"medications": ["drug name"]}'
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
        medications = payload.get("medications", [])
        if not isinstance(medications, list):
            return []
        return [str(item).strip() for item in medications if str(item).strip()][:6]

    def build_structured_triage(
        self,
        question: str,
        answer_markdown: str,
        fallback_triage: dict,
        intent_summary: str = "",
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You produce a compact structured triage summary for a health assistant. "
                        "Use only the supplied answer and fallback safety route. "
                        "Never lower the acuity below the fallback next step. "
                        "Return a JSON object with these exact keys: urgency_level, next_step, what_to_monitor, rationale. "
                        "The next_step must be one of: Self-care, GP, 111, 999. "
                        "what_to_monitor must be an array of up to 3 short phrases."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Intent summary:\n{intent_summary or 'Not available'}\n\n"
                        f"Fallback triage (minimum safe acuity):\n{json.dumps(fallback_triage)}\n\n"
                        f"Assistant answer:\n{answer_markdown}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content.strip())

    def check_claim_source_alignment(
        self,
        answer_markdown: str,
        source_briefings: list[dict],
    ) -> list[dict]:
        """
        Reviews the answer and checks whether each factual claim is backed by
        a retrieved source. Returns a list of dicts:
          {"claim": "...", "status": "supported"|"general_knowledge", "source_ids": [...]}
        Only the top 5 claims are checked to keep latency low.
        """
        if not answer_markdown or not source_briefings:
            return []

        source_block = "\n".join(
            f"[{s['source_id']}] {s.get('title', '')} — {s.get('snippet', '')[:200]}"
            for s in source_briefings[:8]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You check whether factual claims in a clinical answer are backed by the listed sources. "
                        "Extract up to 5 specific factual or clinical claims from the answer. "
                        "For each claim, decide: "
                        "'supported' (a listed source clearly backs it), or "
                        "'general_knowledge' (plausible but not directly in any listed source). "
                        "Return a JSON object with one key: claims. "
                        "Each claim is: {\"claim\": str, \"status\": str, \"source_ids\": [str]}."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Sources:\n{source_block}\n\n"
                        f"Answer (first 1200 chars):\n{answer_markdown[:1200]}\n\n"
                        "Return only valid JSON."
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        payload = json.loads(raw)
        items = payload.get("claims", [])
        if not isinstance(items, list):
            return []
        result = []
        for item in items[:5]:
            if isinstance(item, dict) and item.get("claim"):
                result.append({
                    "claim": str(item.get("claim", "")).strip(),
                    "status": str(item.get("status", "general_knowledge")).strip(),
                    "source_ids": [str(s) for s in item.get("source_ids", [])],
                })
        return result

    def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        chat_history: Optional[list[dict]] = None,
        user_profile: Optional[dict] = None,
        patient_context: Optional[str] = None,
        role_key: str = "patient",
    ) -> list[str]:
        profile_text = self._render_profile_summary(user_profile)
        patient_data = (patient_context or "").strip() or "No structured patient data available."
        is_clinician = role_key in ("doctor", "nurse", "midwife", "physiotherapist")

        # Last 3 conversation turns — full content, no truncation
        recent_turns = ""
        if chat_history:
            turns = [
                m for m in chat_history
                if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
            ][-6:]  # last 3 pairs = 6 messages
            if turns:
                recent_turns = "\n".join(
                    f"{m['role'].title()}: {m['content'].strip()}"
                    for m in turns
                )

        if is_clinician:
            framing = (
                "You are a senior clinician generating the exact follow-up questions you would ask "
                "in a real outpatient consultation after hearing this patient's complaint.\n\n"
                "Your task: read the complaint and answer, identify the most likely differentials, "
                "then generate the questions that would distinguish between them or fill the most "
                "important gaps in the history. These must be the questions a real consultant "
                "would actually ask — not generic categories.\n\n"
                "Draw from standard clinical clerking: site, onset, character, radiation, "
                "associated features, timing, exacerbating/relieving factors, severity — "
                "but ONLY the ones that matter for this specific presentation. Also include "
                "targeted social history (smoking pack-years, alcohol units, occupation, travel) "
                "and systems review questions only when directly relevant to the differentials.\n\n"
                "Phrase as direct consultation questions, e.g. 'Does the pain radiate anywhere?' "
                "or 'Any haemoptysis or night sweats?'. Up to 5 questions. Each must be specific "
                "to this presentation — never generic."
            )
        else:
            framing = (
                "You are a GP generating the follow-up questions you would ask this patient next "
                "in a real consultation, based on what they have just told you.\n\n"
                "Your task: identify what the complaint is, think about the most likely causes, "
                "then generate the questions that would help narrow it down or fill the key gaps "
                "in their history. These must be the questions a real GP would genuinely ask — "
                "natural, targeted, and specific to this presentation.\n\n"
                "Examples of the kind of questions to generate (pick those relevant to this case):\n"
                "- Onset and character: 'When exactly did the chest pain start and is it sharp, "
                "crushing, or burning?'\n"
                "- Radiation and associated symptoms: 'Does it spread to your arm or jaw, and do "
                "you feel short of breath with it?'\n"
                "- Triggers and relief: 'Does it come on with exertion, eating, or lying down, "
                "and does anything make it better?'\n"
                "- Diet and bowel habit: 'Have you noticed any change in your bowel habit or "
                "blood in your stools recently?'\n"
                "- Lifestyle: 'How many cigarettes do you smoke per day and roughly how much "
                "alcohol do you drink in a week?'\n"
                "- Family history: 'Has anyone in your immediate family had a heart attack, "
                "stroke, or bowel cancer?'\n"
                "- Medication: 'Are you still taking [named medication] and have you noticed "
                "any changes since starting it?'\n\n"
                "Phrase in plain conversational language the patient would naturally say when "
                "asked. Up to 5 questions. Every question must be specific to this patient's "
                "complaint — never generic filler."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{framing}\n\n"
                    "Return JSON with one key: 'questions' — an array of up to 5 question strings."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Role: {role_key}\n"
                    f"Patient profile:\n{profile_text}\n\n"
                    f"Patient health record:\n{patient_data}\n\n"
                    + (f"Recent conversation (last 3 turns):\n{recent_turns}\n\n" if recent_turns else "")
                    + f"Complaint / question:\n{question}\n\n"
                    f"Answer already given:\n{answer}\n\n"
                    "Now generate up to 5 follow-up questions a real consultant would ask next "
                    "to build the history and narrow the diagnosis. Base them on the specific "
                    "complaint above — do not generate questions that could apply to any patient. "
                    "Return only valid JSON."
                ),
            },
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.AUX_MODEL,
                messages=messages,
                temperature=0.4,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content.strip())
            questions = payload.get("questions", [])
            if not isinstance(questions, list):
                return []
            return [str(q).strip() for q in questions[:5] if str(q).strip()]
        except Exception as exc:
            print(f"Follow-up question generation failed: {exc}")
            return []

    def _complete_response(self, messages, model: Optional[str] = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _stream_response(self, messages, model: Optional[str] = None) -> Generator[str, None, None]:
        stream = self.client.chat.completions.create(
            model=model or self.model,
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

        # Demographics — listed first as they modify almost every clinical guideline
        dob = (user_profile.get("date_of_birth") or "").strip()
        age = compute_current_age(dob)
        if age is not None:
            fragments.append(f"Age: {age} years")
        sex = (user_profile.get("biological_sex") or "").strip()
        if sex and sex != "Prefer not to say":
            fragments.append(f"Biological sex: {sex}")

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
