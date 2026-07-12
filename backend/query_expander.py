import os
from typing import List

import openai
from dotenv import load_dotenv


class QueryExpander:
    """
    Uses an LLM to turn a user's question into retrieval-friendly PubMed search phrases.
    When patient history context is supplied, generates additional queries that capture
    causal and dependency relationships between known conditions and the current question.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.cache: dict[str, List[str]] = {}

    def expand(self, user_question: str) -> List[str]:
        """
        Generates focused PubMed search topics from a user question (no patient history).
        """
        normalized_question = " ".join((user_question or "").split()).strip()
        cached = self.cache.get(normalized_question)
        if cached is not None:
            return cached

        prompt = (
            "You are helping a clinical evidence platform search PubMed Central. "
            "Generate exactly 2 short, precise search queries that capture the most useful "
            "condition, population, treatment, diagnostic, or outcome concepts in the user's request.\n\n"
            f"User question: {normalized_question}\n\n"
            "Search queries:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        queries = self._parse_response(content) or [normalized_question]
        self.cache[normalized_question] = queries
        return queries

    def expand_with_patient_context(
        self,
        user_question: str,
        patient_history_summary: str,
    ) -> List[str]:
        """
        Generates history-enriched PubMed queries that capture causal and dependency
        relationships between the patient's known conditions and their current question.
        The LLM uses the patient's actual stored history to form combined queries --
        no conditions or symptom pairs are hardcoded here.
        """
        normalized_question = " ".join((user_question or "").split()).strip()
        cleaned_history = (patient_history_summary or "").strip()
        if not cleaned_history:
            return self.expand(normalized_question)

        cache_key = f"{normalized_question}||{cleaned_history[:120]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = (
            "You are helping a clinical evidence platform search PubMed Central for a specific patient.\n\n"
            "The patient has this known health background:\n"
            f"{cleaned_history}\n\n"
            "Their current question is:\n"
            f"{normalized_question}\n\n"
            "Generate exactly 3 short, precise PubMed search queries.\n"
            "Rules:\n"
            "- At least one query must reflect the interaction between the patient's known history and "
            "their current presentation. Think: how does this patient's background change the clinical "
            "picture, the differential, or the management of what they are asking about? Capture that "
            "relationship in a specific combined query.\n"
            "- The remaining queries should cover the current question from different evidence angles "
            "(e.g. mechanism, management, risk stratification).\n"
            "- Do not repeat the same concept across queries.\n"
            "- If the patient's known history above gives a specific, confirmed meaning for an "
            "otherwise ambiguous term in the question, use that confirmed meaning/terminology in "
            "your queries -- never search using the raw ambiguous wording, since that risks "
            "retrieving guidance for an entirely different meaning of the term.\n"
            "- Return only the bare queries, one per line, no numbering, no quotes.\n\n"
            "Search queries:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        queries = self._parse_response(content) or [normalized_question]
        self.cache[cache_key] = queries
        return queries

    def expand_with_pathway(self, user_question: str, pathway_terms: List[str]) -> List[str]:
        """
        Generates PubMed search queries augmented with pathway-specific terminology.
        """
        base_queries = self.expand(user_question)
        if not pathway_terms:
            return base_queries

        augmented = list(base_queries)
        for term in pathway_terms[:2]:
            combined = f"{user_question} {term}"
            if combined not in augmented:
                augmented.append(combined)
        return augmented[:5]

    def _parse_response(self, text: str) -> List[str]:
        """
        Parses numbered or bullet lists from LLM output and removes noisy characters.
        """
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            line = line.strip().lstrip("-*0123456789.").strip()
            line = line.replace('"', "").replace("**", "").strip()
            if line:
                cleaned.append(line)
        return cleaned[:5]
