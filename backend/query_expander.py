import os
from typing import List

import openai
from dotenv import load_dotenv


class QueryExpander:
    """
    Uses an LLM to turn a user's question into retrieval-friendly PubMed search phrases.
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
        Generates focused PubMed search topics from a user question.
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
