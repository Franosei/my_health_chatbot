# backend/query_expander.py

import openai
from typing import List
from dotenv import load_dotenv
import os


class QueryExpander:
    """
    Uses an LLM to rewrite a user's long or complex question into 3–5 targeted PubMed search phrases.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def expand(self, user_question: str) -> List[str]:
        """
        Generates 3–5 focused PubMed-style search topics from a user question.

        Args:
            user_question (str): The user's health-related natural language question.

        Returns:
            List[str]: A list of search-optimized queries.
        """
        prompt = (
            "You are a biomedical research assistant. A user asks a health-related question, "
            "but we want to search PubMed Central using more focused search queries. "
            "Generate 3 distinct and precise PubMed search terms that could return relevant articles.\n\n"
            f"User question: {user_question}\n\n"
            "Search queries:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()
        queries = self._parse_response(content)
        return queries

    def _parse_response(self, text: str) -> List[str]:
        """
        Parses numbered or bullet list of search queries from LLM output.
        Cleans up quotes and special characters to avoid malformed PubMed queries.
        """
        lines = text.strip().split("\n")
        cleaned = []
        for line in lines:
            line = line.strip().lstrip("-•0123456789.").strip()
            # Remove quotation marks or markdown symbols
            line = line.replace('"', "").replace("**", "").strip()
            if line:
                cleaned.append(line)
        return cleaned[:5]
