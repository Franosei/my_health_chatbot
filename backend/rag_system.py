# backend/rag_system.py

from backend.query_expander import QueryExpander
from backend.memory_store import MemoryStore
from backend.pubmed_search import PubMedCentralSearcher
from backend.anonymizer import DocumentAnonymizer
from backend.utils import extract_text_from_pdf
from backend.summarizer import LLMHelper
from backend.moderation_ml import ModerationEnsemble

from typing import List, Optional, Generator, Union
import os


class RAGEngine:
    """
    Core class for question-answering using uploaded documents,
    PubMed literature, and long-term memory search.
    """

    def __init__(self, embedding_dir: str = "sample_data"):
        """
        Initializes the RAG engine with all components including the
        document anonymizer, embedding memory, PubMed search, and LLM wrapper.
        """
        self.embedding_dir = embedding_dir
        self.query_expander = QueryExpander()
        self.memory = MemoryStore()
        self.pubmed = PubMedCentralSearcher()
        self.anonymizer = DocumentAnonymizer()
        self.llm = LLMHelper()
        self.moderation = ModerationEnsemble()

    def ingest_documents(self) -> None:
        """
        Loads uploaded documents, strips sensitive information, summarizes them,
        and stores their embeddings into memory.
        Additionally, generates PubMed search queries based on summaries and stores literature.
        """
        for filename in os.listdir(self.embedding_dir):
            if filename.endswith(".pdf"):
                path = os.path.join(self.embedding_dir, filename)
                raw_text = extract_text_from_pdf(path)
                anonymized = self.anonymizer.anonymize(raw_text)

                # Summarize and embed user health history
                summary = self.llm.summarize_user_health_record(anonymized)
                self.memory.add_entry(summary, {"type": "user_summary", "source": filename})

                # Generate PubMed queries based on user records
                search_terms = self.query_expander.expand(summary)
                for q in search_terms:
                    self._search_and_store_pubmed(q)

    def _search_and_store_pubmed(self, query: str) -> None:
        """
        Internal method to search PubMed and store resulting articles into memory.

        Args:
            query (str): A search-friendly PubMed query.
        """
        print(f"PubMed Search Query: {query}")
        pmcids = self.pubmed.search_articles(query)
        print(f"PMC IDs Found: {pmcids}")

        for pmcid in pmcids:
            print(f"Fetching full text for {pmcid}")
            sections = self.pubmed.fetch_article_sections(pmcid)
            for section, text in sections.items():
                if text:
                    self.memory.add_entry(text, {
                        "type": "pubmed",
                        "pmcid": pmcid,
                        "section": section
                    })

    def handle_user_question(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Responds to user queries using relevant memory and literature,
        while incorporating chat history for follow-up continuity.

        Args:
            question (str): The health-related user input.
            chat_history (Optional[List[dict]]): List of prior user-assistant messages.
            stream (bool): Whether to stream LLM output.

        Returns:
            str or Generator: Final answer or streamable chunks.
        """
        blocked, category, safe_msg, details = self.moderation.decide(question)
        if blocked:
            if stream:
                # Return a one-shot generator so your Streamlit loop can iterate safely
                def _blocked_once():
                    yield safe_msg
                return _blocked_once()
            return safe_msg
        
        # Step 1: Search memory embeddings
        matches = self.memory.search(question)
        if matches:
            context = "\n\n".join([m[0]["text"] for m in matches])
            return self.llm.answer_question(
                question=question,
                context=context,
                chat_history=chat_history,
                stream=stream
            )

        # Step 2: No memory match — perform PubMed search
        expanded_queries = self.query_expander.expand(question)
        retrieved_sections = []

        for q in expanded_queries:
            pmcids = self.pubmed.search_articles(q)
            for pmcid in pmcids:
                sections = self.pubmed.fetch_article_sections(pmcid)
                for sec_name, text in sections.items():
                    if text:
                        self.memory.add_entry(
                            text,
                            {"type": "pubmed", "pmcid": pmcid, "section": sec_name}
                        )
                        retrieved_sections.append(text)

        if not retrieved_sections:
            return "I couldn't find relevant biomedical literature at the moment."

        combined_context = "\n\n".join(retrieved_sections[:3])
        return self.llm.answer_question(
            question=question,
            context=combined_context,
            chat_history=chat_history,
            stream=stream
        )
