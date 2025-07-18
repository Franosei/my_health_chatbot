# backend/memory_store.py

from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np


class MemoryStore:
    """
    Stores embedded biomedical content and allows similarity-based retrieval.
    Useful for answering new questions from previously seen PubMed articles or user-uploaded summaries.
    """

    def __init__(self, embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.entries: List[Dict] = []  # Each entry contains text, embedding, and metadata

    def add_entry(self, text: str, metadata: Dict[str, str]) -> None:
        """
        Adds a new document entry with its embedding to memory.

        Args:
            text (str): Full text (e.g., abstract, intro, discussion).
            metadata (Dict): e.g., {"pmcid": "PMC12345", "section": "discussion", "title": "Title"}
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        self.entries.append({"text": text, "embedding": embedding, "metadata": metadata})

    def search(self, query: str, top_k: int = 3, similarity_threshold: float = 0.75) -> List[Tuple[Dict, float]]:
        """
        Searches memory for relevant texts based on semantic similarity to the query.

        Args:
            query (str): The new user question.
            top_k (int): Number of most similar entries to return.
            similarity_threshold (float): Minimum similarity to consider a match.

        Returns:
            List[Tuple[Dict, float]]: List of (entry, score) where entry includes text + metadata.
        """
        if not self.entries:
            return []

        query_vec = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        all_vectors = np.array([entry["embedding"] for entry in self.entries])
        similarities = util.cos_sim(query_vec, all_vectors)[0].cpu().numpy()

        # Collect matches above threshold
        matches = [
            (self.entries[i], float(score))
            for i, score in enumerate(similarities)
            if score >= similarity_threshold
        ]

        # Sort by highest similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def clear(self):
        """Wipe all memory (used in testing or reset)."""
        self.entries.clear()
