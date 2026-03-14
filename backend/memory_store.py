import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class MemoryStore:
    """
    Stores embedded biomedical content and supports similarity retrieval using OpenAI embeddings.
    """

    def __init__(self, embedding_model: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        base_url = os.getenv("OPENAI_BASE_URL")
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.entries: List[Dict] = []
        self.entry_keys: set[str] = set()
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def add_entry(
        self,
        text: str,
        metadata: Dict[str, str],
        user: Optional[str] = None,
        entry_key: Optional[str] = None,
    ) -> None:
        stable_key = entry_key or metadata.get("entry_key")
        if stable_key and stable_key in self.entry_keys:
            return

        embedding = self._embed_text(text)
        payload = {"text": text, "embedding": embedding, "metadata": metadata, "user": user}
        self.entries.append(payload)
        if stable_key:
            self.entry_keys.add(stable_key)

    def search(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.45,
        user: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        if not self.entries:
            return []

        filtered_entries = [entry for entry in self.entries if user is None or entry.get("user") == user]
        if not filtered_entries:
            return []

        query_vec = self._embed_text(query)
        all_vectors = np.array([entry["embedding"] for entry in filtered_entries], dtype=np.float32)
        similarities = np.dot(all_vectors, query_vec)

        matches = [
            (filtered_entries[index], float(score))
            for index, score in enumerate(similarities)
            if score >= similarity_threshold
        ]
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:top_k]

    def clear(self):
        self.entries.clear()
        self.entry_keys.clear()
        self.embedding_cache.clear()

    def _embed_text(self, text: str) -> np.ndarray:
        key = text.strip()
        cached = self.embedding_cache.get(key)
        if cached is not None:
            return cached

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=key,
        )
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vector)
        normalized = vector if norm == 0 else vector / norm
        self.embedding_cache[key] = normalized
        return normalized
