# backend/summarizer.py

from typing import Optional, Generator
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env")
client = OpenAI(api_key=api_key)


class LLMHelper:
    """
    Wrapper around OpenAI's Chat API for question answering and summarization.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def answer_question(
        self,
        question: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Uses LLM to answer a user’s question with optional chat history.
        """
        messages = [
            {"role": "system", "content": (
                "You are a helpful medical assistant. Use the biomedical context and chat history "
                "to answer the user's health question with clarity and transparency."
            )}
        ]

        # Include past chat messages (up to last 3 turns for brevity)
        if chat_history:
            for msg in chat_history[-6:]:  # 3 user-assistant turns
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current question with relevant context
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely using the provided context and prior conversation."
        })

        return self._stream_response(messages) if stream else self._complete_response(messages)


    def summarize_user_health_record(self, record_text: str) -> str:
        """
        Summarizes a user's anonymized medical record into a brief clinical overview.
        """
        messages = [
            {"role": "system", "content": (
                "You are a medical assistant summarizing a patient's health record. "
                "Produce a brief clinical summary suitable for finding related biomedical research."
            )},
            {"role": "user", "content": (
                f"Patient Record:\n{record_text}\n\n"
                "Summarize their key conditions, lab findings, and relevant clinical info in 1 paragraph."
            )}
        ]
        return self._complete_response(messages)

    def _complete_response(self, messages) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def _stream_response(self, messages) -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
