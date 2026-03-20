"""
Medical illustration generator using GPT-4o image generation (DALL-E 3).
Detects when a question would benefit from a visual and generates an appropriate
safe, clinical-style illustration inline in the chat.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Illustration trigger keywords ──────────────────────────────────────────────
_ILLUSTRATION_PATTERNS = [
    # Exercises and movements
    re.compile(
        r"\b(show|demonstrate|illustration|diagram|picture|image|visual|how to do|"
        r"how to perform|how to apply|what does .{0,20} look like|"
        r"can you show|draw|sketch)\b",
        re.IGNORECASE,
    ),
    # Physical therapy / exercise positions
    re.compile(
        r"\b(exercise|stretch|pose|posture|position|movement|technique|"
        r"plank|squat|lunge|deadlift|push.?up|sit.?up|crunch|bridge|"
        r"yoga|pilates|physiotherapy|rehab|RICE|bandage|splint|sling|"
        r"massage|manipulation|mobilisation|mobilization)\b",
        re.IGNORECASE,
    ),
    # Anatomy and medical diagrams
    re.compile(
        r"\b(anatomy|anatomical|cross.section|diagram of|structure of|"
        r"where is the|location of|nerve|muscle|tendon|ligament|bone|joint|"
        r"organ|heart|lung|kidney|liver|spine|vertebra|pelvis|shoulder|knee|"
        r"hip|ankle|wrist|elbow)\b.{0,60}\b(diagram|structure|anatomy|look like|located)\b",
        re.IGNORECASE,
    ),
    # First aid and procedures
    re.compile(
        r"\b(first aid|CPR|heimlich|recovery position|pressure|wound|"
        r"bandaging|dressing|injection|epipen|inhaler technique|"
        r"blood pressure cuff|peak flow)\b.{0,30}\b(how|technique|position|apply|use)\b",
        re.IGNORECASE,
    ),
]

# Safety keywords that should not appear in generated images
_UNSAFE_TOPICS = re.compile(
    r"\b(blood|wound|injury|trauma|gore|violent|nude|naked|explicit)\b",
    re.IGNORECASE,
)


@dataclass
class IllustrationResult:
    image_url: str
    prompt_used: str
    caption: str
    generated: bool = True


class ImageGenerator:
    """
    Generates clinical-style illustrations via GPT-4o / DALL-E 3.
    Call detect_illustration_need() first, then generate_illustration() if True.
    """

    IMAGE_MODEL = "gpt-image-1"
    IMAGE_SIZE = "1024x1024"
    IMAGE_QUALITY = "medium"

    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def detect_illustration_need(self, question: str) -> bool:
        """
        Returns True if the question is likely to benefit from a visual illustration.
        Uses fast regex — no LLM call needed.
        """
        text = (question or "").strip()
        return any(pattern.search(text) for pattern in _ILLUSTRATION_PATTERNS)

    def generate_illustration(
        self,
        question: str,
        context_answer: str = "",
    ) -> Optional[IllustrationResult]:
        """
        Generates a clinical-style image for the given question.
        Returns None if generation fails or content is deemed unsafe.
        """
        question = (question or "").strip()
        if not question:
            return None

        # Build a safe, clinical illustration prompt
        prompt = self._build_prompt(question, context_answer)
        if not prompt:
            return None

        try:
            response = self.client.images.generate(
                model=self.IMAGE_MODEL,
                prompt=prompt,
                size=self.IMAGE_SIZE,
                quality=self.IMAGE_QUALITY,
                n=1,
            )
            image_url = response.data[0].url
            caption = self._build_caption(question)
            return IllustrationResult(
                image_url=image_url,
                prompt_used=prompt,
                caption=caption,
            )
        except Exception as exc:
            print(f"ImageGenerator: generation failed — {exc}")
            return None

    def _build_prompt(self, question: str, answer_context: str) -> str:
        """
        Constructs a safe, clinically-appropriate DALL-E prompt.
        Sanitises the question and enforces medical illustration style.
        """
        # Extract the core topic from the question
        topic = self._extract_topic(question)
        if not topic:
            return ""

        # Base style: clean medical illustration, no graphic content
        style = (
            "Clean medical education illustration, vector art style, "
            "white background, anatomically accurate, no blood or wounds, "
            "suitable for a healthcare information platform. "
            "Professional diagram style, labelled if appropriate."
        )

        # Route to the right kind of illustration
        q_lower = question.lower()

        if any(w in q_lower for w in ("exercise", "stretch", "pose", "plank", "squat",
                                       "yoga", "pilates", "movement", "technique", "rehab")):
            return (
                f"Clear step-by-step exercise illustration showing {topic}. "
                f"Show correct body positioning and form with a simple human figure. "
                f"{style}"
            )

        if any(w in q_lower for w in ("anatomy", "anatomical", "diagram", "structure",
                                       "muscle", "bone", "joint", "nerve", "organ")):
            return (
                f"Medical anatomy diagram of {topic}. "
                f"Educational illustration with clear labels. "
                f"{style}"
            )

        if any(w in q_lower for w in ("first aid", "cpr", "recovery position",
                                       "heimlich", "bandage", "dressing", "injection")):
            return (
                f"Medical first aid illustration showing {topic}. "
                f"Step-by-step instructional diagram. "
                f"{style}"
            )

        if any(w in q_lower for w in ("posture", "position", "rice", "sling", "splint")):
            return (
                f"Medical illustration showing correct {topic}. "
                f"Clear instructional diagram with positioning guide. "
                f"{style}"
            )

        # Generic fallback
        return f"Medical education illustration showing {topic}. {style}"

    @staticmethod
    def _extract_topic(question: str) -> str:
        """Extract the core illustration subject from the question."""
        # Remove common question prefixes
        cleaned = re.sub(
            r"^(show me|can you show|demonstrate|draw|illustrate|"
            r"what does|how to do|how to perform|how to apply|"
            r"diagram of|picture of|image of|visual of)",
            "",
            question.strip(),
            flags=re.IGNORECASE,
        ).strip()

        # Remove trailing question marks and filler
        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        cleaned = re.sub(r"\b(please|for me|to me|me|a|an|the)\b", "", cleaned).strip()
        cleaned = " ".join(cleaned.split())  # normalise whitespace

        # Safety check — avoid generating anything inappropriate
        if _UNSAFE_TOPICS.search(cleaned):
            return ""

        return cleaned[:200]  # cap prompt topic length

    @staticmethod
    def _build_caption(question: str) -> str:
        """Builds a short descriptive caption for the generated image."""
        cleaned = re.sub(r"^(show me|can you|please|how to|what does)", "", question, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\?+$", "", cleaned).strip()
        cleaned = cleaned[:80].capitalize()
        return f"Illustration: {cleaned}" if cleaned else "Generated illustration"
