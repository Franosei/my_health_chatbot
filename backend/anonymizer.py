import re
from importlib import import_module
from typing import Any, List, Optional


class DocumentAnonymizer:
    """
    Anonymizes sensitive personal health information from documents.
    It prefers spaCy NER when available, but falls back to regex-only redaction
    so the app can still run in environments where spaCy is unavailable.
    """

    def __init__(self, language_model: str = "en_core_web_sm"):
        self.nlp: Optional[Any] = self._load_spacy_model(language_model)

        self.patterns = {
            "DOB": re.compile(r"\b(?:DOB|Date of Birth)[:\s]*\d{2,4}[-/]\d{1,2}[-/]\d{1,2}", re.IGNORECASE),
            "DATE": re.compile(r"\b(?:\d{1,2}[-/th|st|rd|nd\s]?){1,3}(?:\d{2,4})?\b", re.IGNORECASE),
            "PHONE": re.compile(r"\+?\d{1,4}?[-.\s]??(?:\d{3}[-.\s]?){2,4}\d{3,4}"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "ADDRESS": re.compile(
                r"\d{1,5}\s[\w\s]{1,30}(Street|St|Road|Rd|Ave|Avenue|Boulevard|Blvd|Lane|Ln|Way)",
                re.IGNORECASE,
            ),
            "ID_NUMBER": re.compile(r"\b(?:ID|Patient ID|SSN|MRN)[:\s]*[A-Z0-9\-]{5,}", re.IGNORECASE),
        }

        self.ner_entity_tags = {
            "PERSON": "REDACTED_NAME",
            "GPE": "REDACTED_LOCATION",
            "ORG": "REDACTED_ORGANIZATION",
            "LOC": "REDACTED_LOCATION",
            "NORP": "REDACTED_AFFILIATION",
        }

    def anonymize(self, text: str) -> str:
        """
        Anonymizes sensitive data in a medical document.
        """
        redacted_text = text

        for label, pattern in self.patterns.items():
            redacted_text = pattern.sub(f"[REDACTED_{label}]", redacted_text)

        if not self.nlp:
            return redacted_text

        doc = self.nlp(redacted_text)
        for ent in doc.ents:
            label = ent.label_
            if label in self.ner_entity_tags:
                tag = f"[{self.ner_entity_tags[label]}]"
                redacted_text = redacted_text.replace(ent.text, tag)

        return redacted_text

    def anonymize_batch(self, texts: List[str]) -> List[str]:
        return [self.anonymize(text) for text in texts]

    @staticmethod
    def _load_spacy_model(language_model: str) -> Optional[Any]:
        try:
            spacy = import_module("spacy")
            return spacy.load(language_model)
        except Exception as exc:
            print(
                "DocumentAnonymizer: spaCy is unavailable; using regex-only anonymization. "
                f"Reason: {exc}"
            )
            return None
