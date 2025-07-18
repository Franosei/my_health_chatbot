# backend/anonymizer.py

import re
import spacy
from typing import List


class DocumentAnonymizer:
    """
    A class to anonymize sensitive personal health information (PHI) from medical text documents.

    This includes names, dates of birth, addresses, emails, phone numbers, and other identifiers
    using a combination of pattern matching and spaCy's Named Entity Recognition (NER).
    """

    def __init__(self, language_model: str = "en_core_web_sm"):
        """
        Initialize the anonymizer with a spaCy NLP model.

        Args:
            language_model (str): spaCy language model for NER.
        """
        self.nlp = spacy.load(language_model)

        self.patterns = {
            "DOB": re.compile(r"\b(?:DOB|Date of Birth)[:\s]*\d{2,4}[-/]\d{1,2}[-/]\d{1,2}", re.IGNORECASE),
            "DATE": re.compile(r"\b(?:\d{1,2}[-/th|st|rd|nd\s]?){1,3}(?:\d{2,4})?\b", re.IGNORECASE),
            "PHONE": re.compile(r"\+?\d{1,4}?[-.\s]??(?:\d{3}[-.\s]?){2,4}\d{3,4}"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "ADDRESS": re.compile(r"\d{1,5}\s[\w\s]{1,30}(Street|St|Road|Rd|Ave|Avenue|Boulevard|Blvd|Lane|Ln|Way)", re.IGNORECASE),
            "ID_NUMBER": re.compile(r"\b(?:ID|Patient ID|SSN|MRN)[:\s]*[A-Z0-9\-]{5,}", re.IGNORECASE)
        }

        self.ner_entity_tags = {
            "PERSON": "REDACTED_NAME",
            "GPE": "REDACTED_LOCATION",
            "ORG": "REDACTED_ORGANIZATION",
            "LOC": "REDACTED_LOCATION",
            "NORP": "REDACTED_AFFILIATION"
        }

    def anonymize(self, text: str) -> str:
        """
        Anonymizes sensitive data in a given medical document.

        Args:
            text (str): The input text containing potentially sensitive health info.

        Returns:
            str: The anonymized text with PHI removed or replaced with tags.
        """
        # Step 1: Regex-based anonymization
        for label, pattern in self.patterns.items():
            text = pattern.sub(f"[REDACTED_{label}]", text)

        # Step 2: NER-based anonymization
        doc = self.nlp(text)
        redacted_text = text

        for ent in doc.ents:
            label = ent.label_
            if label in self.ner_entity_tags:
                tag = f"[{self.ner_entity_tags[label]}]"
                redacted_text = redacted_text.replace(ent.text, tag)

        return redacted_text

    def anonymize_batch(self, texts: List[str]) -> List[str]:
        """
        Anonymizes a list of text documents.

        Args:
            texts (List[str]): List of raw text strings.

        Returns:
            List[str]: Anonymized version of each document.
        """
        return [self.anonymize(text) for text in texts]
