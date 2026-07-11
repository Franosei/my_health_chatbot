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

        # Order matters: DATE is a broad catch-all that matches loose groups of
        # 1-4 digits, so if it ran first it would eat the leading digits out of
        # phone numbers ("020 7946 0958") and house numbers ("42 Oak Street")
        # before PHONE/ADDRESS ever got a chance to match the full pattern,
        # silently leaving those un-redacted. More specific patterns must run
        # before DATE; DATE goes last so it only mops up genuine bare dates.
        self.patterns = {
            "DOB": re.compile(r"\b(?:DOB|Date of Birth)[:\s]*\d{2,4}[-/]\d{1,2}[-/]\d{1,2}", re.IGNORECASE),
            "ADDRESS": re.compile(
                r"\b\d{1,5}[ \t]+[\w][\w \t]{0,28}\b(?:Street|St|Road|Rd|Ave|Avenue|Boulevard|Blvd|Lane|Ln|Way)\b",
                re.IGNORECASE,
            ),
            # Covers common groupings (3-3-4, 3-4-4, 2-4-4 with UK codes, etc.)
            # via variable-width digit groups rather than assuming exactly 3
            # digits per group -- the original pattern never matched UK-style
            # numbers like "020 7946 0958" (a 3-4-4 grouping) at all. "/" is
            # deliberately excluded from the separator class so this can't
            # collide with DATE (e.g. "12/05/2024").
            "PHONE": re.compile(r"\b\+?\d{1,3}?[\s.\-]?\(?\d{2,5}\)?[\s.\-]\d{3,4}[\s.\-]?\d{3,4}\b"),
            "ID_NUMBER": re.compile(r"\b(?:ID|Patient ID|SSN|MRN)[:\s]*[A-Z0-9\-]{5,}", re.IGNORECASE),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            # The previous DATE pattern (`(?:\d{1,2}[-/th|st|rd|nd\s]?){1,3}(?:\d{2,4})?`)
            # had a character-class bug -- "[...th|st|rd|nd...]" inside [] is a
            # set of individual characters (t,h,s,r,d,n,|,-,/,\s), not the
            # intended ordinal-suffix alternation -- and every component after
            # the first \d{1,2} was optional. Net effect: it matched almost any
            # bare 1-2 digit number in the document, destroying clinical data
            # that has nothing to do with dates (blood pressure "120/80",
            # ages "65", vitals "72 bpm", doses/durations "10 days"). Requiring
            # an actual date structure -- a day/month/year triple or a month
            # name -- fixes that; DOB has its own dedicated pattern above so a
            # bare unlabeled "day/month" with no year is deliberately left
            # alone rather than risk eating a measurement or ratio.
            "DATE": re.compile(
                r"\b(?:"
                r"(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-]\d{2,4}"
                r"|(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-]\d{2,4}"
                r"|(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?[ \t]+"
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
                r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                r"[ \t,]+\d{2,4}"
                r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
                r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                r"[ \t]+(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?[ \t]+\d{2,4}"
                r")\b",
                re.IGNORECASE,
            ),
        }

        # Regex fallback for names when spaCy NER is unavailable (e.g. spaCy
        # currently fails to import under Python 3.14 -- see requirements.txt).
        # Regex can't recognize arbitrary names the way NER does; this only
        # catches names next to a labeled field or an honorific, which covers
        # the common "Patient Name: John Smith" / "Dr. Jane Doe" cases found
        # in clinical documents. It is a mitigation, not a full replacement
        # for NER -- an unlabeled mid-sentence name will still slip through.
        # [ \t] (not \s) between name words on purpose: \s also matches
        # newlines, which let the capture bleed across a line break and
        # swallow the next line's label (e.g. "Sarah Jones\nAddress" was
        # captured as a single four-word "name"). Keep these matches
        # confined to one line.
        self._labeled_name_pattern = re.compile(
            r"\b(Patient(?:'s)? Name|Full Name|Name|Next of Kin|Emergency Contact|"
            r"GP|G\.P\.|Consultant|Physician|Doctor|Under the care of|Referred by|Seen by)"
            r"[ \t]*[:\-][ \t]*(?:(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?[ \t]+)?"
            r"([A-Z][a-zA-Z'\-]+(?:[ \t]+[A-Z][a-zA-Z'\-]+){0,3})",
            re.IGNORECASE,
        )
        self._titled_name_pattern = re.compile(
            r"\b(Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?[ \t]+([A-Z][a-zA-Z'\-]+(?:[ \t]+[A-Z][a-zA-Z'\-]+)?)\b"
        )

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

        redacted_text = self._labeled_name_pattern.sub(
            lambda m: f"{m.group(1)}: [REDACTED_NAME]", redacted_text
        )
        redacted_text = self._titled_name_pattern.sub(
            lambda m: f"{m.group(1)}. [REDACTED_NAME]", redacted_text
        )

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
