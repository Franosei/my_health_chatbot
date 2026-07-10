"""
Evidence extraction schema.

Every retrieved article/source is processed through the evidence extractor
to produce an ArticleEvidence instance before being passed to the LLM.

This prevents:
- Hallucination (model inventing facts not in the source)
- Wrong-patient facts (population mismatches silently reaching the LLM)
- Information overload (only patient-relevant facts are forwarded)
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PatientAlignmentFact(BaseModel):
    """One fact from the article that maps to a specific value in this patient's profile."""
    category: str  # condition | medication | vital | demographic | allergy | symptom
    patient_value: str  # e.g. "Type 2 diabetes", "Metformin 500mg", "HbA1c 72mmol/mol"
    article_statement: str  # what the article actually says about this
    relevance_type: str  # direct_evidence | contraindication | drug_interaction | population_match


class ArticleEvidence(BaseModel):
    """
    Structured evidence extracted from one source, filtered and mapped to this patient.
    All LLM prompts receive this schema -- never raw unprocessed snippets.
    """
    source_id: str
    title: str
    journal: Optional[str] = None
    year: Optional[str] = None
    url: Optional[str] = None
    evidence_tier: int = Field(default=3, description="1=NHS/NICE, 2=systematic reviews, 3=primary research")
    tier_label: str = ""

    # Does the article directly address the patient's question?
    answers_question: bool = False

    # Specific facts that answer the current question
    question_facts: List[str] = Field(
        default_factory=list,
        description="Precise facts from this article that answer the patient's question"
    )

    # Patient profile alignment
    patient_aligned_facts: List[PatientAlignmentFact] = Field(
        default_factory=list,
        description="Facts relevant to this specific patient's conditions, meds, vitals, demographics"
    )

    # Safety
    contraindications: List[str] = Field(
        default_factory=list,
        description="Contraindications or warnings relevant to this patient specifically"
    )
    drug_interactions: List[str] = Field(
        default_factory=list,
        description="Drug interactions mentioned that involve this patient's medications"
    )

    # Synthesis for the LLM prompt
    patient_relevant_summary: str = Field(
        default="",
        description="2-3 sentence summary of what this article contributes for this patient"
    )

    # Quality signal
    alignment_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Raw source passage used for extraction
    source_snippet: str = ""


class ExtractedEvidenceDossier(BaseModel):
    """
    The complete set of structured evidence for one question + patient.
    Serialised into the LLM system prompt as the evidence context block.
    """
    question: str
    patient_profile_summary: str
    articles: List[ArticleEvidence] = Field(default_factory=list)
    extraction_notes: str = ""

    def to_prompt_context(self) -> str:
        """Render the dossier as a structured LLM context block."""
        if not self.articles:
            return "No retrieved evidence was available for this question."

        lines = [
            f"PATIENT CONTEXT: {self.patient_profile_summary}",
            f"QUESTION: {self.question}",
            "",
            "EVIDENCE DOSSIER -- structured extraction matched to this patient.",
            "RULE: Only cite facts explicitly present below. Do not invent or extend.",
            "",
        ]
        for i, art in enumerate(self.articles, 1):
            meta = " | ".join(filter(None, [art.journal, str(art.year or ""), f"Tier {art.evidence_tier}: {art.tier_label}"]))
            lines.append(f"[S{i}] {art.title}")
            if meta:
                lines.append(f"     {meta}")
            if art.question_facts:
                lines.append(f"     ANSWER FACTS: {' | '.join(art.question_facts[:5])}")
            for fact in art.patient_aligned_facts[:3]:
                lines.append(
                    f"     PATIENT MATCH [{fact.category} -- {fact.patient_value}]: {fact.article_statement}"
                )
            if art.contraindications:
                lines.append(f"     CONTRAINDICATIONS: {'; '.join(art.contraindications[:3])}")
            if art.drug_interactions:
                lines.append(f"     DRUG INTERACTIONS: {'; '.join(art.drug_interactions[:3])}")
            if art.patient_relevant_summary:
                lines.append(f"     SUMMARY: {art.patient_relevant_summary}")
            if not art.answers_question:
                lines.append("     NOTE: This source provides background context, not direct answer evidence.")
            lines.append("")

        if self.extraction_notes:
            lines.append(f"EXTRACTION NOTE: {self.extraction_notes}")

        return "\n".join(lines)
