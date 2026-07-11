from backend.evidence_schema import ArticleEvidence, ExtractedEvidenceDossier


def test_to_prompt_context_labels_by_real_source_id_not_position():
    """
    Articles are stored sorted by alignment_confidence, which is a different order
    than the source_id assigned by evidence_ranker (based on combined_score). If the
    prompt labeled sources positionally ([S1], [S2], ...) instead of by their real
    source_id, an LLM citation like [S1] would resolve against the wrong entry in
    combined_sources when _link_citations() renders the final answer -- a citation
    pointing at the wrong source entirely.
    """
    dossier = ExtractedEvidenceDossier(
        question="What does my peak flow of 18 mean?",
        patient_profile_summary="Adult",
        articles=[
            ArticleEvidence(source_id="S3", title="Highest-confidence article", alignment_confidence=0.9),
            ArticleEvidence(source_id="S1", title="Lower-confidence article", alignment_confidence=0.4),
        ],
    )

    context = dossier.to_prompt_context()

    assert "[S3] Highest-confidence article" in context
    assert "[S1] Lower-confidence article" in context
    assert "[S1] Highest-confidence article" not in context
