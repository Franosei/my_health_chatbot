from backend import moderation_ml


def test_moderation_falls_back_to_rules_when_detoxify_is_unavailable():
    original_detoxify = moderation_ml.Detoxify
    original_import_error = moderation_ml._DETOXIFY_IMPORT_ERROR

    moderation_ml.Detoxify = None
    moderation_ml._DETOXIFY_IMPORT_ERROR = ImportError("torch/transformers stack is incompatible")

    try:
        moderation = moderation_ml.ModerationEnsemble()
        blocked, category, _, details = moderation.decide("How can I kill myself?")
    finally:
        moderation_ml.Detoxify = original_detoxify
        moderation_ml._DETOXIFY_IMPORT_ERROR = original_import_error

    assert moderation.detox is None
    assert blocked is True
    assert category == "self_harm"
    assert details["moderation_backend"] == "rules_only"
    assert "ImportError" in details["moderation_backend_error"]


def test_moderation_rules_only_mode_allows_neutral_text():
    original_detoxify = moderation_ml.Detoxify
    original_import_error = moderation_ml._DETOXIFY_IMPORT_ERROR

    moderation_ml.Detoxify = None
    moderation_ml._DETOXIFY_IMPORT_ERROR = ImportError("detoxify disabled for this test")

    try:
        moderation = moderation_ml.ModerationEnsemble()
        blocked, category, _, details = moderation.decide("What are common symptoms of dehydration?")
    finally:
        moderation_ml.Detoxify = original_detoxify
        moderation_ml._DETOXIFY_IMPORT_ERROR = original_import_error

    assert blocked is False
    assert category == "allow"
    assert details["moderation_backend"] == "rules_only"
    assert details["detoxify"]["toxicity"] == 0.0
