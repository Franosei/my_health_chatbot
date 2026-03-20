from backend.image_generator import ImageGenerator


def test_detect_illustration_need_for_explicit_visual_exercise_request():
    assert ImageGenerator.detect_illustration_need(
        "Can you show me example exercises in picture form that might help my back?"
    )


def test_detect_illustration_need_for_explicit_anatomy_diagram_question():
    assert ImageGenerator.detect_illustration_need(
        "Show me a diagram of the shoulder joint."
    )


def test_detect_illustration_need_does_not_trigger_for_procedural_question_without_visual_request():
    assert not ImageGenerator.detect_illustration_need(
        "How do I perform a bridge exercise correctly?"
    )


def test_detect_illustration_need_does_not_trigger_for_generic_exercise_question():
    assert not ImageGenerator.detect_illustration_need(
        "What exercises help lower back pain?"
    )


def test_detect_illustration_need_does_not_trigger_for_generic_information_question():
    assert not ImageGenerator.detect_illustration_need(
        "What are the symptoms of dehydration?"
    )


def test_detect_illustration_need_does_not_trigger_for_non_visual_treatment_question():
    assert not ImageGenerator.detect_illustration_need(
        "What treatment helps sciatica?"
    )
