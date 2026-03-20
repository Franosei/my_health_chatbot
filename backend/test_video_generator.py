from backend.video_generator import VideoGenerator


def test_detect_video_request_for_explicit_video_request():
    assert VideoGenerator.detect_video_request(
        "Can you show me a video of these shoulder exercises?"
    )


def test_detect_video_request_for_explicit_animation_request():
    assert VideoGenerator.detect_video_request(
        "Please animate how the knee joint moves."
    )


def test_detect_video_request_does_not_trigger_for_generic_exercise_question():
    assert not VideoGenerator.detect_video_request(
        "What exercises help lower back pain?"
    )


def test_detect_video_request_does_not_trigger_for_generic_information_question():
    assert not VideoGenerator.detect_video_request(
        "Explain shoulder impingement."
    )
