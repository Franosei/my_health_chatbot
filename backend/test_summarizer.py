from backend.summarizer import LLMHelper


def test_answer_generation():
    helper = LLMHelper()
    question = "Is dexamethasone safe for elderly patients?"
    sources = [
        {
            "source_id": "S1",
            "title": "Example corticosteroid safety study",
            "journal": "Example Journal",
            "year": "2024",
            "section": "Discussion",
            "snippet": "Older adults may require closer monitoring because adverse effects can be more frequent in frail populations.",
        }
    ]

    response = helper.answer_question(
        question=question,
        context="",
        source_briefings=sources,
        stream=False,
    )
    print(response)


if __name__ == "__main__":
    test_answer_generation()
