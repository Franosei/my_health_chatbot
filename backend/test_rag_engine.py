from backend.rag_system import RAGEngine


def test_rag_pipeline():
    print("Initializing RAG engine...")
    rag = RAGEngine(embedding_dir="data/uploads")

    question = "What does recent evidence say about hypertension treatment in older adults?"
    print(f"Asking question: {question}")

    payload = rag.handle_user_question(question=question, stream=False)

    print("\nAnswer:\n")
    print(payload["answer_markdown"])
    print("\nTrace:\n")
    print(payload["trace"])


if __name__ == "__main__":
    test_rag_pipeline()
