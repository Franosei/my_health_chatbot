# backend/test_rag_engine.py

from rag_system import RAGEngine

def test_rag_pipeline():
    print("Initializing RAG Engine...")
    rag = RAGEngine(embedding_dir="sample_data")  # Folder with uploaded health docs

    print("Ingesting documents...")
    rag.ingest_documents()

    print("Asking health question...")
    question = "Is dexamethasone safe for elderly patients?"

    print(f"Question: {question}")
    response = rag.handle_user_question(question, stream=False)

    print("\n Response:\n")
    print(response)

    print("\n Chat History:")
    for turn in rag.get_chat_history():
        print(f"{turn['role']}: {turn['content'][:200]}{'...' if len(turn['content']) > 200 else ''}")

if __name__ == "__main__":
    test_rag_pipeline()
