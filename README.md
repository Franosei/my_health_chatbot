# My Health Chatbot

**My Health Chatbot** is an AI-powered assistant guided by **Dr. Charlotte**, built to help users ask health-related questions and receive scientifically grounded answers from PubMed Central.

This tool supports both the general public and health professionals by leveraging real-time biomedical research and large language models (LLMs) to generate personalized, accurate responses.

---

## Features

- Real-time access to biomedical literature from PubMed Central (PMC)
- Natural language question answering
- Summarization using Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG) pipeline
- Dr. Charlotte as your AI health companion

---

## Example Questions

- What are the current treatments for hypertension?
- Is metformin safe for elderly patients?
- What does recent research say about probiotics and gut health?
- Are there new therapies for chronic kidney disease?

---

## How It Works

1. The user submits a health question.
2. The system reformulates the question into a PubMed-compatible search query.
3. Relevant open-access articles are retrieved using the Entrez API.
4. Extracted sections are summarized using an LLM.
5. The chatbot responds with a concise, evidence-based summary (optionally with sources).

---

## Tech Stack

| Component     | Technology                    |
|---------------|-------------------------------|
| Frontend      | Streamlit                     |
| Backend       | Python                        |
| Retrieval     | Entrez API (PubMed Central)   |
| Language Model| OpenAI (or custom)            |
| Architecture  | Retrieval-Augmented Generation (RAG) |

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/my_health_chatbot.git
cd my_health_chatbot

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
python -m streamlit run Home.py --logger.level error

```
---

## Project Structure

```
my_health_chatbot/
│
├── app_ui/ # Static assets, uploader, styles
│ ├── static/
│ ├── init.py
│ ├── app.py
│ ├── landing.py
│ └── uploader.py
│
├── backend/ # RAG pipeline and utils
│ ├── init.py
│ ├── anonymizer.py
│ ├── memory_store.py
│ ├── pubmed_search.py
│ ├── query_expander.py
│ ├── rag_system.py
│ ├── summarizer.py
│ ├── test_pubmed_search.py
│ ├── test_rag_engine.py
│ ├── test_summarizer.py
│ └── utils.py
│
├── pages/ # Streamlit multipage files
│ ├── 1_Landing.py
│ └── 2_Chatbot.py
│
├── sample_data/ # Sample data for testing
│
├── chat_history.json # Saved conversation history
├── .env # Optional environment variables
├── .gitignore # Git ignore file
├── Home.py # Main entry point that redirects to Landing
├── LICENSE
├── README.md
└── requirements.txt # Python dependencies
```

---

## Key Features

- **Streamlit UI** with file upload and multipage support
- **PubMed integration** with query expansion and LLM summarization
- **Memory and history** for continuous context in chat
- **Testing suite** for RAG modules
- **Easily extendable** backend structure

---

## Setup Instructions

```bash
# Clone repository
git clone https://github.com/your_username/my_health_chatbot.git
cd my_health_chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py

```
---

## Environment Variables

Create a .env file with your API keys and other secrets, such as:
OPENAI_API_KEY=your_openai_key

---

