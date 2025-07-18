from summarizer import MedicalSummarizer
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

summarizer = MedicalSummarizer(api_key=api_key)

sections = {
    "abstract": "This study evaluates dexamethasone effects in elderly...",
    "introduction": "Dexamethasone is commonly used in elderly care settings...",
    "discussion": "Our findings show increased side effects in patients over 70...",
    "conclusion": "The results highlight age-related risk and need for dose adjustments..."
}

question = "Is dexamethasone safe for elderly patients?"

response = summarizer.summarize_sections(question, sections)
print(response)
