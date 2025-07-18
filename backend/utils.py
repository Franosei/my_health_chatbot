# backend/utils.py

from pathlib import Path
from typing import List
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extracts and returns full text content from a PDF file using PyMuPDF.

    Args:
        file_path (Path): Path to the PDF file.

    Returns:
        str: Extracted plain text content.
    """
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF file: {file_path}")

    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text
