# frontend/uploader.py

import streamlit as st
import os
import shutil
from pathlib import Path


def upload_documents():
    """
    Streamlit file uploader for medical PDFs. 
    Saves them to the embedding directory (e.g., sample_data/).
    """
    st.subheader("Upload Medical Documents")
    uploaded_files = st.file_uploader(
        label="Upload PDFs (e.g. doctor's report, lab results, etc.)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_dir = Path("sample_data")
        save_dir.mkdir(exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = save_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")
        st.rerun()  # Force refresh to reinitialize RAG with new docs
