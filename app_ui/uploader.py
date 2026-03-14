from pathlib import Path
from typing import List

import streamlit as st

from backend.user_store import UserStore


def upload_documents(current_user: str) -> List[Path]:
    """
    Saves uploaded PDFs into a user-specific directory and returns the stored paths.
    """
    uploaded_files = st.file_uploader(
        label="Upload PDFs such as lab reports, discharge notes, or specialist letters",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return []

    if not st.button("Securely save documents", type="primary"):
        return []

    save_dir = UserStore.get_upload_dir(current_user)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        UserStore.add_upload(current_user, uploaded_file.name, stored_path=str(file_path))
        saved_paths.append(file_path)

    st.success(f"Saved {len(saved_paths)} document(s) to this user's secure workspace.")
    return saved_paths
