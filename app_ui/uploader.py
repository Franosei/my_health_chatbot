import html
import re
from pathlib import Path
from typing import List

import streamlit as st

from backend.user_store import UserStore
from backend.utils import extract_text_from_pdf


_NAME_LABEL_PATTERNS = [
    re.compile(
        r"\b(?:patient|client)\s*(?:name|full name)?\s*[:\-]\s*"
        r"([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){0,4})"
    ),
    re.compile(
        r"\b(?:name|full name)\s*[:\-]\s*"
        r"([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+){1,4})"
    ),
]
_NON_NAME_WORDS = {
    "address", "blood", "clinic", "date", "diagnosis", "discharge", "doctor",
    "dob", "gender", "hospital", "laboratory", "medical", "nhs", "patient",
    "report", "result", "results", "sample", "specimen",
}
_FILENAME_PREFIX_RE = re.compile(r"\b(patient|client)\s*(name|full\s*name)?\b", re.IGNORECASE)


def _normalize_name_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z]+", (value or "").lower())
        if len(token) > 1 and token not in _NON_NAME_WORDS
    ]


def _clean_candidate_name(value: str) -> str:
    cleaned = " ".join((value or "").replace("\t", " ").split())
    cleaned = re.split(r"\s{2,}|,|\||\bDOB\b|\bDate\b|\bNHS\b", cleaned, flags=re.IGNORECASE)[0]
    words = []
    for word in cleaned.split():
        token = re.sub(r"[^A-Za-z'.\-]", "", word)
        if not token:
            continue
        if token.lower().strip(".-") in _NON_NAME_WORDS:
            break
        words.append(token)
        if len(words) >= 5:
            break
    return " ".join(words).strip()


def _extract_patient_names(text: str) -> list[str]:
    candidates: list[str] = []
    sample = "\n".join((text or "").splitlines()[:80])
    for pattern in _NAME_LABEL_PATTERNS:
        for match in pattern.finditer(sample):
            candidate = _clean_candidate_name(match.group(1))
            tokens = _normalize_name_tokens(candidate)
            if candidate and len(tokens) >= 2:
                candidates.append(candidate)

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = " ".join(_normalize_name_tokens(candidate))
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    return unique_candidates[:3]


def _extract_patient_names_from_filename(filename: str) -> list[str]:
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = _FILENAME_PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :.")
    candidate = _clean_candidate_name(cleaned.title())
    return [candidate] if len(_normalize_name_tokens(candidate)) >= 2 else []


def _is_usable_account_name(expected_name: str) -> bool:
    return len(_normalize_name_tokens(expected_name)) >= 2


def _name_matches(expected_name: str, candidate_names: list[str]) -> bool:
    expected_tokens = _normalize_name_tokens(expected_name)
    if not expected_tokens:
        return False

    expected_set = set(expected_tokens)
    for candidate in candidate_names:
        candidate_set = set(_normalize_name_tokens(candidate))
        if not candidate_set:
            continue
        if len(expected_set) >= 2:
            if expected_set.issubset(candidate_set) or (
                expected_tokens[0] in candidate_set and expected_tokens[-1] in candidate_set
            ):
                return True
        elif expected_tokens[0] in candidate_set:
            return True
    return False


def _pending_upload_key(current_user: str) -> str:
    return f"pending_upload_verification::{current_user.strip().lower()}"


def _verify_saved_pdf(file_path: Path, expected_name: str) -> dict:
    filename_names = _extract_patient_names_from_filename(file_path.name)
    try:
        text = extract_text_from_pdf(file_path)
    except Exception as exc:
        if not _is_usable_account_name(expected_name):
            return {
                "path": str(file_path),
                "file": file_path.name,
                "status": "missing_account_name",
                "message": "No full account name is saved, so the document cannot be auto-verified.",
                "detected_names": filename_names,
            }
        return {
            "path": str(file_path),
            "file": file_path.name,
            "status": "unreadable",
            "message": f"The PDF text could not be read: {exc}",
            "detected_names": filename_names,
        }

    detected_names = filename_names + [
        name for name in _extract_patient_names(text)
        if " ".join(_normalize_name_tokens(name)) not in {
            " ".join(_normalize_name_tokens(existing)) for existing in filename_names
        }
    ]
    if not _is_usable_account_name(expected_name):
        return {
            "path": str(file_path),
            "file": file_path.name,
            "status": "missing_account_name",
            "message": "No full account name is saved, so the document cannot be auto-verified.",
            "detected_names": detected_names,
        }
    if _name_matches(expected_name, detected_names):
        return {
            "path": str(file_path),
            "file": file_path.name,
            "status": "matched",
            "message": "Patient name matched the signed-in account.",
            "detected_names": detected_names,
        }
    if detected_names:
        return {
            "path": str(file_path),
            "file": file_path.name,
            "status": "mismatch",
            "message": (
                f"Detected patient name {', '.join(detected_names)} does not match "
                f"the signed-in account name {expected_name}."
            ),
            "detected_names": detected_names,
        }
    return {
        "path": str(file_path),
        "file": file_path.name,
        "status": "missing_name",
        "message": "No patient name could be confidently found in this PDF.",
        "detected_names": [],
    }


def _render_pending_verifications(current_user: str, expected_name: str) -> list[Path]:
    key = _pending_upload_key(current_user)
    pending = st.session_state.get(key, [])
    if not pending:
        return []

    ready_paths: list[Path] = []
    remaining = []
    refreshed_pending = []
    for item in pending:
        should_refresh = (
            item.get("path")
            and item.get("status") in {"missing_name", "missing_account_name"}
            and not item.get("detected_names")
        )
        refreshed_pending.append(
            _verify_saved_pdf(Path(item["path"]), expected_name)
            if should_refresh else item
        )
    pending = refreshed_pending

    st.warning("Some uploaded document(s) need verification before extraction.")
    for index, item in enumerate(pending):
        if item.get("status") == "matched":
            ready_paths.append(Path(item["path"]))
            continue
        file_label = item.get("file", "Document")
        detected = ", ".join(item.get("detected_names") or []) or "No name detected"
        account_label = expected_name if _is_usable_account_name(expected_name) else "Full name not set"
        st.markdown(
            f"""
            <div class="upload-verification-card">
                <strong>{html.escape(file_label)}</strong>
                <span>Account full name: {html.escape(account_label)}</span>
                <span>Document name: {html.escape(detected)}</span>
                <p>{html.escape(item.get('message', ''))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        process_col, skip_col = st.columns(2)
        if process_col.button(
            "Process this document",
            key=f"process_pending_{index}_{file_label}",
            use_container_width=True,
        ):
            ready_paths.append(Path(item["path"]))
        elif skip_col.button(
            "Do not process",
            key=f"skip_pending_{index}_{file_label}",
            use_container_width=True,
        ):
            st.info(f"{file_label} was left out of extraction.")
        else:
            remaining.append(item)

    st.session_state[key] = remaining
    return ready_paths


def upload_documents(current_user: str, expected_name: str = "") -> List[Path]:
    """
    Save uploaded PDFs and return only paths approved for extraction.
    Documents with a missing or mismatched patient name are held for confirmation.
    """
    expected_name = (expected_name or "").strip()
    ready_paths = _render_pending_verifications(current_user, expected_name)

    uploaded_files = st.file_uploader(
        label="Upload PDFs such as lab reports, discharge notes, or specialist letters",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return ready_paths

    if not st.button("Securely save documents", type="primary"):
        return ready_paths

    save_dir = UserStore.get_upload_dir(current_user)
    pending_key = _pending_upload_key(current_user)
    pending = st.session_state.get(pending_key, [])

    for uploaded_file in uploaded_files:
        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        verification = _verify_saved_pdf(file_path, expected_name)
        if verification["status"] == "matched":
            ready_paths.append(file_path)
        else:
            pending.append(verification)

    st.session_state[pending_key] = pending
    if ready_paths:
        st.success(f"{len(ready_paths)} verified document(s) ready for extraction.")
    if pending:
        st.warning(f"{len(pending)} document(s) need review before extraction.")
    return ready_paths
