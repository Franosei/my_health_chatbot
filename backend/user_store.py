import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

DATA_DIR = Path("data")
UPLOAD_ROOT = DATA_DIR / "uploads"
USER_DB_PATH = Path("users.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_db() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    if not USER_DB_PATH.exists():
        USER_DB_PATH.write_text(json.dumps({"users": {}}, indent=2), encoding="utf-8")


def _normalize_message(message: Dict) -> Dict:
    normalized = dict(message)
    normalized.setdefault("message_id", f"msg-{uuid4().hex[:12]}")
    normalized.setdefault("timestamp", _utc_now())
    normalized.setdefault("sources", [])
    normalized.setdefault("trace_id", None)
    normalized.setdefault("metadata", {})
    return normalized


def _default_profile(username: str, display_name: Optional[str] = None) -> Dict[str, str]:
    name = (display_name or username).strip() or username
    return {
        "display_name": name,
        "email": "",
        "care_context": "Personal health guidance",
        "role": "Individual",
        "organization": "",
        "follow_up_preferences": "",
    }


def _normalize_user_record(username: str, record: Dict) -> Dict:
    normalized = dict(record)
    profile = dict(normalized.get("profile", {}))
    display_name = (
        normalized.get("display_name")
        or profile.get("display_name")
        or username
    )

    default_profile = _default_profile(username, display_name)
    for key, value in default_profile.items():
        profile.setdefault(key, value)

    normalized["username"] = normalized.get("username", username)
    normalized["display_name"] = display_name
    normalized["profile"] = profile
    normalized.setdefault("created_at", _utc_now())
    normalized.setdefault("last_login", None)
    normalized.setdefault("conversation", [])
    normalized.setdefault("audit", [])
    normalized.setdefault("uploads", [])
    normalized.setdefault("doc_summaries", [])
    normalized.setdefault("traces", [])
    normalized.setdefault("active_conversation_id", f"conv-{uuid4().hex[:12]}")

    normalized["conversation"] = [
        _normalize_message(message)
        for message in normalized.get("conversation", [])
        if isinstance(message, dict)
    ]

    for upload in normalized["uploads"]:
        upload.setdefault("uploaded_at", _utc_now())
        upload.setdefault("stored_path", "")
        upload.setdefault("summary_available", False)

    for summary in normalized["doc_summaries"]:
        summary.setdefault("stored_path", "")
        summary.setdefault("updated_at", _utc_now())

    for trace in normalized["traces"]:
        trace.setdefault("trace_id", f"trace-{uuid4().hex[:12]}")
        trace.setdefault("created_at", _utc_now())
        trace.setdefault("sources", [])

    return normalized


def _load_db() -> Dict:
    _ensure_db()
    with open(USER_DB_PATH, "r", encoding="utf-8") as file:
        db = json.load(file)

    if "users" not in db or not isinstance(db["users"], dict):
        db = {"users": {}}

    changed = False
    normalized_users = {}
    for key, record in db["users"].items():
        normalized_record = _normalize_user_record(key, record)
        normalized_users[key] = normalized_record
        if normalized_record != record:
            changed = True

    db["users"] = normalized_users
    if changed:
        _save_db(db)
    return db


def _save_db(db: Dict) -> None:
    with open(USER_DB_PATH, "w", encoding="utf-8") as file:
        json.dump(db, file, indent=2)


def _hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200000,
    )
    return {"salt": salt, "hash": hashed.hex()}


def _append_audit(
    user_record: Dict,
    event: str,
    details: str,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    user_record["audit"].append(
        {
            "time": _utc_now(),
            "event": event,
            "details": details,
            "trace_id": trace_id,
            "metadata": metadata or {},
        }
    )


class UserStore:
    """Persistent local store for user profiles, conversations, uploads, and audit traces."""

    @staticmethod
    def create_user(
        username: str,
        password: str,
        display_name: Optional[str] = None,
        email: str = "",
        care_context: str = "Personal health guidance",
        role: str = "Individual",
        organization: str = "",
    ) -> bool:
        db = _load_db()
        key = username.strip().lower()
        if not key or key in db["users"] or len(password) < 8:
            return False

        pwh = _hash_password(password)
        profile = _default_profile(key, display_name)
        profile.update(
            {
                "email": email.strip(),
                "care_context": care_context.strip() or "Personal health guidance",
                "role": role.strip() or "Individual",
                "organization": organization.strip(),
            }
        )

        user_record = _normalize_user_record(
            key,
            {
                "username": key,
                "display_name": profile["display_name"],
                "password_hash": pwh["hash"],
                "salt": pwh["salt"],
                "created_at": _utc_now(),
                "last_login": None,
                "profile": profile,
                "conversation": [],
                "audit": [],
                "uploads": [],
                "doc_summaries": [],
                "traces": [],
            },
        )
        _append_audit(user_record, "account_created", "Account created")
        db["users"][key] = user_record
        _save_db(db)
        return True

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return False

        pwh = _hash_password(password, salt=user["salt"])
        return pwh["hash"] == user["password_hash"]

    @staticmethod
    def update_last_login(username: str) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return
        user["last_login"] = _utc_now()
        _append_audit(user, "login", "User logged in")
        _save_db(db)

    @staticmethod
    def get_user_profile(username: str) -> Dict:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return {}

        profile = deepcopy(user.get("profile", {}))
        profile["created_at"] = user.get("created_at")
        profile["last_login"] = user.get("last_login")
        profile["active_conversation_id"] = user.get("active_conversation_id")
        return profile

    @staticmethod
    def update_profile(username: str, updates: Dict[str, str]) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return

        profile = user.setdefault("profile", _default_profile(key))
        allowed_keys = {
            "display_name",
            "email",
            "care_context",
            "role",
            "organization",
            "follow_up_preferences",
        }
        applied_updates = {}
        for field, value in updates.items():
            if field in allowed_keys:
                profile[field] = (value or "").strip()
                applied_updates[field] = profile[field]

        if "display_name" in applied_updates and applied_updates["display_name"]:
            user["display_name"] = applied_updates["display_name"]
        _append_audit(user, "profile_updated", "Profile details updated", metadata=applied_updates)
        _save_db(db)

    @staticmethod
    def get_upload_dir(username: str) -> Path:
        key = username.strip().lower()
        upload_dir = UPLOAD_ROOT / key
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir

    @staticmethod
    def get_chat_history(username: str) -> List[Dict]:
        db = _load_db()
        key = username.strip().lower()
        return deepcopy(db["users"].get(key, {}).get("conversation", []))

    @staticmethod
    def set_chat_history(username: str, history: List[Dict]) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return
        user["conversation"] = [_normalize_message(message) for message in history]
        _append_audit(user, "conversation_replaced", "Conversation history replaced")
        _save_db(db)

    @staticmethod
    def clear_chat_history(username: str) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return
        user["conversation"] = []
        user["active_conversation_id"] = f"conv-{uuid4().hex[:12]}"
        _append_audit(user, "conversation_cleared", "Conversation history cleared")
        _save_db(db)

    @staticmethod
    def append_chat(username: str, message: Dict) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return
        normalized_message = _normalize_message(message)
        user["conversation"].append(normalized_message)
        _append_audit(
            user,
            "chat_message",
            f"{normalized_message.get('role', 'unknown')} message stored",
            trace_id=normalized_message.get("trace_id"),
            metadata={
                "message_id": normalized_message.get("message_id"),
                "source_count": len(normalized_message.get("sources", [])),
            },
        )
        _save_db(db)

    @staticmethod
    def add_upload(username: str, upload_name: str, stored_path: Optional[str] = None) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return

        uploads = user.setdefault("uploads", [])
        existing = next((item for item in uploads if item.get("file") == upload_name), None)
        if existing:
            existing["uploaded_at"] = _utc_now()
            if stored_path is not None:
                existing["stored_path"] = stored_path
        else:
            uploads.append(
                {
                    "file": upload_name,
                    "uploaded_at": _utc_now(),
                    "stored_path": stored_path or "",
                    "summary_available": False,
                }
            )

        _append_audit(
            user,
            "upload",
            f"Uploaded {upload_name}",
            metadata={"file": upload_name, "stored_path": stored_path or ""},
        )
        _save_db(db)

    @staticmethod
    def save_document_summary(
        username: str,
        filename: str,
        summary: str,
        stored_path: Optional[str] = None,
    ) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return

        doc_summaries = user.setdefault("doc_summaries", [])
        existing = next((item for item in doc_summaries if item.get("file") == filename), None)
        payload = {
            "file": filename,
            "summary": summary,
            "stored_path": stored_path or "",
            "updated_at": _utc_now(),
        }
        if existing:
            existing.update(payload)
        else:
            doc_summaries.append(payload)

        upload_record = next((item for item in user.get("uploads", []) if item.get("file") == filename), None)
        if upload_record:
            upload_record["summary_available"] = True
            if stored_path is not None:
                upload_record["stored_path"] = stored_path

        _append_audit(
            user,
            "document_indexed",
            f"Indexed upload {filename}",
            metadata={"file": filename},
        )
        _save_db(db)

    @staticmethod
    def get_document_summaries(username: str) -> List[Dict]:
        db = _load_db()
        key = username.strip().lower()
        return deepcopy(db["users"].get(key, {}).get("doc_summaries", []))

    @staticmethod
    def get_uploads(username: str) -> List[Dict]:
        db = _load_db()
        key = username.strip().lower()
        return deepcopy(db["users"].get(key, {}).get("uploads", []))

    @staticmethod
    def save_interaction_trace(username: str, trace: Dict) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return

        trace_payload = deepcopy(trace)
        trace_payload.setdefault("trace_id", f"trace-{uuid4().hex[:12]}")
        trace_payload.setdefault("created_at", _utc_now())
        trace_payload.setdefault("sources", [])
        trace_payload.setdefault("question", "")
        trace_payload.setdefault("answer_preview", "")
        user.setdefault("traces", []).append(trace_payload)

        _append_audit(
            user,
            "trace_saved",
            f"Trace saved for question: {trace_payload.get('question', '')[:80]}",
            trace_id=trace_payload["trace_id"],
            metadata={"source_count": len(trace_payload.get("sources", []))},
        )
        _save_db(db)

    @staticmethod
    def get_interaction_traces(username: str, limit: Optional[int] = 25) -> List[Dict]:
        db = _load_db()
        key = username.strip().lower()
        traces = deepcopy(db["users"].get(key, {}).get("traces", []))
        traces.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        if limit is None:
            return traces
        return traces[:limit]

    @staticmethod
    def add_audit(
        username: str,
        event: str,
        details: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return
        _append_audit(user, event, details, trace_id=trace_id, metadata=metadata)
        _save_db(db)

    @staticmethod
    def get_audit(username: str, limit: Optional[int] = 50) -> List[Dict]:
        db = _load_db()
        key = username.strip().lower()
        audit = deepcopy(db["users"].get(key, {}).get("audit", []))
        audit.sort(key=lambda item: item.get("time", ""), reverse=True)
        if limit is None:
            return audit
        return audit[:limit]

    @staticmethod
    def export_user_snapshot(username: str) -> Dict:
        db = _load_db()
        key = username.strip().lower()
        user = db["users"].get(key)
        if not user:
            return {}

        exported = deepcopy(user)
        exported.pop("password_hash", None)
        exported.pop("salt", None)
        return exported
