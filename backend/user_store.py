import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
UPLOAD_ROOT = DATA_DIR / "uploads"
USER_DB_PATH = Path("users.json")
USER_TABLE_NAME = "app_user_store"
_USER_BACKEND = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_upload_root() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def _get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return default


def _normalize_message(message: Dict) -> Dict:
    normalized = dict(message)
    normalized.setdefault("message_id", f"msg-{uuid4().hex[:12]}")
    normalized.setdefault("timestamp", _utc_now())
    normalized.setdefault("sources", [])
    normalized.setdefault("trace_id", None)
    normalized.setdefault("metadata", {})
    return normalized


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _is_valid_email(email: str) -> bool:
    normalized = _normalize_email(email)
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized))


def _default_profile(username: str, display_name: Optional[str] = None) -> Dict[str, str]:
    name = (display_name or username).strip() or username
    return {
        "display_name": name,
        "email": "",
        "care_context": "Personal health guidance",
        "role": "Individual",
        "clinical_role": "",   # 5-tier clinical role (patient, doctor, nurse, midwife, physiotherapist)
        "organization": "",
        "follow_up_preferences": "",
        "terms_version": "",
        "terms_role": "",
        "terms_accepted_at": "",
        "privacy_accepted_at": "",
        "last_video_generated_at": "",  # ISO-8601 UTC; enforces 1-video-per-hour rate limit
    }


def _default_longitudinal_memory() -> Dict[str, Optional[str]]:
    return {
        "summary": "",
        "updated_at": None,
        "source": "",
    }


def _normalize_symptom_log(entry: Dict) -> Dict:
    logged_for = str(entry.get("logged_for") or "").strip()
    severity = entry.get("severity", 0)
    try:
        severity_value = max(0, min(10, int(severity)))
    except (TypeError, ValueError):
        severity_value = 0

    return {
        "log_id": entry.get("log_id") or f"sym-{uuid4().hex[:12]}",
        "symptom": str(entry.get("symptom") or "").strip(),
        "logged_for": logged_for,
        "severity": severity_value,
        "triggers": str(entry.get("triggers") or "").strip(),
        "notes": str(entry.get("notes") or "").strip(),
        "created_at": entry.get("created_at") or _utc_now(),
    }


def _normalize_medication(entry: Dict) -> Dict:
    return {
        "medication_id": entry.get("medication_id") or f"med-{uuid4().hex[:12]}",
        "name": str(entry.get("name") or "").strip(),
        "dose": str(entry.get("dose") or "").strip(),
        "schedule": str(entry.get("schedule") or "").strip(),
        "reason": str(entry.get("reason") or "").strip(),
        "started_on": str(entry.get("started_on") or "").strip(),
        "notes": str(entry.get("notes") or "").strip(),
        "created_at": entry.get("created_at") or _utc_now(),
        "updated_at": entry.get("updated_at") or _utc_now(),
    }


def _normalize_triage_summary(entry: Dict) -> Dict:
    monitor = entry.get("what_to_monitor", [])
    if not isinstance(monitor, list):
        monitor = []

    return {
        "summary_id": entry.get("summary_id") or f"triage-{uuid4().hex[:12]}",
        "question": str(entry.get("question") or "").strip(),
        "urgency_level": str(entry.get("urgency_level") or "").strip(),
        "next_step": str(entry.get("next_step") or "").strip(),
        "what_to_monitor": [str(item).strip() for item in monitor if str(item).strip()],
        "rationale": str(entry.get("rationale") or "").strip(),
        "trace_id": entry.get("trace_id"),
        "created_at": entry.get("created_at") or _utc_now(),
    }


def _normalize_user_record(username: str, record: Dict) -> Dict:
    normalized = dict(record)
    profile = dict(normalized.get("profile", {}))
    display_name = normalized.get("display_name") or profile.get("display_name") or username
    longitudinal_memory = normalized.get("longitudinal_memory", {})
    if not isinstance(longitudinal_memory, dict):
        longitudinal_memory = {"summary": str(longitudinal_memory or "").strip()}

    default_profile = _default_profile(username, display_name)
    for key, value in default_profile.items():
        profile.setdefault(key, value)
    default_memory = _default_longitudinal_memory()
    for key, value in default_memory.items():
        longitudinal_memory.setdefault(key, value)

    normalized["username"] = normalized.get("username", username)
    normalized["display_name"] = display_name
    normalized["profile"] = profile
    normalized["longitudinal_memory"] = longitudinal_memory
    normalized.setdefault("created_at", _utc_now())
    normalized.setdefault("last_login", None)
    normalized.setdefault("conversation", [])
    normalized.setdefault("audit", [])
    normalized.setdefault("uploads", [])
    normalized.setdefault("doc_summaries", [])
    normalized.setdefault("traces", [])
    normalized.setdefault("symptom_logs", [])
    normalized.setdefault("medications", [])
    normalized.setdefault("triage_summaries", [])
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

    normalized["symptom_logs"] = [
        _normalize_symptom_log(entry)
        for entry in normalized.get("symptom_logs", [])
        if isinstance(entry, dict)
    ]
    normalized["medications"] = [
        _normalize_medication(entry)
        for entry in normalized.get("medications", [])
        if isinstance(entry, dict)
    ]
    normalized["triage_summaries"] = [
        _normalize_triage_summary(entry)
        for entry in normalized.get("triage_summaries", [])
        if isinstance(entry, dict)
    ]

    return normalized


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


class _UserBackend(Protocol):
    def get_user(self, username: str) -> Optional[Dict]:
        ...

    def find_username_by_email(self, email: str) -> Optional[str]:
        ...

    def save_user(self, username: str, record: Dict) -> None:
        ...


class _LocalJSONUserBackend:
    def __init__(self) -> None:
        _ensure_upload_root()
        if not USER_DB_PATH.exists():
            USER_DB_PATH.write_text(json.dumps({"users": {}}, indent=2), encoding="utf-8")

    def _load_all_users(self) -> Dict[str, Dict]:
        with open(USER_DB_PATH, "r", encoding="utf-8") as file:
            db = json.load(file)

        users = db.get("users", {})
        if not isinstance(users, dict):
            users = {}

        changed = False
        normalized_users = {}
        for key, record in users.items():
            normalized_record = _normalize_user_record(key, record)
            normalized_users[key] = normalized_record
            if normalized_record != record:
                changed = True

        if changed:
            self._save_all_users(normalized_users)

        return normalized_users

    @staticmethod
    def _save_all_users(users: Dict[str, Dict]) -> None:
        with open(USER_DB_PATH, "w", encoding="utf-8") as file:
            json.dump({"users": users}, file, indent=2)

    def get_user(self, username: str) -> Optional[Dict]:
        return self._load_all_users().get(username)

    def find_username_by_email(self, email: str) -> Optional[str]:
        normalized_email = _normalize_email(email)
        if not normalized_email:
            return None

        for username, record in self._load_all_users().items():
            profile_email = _normalize_email(record.get("profile", {}).get("email", ""))
            if profile_email == normalized_email:
                return username
        return None

    def save_user(self, username: str, record: Dict) -> None:
        users = self._load_all_users()
        users[username] = _normalize_user_record(username, record)
        self._save_all_users(users)


class _PostgresUserBackend:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._ready = False
        _ensure_upload_root()

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "DATABASE_URL is set, but psycopg is not installed. "
                "Add `psycopg[binary]` to requirements.txt."
            ) from exc

        return psycopg.connect(self.database_url)

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {USER_TABLE_NAME} (
                        username TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            conn.commit()

        self._ready = True

    def get_user(self, username: str) -> Optional[Dict]:
        self._ensure_ready()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT payload FROM {USER_TABLE_NAME} WHERE username = %s",
                    (username,),
                )
                row = cur.fetchone()

        if not row:
            return None

        payload = row[0]
        record = json.loads(payload) if isinstance(payload, str) else payload
        normalized = _normalize_user_record(username, record)
        if normalized != record:
            self.save_user(username, normalized)
        return normalized

    def find_username_by_email(self, email: str) -> Optional[str]:
        normalized_email = _normalize_email(email)
        if not normalized_email:
            return None

        self._ensure_ready()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT username, payload FROM {USER_TABLE_NAME}")
                rows = cur.fetchall()

        for username, payload in rows:
            record = json.loads(payload) if isinstance(payload, str) else payload
            normalized_record = _normalize_user_record(username, record)
            profile_email = _normalize_email(normalized_record.get("profile", {}).get("email", ""))
            if profile_email == normalized_email:
                if normalized_record != record:
                    self.save_user(username, normalized_record)
                return username
        return None

    def save_user(self, username: str, record: Dict) -> None:
        self._ensure_ready()
        payload = json.dumps(_normalize_user_record(username, record))
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {USER_TABLE_NAME} (username, payload, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (username)
                    DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                    """,
                    (username, payload),
                )
            conn.commit()


def _get_backend() -> _UserBackend:
    global _USER_BACKEND
    if _USER_BACKEND is None:
        database_url = _get_setting("DATABASE_URL")
        _USER_BACKEND = _PostgresUserBackend(database_url) if database_url else _LocalJSONUserBackend()
    return _USER_BACKEND


def _get_user_record(username: str) -> Optional[Dict]:
    key = username.strip().lower()
    if not key:
        return None
    return _get_backend().get_user(key)


def _resolve_username(identifier: str) -> Optional[str]:
    key = (identifier or "").strip().lower()
    if not key:
        return None

    if _get_backend().get_user(key):
        return key

    return _get_backend().find_username_by_email(key)


def _save_user_record(username: str, record: Dict) -> None:
    key = username.strip().lower()
    if not key:
        return
    _get_backend().save_user(key, _normalize_user_record(key, record))


class UserStore:
    """Persistent store for user profiles, conversations, uploads, and audit traces."""

    @staticmethod
    def create_user(
        username: str,
        password: str,
        display_name: Optional[str] = None,
        email: str = "",
        care_context: str = "Personal health guidance",
        role: str = "Individual",
        clinical_role: str = "",
        organization: str = "",
        terms_version: str = "",
        terms_role: str = "",
        terms_accepted_at: str = "",
        privacy_accepted_at: str = "",
    ) -> bool:
        key = username.strip().lower()
        normalized_email = _normalize_email(email)

        if (
            not key
            or _get_user_record(key)
            or len(password) < 8
            or not _is_valid_email(normalized_email)
            or _get_backend().find_username_by_email(normalized_email)
        ):
            return False

        pwh = _hash_password(password)
        profile = _default_profile(key, display_name)
        profile.update(
            {
                "email": normalized_email,
                "care_context": care_context.strip() or "Personal health guidance",
                "role": role.strip() or "Individual",
                "clinical_role": clinical_role.strip(),
                "organization": organization.strip(),
                "terms_version": terms_version.strip(),
                "terms_role": (terms_role or clinical_role or role).strip(),
                "terms_accepted_at": terms_accepted_at or _utc_now(),
                "privacy_accepted_at": privacy_accepted_at or _utc_now(),
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
                "symptom_logs": [],
                "medications": [],
                "triage_summaries": [],
                "longitudinal_memory": _default_longitudinal_memory(),
            },
        )
        _append_audit(
            user_record,
            "account_created",
            "Account created",
            metadata={
                "terms_version": profile.get("terms_version", ""),
                "terms_role": profile.get("terms_role", ""),
                "email": profile.get("email", ""),
            },
        )
        _save_user_record(key, user_record)
        return True

    @staticmethod
    def resolve_login_username(identifier: str) -> Optional[str]:
        return _resolve_username(identifier)

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        resolved_username = _resolve_username(username)
        user = _get_user_record(resolved_username or "")
        if not user:
            return False

        pwh = _hash_password(password, salt=user["salt"])
        return pwh["hash"] == user["password_hash"]

    @staticmethod
    def update_last_login(username: str) -> None:
        user = _get_user_record(username)
        if not user:
            return
        user["last_login"] = _utc_now()
        _append_audit(user, "login", "User logged in")
        _save_user_record(username, user)

    @staticmethod
    def get_user_profile(username: str) -> Dict:
        user = _get_user_record(username)
        if not user:
            return {}

        profile = deepcopy(user.get("profile", {}))
        profile["created_at"] = user.get("created_at")
        profile["last_login"] = user.get("last_login")
        profile["active_conversation_id"] = user.get("active_conversation_id")
        return profile

    @staticmethod
    def update_profile(username: str, updates: Dict[str, str]) -> bool:
        user = _get_user_record(username)
        if not user:
            return False

        key = username.strip().lower()
        profile = user.setdefault("profile", _default_profile(key))
        allowed_keys = {
            "display_name",
            "email",
            "care_context",
            "role",
            "clinical_role",
            "organization",
            "follow_up_preferences",
            "last_video_generated_at",
        }
        applied_updates = {}
        for field, value in updates.items():
            if field in allowed_keys:
                if field == "email":
                    normalized_email = _normalize_email(value)
                    existing_owner = _get_backend().find_username_by_email(normalized_email) if normalized_email else None
                    if existing_owner and existing_owner != key:
                        return False
                    profile[field] = normalized_email
                else:
                    profile[field] = (value or "").strip()
                applied_updates[field] = profile[field]

        if "display_name" in applied_updates and applied_updates["display_name"]:
            user["display_name"] = applied_updates["display_name"]
        _append_audit(user, "profile_updated", "Profile details updated", metadata=applied_updates)
        _save_user_record(username, user)
        return True

    @staticmethod
    def get_upload_dir(username: str) -> Path:
        _ensure_upload_root()
        key = username.strip().lower()
        upload_dir = UPLOAD_ROOT / key
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir

    @staticmethod
    def add_symptom_log(
        username: str,
        symptom: str,
        logged_for: str,
        severity: int,
        triggers: str = "",
        notes: str = "",
    ) -> Optional[Dict]:
        user = _get_user_record(username)
        if not user:
            return None

        payload = _normalize_symptom_log(
            {
                "symptom": symptom,
                "logged_for": logged_for,
                "severity": severity,
                "triggers": triggers,
                "notes": notes,
            }
        )
        if not payload["symptom"] or not payload["logged_for"]:
            return None

        user.setdefault("symptom_logs", []).append(payload)
        _append_audit(
            user,
            "symptom_logged",
            f"Tracked symptom: {payload['symptom']}",
            metadata={
                "log_id": payload["log_id"],
                "logged_for": payload["logged_for"],
                "severity": payload["severity"],
            },
        )
        _save_user_record(username, user)
        return payload

    @staticmethod
    def get_symptom_logs(username: str, limit: Optional[int] = 50) -> List[Dict]:
        user = _get_user_record(username)
        logs = deepcopy(user.get("symptom_logs", [])) if user else []
        logs.sort(
            key=lambda item: (
                item.get("logged_for", ""),
                item.get("created_at", ""),
            ),
            reverse=True,
        )
        if limit is None:
            return logs
        return logs[:limit]

    @staticmethod
    def delete_symptom_log(username: str, log_id: str) -> bool:
        user = _get_user_record(username)
        if not user:
            return False

        logs = user.setdefault("symptom_logs", [])
        kept = [entry for entry in logs if entry.get("log_id") != log_id]
        if len(kept) == len(logs):
            return False

        user["symptom_logs"] = kept
        _append_audit(
            user,
            "symptom_deleted",
            "Removed symptom tracker entry",
            metadata={"log_id": log_id},
        )
        _save_user_record(username, user)
        return True

    @staticmethod
    def save_medication(username: str, medication: Dict) -> Optional[Dict]:
        user = _get_user_record(username)
        if not user:
            return None

        normalized = _normalize_medication(medication)
        if not normalized["name"]:
            return None

        medications = user.setdefault("medications", [])
        updated = False
        for index, existing in enumerate(medications):
            same_id = existing.get("medication_id") == normalized.get("medication_id")
            same_name = existing.get("name", "").strip().lower() == normalized["name"].lower()
            if same_id or same_name:
                normalized["created_at"] = existing.get("created_at") or normalized["created_at"]
                medications[index] = normalized
                updated = True
                break

        if not updated:
            medications.append(normalized)

        _append_audit(
            user,
            "medication_saved",
            f"Saved medication: {normalized['name']}",
            metadata={
                "medication_id": normalized["medication_id"],
                "dose": normalized["dose"],
                "schedule": normalized["schedule"],
            },
        )
        _save_user_record(username, user)
        return normalized

    @staticmethod
    def get_medications(username: str) -> List[Dict]:
        user = _get_user_record(username)
        medications = deepcopy(user.get("medications", [])) if user else []
        medications.sort(
            key=lambda item: (
                item.get("name", "").lower(),
                item.get("updated_at", ""),
            )
        )
        return medications

    @staticmethod
    def delete_medication(username: str, medication_id: str) -> bool:
        user = _get_user_record(username)
        if not user:
            return False

        medications = user.setdefault("medications", [])
        kept = [entry for entry in medications if entry.get("medication_id") != medication_id]
        if len(kept) == len(medications):
            return False

        user["medications"] = kept
        _append_audit(
            user,
            "medication_deleted",
            "Removed medication from list",
            metadata={"medication_id": medication_id},
        )
        _save_user_record(username, user)
        return True

    @staticmethod
    def save_triage_summary(username: str, summary: Dict) -> Optional[Dict]:
        user = _get_user_record(username)
        if not user:
            return None

        payload = _normalize_triage_summary(summary)
        triage_summaries = user.setdefault("triage_summaries", [])
        triage_summaries.append(payload)
        triage_summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        user["triage_summaries"] = triage_summaries[:25]
        _append_audit(
            user,
            "triage_saved",
            f"Saved triage summary: {payload['urgency_level']} -> {payload['next_step']}",
            trace_id=payload.get("trace_id"),
            metadata={"summary_id": payload["summary_id"]},
        )
        _save_user_record(username, user)
        return payload

    @staticmethod
    def get_triage_summaries(username: str, limit: Optional[int] = 10) -> List[Dict]:
        user = _get_user_record(username)
        triage_summaries = deepcopy(user.get("triage_summaries", [])) if user else []
        triage_summaries.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        if limit is None:
            return triage_summaries
        return triage_summaries[:limit]

    @staticmethod
    def get_latest_triage_summary(username: str) -> Dict:
        summaries = UserStore.get_triage_summaries(username, limit=1)
        return summaries[0] if summaries else {}

    @staticmethod
    def get_longitudinal_memory(username: str) -> Dict:
        user = _get_user_record(username)
        if not user:
            return _default_longitudinal_memory()
        memory = deepcopy(user.get("longitudinal_memory", {}))
        default_memory = _default_longitudinal_memory()
        for key, value in default_memory.items():
            memory.setdefault(key, value)
        return memory

    @staticmethod
    def save_longitudinal_memory(
        username: str,
        summary: str,
        source: str = "conversation",
        metadata: Optional[Dict] = None,
    ) -> None:
        user = _get_user_record(username)
        if not user:
            return

        cleaned_summary = (summary or "").strip()
        memory = user.setdefault("longitudinal_memory", _default_longitudinal_memory())
        previous_summary = (memory.get("summary") or "").strip()
        if cleaned_summary == previous_summary:
            return

        memory["summary"] = cleaned_summary
        memory["updated_at"] = _utc_now()
        memory["source"] = source
        _append_audit(
            user,
            "longitudinal_memory_updated",
            f"Longitudinal memory refreshed from {source}",
            metadata=metadata or {"summary_length": len(cleaned_summary)},
        )
        _save_user_record(username, user)

    @staticmethod
    def get_chat_history(username: str) -> List[Dict]:
        user = _get_user_record(username)
        return deepcopy(user.get("conversation", [])) if user else []

    @staticmethod
    def set_chat_history(username: str, history: List[Dict]) -> None:
        user = _get_user_record(username)
        if not user:
            return
        user["conversation"] = [_normalize_message(message) for message in history]
        _append_audit(user, "conversation_replaced", "Conversation history replaced")
        _save_user_record(username, user)

    @staticmethod
    def clear_chat_history(username: str) -> None:
        user = _get_user_record(username)
        if not user:
            return
        user["conversation"] = []
        user["active_conversation_id"] = f"conv-{uuid4().hex[:12]}"
        _append_audit(user, "conversation_cleared", "Conversation history cleared")
        _save_user_record(username, user)

    @staticmethod
    def append_chat(username: str, message: Dict) -> None:
        user = _get_user_record(username)
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
        _save_user_record(username, user)

    @staticmethod
    def add_upload(username: str, upload_name: str, stored_path: Optional[str] = None) -> None:
        user = _get_user_record(username)
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
        _save_user_record(username, user)

    @staticmethod
    def save_document_summary(
        username: str,
        filename: str,
        summary: str,
        stored_path: Optional[str] = None,
    ) -> None:
        user = _get_user_record(username)
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
        _save_user_record(username, user)

    @staticmethod
    def get_document_summaries(username: str) -> List[Dict]:
        user = _get_user_record(username)
        return deepcopy(user.get("doc_summaries", [])) if user else []

    @staticmethod
    def get_uploads(username: str) -> List[Dict]:
        user = _get_user_record(username)
        return deepcopy(user.get("uploads", [])) if user else []

    @staticmethod
    def save_interaction_trace(username: str, trace: Dict) -> None:
        user = _get_user_record(username)
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
        _save_user_record(username, user)

    @staticmethod
    def get_interaction_traces(username: str, limit: Optional[int] = 25) -> List[Dict]:
        user = _get_user_record(username)
        traces = deepcopy(user.get("traces", [])) if user else []
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
        user = _get_user_record(username)
        if not user:
            return
        _append_audit(user, event, details, trace_id=trace_id, metadata=metadata)
        _save_user_record(username, user)

    @staticmethod
    def get_audit(username: str, limit: Optional[int] = 50) -> List[Dict]:
        user = _get_user_record(username)
        audit = deepcopy(user.get("audit", [])) if user else []
        audit.sort(key=lambda item: item.get("time", ""), reverse=True)
        if limit is None:
            return audit
        return audit[:limit]

    @staticmethod
    def record_video_generated(username: str) -> None:
        """Stamps the current UTC time as the last video generation timestamp."""
        user = _get_user_record(username)
        if not user:
            return
        user.setdefault("profile", {})["last_video_generated_at"] = _utc_now()
        _append_audit(user, "video_generated", "Sora-2 video generated")
        _save_user_record(username, user)

    @staticmethod
    def get_last_video_generated_at(username: str) -> str:
        """Returns the ISO-8601 UTC timestamp of the last video generation, or empty string."""
        user = _get_user_record(username)
        if not user:
            return ""
        return user.get("profile", {}).get("last_video_generated_at", "")

    @staticmethod
    def export_user_snapshot(username: str) -> Dict:
        user = _get_user_record(username)
        if not user:
            return {}

        exported = deepcopy(user)
        exported.pop("password_hash", None)
        exported.pop("salt", None)
        return exported
