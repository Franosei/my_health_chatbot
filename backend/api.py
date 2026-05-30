from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.clinical_trials import build_trial_search_profile, find_matching_trials
from backend.product_config import (
    FOUNDER_NAME,
    PRIVACY_NOTICE_POINTS,
    PRODUCT_NAME,
    PRODUCT_SUBTITLE,
    PRODUCT_TAGLINE,
    ROLE_OPTIONS,
    ROLE_TERMS,
    SUPPORT_EMAIL,
    TERMS_VERSION,
    default_care_context_for_role,
    get_terms_for_role,
    is_clinician_role,
)
from backend.rag_system import RAGEngine
from backend.upload_verification import verify_saved_pdf
from backend.user_store import UserStore
from backend.voice_transcriber import VoiceTranscriber

load_dotenv()

app = FastAPI(title=f"{PRODUCT_NAME} API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RAG_ENGINE: Optional[RAGEngine] = None
_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _token_secret() -> bytes:
    secret = os.getenv("APP_SECRET") or os.getenv("SECRET_KEY") or "dr-charlotte-local-dev-secret"
    return secret.encode("utf-8")


def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _create_token(username: str) -> str:
    payload = {
        "sub": username.strip().lower(),
        "exp": int(time.time()) + _TOKEN_TTL_SECONDS,
    }
    payload_part = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(_token_secret(), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{_b64_encode(signature)}"


def _read_token(token: str) -> str:
    try:
        payload_part, signature_part = token.split(".", 1)
        expected_signature = hmac.new(
            _token_secret(),
            payload_part.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = _b64_decode(signature_part)
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise ValueError("bad signature")
        payload = json.loads(_b64_decode(payload_part))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        username = str(payload.get("sub", "")).strip().lower()
        if not username or not UserStore.get_user_profile(username):
            raise ValueError("unknown user")
        return username
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Sign in again to continue.") from exc


def current_user(authorization: str = Header(default="")) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing access token.")
    return _read_token(token)


def _get_rag_engine() -> RAGEngine:
    global _RAG_ENGINE
    if _RAG_ENGINE is None:
        _RAG_ENGINE = RAGEngine()
    return _RAG_ENGINE


def _json_line(payload: Dict) -> bytes:
    return (json.dumps(payload, default=str) + "\n").encode("utf-8")


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip()
    return cleaned or "upload.pdf"


def _public_profile(username: str) -> Dict:
    profile = UserStore.get_user_profile(username)
    return {
        "username": username,
        **profile,
    }


def _snapshot(username: str) -> Dict:
    uploads = UserStore.get_uploads(username)
    symptom_logs = UserStore.get_symptom_logs(username, limit=None)
    medications = UserStore.get_medications(username)
    allergies = UserStore.get_allergies(username)
    conditions = UserStore.get_conditions(username)
    vitals = UserStore.get_vitals(username, limit=None)
    triage_summaries = UserStore.get_triage_summaries(username, limit=None)
    latest_triage = triage_summaries[0] if triage_summaries else {}
    chat_history = UserStore.get_chat_history(username)

    return {
        "product": {
            "name": PRODUCT_NAME,
            "tagline": PRODUCT_TAGLINE,
            "subtitle": PRODUCT_SUBTITLE,
            "support_email": SUPPORT_EMAIL,
        },
        "user": username,
        "profile": _public_profile(username),
        "metrics": {
            "messages": len(chat_history),
            "documents": len(uploads),
            "symptoms": len(symptom_logs),
            "conditions": len(conditions),
            "medications": len(medications),
            "allergies": len(allergies),
            "vitals": len(vitals),
            "triage_records": len(triage_summaries),
        },
        "latest_triage": latest_triage,
        "chat_history": chat_history,
        "uploads": uploads,
        "document_summaries": UserStore.get_document_summaries(username),
        "symptom_logs": symptom_logs,
        "medications": medications,
        "allergies": allergies,
        "conditions": conditions,
        "vitals": vitals,
        "triage_summaries": triage_summaries,
        "traces": UserStore.get_interaction_traces(username, limit=10),
        "audit": UserStore.get_audit(username, limit=20),
        "memory": UserStore.get_longitudinal_memory(username),
        "trial_search_result": UserStore.get_trial_search_result(username),
    }


class LoginPayload(BaseModel):
    identifier: str
    password: str


class SignupPayload(BaseModel):
    full_name: str
    email: str
    username: str
    role: str
    password: str
    confirm_password: str
    organization: str = ""
    date_of_birth: str = ""
    biological_sex: str = ""
    accept_role_terms: bool = False
    accept_privacy: bool = False


class ProfilePayload(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    care_context: Optional[str] = None
    role: Optional[str] = None
    clinical_role: Optional[str] = None
    organization: Optional[str] = None
    follow_up_preferences: Optional[str] = None
    date_of_birth: Optional[str] = None
    biological_sex: Optional[str] = None


class ChatPayload(BaseModel):
    message: str


class SymptomPayload(BaseModel):
    symptom: str
    logged_for: str
    severity: int
    triggers: str = ""
    notes: str = ""


class ConditionPayload(BaseModel):
    name: str
    status: str = "active"
    recorded_on: str = ""
    notes: str = ""
    condition_id: str = ""


class MedicationPayload(BaseModel):
    name: str
    dose: str = ""
    schedule: str = ""
    reason: str = ""
    started_on: str = ""
    notes: str = ""
    medication_id: str = ""


class AllergyPayload(BaseModel):
    name: str
    reaction: str = ""
    severity: str = "unknown"
    allergy_type: str = "other"
    confirmed: bool = True
    notes: str = ""
    allergy_id: str = ""


class VitalsPayload(BaseModel):
    type: str
    value: str
    unit: str = ""
    recorded_on: str = ""
    notes: str = ""


class TrialSearchPayload(BaseModel):
    location: str
    max_results: int = 10


@app.get("/api/health")
def health() -> Dict:
    return {"ok": True, "product": PRODUCT_NAME}


@app.get("/api/config")
def config() -> Dict:
    return {
        "product_name": PRODUCT_NAME,
        "product_tagline": PRODUCT_TAGLINE,
        "product_subtitle": PRODUCT_SUBTITLE,
        "founder_name": FOUNDER_NAME,
        "support_email": SUPPORT_EMAIL,
        "terms_version": TERMS_VERSION,
        "role_options": ROLE_OPTIONS,
        "role_terms": ROLE_TERMS,
        "privacy_notice_points": PRIVACY_NOTICE_POINTS,
    }


@app.post("/api/auth/login")
def login(payload: LoginPayload) -> Dict:
    if not payload.identifier.strip() or not payload.password:
        raise HTTPException(status_code=400, detail="Enter your email or username and password.")
    if not UserStore.authenticate(payload.identifier, payload.password):
        raise HTTPException(status_code=401, detail="The email, username, or password is incorrect.")
    username = UserStore.resolve_login_username(payload.identifier)
    if not username:
        raise HTTPException(status_code=401, detail="We could not open your account.")
    UserStore.update_last_login(username)
    return {
        "token": _create_token(username),
        "profile": _public_profile(username),
        "snapshot": _snapshot(username),
    }


@app.post("/api/auth/signup")
def signup(payload: SignupPayload) -> Dict:
    full_name = payload.full_name.strip()
    username = payload.username.strip().lower()
    email = payload.email.strip().lower()
    name_tokens = re.findall(r"[A-Za-z]{2,}", full_name)

    if not full_name or not email or not username or not payload.password or not payload.confirm_password:
        raise HTTPException(status_code=400, detail="Full name, email, username, and password are required.")
    if len(name_tokens) < 2:
        raise HTTPException(status_code=400, detail="Enter your full name with at least first and last name.")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="The password and confirmation fields must match.")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Use a password with at least 8 characters.")
    if payload.role not in ROLE_OPTIONS:
        raise HTTPException(status_code=400, detail="Choose a valid account role.")
    if not payload.accept_role_terms or not payload.accept_privacy:
        raise HTTPException(status_code=400, detail="Accept the role terms and privacy notice before creating the account.")
    if UserStore.resolve_login_username(username):
        raise HTTPException(status_code=409, detail="That username is already taken.")
    if UserStore.resolve_login_username(email):
        raise HTTPException(status_code=409, detail="That email address is already registered.")

    accepted_at = _utc_now()
    created = UserStore.create_user(
        username,
        payload.password,
        display_name=full_name,
        email=email,
        care_context=default_care_context_for_role(payload.role),
        role=payload.role,
        clinical_role=payload.role,
        organization=payload.organization if is_clinician_role(payload.role) else "",
        terms_version=TERMS_VERSION,
        terms_role=payload.role,
        terms_accepted_at=accepted_at,
        privacy_accepted_at=accepted_at,
        date_of_birth=payload.date_of_birth,
        biological_sex=payload.biological_sex,
    )
    if not created:
        raise HTTPException(status_code=400, detail="Account creation failed. Try another username or email.")

    UserStore.update_last_login(username)
    return {
        "token": _create_token(username),
        "profile": _public_profile(username),
        "snapshot": _snapshot(username),
    }


@app.get("/api/me")
def me(username: str = Depends(current_user)) -> Dict:
    return {"profile": _public_profile(username), "snapshot": _snapshot(username)}


@app.get("/api/snapshot")
def snapshot(username: str = Depends(current_user)) -> Dict:
    return _snapshot(username)


@app.put("/api/profile")
def update_profile(payload: ProfilePayload, username: str = Depends(current_user)) -> Dict:
    updates = {key: value for key, value in payload.dict().items() if value is not None}
    if not UserStore.update_profile(username, updates):
        raise HTTPException(status_code=400, detail="Profile update failed.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/chat")
def clear_chat(username: str = Depends(current_user)) -> Dict:
    UserStore.clear_chat_history(username)
    return _snapshot(username)


@app.post("/api/chat/stream")
def stream_chat(payload: ChatPayload, username: str = Depends(current_user)) -> StreamingResponse:
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a message before sending.")

    def generate() -> Generator[bytes, None, None]:
        try:
            rag_engine = _get_rag_engine()
            chat_history = UserStore.get_chat_history(username)
        except Exception as exc:
            yield _json_line({"type": "error", "message": f"Failed to start the answer pipeline: {exc}"})
            yield _json_line({"type": "done"})
            return

        now = _utc_now()
        user_entry = {
            "role": "user",
            "content": question,
            "timestamp": now,
            "sources": [],
            "metadata": {},
        }
        UserStore.append_chat(username, user_entry)
        chat_history.append(user_entry)
        yield _json_line({"type": "user_message", "message": user_entry})

        try:
            payload_final = None
            streamed_answer_parts: List[str] = []
            for event in rag_engine.stream_user_question_events(
                question=question,
                chat_history=chat_history,
                user=username,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    yield _json_line({"type": "status", "message": event.get("message", "Working...")})
                elif event_type == "token":
                    delta = event.get("delta", "")
                    streamed_answer_parts.append(delta)
                    yield _json_line({"type": "token", "delta": delta})
                elif event_type == "final":
                    payload_final = event.get("payload")

            if not payload_final:
                raise RuntimeError("The answer pipeline did not return a payload.")

            image_b64 = ""
            if payload_final.get("image_bytes"):
                image_b64 = base64.b64encode(payload_final["image_bytes"]).decode("ascii")

            assistant_entry = {
                "role": "assistant",
                "content": payload_final["answer_markdown"],
                "timestamp": _utc_now(),
                "sources": payload_final.get("sources", []),
                "trace_id": payload_final.get("trace", {}).get("trace_id"),
                "metadata": {
                    "personal_context": payload_final.get("personal_context", []),
                    "longitudinal_memory": payload_final.get("longitudinal_memory", ""),
                    "triage_summary": payload_final.get("triage_summary", {}),
                    "medication_alerts": payload_final.get("medication_alerts", []),
                    "resolved_medications": payload_final.get("resolved_medications", []),
                    "trace": payload_final.get("trace", {}),
                    "image_url": payload_final.get("image_url", ""),
                    "image_b64": image_b64,
                    "image_caption": payload_final.get("image_caption", ""),
                    "video_url": payload_final.get("video_url", ""),
                    "video_caption": payload_final.get("video_caption", ""),
                    "video_rate_limit_msg": payload_final.get("video_rate_limit_msg", ""),
                    "follow_up_questions": payload_final.get("follow_up_questions", []),
                },
            }

            try:
                refreshed_memory = rag_engine.refresh_longitudinal_memory_from_turn(
                    user=username,
                    user_message=question,
                    personal_context=payload_final.get("personal_context", []),
                )
                if refreshed_memory:
                    assistant_entry["metadata"]["longitudinal_memory"] = refreshed_memory
            except Exception as exc:
                print(f"Longitudinal memory refresh failed: {exc}")

            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "assistant_message", "message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})
            yield _json_line({"type": "done"})
        except Exception as exc:
            error_message = (
                "## Response unavailable\n"
                f"I ran into an issue while building the answer: `{exc}`.\n\n"
                "Please try again, or narrow the question if the request is very broad."
            )
            assistant_entry = {
                "role": "assistant",
                "content": error_message,
                "timestamp": _utc_now(),
                "sources": [],
                "metadata": {},
            }
            UserStore.append_chat(username, assistant_entry)
            yield _json_line({"type": "error", "message": str(exc), "assistant_message": assistant_entry})
            yield _json_line({"type": "snapshot", "snapshot": _snapshot(username)})

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.post("/api/uploads")
async def upload_documents(
    files: List[UploadFile] = File(...),
    process_unverified: bool = Form(False),
    username: str = Depends(current_user),
) -> Dict:
    profile = UserStore.get_user_profile(username)
    expected_name = profile.get("display_name", username)
    save_dir = UserStore.get_upload_dir(username)
    ready_paths: List[Path] = []
    pending: List[Dict] = []

    for upload in files:
        filename = _safe_filename(upload.filename or "upload.pdf")
        if not filename.lower().endswith(".pdf"):
            pending.append(
                {
                    "file": filename,
                    "status": "unsupported",
                    "message": "Only PDF uploads are supported.",
                    "detected_names": [],
                }
            )
            continue
        path = save_dir / filename
        content = await upload.read()
        path.write_bytes(content)
        verification = verify_saved_pdf(path, expected_name)
        if verification.get("status") == "matched" or process_unverified:
            ready_paths.append(path)
        else:
            pending.append(verification)

    indexed = []
    if ready_paths:
        indexed = _get_rag_engine().ingest_documents(user=username, file_paths=ready_paths)

    return {
        "processed": indexed,
        "pending": pending,
        "snapshot": _snapshot(username),
    }


@app.post("/api/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...), username: str = Depends(current_user)) -> Dict:
    del username
    data = await audio.read()
    try:
        transcriber = VoiceTranscriber()
        return {"text": transcriber.transcribe(data, filename=audio.filename or "recording.webm")}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Voice transcription is unavailable: {exc}") from exc


@app.post("/api/symptoms")
def add_symptom(payload: SymptomPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.add_symptom_log(
        username,
        symptom=payload.symptom,
        logged_for=payload.logged_for,
        severity=payload.severity,
        triggers=payload.triggers,
        notes=payload.notes,
    )
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a symptom and date.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/symptoms/{log_id}")
def delete_symptom(log_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_symptom_log(username, log_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/conditions")
def save_condition(payload: ConditionPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_condition(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a condition name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/conditions/{condition_id}")
def delete_condition(condition_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_condition(username, condition_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/medications")
def save_medication(payload: MedicationPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_medication(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a medication name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/medications/{medication_id}")
def delete_medication(medication_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_medication(username, medication_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/allergies")
def save_allergy(payload: AllergyPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_allergy(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter an allergy name.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/allergies/{allergy_id}")
def delete_allergy(allergy_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_allergy(username, allergy_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.post("/api/vitals")
def save_vitals(payload: VitalsPayload, username: str = Depends(current_user)) -> Dict:
    saved = UserStore.save_vitals_entry(username, payload.dict(exclude_none=True))
    if not saved:
        raise HTTPException(status_code=400, detail="Enter a measurement type and value.")
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.delete("/api/vitals/{vitals_id}")
def delete_vitals(vitals_id: str, username: str = Depends(current_user)) -> Dict:
    UserStore.delete_vitals_entry(username, vitals_id)
    _get_rag_engine().restore_user_context(username)
    return _snapshot(username)


@app.get("/api/export/account")
def export_account(username: str = Depends(current_user)) -> JSONResponse:
    return JSONResponse(UserStore.export_user_snapshot(username))


@app.get("/api/export/summary.pdf")
def export_summary(username: str = Depends(current_user)) -> Response:
    UserStore.add_audit(username, "summary_generated", "Health summary generated")
    pdf = _get_rag_engine().build_summary_pdf_for_user(username)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{username}-health-summary.pdf"'},
    )


@app.get("/api/terms/{role_label}")
def terms_for_role(role_label: str) -> Dict:
    return get_terms_for_role(role_label)


@app.post("/api/trials/search")
def search_trials(payload: TrialSearchPayload, username: str = Depends(current_user)) -> Dict:
    profile = UserStore.get_user_profile(username)
    trial_profile = build_trial_search_profile(
        profile=profile,
        memory=UserStore.get_longitudinal_memory(username),
        symptom_logs=UserStore.get_symptom_logs(username, limit=None),
        medications=UserStore.get_medications(username),
        allergies=UserStore.get_allergies(username),
        conditions=UserStore.get_conditions(username),
        vitals=UserStore.get_vitals(username, limit=None),
        triage_summaries=UserStore.get_triage_summaries(username, limit=None),
    )
    result = find_matching_trials(
        profile=trial_profile,
        location_query=payload.location,
        max_results=max(1, min(payload.max_results, 25)),
    )
    UserStore.save_trial_search_result(username, result)
    return {"result": result, "snapshot": _snapshot(username)}


@app.get("/api/trials/result")
def trial_result(username: str = Depends(current_user)) -> Dict:
    return {"result": UserStore.get_trial_search_result(username)}


_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
