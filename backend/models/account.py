from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Enum, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin


class AccountKind(str, enum.Enum):
    """Authoritative authorization signal -- distinct from the free-text,
    cosmetic `role` field (Doctor/Nurse/Patient/...) that only shapes AI
    persona. A "Doctor" `role` with `account_kind=patient` can only ever
    view their own data; only `account_kind=clinician` can request
    cross-patient access, and only through a consent grant."""

    patient = "patient"
    clinician = "clinician"


class Account(Base, TimestampMixin):
    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    password_salt: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    password_algo: Mapped[str] = mapped_column(String(64), nullable=False, default="argon2id")

    account_kind: Mapped[AccountKind] = mapped_column(
        Enum(AccountKind, name="account_kind"), nullable=False
    )
    role_label: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    clinical_role: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    organization: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    care_context: Mapped[str] = mapped_column(String(255), nullable=False, default="Personal health guidance")
    follow_up_preferences: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    email_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    terms_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    terms_role: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    terms_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    privacy_accepted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Temporary bridge for the JSON -> SQL migration (see backend/scripts/migrate_json_to_sql.py).
    # Kept for one release cycle post-cutover so re-runs are idempotent; safe to drop after that.
    legacy_username: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True, index=True)

    patient: Mapped[Optional["Patient"]] = relationship(  # noqa: F821
        back_populates="account", uselist=False, cascade="all, delete-orphan"
    )
