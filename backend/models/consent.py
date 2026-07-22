from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base, TimestampMixin


class ConsentStatus(str, enum.Enum):
    pending = "pending"
    active = "active"
    denied = "denied"
    revoked = "revoked"
    expired = "expired"


class ConsentScope(str, enum.Enum):
    """Fixed consent bundles a patient approves -- not per-field checkboxes.
    `previsit_summary` covers overview + care plans + notes; `chat_history`
    is a separate opt-in a patient can additionally grant on the same
    approval. Values are deliberately shaped like SMART-on-FHIR scopes
    (see backend/fhir/) so a later phase can map to `patient/*.read`
    without a schema change."""

    previsit_summary = "previsit_summary"
    chat_history = "chat_history"


class ConsentGrant(Base, TimestampMixin):
    __tablename__ = "consent_grants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clinician_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[ConsentStatus] = mapped_column(
        Enum(ConsentStatus, name="consent_status"), nullable=False, default=ConsentStatus.pending
    )
    # List[ConsentScope] values, e.g. ["previsit_summary"] or ["previsit_summary", "chat_history"].
    scope: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    request_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    decision_note: Mapped[str] = mapped_column(Text, nullable=False, default="")

    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )

    # At most one pending/active grant per (patient, clinician) pair is
    # enforced by a partial unique index (WHERE status IN ('pending','active'))
    # added directly in migrations/versions/0001_initial_schema.py -- a plain
    # UniqueConstraint can't express "unique only among certain statuses" and
    # would wrongly block a new request after a prior grant was denied/
    # revoked/expired.
