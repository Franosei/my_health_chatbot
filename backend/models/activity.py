from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AccountActivityLog(Base):
    """General-purpose self-activity log for an account's own actions
    (login, profile_updated, medication_saved, ...) -- mirrors the legacy
    per-user `audit` list (backend/user_store.py's `_append_audit`).

    Distinct from backend/models/audit.py's AuditLogEntry, which is scoped
    specifically to a *clinician* accessing *another* patient's record.
    This table has no security-gating role; it's informational/UX history,
    same as it was in the legacy store."""

    __tablename__ = "account_activity_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    event: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[str] = mapped_column(Text, nullable=False, default="")
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    event_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
