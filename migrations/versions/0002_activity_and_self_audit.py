"""Triage summaries, interaction traces, cached trial search, video-gen
rate-limit timestamp, and the general per-account self-activity log --
schema gaps found while designing PR5's repository swap (these exist in the
legacy JSON store but had no PR1 table/column).

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("patients", sa.Column("last_video_generated_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("patients", sa.Column("last_trial_search", postgresql.JSONB, nullable=True))

    op.create_table(
        "triage_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("urgency_level", sa.String(64), nullable=False),
        sa.Column("next_step", sa.Text, nullable=False),
        sa.Column("what_to_monitor", postgresql.JSONB, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("pathway_label", sa.String(128), nullable=False),
        sa.Column("decision_summary", sa.Text, nullable=False),
        sa.Column("immediate_actions", postgresql.JSONB, nullable=False),
        sa.Column("escalation_triggers", postgresql.JSONB, nullable=False),
        sa.Column("communication_points", postgresql.JSONB, nullable=False),
        sa.Column("rule_hits", postgresql.JSONB, nullable=False),
        sa.Column("guideline_references", postgresql.JSONB, nullable=False),
        sa.Column("logic_version", sa.String(64), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_triage_summaries_patient_id", "triage_summaries", ["patient_id"])

    op.create_table(
        "interaction_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_interaction_traces_patient_id", "interaction_traces", ["patient_id"])

    op.create_table(
        "account_activity_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event", sa.String(64), nullable=False),
        sa.Column("details", sa.Text, nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("event_metadata", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_account_activity_log_account_id", "account_activity_log", ["account_id"])


def downgrade() -> None:
    op.drop_table("account_activity_log")
    op.drop_table("interaction_traces")
    op.drop_table("triage_summaries")
    op.drop_column("patients", "last_trial_search")
    op.drop_column("patients", "last_video_generated_at")
