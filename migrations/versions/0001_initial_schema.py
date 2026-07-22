"""Initial relational schema: accounts, patients, clinical records, consent grants, audit log.

Revision ID: 0001
Revises:
Create Date: 2026-07-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(512), nullable=False),
        sa.Column("password_salt", sa.String(255), nullable=True),
        sa.Column("password_algo", sa.String(64), nullable=False),
        sa.Column("account_kind", sa.Enum("patient", "clinician", name="account_kind"), nullable=False),
        sa.Column("role_label", sa.String(64), nullable=False),
        sa.Column("clinical_role", sa.String(64), nullable=False),
        sa.Column("organization", sa.String(255), nullable=False),
        sa.Column("email_verified", sa.Boolean, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False),
        sa.Column("terms_version", sa.String(64), nullable=False),
        sa.Column("terms_role", sa.String(64), nullable=False),
        sa.Column("terms_accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("privacy_accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("legacy_username", sa.String(255), nullable=True, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_accounts_username", "accounts", ["username"])
    op.create_index("ix_accounts_email", "accounts", ["email"])
    op.create_index("ix_accounts_legacy_username", "accounts", ["legacy_username"])

    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("patient_id", sa.String(32), nullable=False, unique=True),
        sa.Column("date_of_birth", sa.Date, nullable=True),
        sa.Column("biological_sex", sa.String(32), nullable=False),
        sa.Column("longitudinal_memory", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_patients_patient_id", "patients", ["patient_id"])

    def _child_table(name: str, columns: list[sa.Column], extra_indexes: list[tuple] | None = None):
        op.create_table(
            name,
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "patient_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("patients.id", ondelete="CASCADE"),
                nullable=False,
            ),
            *columns,
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(f"ix_{name}_patient_id", name, ["patient_id"])
        for index_name, cols in extra_indexes or []:
            op.create_index(index_name, name, list(cols))

    _child_table(
        "medications",
        [
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("dose", sa.String(255), nullable=False),
            sa.Column("schedule", sa.String(255), nullable=False),
            sa.Column("reason", sa.String(255), nullable=False),
            sa.Column("started_on", sa.String(32), nullable=False),
            sa.Column("notes", sa.Text, nullable=False),
        ],
    )
    _child_table(
        "conditions",
        [
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("recorded_on", sa.String(32), nullable=False),
            sa.Column("notes", sa.Text, nullable=False),
        ],
    )
    _child_table(
        "allergies",
        [
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("reaction", sa.String(255), nullable=False),
            sa.Column("severity", sa.String(32), nullable=False),
            sa.Column("allergy_type", sa.String(32), nullable=False),
            sa.Column("confirmed", sa.Boolean, nullable=False),
            sa.Column("notes", sa.Text, nullable=False),
        ],
    )
    _child_table(
        "vitals_entries",
        [
            sa.Column("recorded_on", sa.String(32), nullable=False),
            sa.Column("type", sa.String(64), nullable=False),
            sa.Column("value", sa.String(64), nullable=False),
            sa.Column("unit", sa.String(32), nullable=False),
            sa.Column("notes", sa.Text, nullable=False),
        ],
        extra_indexes=[("ix_vitals_patient_recorded", ("patient_id", "recorded_on"))],
    )
    _child_table(
        "symptom_logs",
        [
            sa.Column("symptom", sa.String(255), nullable=False),
            sa.Column("logged_for", sa.String(32), nullable=False),
            sa.Column("severity", sa.Integer, nullable=False),
            sa.Column("triggers", sa.Text, nullable=False),
            sa.Column("notes", sa.Text, nullable=False),
        ],
    )
    _child_table(
        "chat_messages",
        [
            sa.Column("role", sa.String(32), nullable=False),
            sa.Column("content", sa.Text, nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("sources", postgresql.JSONB, nullable=False),
            sa.Column("trace_id", sa.String(64), nullable=True),
            sa.Column("message_metadata", postgresql.JSONB, nullable=False),
        ],
        extra_indexes=[("ix_chat_patient_timestamp", ("patient_id", "timestamp"))],
    )
    _child_table(
        "care_plans",
        [
            sa.Column("condition", sa.String(255), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("body", postgresql.JSONB, nullable=False),
            sa.Column("clinical_context", postgresql.JSONB, nullable=False),
            sa.Column("validation", postgresql.JSONB, nullable=False),
            sa.Column("gp_prep_summary", sa.Text, nullable=True),
        ],
    )
    _child_table(
        "clinical_notes",
        [
            sa.Column("subjective", sa.Text, nullable=False),
            sa.Column("objective", sa.Text, nullable=False),
            sa.Column("assessment", sa.Text, nullable=False),
            sa.Column("plan", sa.Text, nullable=False),
            sa.Column("urgency_level", sa.String(32), nullable=False),
            sa.Column("requires_gp_visit", sa.Boolean, nullable=False),
            sa.Column("gp_visit_reason", sa.Text, nullable=False),
            sa.Column(
                "edited_by_account_id",
                postgresql.UUID(as_uuid=True),
                sa.ForeignKey("accounts.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("email_sent", sa.Boolean, nullable=False),
            sa.Column("email_sent_at", sa.DateTime(timezone=True), nullable=True),
        ],
    )
    _child_table(
        "uploads",
        [
            sa.Column("file_name", sa.String(512), nullable=False),
            sa.Column("stored_path", sa.String(1024), nullable=False),
            sa.Column("content_hash", sa.String(128), nullable=False),
            sa.Column("summary_available", sa.Boolean, nullable=False),
        ],
    )

    op.create_table(
        "document_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "upload_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("uploads.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "consent_grants",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "clinician_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum("pending", "active", "denied", "revoked", "expired", name="consent_status"),
            nullable=False,
        ),
        sa.Column("scope", postgresql.JSONB, nullable=False),
        sa.Column("request_reason", sa.Text, nullable=False),
        sa.Column("decision_note", sa.Text, nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "revoked_by_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_consent_grants_patient_id", "consent_grants", ["patient_id"])
    op.create_index("ix_consent_grants_clinician_account_id", "consent_grants", ["clinician_account_id"])
    # At most one pending/active grant per (patient, clinician) pair.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_consent_grant_active_pair
        ON consent_grants (patient_id, clinician_account_id)
        WHERE status IN ('pending', 'active')
        """
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "actor_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("actor_role_at_time", sa.String(64), nullable=False),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "action",
            sa.Enum(
                "login",
                "patient_read_self",
                "clinician_access_requested",
                "clinician_access_granted",
                "clinician_access_denied",
                "clinician_access_revoked",
                "clinician_read_previsit_summary",
                "clinician_read_gp_prep",
                "clinician_read_chat_history",
                "clinician_read_clinical_notes",
                name="audit_action",
            ),
            nullable=False,
        ),
        sa.Column("resource_type", sa.String(64), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=True),
        sa.Column("outcome", sa.Enum("success", "denied", "error", name="audit_outcome"), nullable=False),
        sa.Column(
            "consent_grant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consent_grants.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("request_ip", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_log_actor_account_id", "audit_log", ["actor_account_id"])
    op.create_index("ix_audit_log_patient_id", "audit_log", ["patient_id"])

    # Immutability: the app's runtime DB role only gets INSERT/SELECT on
    # audit_log -- UPDATE/DELETE must go through a separate, more privileged
    # role (e.g. for retention/compliance tooling), never the app itself.
    # APP_DB_ROLE defaults to the role that ran this migration if unset, so
    # this is a no-op grant (harmless) until a dedicated least-privilege
    # runtime role is provisioned -- revisit as part of the PR9 hardening pass.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_setting('app.runtime_role', true)) THEN
                EXECUTE format(
                    'REVOKE UPDATE, DELETE ON audit_log FROM %I',
                    current_setting('app.runtime_role')
                );
                EXECUTE format(
                    'GRANT INSERT, SELECT ON audit_log TO %I',
                    current_setting('app.runtime_role')
                );
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.execute("DROP INDEX IF EXISTS uq_consent_grant_active_pair")
    op.drop_table("consent_grants")
    op.drop_table("document_summaries")
    for table in (
        "uploads",
        "clinical_notes",
        "care_plans",
        "chat_messages",
        "symptom_logs",
        "vitals_entries",
        "allergies",
        "conditions",
        "medications",
    ):
        op.drop_table(table)
    op.drop_table("patients")
    op.drop_table("accounts")
    op.execute("DROP TYPE IF EXISTS audit_outcome")
    op.execute("DROP TYPE IF EXISTS audit_action")
    op.execute("DROP TYPE IF EXISTS consent_status")
    op.execute("DROP TYPE IF EXISTS account_kind")
