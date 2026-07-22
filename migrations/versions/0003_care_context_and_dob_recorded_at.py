"""care_context/follow_up_preferences on accounts, dob_recorded_at on
patients -- more legacy profile fields found while building PR5's SQL
repository layer.

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "accounts",
        sa.Column("care_context", sa.String(255), nullable=False, server_default="Personal health guidance"),
    )
    op.add_column(
        "accounts",
        sa.Column("follow_up_preferences", sa.String(255), nullable=False, server_default=""),
    )
    op.add_column("patients", sa.Column("dob_recorded_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("patients", "dob_recorded_at")
    op.drop_column("accounts", "follow_up_preferences")
    op.drop_column("accounts", "care_context")
