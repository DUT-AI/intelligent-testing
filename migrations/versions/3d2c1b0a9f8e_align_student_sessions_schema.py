"""align student_sessions schema

Revision ID: 3d2c1b0a9f8e
Revises: 9a7b6c5d4e3f
Create Date: 2026-05-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3d2c1b0a9f8e"
down_revision: Union[str, Sequence[str], None] = "9a7b6c5d4e3f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "student_sessions",
        "response_times",
        new_column_name="response_time",
        existing_type=sa.Text(),
        existing_nullable=True,
    )
    op.drop_column("student_sessions", "cidxs")
    op.drop_column("student_sessions", "selectmasks")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "student_sessions",
        sa.Column("selectmasks", sa.Text(), nullable=True),
    )
    op.add_column(
        "student_sessions",
        sa.Column("cidxs", sa.Text(), nullable=True),
    )
    op.alter_column(
        "student_sessions",
        "response_time",
        new_column_name="response_times",
        existing_type=sa.Text(),
        existing_nullable=True,
    )