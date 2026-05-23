"""add student_sessions table

Revision ID: 9a7b6c5d4e3f
Revises: f3c1a8b0c2a1
Create Date: 2026-05-24 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a7b6c5d4e3f"
down_revision: Union[str, Sequence[str], None] = "f3c1a8b0c2a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "student_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dataset_type", sa.String(length=50), nullable=False),
        sa.Column("fold", sa.Integer(), nullable=True),
        sa.Column("uid", sa.Integer(), nullable=False),
        sa.Column("questions", sa.Text(), nullable=False),
        sa.Column("concepts", sa.Text(), nullable=False),
        sa.Column("responses", sa.Text(), nullable=False),
        sa.Column("timestamps", sa.Text(), nullable=False),
        sa.Column("is_repeat", sa.Text(), nullable=False),
        sa.Column("response_times", sa.Text(), nullable=True),
        sa.Column("cidxs", sa.Text(), nullable=True),
        sa.Column("selectmasks", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_student_sessions_dataset_type"),
        "student_sessions",
        ["dataset_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_student_sessions_fold"),
        "student_sessions",
        ["fold"],
        unique=False,
    )
    op.create_index(
        op.f("ix_student_sessions_uid"),
        "student_sessions",
        ["uid"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_student_sessions_uid"), table_name="student_sessions")
    op.drop_index(op.f("ix_student_sessions_fold"), table_name="student_sessions")
    op.drop_index(
        op.f("ix_student_sessions_dataset_type"),
        table_name="student_sessions",
    )
    op.drop_table("student_sessions")
