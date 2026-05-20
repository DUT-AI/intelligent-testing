"""add exam sessions and interactions tables

Revision ID: e1a2b3c4d5f6
Revises: d3b1e4a9c2f1
Create Date: 2026-05-20 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e1a2b3c4d5f6"
down_revision: Union[str, Sequence[str], None] = "d3b1e4a9c2f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create exam_sessions table
    op.create_table(
        "exam_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=False),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        sa.Column("total_questions", sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column("final_theta", sa.Float(), nullable=True),
        sa.Column("is_completed", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_exam_sessions_id"), "exam_sessions", ["id"], unique=False)
    op.create_index(op.f("ix_exam_sessions_user_id"), "exam_sessions", ["user_id"], unique=False)

    # Create exam_interactions table
    op.create_table(
        "exam_interactions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.String(length=50), nullable=False),
        sa.Column("step_order", sa.Integer(), nullable=False),
        sa.Column("is_correct", sa.Boolean(), nullable=False),
        sa.Column("response_time_sec", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("theta_after", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["session_id"], ["exam_sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"], ondelete="RESTRICT"),
    )
    op.create_index(op.f("ix_exam_interactions_id"), "exam_interactions", ["id"], unique=False)
    op.create_index(op.f("ix_exam_interactions_session_id"), "exam_interactions", ["session_id"], unique=False)
    op.create_index(op.f("ix_exam_interactions_question_id"), "exam_interactions", ["question_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_exam_interactions_question_id"), table_name="exam_interactions")
    op.drop_index(op.f("ix_exam_interactions_session_id"), table_name="exam_interactions")
    op.drop_index(op.f("ix_exam_interactions_id"), table_name="exam_interactions")
    op.drop_table("exam_interactions")

    op.drop_index(op.f("ix_exam_sessions_user_id"), table_name="exam_sessions")
    op.drop_index(op.f("ix_exam_sessions_id"), table_name="exam_sessions")
    op.drop_table("exam_sessions")
