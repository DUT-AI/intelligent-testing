"""add kc_level_records table

Revision ID: f3c1a8b0c2a1
Revises: 0afbf46445e8
Create Date: 2026-05-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f3c1a8b0c2a1"
down_revision: Union[str, Sequence[str], None] = "0afbf46445e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "kc_level_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fold", sa.Integer(), nullable=False),
        sa.Column("uid", sa.Integer(), nullable=False),
        sa.Column("questions", sa.Text(), nullable=False),
        sa.Column("concepts", sa.Text(), nullable=False),
        sa.Column("responses", sa.Text(), nullable=False),
        sa.Column("timestamps", sa.Text(), nullable=False),
        sa.Column("selectmasks", sa.Text(), nullable=False),
        sa.Column("is_repeat", sa.Text(), nullable=False),
        sa.Column("dataset_type", sa.String(length=50), nullable=False),
        sa.Column("qidxs", sa.Text(), nullable=True),
        sa.Column("rest", sa.Text(), nullable=True),
        sa.Column("orirow", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_kc_level_records_dataset_type"),
        "kc_level_records",
        ["dataset_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_kc_level_records_fold"),
        "kc_level_records",
        ["fold"],
        unique=False,
    )
    op.create_index(
        op.f("ix_kc_level_records_uid"),
        "kc_level_records",
        ["uid"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_kc_level_records_uid"), table_name="kc_level_records")
    op.drop_index(op.f("ix_kc_level_records_fold"), table_name="kc_level_records")
    op.drop_index(
        op.f("ix_kc_level_records_dataset_type"),
        table_name="kc_level_records",
    )
    op.drop_table("kc_level_records")
