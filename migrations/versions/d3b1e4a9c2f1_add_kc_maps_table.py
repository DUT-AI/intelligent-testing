"""add kc_maps table

Revision ID: d3b1e4a9c2f1
Revises: cbb75f345e2c
Create Date: 2026-05-20 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d3b1e4a9c2f1"
down_revision: Union[str, Sequence[str], None] = "cbb75f345e2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "kc_maps",
        sa.Column("concept_id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
    )
    op.drop_column("questions", "kc_routes_cleaned")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column("questions", sa.Column("kc_routes_cleaned", sa.JSON(), nullable=True))
    op.drop_table("kc_maps")
