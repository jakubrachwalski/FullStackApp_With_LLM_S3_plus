"""create pdf table

Revision ID: 6224ffbc9ae2
Revises: 677841be8ed6
Create Date: 2025-07-20 15:48:39.269257

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6224ffbc9ae2'
down_revision: Union[str, Sequence[str], None] = '677841be8ed6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
