"""create pdfs table

Revision ID: 677841be8ed6
Revises: 60b85b1ce809
Create Date: 2025-07-13 15:53:03.967572

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '677841be8ed6'
down_revision: Union[str, Sequence[str], None] = '60b85b1ce809'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
