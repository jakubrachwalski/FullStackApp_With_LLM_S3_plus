"""create pdfs table

Revision ID: 2bd5421e28e8
Revises: 0c12f9ce869f
Create Date: 2025-07-20 16:29:24.363745

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2bd5421e28e8'
down_revision: Union[str, Sequence[str], None] = '0c12f9ce869f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
