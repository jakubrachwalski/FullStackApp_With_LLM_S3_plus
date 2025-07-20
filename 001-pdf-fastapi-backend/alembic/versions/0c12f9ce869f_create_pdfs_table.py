"""create pdfs table

Revision ID: 0c12f9ce869f
Revises: 497c4df89e2f
Create Date: 2025-07-20 16:23:44.504510

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0c12f9ce869f'
down_revision: Union[str, Sequence[str], None] = '497c4df89e2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
