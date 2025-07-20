"""create pdfs table

Revision ID: 497c4df89e2f
Revises: 6224ffbc9ae2
Create Date: 2025-07-20 16:14:10.641699

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '497c4df89e2f'
down_revision: Union[str, Sequence[str], None] = '6224ffbc9ae2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
