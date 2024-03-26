"""Database repository (directly interacting with db) base class."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio.session import AsyncSession


class DatabaseRepository:
    """Base class for all repositories."""

    def __init__(self, session: AsyncSession) -> None:
        """Initizalize session."""
        self.session = session
