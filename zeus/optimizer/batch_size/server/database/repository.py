from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio.session import AsyncSession


class DatabaseRepository:
    def __init__(self, session: "AsyncSession"):
        self.session = session
