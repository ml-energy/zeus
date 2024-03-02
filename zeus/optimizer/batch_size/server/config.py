import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

"""
TODO: Add app setting too
"""


class Setting(BaseModel):
    """DB setting"""

    database_url: str
    echo_sql: bool = True


setting = Setting(database_url=os.getenv("DATABASE_URL"))
