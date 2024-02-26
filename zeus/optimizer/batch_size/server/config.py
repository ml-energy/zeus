import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Setting(BaseModel):
    """App setting"""

    database_url: str
    echo_sql: bool = True


setting = Setting(database_url=os.getenv("DATABASE_URL"))
