"""Server global configurations."""

from dotenv import find_dotenv
from zeus.util.pydantic_v1 import BaseSettings


class ZeusBsoSettings(BaseSettings):
    """App setting.

    Attributes:
        database_url: url of database for the server
        echo_sql: log sql statements it executes
        log_level: level of log
    """

    database_url: str
    echo_sql: bool = False
    log_level: str = "DEBUG"

    class Config:
        """Model configuration.

        Set how to find the env variables and how to parse it.
        """

        env_prefix = "ZEUS_BSO_"
        env_file = find_dotenv(filename=".env")
        env_file_encoding = "utf-8"


settings = ZeusBsoSettings()
