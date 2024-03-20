"""App setting is read here."""

from pydantic.env_settings import BaseSettings
from dotenv import find_dotenv


class ZeusBsoSettings(BaseSettings):
    """App setting.

    Attributes:
        database_url: url of database for the server
        database_password: password of database if any
        echo_sql: log sql statements it executes
        log_level: level of log
    """

    database_url: str
    database_password: str = ""
    echo_sql: bool = True
    log_level: str = "DEBUG"

    class Config:
        """Model configuration.

        Set how to find the env variables and how to parse it.
        """

        env_prefix = "ZEUS_"
        env_file = find_dotenv(filename=".env")
        env_file_encoding = "utf-8"


settings = ZeusBsoSettings()
