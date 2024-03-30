"""Server global configurations."""

from dotenv import find_dotenv
from zeus.util.pydantic_v1 import BaseSettings, validator


class ZeusBsoSettings(BaseSettings):
    """App setting.

    Attributes:
        database_url: url of database for the server
        echo_sql: log sql statements it executes
        log_level: level of log
    """

    database_url: str
    echo_sql: bool = False
    log_level: str = "INFO"

    class Config:
        """Model configuration.

        Set how to find the env variables and how to parse it.
        """

        env_prefix = "ZEUS_BSO_"
        env_file = find_dotenv(filename=".env")
        env_file_encoding = "utf-8"

    @validator("echo_sql")
    def _validate_echo_sql(cls, v) -> bool:
        if v is None or not isinstance(v, bool):
            # Set default to false
            return False
        return v

    @validator("log_level")
    def _validate_log_level(cls, v) -> bool:
        if v is None or (
            v != "NOTSET"
            and v != "DEBUG"
            and v != "INFO"
            and v != "WARN"
            and v != "ERROR"
            and v != "CRITICAL"
        ):
            # Default log level
            return "INFO"
        return v


settings = ZeusBsoSettings()
print(settings)
