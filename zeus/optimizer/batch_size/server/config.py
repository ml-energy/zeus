"""Server global configurations."""

from __future__ import annotations
from typing import Union

from dotenv import find_dotenv
from zeus.utils.pydantic_v1 import BaseSettings, validator


class ZeusBsoSettings(BaseSettings):
    """App setting.

    Attributes:
        database_url: url of database for the server
        echo_sql: log sql statements it executes
        log_level: level of log
    """

    database_url: str
    echo_sql: Union[bool, str] = False  # To prevent conversion error for empty string
    log_level: str = "INFO"

    class Config:  # type: ignore
        """Model configuration.

        Set how to find the env variables and how to parse it.
        """

        env_prefix = "ZEUS_BSO_"
        env_file = find_dotenv(filename=".env")
        env_file_encoding = "utf-8"

    @validator("echo_sql")
    def _validate_echo_sql(cls, v) -> bool:
        if v is not None and isinstance(v, bool):
            return v
        elif v is not None and isinstance(v, str):
            if v.lower() == "false":
                return False
            elif v.lower() == "true":
                return True
        return False

    @validator("log_level")
    def _validate_log_level(cls, v) -> str:
        if v is None or v not in {
            "NOTSET",
            "DEBUG",
            "INFO",
            "WARN",
            "ERROR",
            "CRITICAL",
        }:
            # Default log level
            return "INFO"
        return v


settings = ZeusBsoSettings()  # type: ignore
