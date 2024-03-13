from pydantic.env_settings import BaseSettings


class ZeusBsoSettings(BaseSettings):
    """App setting"""

    database_url: str
    database_password: str = ""
    echo_sql: bool = True
    log_level: str = "DEBUG"

    class Config:
        env_prefix = "ZEUS_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = ZeusBsoSettings()
