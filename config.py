from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings class using Pydantic Settings.
    It automatically parses configurations from environment variables
    or a local .env file with validation.
    """

    # PostgreSQL Configuration
    postgres_user: str = Field("admin", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(
        "secure_postgres_password_123", validation_alias="POSTGRES_PASSWORD"
    )
    db_host: str = Field("localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field("cpp_database", validation_alias="POSTGRES_DB")

    @property
    def database_url(self) -> str:
        """
        Dynamically constructs the PostgreSQL connection string.
        """
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.db_host}:{self.postgres_port}/{self.postgres_db}"

    # Tell Pydantic to read from .env if present
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a singleton settings object to be imported across the application
settings = Settings()
