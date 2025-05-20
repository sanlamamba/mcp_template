"""Configuration management for the MCP server."""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # OpenAI API
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    LANGCHAIN_TRACING: str = os.getenv("LANGCHAIN_TRACING", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"


settings = Settings()
