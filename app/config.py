import os

from dotenv import load_dotenv

# from pydantic import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "ENTERPRISE_RAG"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")

    # PostgreSQL Database URL for persistent session memory
    DATABASE_URL = os.getenv("DATABASE_URL")

    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(
        ","
    )  # Default to all origins if not set


settings = Settings()
