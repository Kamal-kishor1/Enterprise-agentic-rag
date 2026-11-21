import os

from dotenv import load_dotenv

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

    # Redis / In-Memory Response Cache
    REDIS_URL = os.getenv("REDIS_URL")
    CACHE_TTL_SECONDS = int(
        os.getenv("CACHE_TTL_SECONDS") or "3600"
    )  # Default to 1 hour if not set

    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(
        ","
    )  # Default to all origins if not set


settings = Settings()
