"""
Application configuration for the Multi-Format Knowledge Base.

All settings are loaded from environment variables (the .env file).
This means the same code works everywhere — only the .env changes:
- Your laptop: uses your personal API keys, debug logging
- Docker: uses service hostnames (qdrant, redis), production settings
- Testing: could use mock API keys and in-memory stores
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Each field maps to an environment variable.
    For example, 'openai_api_key' reads from the OPENAI_API_KEY variable.
    Pydantic automatically handles the uppercase/lowercase conversion.
    """

    # Application metadata
    app_name: str = "Multi-Format Knowledge Base"
    app_version: str = "1.0.0"
    debug: bool = False

    # OpenAI settings
    openai_api_key: str = ""  # Will be loaded from .env
    openai_model: str = "gpt-4o-mini"  # Good balance of quality and cost
    embedding_model: str = "text-embedding-3-small"  # Same as Phase 2
    max_tokens: int = 2000

    # Qdrant settings (vector database — replaces ChromaDB for hybrid search)
    qdrant_host: str = "localhost"  # Changes to "qdrant" inside Docker
    qdrant_port: int = 6333
    collection_name: str = "knowledge_base"

    # Redis settings (semantic cache)
    redis_host: str = "localhost"  # Changes to "redis" inside Docker
    redis_port: int = 6379

    # Cohere settings (re-ranking)
    cohere_api_key: str = ""  # For the re-ranker

    # Retrieval settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 20  # Get 20 candidates from vector search
    rerank_top_n: int = 5      # Re-ranker returns top 5

    # File upload settings
    max_upload_size_mb: int = 50
    upload_dir: str = "data/uploads"

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a single instance — imported everywhere in the app
settings = Settings()
