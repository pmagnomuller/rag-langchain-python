from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the RAG system.

    Values can be overridden via environment variables with the `RAG_` prefix
    (e.g. `RAG_OLLAMA_MODEL`, `RAG_DATA_DIR`) and/or a `.env` file.
    """

    # Where to read documents from
    data_dir: Path = Path("data")

    # Where to persist the Chroma vector store
    chroma_dir: Path = Path("chroma")

    # Ollama model used for generation (chat model)
    ollama_model: str = "llama3"

    # Ollama model used for embeddings
    embedding_model: str = "nomic-embed-text"

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()

