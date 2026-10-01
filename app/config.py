from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data
    data_dir: str = "data/scifact"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection: str = "scifact_V1"

    # Models (FastEmbed, executed locally on CPU)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Search
    retrieve_k: int = 50
    final_k: int = 5
    use_rerank: bool = True

    # LLM
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 600

    # Logs
    log_file: str = "logs/requests.jsonl"


settings = Settings()
