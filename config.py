# ============================================================
# config.py  — Project-wide settings (Pydantic BaseSettings)
# Reads from environment variables / .env file automatically.
# Import: from config import settings
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """
    Single source of truth for all configuration.
    Values are loaded from .env (or environment variables).
    Pydantic validates types at startup — fail fast on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # Ignore unknown env vars silently
    )

    # --- LLM Providers ---
    groq_api_key: str = Field(default="", description="Groq API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    google_api_key: str = Field(default="", description="Google API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    qwen_api_key: str = Field(default="", description="Qwen API key")

    # --- Web Search ---
    tavily_api_key: str = Field(default="", description="Tavily API key")

    # --- Model Configuration ---
    default_llm_provider: str = Field(default="groq")
    groq_model_name: str = Field(default="llama-3.3-70b-versatile")
    google_model_name: str = Field(default="models/gemma-4-31b-it")
    deepseek_model_name: str = Field(default="deepseek/deepseek-chat")
    qwen_model_name: str = Field(default="qwen/qwen3-next-80b-a3b-instruct:free")

    # --- Agent Behavior ---
    agent_timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    retry_base_delay: float = Field(default=1.0)

    # --- Vector Store ---
    faiss_index_path: str = Field(default="./data/faiss_index")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)

    # --- Evaluation ---
    eval_output_path: str = Field(
        default="./evaluation/results/benchmark_report.json"
    )

    # --- FastAPI ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # --- Streamlit ---
    streamlit_port: int = Field(default=8501)

    def get_llm_model_name(self) -> str:
        """Returns the primary model name based on provider setting."""
        if self.default_llm_provider == "groq":
            return self.groq_model_name
        elif self.default_llm_provider == "google":
            return self.google_model_name
        elif self.default_llm_provider == "deepseek":
            return self.deepseek_model_name
        elif self.default_llm_provider == "qwen":
            return self.qwen_model_name
        return self.groq_model_name  # fallback


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance — only parsed once per process.
    Use this everywhere: from config import settings
    """
    return Settings()


# Module-level singleton for convenience
settings = get_settings()
