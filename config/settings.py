from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # GCP project
    gcp_project_id: str
    gcp_region: str = "us-central1"

    # GCP resources
    gcs_bucket: str
    bq_dataset: str = "nlp_demo"

    # Optional: impersonate a service account instead of using caller's ADC identity
    impersonate_sa: str | None = None

    # Gemini model identifiers (referenced later in extraction / summarization)
    gemini_model: str = "gemini-2.5-pro"
    gemini_model_flash: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
