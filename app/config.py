"""
Application configuration and artifact loading.

Settings are loaded from environment variables and/or a .env file
using pydantic-settings. Artifact JSON files are loaded into memory
once at startup via load_artifacts().
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------
    # App / API settings
    # -------------------------
    app_name: str = "Used Car Price API"
    app_version: str = "0.1.0"
    environment: Literal["dev", "test", "prod"] = "dev"
    debug: bool = False
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000

    # Can be passed as:
    # 1) comma-separated string
    #    CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
    # 2) JSON list
    #    CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
    cors_origins: list[str] = Field(default_factory=list)

    # -------------------------
    # Model / artifact locations
    # -------------------------
    model_dir: Path = ROOT_DIR / "models"
    api_artifacts_dir: Path = ROOT_DIR / "models" / "api_artifacts"

    catboost_model_name: str = "catboost_final.cbm"
    model_metadata_name: str = "model_metadata.json"
    categorical_vocabularies_name: str = "categorical_vocabularies.json"
    models_by_manufacturer_name: str = "models_by_manufacturer.json"
    numeric_medians_name: str = "numeric_medians.json"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                return json.loads(value)
            return [item.strip() for item in value.split(",") if item.strip()]
        raise TypeError("cors_origins must be a list or string")

    # -------------------------
    # Computed paths
    # -------------------------
    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    @property
    def catboost_model_path(self) -> Path:
        return self.model_dir / self.catboost_model_name

    @property
    def model_metadata_path(self) -> Path:
        return self.api_artifacts_dir / self.model_metadata_name

    @property
    def categorical_vocabularies_path(self) -> Path:
        return self.api_artifacts_dir / self.categorical_vocabularies_name

    @property
    def models_by_manufacturer_path(self) -> Path:
        return self.api_artifacts_dir / self.models_by_manufacturer_name

    @property
    def numeric_medians_path(self) -> Path:
        return self.api_artifacts_dir / self.numeric_medians_name

    # -------------------------
    # Startup validation
    # -------------------------
    def validate_required_files(self) -> None:
        """Check that all required model and artifact files exist."""
        required_paths = [
            self.catboost_model_path,
            self.model_metadata_path,
            self.categorical_vocabularies_path,
            self.models_by_manufacturer_path,
            self.numeric_medians_path,
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            joined = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "Missing required model/artifact files:\n"
                f"{joined}"
            )


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (created once, reused everywhere)."""
    return Settings()


# ===============================
# Artifact loading
# ===============================

def load_artifacts(settings: Settings | None = None) -> dict:
    """Load all JSON artifacts into memory. Called once at app startup.

    Returns a dict with keys:
        model_metadata, categorical_vocabs, models_by_manufacturer, numeric_medians
    """
    if settings is None:
        settings = get_settings()
    settings.validate_required_files()

    def _load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    return {
        "model_metadata": _load(settings.model_metadata_path),
        "categorical_vocabs": _load(settings.categorical_vocabularies_path),
        "models_by_manufacturer": _load(settings.models_by_manufacturer_path),
        "numeric_medians": _load(settings.numeric_medians_path),
    }