from pathlib import Path
from typing import List, Optional, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Universo y configuración SEC (con "Any" para compatibilidad)
    ticker_universe: Any = Field(default_factory=list)
    sec_user_agent: str = Field(..., description="Required by SEC API (e.g. 'Name (email@domain.com)')")

    @field_validator("ticker_universe", mode="before")
    @classmethod
    def _split_universe(cls, value):
        if isinstance(value, str):
            return [t.strip().upper() for t in value.split(",") if t.strip()]
        return value

    # Filtros
    margen_bruto_minimo: float = Field(default=0.50, ge=0.0, le=1.0)
    eps_years_lookback: int = Field(default=5, ge=2, le=20)
    shares_years_lookback: int = Field(default=3, ge=2, le=10)
    per_discount_min: float = Field(default=0.20, ge=0.0, le=1.0)
    per_history_years: int = Field(default=5, ge=2, le=20)
    rsi_oversold_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    rsi_extreme_oversold: float = Field(default=20.0, ge=0.0, le=100.0)
    ma_period: int = Field(default=200, ge=20, le=500)
    rsi_period: int = Field(default=14, ge=2, le=50)

    # Ejecución y persistencia
    concurrency_limit: int = Field(default=5, ge=1, le=50)
    retry_max_attempts: int = Field(default=5, ge=1, le=20)
    retry_base_delay: float = Field(default=2.0, ge=0.1, le=60.0)
    scan_interval_seconds: int = Field(default=3600, ge=60)
    database_path: Path = Field(default=Path("./data/arquitecto.db"))
    log_level: str = Field(default="INFO")
    log_path: Path = Field(default=Path("./logs/arquitecto.jsonl"))

    # ---------------------------------------------------------------------
    # Alertas (Lo que faltaba para que main.py no falle)
    # ---------------------------------------------------------------------
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from: Optional[str] = None
    smtp_to: Optional[str] = None

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def smtp_enabled(self) -> bool:
        return bool(
            self.smtp_host
            and self.smtp_user
            and self.smtp_password
            and self.smtp_from
            and self.smtp_to
        )

_settings_singleton: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton