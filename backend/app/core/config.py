"""환경 설정 — backend/.env 에서 로드 (.env.example 참고)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    database_url: str = "postgresql+psycopg://wayfinder:wayfinder@localhost:5432/wayfinder"

    # LLM(Anthropic Claude). 임베딩은 로컬 모델(BAAI/bge-m3)이라 키가 필요 없음
    anthropic_api_key: str = ""

    # 공공데이터
    work24_api_key: str = ""
    hrdnet_api_key: str = ""
    ncs_api_key: str = ""
    seoul_job_api_key: str = ""
    gg_joba_api_key: str = ""

    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
