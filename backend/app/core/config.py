"""환경 설정 — backend/.env 에서 로드 (.env.example 참고)."""
from pathlib import Path

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "dev"
    database_url: str = "postgresql+psycopg://wayfinder:wayfinder@localhost:5432/wayfinder"

    # LLM 프로바이더: claude(Anthropic API) 또는 ollama(로컬 모델, 키 불필요)
    # 임베딩은 어느 쪽이든 로컬 모델(BAAI/bge-m3)이라 키가 필요 없음
    llm_provider: Literal["claude", "ollama"] = "claude"
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    # 입력+출력 토큰 합 상한. 프롬프트가 이보다 길면 앞부분이 잘려 지시문이 사라진다
    ollama_num_ctx: int = 8192
    # CPU 추론은 출력 약 5tok/s — 호출 하나가 수 분 걸릴 수 있다
    ollama_timeout_seconds: float = 600

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
