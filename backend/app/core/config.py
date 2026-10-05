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
    # CPU 추론은 출력 약 5tok/s — 호출 하나가 수 분 걸릴 수 있다.
    # 600초는 로드맵 호출(실측 626초)이 넘겨 502가 났다 — 프론트 axios 타임아웃과 같은 30분으로 둔다
    ollama_timeout_seconds: float = 1800
    # 호출 사이에 모델이 메모리에서 내려가면 다음 호출이 재로딩부터 한다 (Ollama 기본 5분)
    ollama_keep_alive: str = "30m"
    # 같은 입력의 LLM 결과를 재사용하는 SQLite 캐시 경로. 빈 문자열이면 캐시를 끈다
    llm_cache_path: str = str(Path(__file__).resolve().parents[2] / ".cache" / "llm_cache.sqlite")
    # 원격 임베딩 서버(scripts/runpod/embed_server.py) 주소. 비우면 로컬 CPU로 계산한다.
    # 원격이 실패하거나 모델 리비전이 다르면 로컬 계산으로 돌아간다
    embedding_api_url: str = ""

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
