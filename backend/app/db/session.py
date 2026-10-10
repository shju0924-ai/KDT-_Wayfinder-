"""SQLAlchemy 엔진·세션 팩토리.

라우터는 세션을 주입받지 않는다 — 동기 쿼리는 embedding.run_db 가 스레드마다 자체 세션을 열어 실행한다."""
import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

# connect_timeout: DB가 꺼져 있으면 연결 시도가 실패하지 않고 수 분 이상 멈췄다(Windows에서
# 닫힌 포트로의 연결이 거부되지 않고 응답 없이 대기). 몇 초 안에 실패시켜 502로 알린다.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": settings.db_connect_timeout_seconds},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def close_quietly(db: Session) -> None:
    """세션을 닫되, 이미 끊긴 연결의 정리 오류는 무시한다.

    결과를 다 읽은 뒤 close 단계에서 난 연결 오류로 성공한 요청을 실패로 바꾸지 않는다.
    죽은 연결은 풀에서 버려지고(pool_pre_ping) 다음 요청은 새 연결을 쓴다."""
    try:
        db.close()
    except OperationalError as e:
        logger.warning("DB 세션 정리 중 연결 오류 무시: %s", e)
