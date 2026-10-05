"""SQLAlchemy 세션 — FastAPI 의존성으로 주입해서 사용."""
import logging
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 라우터에서: db: Session = Depends(get_db)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except OperationalError as e:
            # 응답을 이미 보낸 뒤라 여기서 예외가 나가면 서버가 클라이언트 연결을 끊는다.
            # 죽은 연결은 풀에서 버려지고(pool_pre_ping) 다음 요청은 새 연결을 쓴다.
            logger.warning("DB 세션 정리 중 연결 오류 무시: %s", e)
