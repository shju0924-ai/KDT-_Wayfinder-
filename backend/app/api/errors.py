"""라우터 공용 오류 응답."""
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def upstream_error(message: str, exc: BaseException) -> HTTPException:
    """LLM·DB·임베딩 같은 하위 단계 실패 → 502.

    예외 원문(DB 호스트·연결 문자열·스택 등)은 서버 로그에만 남기고, 응답에는 무엇이 실패했는지와
    예외 종류만 싣는다 — 사용자가 원인 방향(연결 실패·시간 초과 등)은 알 수 있게 한다.
    """
    logger.error("%s: %r", message, exc, exc_info=exc)
    return HTTPException(status_code=502, detail=f"{message} ({type(exc).__name__})")
