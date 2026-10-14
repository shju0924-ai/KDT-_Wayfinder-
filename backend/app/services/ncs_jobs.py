"""교육 후 직무 전환형(STEP 3-B)의 후보 직무 — NCS 세분류(직무) 검색.

이전에는 LLM이 직무명을 자유롭게 지어 냈다. 4b는 경력과 상관없이 '데이터 분석가'·'프로젝트 관리자'·
'고객 경험 디자이너'를 반복해(e2e 3차: 15명 중 7명에게 프로젝트 관리자), 범용 직무 목록으로 막아도 새 이름이
계속 나왔다. 그래서 후보를 NCS 세분류(국가직무능력표준의 '직무' 단위)로 묶는다:

    보유 역량마다 NCS 세분류 색인과 유사도 검색 → 현재 직무와 같은 중분류는 제외(이 트랙은 '새 분야')
    → 대분류마다 몇 개씩만 남겨 분야를 섞음 → LLM은 이 목록 안에서만 번호로 고른다.

색인에는 실제 훈련과정(training_courses.ncs_cd)이 있는 세분류만 넣는다 — 배울 곳이 있는 직무만 후보가 된다.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.services import embedding

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parents[2] / ".cache"
# 역량 하나당 가져올 세분류 수 — 역량마다 따로 찾아야 한 역량(예: 고객 응대)이 후보를 독차지하지 않는다
PER_SKILL = 6
# LLM에 보여줄 후보 수 — 로컬 4b 프롬프트 길이를 고려
CANDIDATES = 12
# 같은 대분류에서 남길 최대 후보 수 — 분야를 섞는다
PER_LARGE_CATEGORY = 3
# 현재 분야 판별 — '직무명: 역량들' 문장과 가장 가까운 세분류 CURRENT_FIELD_TOP 개 중 유사도가
# CURRENT_FIELD_MIN 이상인 것의 중분류를 뺀다. 직무명만 쓰면 콜센터 상담원 → 취업알선, 간호조무사 → 노무관리(0.49)로
# 틀렸다. 결합 문장 실측: 회계 사무원 → 회계·감사 0.71, 간호조무사 → 병원안내 0.64, 호텔 프런트 → 객실관리 0.65 /
# NCS에 번역 분야가 없는 번역가는 최고 0.56(기계품질관리, 무관)이라 빼지 않는다.
CURRENT_FIELD_TOP = 3
CURRENT_FIELD_MIN = 0.6
UNITS_IN_TEXT = 8

_SUB_CATEGORY_SQL = text(
    """
    SELECT left(n.ncs_code, 8) AS cd,
           max(n.large_category) AS large, max(n.middle_category) AS middle,
           max(n.small_category) AS small, max(n.sub_category) AS sub,
           string_agg(n.unit_name, ', ' ORDER BY n.ncs_code)
               FILTER (WHERE n.unit_name NOT LIKE '%구버전%' AND n.unit_name NOT LIKE '%삭제%') AS units
    FROM ncs_units n
    WHERE left(n.ncs_code, 8) IN (SELECT DISTINCT ncs_cd FROM training_courses WHERE ncs_cd IS NOT NULL)
    GROUP BY 1
    ORDER BY 1
    """
)

_index: tuple[list[dict], np.ndarray] | None = None
_lock = threading.Lock()


def _text(row: dict) -> str:
    units = ", ".join((row["units"] or "").split(", ")[:UNITS_IN_TEXT])
    return f"{row['sub']} ({row['middle']} > {row['small']}) — {units}"


def _sub_category_index(db: Session) -> tuple[list[dict], np.ndarray] | None:
    """훈련과정이 있는 NCS 세분류(약 400개) 벡터 — 프로세스당 한 번, 디스크 캐시."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        try:
            rows = [dict(r._mapping) for r in db.execute(_SUB_CATEGORY_SQL).all()]
        except SQLAlchemyError as e:
            db.rollback()
            logger.warning("NCS 세분류 조회 실패: %r", e)
            return None
        rows = [r for r in rows if r["sub"]]
        if not rows:
            return None
        texts = [_text(r) for r in rows]
        key = hashlib.sha256(
            "\n".join([embedding.EMBEDDING_MODEL, embedding.EMBEDDING_REVISION, *texts]).encode("utf-8")
        ).hexdigest()[:16]
        cache = INDEX_DIR / f"ncs_sub_index_{key}.npy"
        if cache.exists():
            _index = (rows, np.load(cache))
            return _index
        logger.info("NCS 세분류 색인 생성 — %d건 (첫 실행만)", len(texts))
        arr = np.asarray(embedding.embed_passages(texts), dtype=np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        except OSError as e:
            logger.warning("NCS 세분류 색인 저장 실패: %r", e)
        _index = (rows, arr)
        return _index


def warm_index() -> None:
    """서버 시작 시 백그라운드로 색인을 준비한다 — 첫 요청이 색인 생성을 기다리지 않게."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        _sub_category_index(db)
    except Exception as e:
        logger.warning("NCS 세분류 색인 사전 준비 실패: %r", e)
    finally:
        db.close()


def select_candidates(
    rows: list[dict],
    unit_vecs: np.ndarray,
    skill_vecs: np.ndarray,
    skill_names: list[str],
    current_vec: np.ndarray | None,
) -> list[dict]:
    """색인·벡터가 주어졌을 때 후보 세분류를 고른다(순수 함수 — 테스트용으로 분리)."""
    excluded: set[tuple[str, str]] = set()
    if current_vec is not None:
        cur = unit_vecs @ current_vec
        for ui in np.argsort(-cur)[:CURRENT_FIELD_TOP]:
            if cur[ui] >= CURRENT_FIELD_MIN:
                excluded.add((rows[ui]["large"], rows[ui]["middle"]))
                logger.info("교육 후 전환 후보에서 현재 분야 제외: %s > %s (%s %.2f)",
                            rows[ui]["large"], rows[ui]["middle"], rows[ui]["sub"], cur[ui])

    sims = skill_vecs @ unit_vecs.T  # 역량 × 세분류, 정규화 벡터라 내적 = 코사인
    best_sim: dict[int, float] = {}
    hits: dict[int, list[str]] = {}
    for si, row in enumerate(sims):
        for ui in np.argsort(-row)[:PER_SKILL]:
            ui = int(ui)
            if (rows[ui]["large"], rows[ui]["middle"]) in excluded:
                continue
            best_sim[ui] = max(best_sim.get(ui, 0.0), float(row[ui]))
            hits.setdefault(ui, []).append(skill_names[si])

    # 여러 역량에 걸린 세분류를 먼저, 같으면 유사도 순
    order = sorted(best_sim, key=lambda ui: (len(hits[ui]), best_sim[ui]), reverse=True)
    per_large: dict[str, int] = {}
    out: list[dict] = []
    for ui in order:
        large = rows[ui]["large"] or ""
        if per_large.get(large, 0) >= PER_LARGE_CATEGORY:
            continue
        per_large[large] = per_large.get(large, 0) + 1
        r = rows[ui]
        out.append({
            "ncs_code": r["cd"],
            "job_title": r["sub"],
            "path": f"{r['large']} > {r['middle']} > {r['small']}",
            "units": ", ".join((r["units"] or "").split(", ")[:5]),
            "similarity": round(best_sim[ui], 3),
            "skills": list(dict.fromkeys(hits[ui])),
        })
        if len(out) == CANDIDATES:
            break
    return out


def find_candidates(
    db: Session, skill_texts: list[str], skill_names: list[str], current_job: str | None
) -> list[dict]:
    """보유 역량(근거 포함 텍스트)마다 NCS 세분류를 찾아 후보 목록을 만든다. 색인이 없으면 []."""
    index = _sub_category_index(db)
    if index is None or not skill_texts:
        return []
    rows, unit_vecs = index
    # 현재 분야는 직무명만이 아니라 '직무명: 역량들'로 찾는다 — CURRENT_FIELD_MIN 주석 참고
    current_text = f"{current_job or '현재 직무'}: " + ", ".join(skill_names)
    vecs = np.asarray(embedding.embed_passages([*skill_texts, current_text]), dtype=np.float32)
    skill_vecs, current_vec = vecs[:-1], vecs[-1]
    out = select_candidates(rows, unit_vecs, skill_vecs, skill_names, current_vec)
    logger.info("교육 후 전환 NCS 후보: %s", ", ".join(f"{c['job_title']} {c['similarity']}" for c in out))
    return out
