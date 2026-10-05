"""임베딩 서비스 — 역량·직무 텍스트를 동일 벡터 공간에 사상 + pgvector 유사도 검색.

로컬 오픈소스 모델 BAAI/bge-m3 사용 (sentence-transformers, 1024차원 — 기존 pgvector 스키마 유지).
API 키가 필요 없고, 첫 호출 시 모델(약 2GB)을 Hugging Face에서 내려받아 캐시한다.
bge-m3는 query/passage 프리픽스 구분이 없어 검색 대상·검색어 모두 같은 방식으로 임베딩한다.
함수는 embed_query / embed_passages 로 역할을 계속 구분해 둔다(프로바이더 교체 대비).

주의: 모델·차원은 data-pipeline/embedding/ 의 배치 적재와 반드시 같은 모델·차원을 사용할 것.
"""
import asyncio
import json
import logging
import threading
from urllib.parse import urlencode

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-m3"
# data-pipeline/embedding/common.py 의 EMBEDDING_REVISION 과 일치시킬 것 — DB 벡터와 같은 가중치 보장
EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIM = 1024  # db/models.py 의 Vector 차원과 일치해야 함
# data-pipeline/embedding/common.py 의 EMBED_MAX_SEQ_LENGTH 와 일치시킬 것
EMBED_MAX_SEQ_LENGTH = 512
WORK24_COURSE_DETAIL_URL = "https://www.work24.go.kr/hr/a/a/3100/selectTracseDetl.do"

_model = None
_model_lock = threading.Lock()


def get_model():
    """모델은 무거워(로드 수 초~수십 초) 프로세스당 한 번만 로드한다."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)
                model.max_seq_length = EMBED_MAX_SEQ_LENGTH
                _model = model
    return _model


def _encode(texts: list[str]) -> list[list[float]]:
    if settings.embedding_api_url:
        vecs = _encode_remote(texts)
        if vecs is not None:
            return vecs
    # 정규화 후 저장·검색 — pgvector 코사인 거리(<=>)와 일관
    vecs = get_model().encode(texts, batch_size=16, normalize_embeddings=True)
    return [[float(x) for x in v] for v in vecs]


def _encode_remote(texts: list[str]) -> list[list[float]] | None:
    """GPU 임베딩 서버로 계산한다. 실패하거나 모델·리비전이 다르면 None — 호출부가 로컬로 계산한다.

    로컬 CPU는 모델 로드만 수 분(RAM 8GB)이라 서버가 있으면 그쪽을 쓴다."""
    try:
        resp = httpx.post(
            f"{settings.embedding_api_url.rstrip('/')}/embed",
            json={"texts": texts},
            timeout=httpx.Timeout(60, connect=3),
        )
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("원격 임베딩 실패, 로컬로 계산: %r", e)
        return None
    if body.get("model") != EMBEDDING_MODEL or body.get("revision") != EMBEDDING_REVISION:
        logger.warning("원격 임베딩 모델 불일치(%s@%s), 로컬로 계산", body.get("model"), body.get("revision"))
        return None
    vecs = body.get("vectors") or []
    if len(vecs) != len(texts) or any(len(v) != EMBEDDING_DIM for v in vecs):
        logger.warning("원격 임베딩 응답 형식 이상, 로컬로 계산")
        return None
    return vecs


async def embed_query(text_input: str) -> list[float]:
    """검색어 임베딩 — 사용자 역량 프로필·역량 격차 등 STEP 3·4의 유사도 검색 입력값."""
    # CPU 연산이 이벤트 루프를 막지 않도록 스레드에서 실행
    return (await asyncio.to_thread(_encode, [text_input]))[0]


async def embed_queries(texts: list[str]) -> list[list[float]]:
    """검색어 여러 개를 한 번에 임베딩 — 호출(원격이면 왕복) 횟수를 줄인다."""
    if not texts:
        return []
    return await asyncio.to_thread(_encode, texts)


async def similarities_to(base: str, texts: list[str]) -> list[float]:
    """base 와 texts 각각의 코사인 유사도 — 직무명끼리 '사실상 같은 직무'인지 비교할 때 쓴다."""
    if not texts:
        return []
    vecs = await asyncio.to_thread(_encode, [base, *texts])
    head = vecs[0]
    return [sum(a * b for a, b in zip(head, v)) for v in vecs[1:]]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """검색 대상 임베딩(배치) — 채용공고·NCS 능력단위·훈련과정 등 저장·검색될 문서."""
    return _encode(texts)


# ── pgvector 유사도 검색 (STEP 3·4) ─────────────────────────
# `<=>` 는 코사인 거리(0=동일). 유사도 = 1 - 거리.
# 청크 행(source_id 에 #c1 접미어)이 섞여 있어 원문 단위로 묶기 위해
# job_title/course_name 기준 최고 유사도만 남긴다.
#
# DISTINCT ON 을 테이블 전체에 바로 걸면 HNSW 인덱스를 못 타고 전 행과 거리를 계산한다
# (공고 5만 행에서 1회 ~20초). 그래서 안쪽 서브쿼리에서 `ORDER BY 거리 LIMIT :candidates`
# 로 HNSW 후보를 먼저 뽑고, 그 후보 안에서만 제목 중복을 제거한다.
# 후보 수는 hnsw.ef_search 이상이어야 다 채워진다 → _execute_hnsw() 참고.
# 공고는 같은 제목(예: '경리 사무원')이 수백 건씩 몰려 있어 후보가 적으면 고유 제목이
# 모자란다(400건 → 회계 질의에서 17개). 1000 = ef_search 상한, 캐시된 상태 ~0.4초.

JOB_CANDIDATES = 1000
COURSE_CANDIDATES = 200

_JOB_SEARCH_SQL = text(
    """
    SELECT DISTINCT ON (job_title)
           posting_id, job_title, company, region, source_url, snippet,
           1 - (embedding <=> CAST(:skillvec AS vector)) AS similarity,
           1 - rank_dist AS ranking_similarity
    FROM (
        SELECT split_part(source_id, '#', 1) AS posting_id,
               job_title, company, region, source_url,
               left(required_skills_text, 400) AS snippet,
               embedding,
               embedding <=> CAST(:rankvec AS vector) AS rank_dist
        FROM job_postings
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:rankvec AS vector)
        LIMIT :candidates
    ) c
    ORDER BY job_title, rank_dist
    """
)

_COURSE_SEARCH_SQL = text(
    """
    SELECT DISTINCT ON (course_name)
           source_id, course_name, institution, ncs_cd, ncs_nm,
           start_date, end_date, tuition, address,
           1 - dist AS similarity
    FROM (
        SELECT source_id, course_name, institution, ncs_cd, ncs_nm,
               start_date, end_date, tuition, address,
               embedding <=> CAST(:qvec AS vector) AS dist
        FROM training_courses
        WHERE embedding IS NOT NULL
          AND end_date::date >= CURRENT_DATE
        ORDER BY embedding <=> CAST(:qvec AS vector)
        LIMIT :candidates
    ) c
    ORDER BY course_name, dist
    """
)


async def run_db(fn, *args):
    """동기 DB 작업을 스레드에서 자체 세션으로 실행한다.

    async 엔드포인트 안에서 동기 쿼리를 바로 부르면 쿼리 시간(벡터 검색 0.5~1초) 동안 이벤트 루프가 멈춰
    다른 사용자의 요청까지 모두 기다렸다(8명 동시 검증에서 GPU 사용률 41%). 스레드마다 세션을 따로 쓰는 건
    SQLAlchemy 세션이 스레드 간 공유에 안전하지 않고, 한 요청 안에서도 검색을 동시에(gather) 보내기 때문이다."""
    def work():
        db = SessionLocal()
        try:
            return fn(db, *args)
        finally:
            db.close()

    return await asyncio.to_thread(work)


def _execute_hnsw(db: Session, sql: str, params: dict, candidates: int) -> list[dict]:
    """HNSW 인덱스로 후보를 뽑는 검색 쿼리를 실행한다.

    - ef_search: 기본 40이라 후보 수만큼 넓혀야 LIMIT :candidates 가 다 채워진다.
    - iterative_scan: end_date 같은 WHERE 필터로 후보가 모자랄 때 인덱스를 더 훑는다
      (pgvector 0.8+). relaxed_order 라 순서가 약간 어긋날 수 있지만 바깥 쿼리가 다시 정렬한다.
    - enable_seqscan=off: 후보가 수백 건 이상이면 플래너가 전체 스캔을 더 싸다고 오판한다
      (실측 전체 스캔 ~4초 vs HNSW ~0.4초).
    SET LOCAL 이라 트랜잭션이 끝나면 원래 값으로 돌아간다.

    읽기 전용 검색이므로 끝나면 바로 트랜잭션을 닫아 연결을 풀에 돌려준다. 열어 둔 채로 두면
    뒤이은 로컬 LLM 호출(최대 30분) 동안 'idle in transaction' 연결이 붙잡혀 있다가, 그 사이 DB 쪽
    연결이 끊기면 요청 종료 시 ROLLBACK 이 실패해 응답 뒤 ASGI 예외·클라이언트 연결 끊김이 났다.
    """
    try:
        db.execute(text(f"SET LOCAL hnsw.ef_search = {int(candidates)}"))
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        db.execute(text("SET LOCAL enable_seqscan = off"))
        rows = db.execute(text(sql), {**params, "candidates": candidates}).mappings().all()
    finally:
        db.rollback()
    return [dict(r) for r in rows]


_SAME_TITLE_SQL = text(
    """
    SELECT job_title, snippet FROM (
        SELECT job_title, left(required_skills_text, :chars) AS snippet,
               row_number() OVER (PARTITION BY job_title ORDER BY collected_at DESC NULLS LAST, id) AS rn
        FROM job_postings
        WHERE job_title = ANY(:titles)
          AND source_id NOT LIKE '%#c%'
          AND coalesce(required_skills_text, '') <> ''
    ) t
    WHERE rn <= :per_title
    """
)


def sample_postings_by_title(db: Session, titles: list[str], per_title: int = 5, chars: int = 300) -> dict[str, list[str]]:
    """직무명마다 같은 이름의 공고 발췌를 몇 건씩 — 공고 한 건의 회사 특수 요구가 아니라
    그 직무에 공통으로 요구되는 역량을 LLM이 가려내도록 근거를 넓힌다."""
    if not titles:
        return {}
    try:
        rows = db.execute(_SAME_TITLE_SQL, {"titles": list(set(titles)), "per_title": per_title, "chars": chars}).all()
    finally:
        db.rollback()
    out: dict[str, list[str]] = {}
    for title, snippet in rows:
        out.setdefault(title, []).append(snippet)
    return out


def build_work24_course_url(source_id: str) -> str | None:
    """HRD-Net 원천 ID(`hrdnet:과정ID#회차[#청크]`)를 고용24 상세 주소로 변환한다."""
    if not source_id.startswith("hrdnet:"):
        return None
    original_id = source_id.split("#c", 1)[0].removeprefix("hrdnet:")
    if "#" not in original_id:
        return None
    course_id, course_round = original_id.rsplit("#", 1)
    if not course_id or not course_round:
        return None
    query = urlencode(
        {
            "crseTracseSe": "",
            "tracseId": course_id,
            "tracseTme": course_round,
            "trainstCstmrId": "",
        }
    )
    return f"{WORK24_COURSE_DETAIL_URL}?{query}"


async def search_similar_jobs(
    db: Session,
    skill_text: str,
    limit: int = 5,
    search_context: str | None = None,
) -> list[dict]:
    """역량 프로필(근거 포함 텍스트) → 유사한 채용공고 top-k (STEP 3).

    역량명만 떼어 넘기면 맥락을 잃는다 — 예: '아웃바운드 커뮤니케이션'만 검색하면
    상담 도메인과 무관하게 아무 '아웃바운드/커뮤니케이션' 공고와 유사도가 잡힌다.
    호출부(jobs.py)가 역량명 + 근거 문장을 합친 텍스트를 넘겨야 이 문제가 줄어든다.

    선택한 탐색 경로가 있으면 후보 검색에는 경로 문맥을 함께 쓰되, 화면에 표시하는
    적합도는 순수한 보유 역량 벡터로 따로 계산한다.
    """
    if not skill_text.strip():
        return []
    if search_context:
        skillvec, rankvec = await asyncio.gather(
            embed_query(skill_text),
            embed_query(f"탐색 방향: {search_context}\n관련 보유 역량: {skill_text}"),
        )
    else:
        skillvec = await embed_query(skill_text)
        rankvec = skillvec
    return await run_db(
        _execute_hnsw,
        f"SELECT * FROM ({_JOB_SEARCH_SQL.text}) t ORDER BY ranking_similarity DESC LIMIT :limit",
        {"skillvec": json.dumps(skillvec), "rankvec": json.dumps(rankvec), "limit": limit},
        JOB_CANDIDATES,
    )


async def search_similar_courses(db: Session, gap_text: str, limit: int = 5) -> list[dict]:
    """역량 격차 텍스트 → 유사한 훈련과정 top-k (STEP 4).

    검색어는 '보유 역량'이 아니라 '부족한 역량'이어야 의미 있는 추천이 나온다.
    """
    if not gap_text.strip():
        return []
    qvec = await embed_query(gap_text)
    courses = await run_db(
        _execute_hnsw,
        f"SELECT * FROM ({_COURSE_SEARCH_SQL.text}) t ORDER BY similarity DESC LIMIT :limit",
        {"qvec": json.dumps(qvec), "limit": limit},
        COURSE_CANDIDATES,
    )
    labels = await run_db(_ncs_labels, [c["ncs_cd"] for c in courses if c.get("ncs_cd")])
    for course in courses:
        course["source_url"] = build_work24_course_url(course["source_id"])
        # HRD-Net 원천에는 과정 설명이 없고 ncs_nm 도 비어 있다 — NCS 세분류명·능력단위명으로 과정 내용을 보완한다
        sub, units = labels.get(course.get("ncs_cd") or "", (None, None))
        course["ncs_nm"] = course.get("ncs_nm") or sub
        course["ncs_units"] = units
    return courses


_NCS_LABEL_SQL = text(
    """
    SELECT left(ncs_code, 8) AS cd, max(sub_category) AS sub,
           string_agg(unit_name, ', ' ORDER BY ncs_code) FILTER (WHERE unit_name NOT LIKE '%구버전%') AS units
    FROM ncs_units
    WHERE left(ncs_code, 8) = ANY(:codes)
    GROUP BY 1
    """
)


def _ncs_labels(db: Session, codes: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """훈련과정 NCS 세분류 코드(8자리) → (세분류명, 능력단위명 앞 몇 개)."""
    codes = list({c for c in codes if c})
    if not codes:
        return {}
    try:
        rows = db.execute(_NCS_LABEL_SQL, {"codes": codes}).all()
    except Exception as e:  # 라벨은 보조 정보 — 실패해도 검색 결과는 돌려준다
        logger.warning("NCS 라벨 조회 실패: %r", e)
        return {}
    finally:
        db.rollback()
    return {cd: (sub, ", ".join((units or "").split(", ")[:5]) or None) for cd, sub, units in rows}
