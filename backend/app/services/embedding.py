"""임베딩 서비스 — 역량·직무 텍스트를 동일 벡터 공간에 사상 + pgvector 유사도 검색.

Upstage Solar Embedding 2 사용 (OpenAI SDK 호환, base_url만 다름 — 별도 패키지 불필요).
query/passage 이원화 모델: 검색 대상(채용공고·NCS·훈련과정)은 passage로,
검색어(사용자 역량·역량 격차)는 query로 임베딩해야 함 — 섞어 쓰면 유사도 품질이 떨어짐.

주의: 모델·차원은 data-pipeline/embedding/ 의 배치 적재와 반드시 같은 모델·차원을 사용할 것.
"""
import asyncio
import json
from urllib.parse import urlencode

from openai import AsyncOpenAI, OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings

QUERY_MODEL = "solar-embedding-2-query"
PASSAGE_MODEL = "solar-embedding-2-passage"
EMBEDDING_DIM = 1024  # db/models.py 의 Vector 차원과 일치해야 함
BASE_URL = "https://api.upstage.ai/v1"
WORK24_COURSE_DETAIL_URL = "https://www.work24.go.kr/hr/a/a/3100/selectTracseDetl.do"

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.upstage_api_key, base_url=BASE_URL)
    return _client


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=settings.upstage_api_key, base_url=BASE_URL)
    return _async_client


async def embed_query(text_input: str) -> list[float]:
    """검색어 임베딩 — 사용자 역량 프로필·역량 격차 등 STEP 3·4의 유사도 검색 입력값."""
    resp = await get_async_client().embeddings.create(model=QUERY_MODEL, input=text_input)
    # 응답에 int/float가 섞여 오는 경우가 있어 pgvector 캐스팅 전에 float로 통일
    return [float(x) for x in resp.data[0].embedding]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """검색 대상 임베딩(배치) — 채용공고·NCS 능력단위·훈련과정 등 저장·검색될 문서.

    TODO:
      - [ ] 배치 한도(요청당 100건 · 204,800토큰) 초과 시 분할 처리
    """
    resp = get_client().embeddings.create(model=PASSAGE_MODEL, input=texts)
    return [[float(x) for x in d.embedding] for d in resp.data]


# ── pgvector 유사도 검색 (STEP 3·4) ─────────────────────────
# `<=>` 는 코사인 거리(0=동일). 유사도 = 1 - 거리.
# 청크 행(source_id 에 #c1 접미어)이 섞여 있어 원문 단위로 묶기 위해
# job_title/course_name 기준 최고 유사도만 남긴다.

_JOB_SEARCH_SQL = text(
    """
    SELECT DISTINCT ON (job_title)
           split_part(source_id, '#', 1) AS posting_id,
           job_title, company, region, source_url,
           left(required_skills_text, 400) AS snippet,
           1 - (embedding <=> CAST(:skillvec AS vector)) AS similarity,
           1 - (embedding <=> CAST(:rankvec AS vector)) AS ranking_similarity
    FROM job_postings
    WHERE embedding IS NOT NULL
    ORDER BY job_title, embedding <=> CAST(:rankvec AS vector)
    """
)

_COURSE_SEARCH_SQL = text(
    """
    SELECT DISTINCT ON (course_name)
           source_id, course_name, institution, ncs_cd, ncs_nm,
           start_date, end_date, tuition, address,
           1 - (embedding <=> CAST(:qvec AS vector)) AS similarity
    FROM training_courses
    WHERE embedding IS NOT NULL
      AND end_date::date >= CURRENT_DATE
    ORDER BY course_name, embedding <=> CAST(:qvec AS vector)
    """
)


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
    rows = db.execute(
        text(
            f"SELECT * FROM ({_JOB_SEARCH_SQL.text}) t "
            "ORDER BY ranking_similarity DESC LIMIT :limit"
        ),
        {
            "skillvec": json.dumps(skillvec),
            "rankvec": json.dumps(rankvec),
            "limit": limit,
        },
    ).mappings().all()
    return [dict(r) for r in rows]


async def search_similar_courses(db: Session, gap_text: str, limit: int = 5) -> list[dict]:
    """역량 격차 텍스트 → 유사한 훈련과정 top-k (STEP 4).

    검색어는 '보유 역량'이 아니라 '부족한 역량'이어야 의미 있는 추천이 나온다.
    """
    if not gap_text.strip():
        return []
    qvec = await embed_query(gap_text)
    rows = db.execute(
        text(f"SELECT * FROM ({_COURSE_SEARCH_SQL.text}) t ORDER BY similarity DESC LIMIT :limit"),
        {"qvec": json.dumps(qvec), "limit": limit},
    ).mappings().all()
    courses = [dict(r) for r in rows]
    for course in courses:
        course["source_url"] = build_work24_course_url(course["source_id"])
    return courses
