"""LLM 오케스트레이션 — 역량 분해·위험도 진단·로드맵 생성 프롬프트 파이프라인.

원칙: 모든 LLM 응답은 근거(evidence/source)를 함께 반환하도록 프롬프트를 설계한다 (환각 억제).
구조화 출력(structured outputs)으로 스키마를 강제해 근거 필드 누락과 파싱 실패를 함께 막는다.

STEP 1·2는 순수 생성, STEP 4는 pgvector로 검색한 훈련과정만 근거로 쓰는 RAG.
단계별 시스템 프롬프트 문구는 prompts.py 에 모여 있다 — 이 파일은 스키마·호출·후처리를 맡는다.

프로바이더는 LLM_PROVIDER 로 고른다 — claude(Anthropic API) 또는 ollama(로컬 모델).
로컬 소형 모델은 출력이 느리고(CPU 약 5tok/s) 지시를 덜 지키므로, 어느 프로바이더든
역량명 필드는 보유 역량 enum 으로 스키마에서 강제하고 출력 개수·길이를 스키마로 묶는다.
"""
import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Literal, TypeVar

import httpx
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, ValidationError, create_model

from app.core.config import settings
from app.schemas.career import SkillItem
from app.services import prompts

logger = logging.getLogger(__name__)

# 2026-09-18: LLM 호출을 OpenAI(gpt-5.6-luna)에서 Anthropic Claude로 전환.
# 2026-10-01: 로컬 모델(Ollama) 선택지 추가 — API 크레딧 없이도 전체 흐름이 돌도록.
# 임베딩(app/services/embedding.py)은 로컬 모델 BAAI/bge-m3 사용 — Anthropic은 임베딩 API를 제공하지 않음.
MAX_TOKENS = 16000

_client: AsyncAnthropic | None = None

T = TypeVar("T", bound=BaseModel)


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def _parse(
    schema: type[T],
    system: str,
    user: str,
    *,
    max_tokens: int = MAX_TOKENS,
    local_max_tokens: int,
) -> T | None:
    """구조화 출력 호출 — 스키마에 맞게 파싱된 결과, 실패하면 None.

    local_max_tokens 는 로컬 모델 출력 상한(num_predict)이다. 상한이 없으면 클라이언트가
    끊겨도 Ollama 서버가 생성을 계속해 다음 요청이 대기열에 막힌다 — 반드시 지정한다.

    같은 입력(프로바이더·모델·스키마·프롬프트)이면 캐시된 결과를 돌려준다 — 로컬 모델은
    호출당 수 분이라, 같은 경력으로 다시 보거나 트랙을 오가는 경우 대기 시간을 없앤다.
    """
    system = f"{prompts.KOREAN_ONLY_RULE}\n\n{system}"
    user = f"{user}{prompts.KOREAN_ONLY_REMINDER}"
    key = _cache_key(schema, system, user)
    cached = _cache_get(key)
    if cached is not None:
        try:
            return schema.model_validate_json(cached)
        except ValidationError:
            pass  # 스키마가 바뀐 뒤의 옛 캐시 — 새로 호출한다

    if settings.llm_provider == "ollama":
        out = await _ollama_parse(schema, system, user, local_max_tokens)
    else:
        resp = await get_client().messages.parse(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        out = resp.parsed_output
    if out is not None:
        _cache_put(key, out.model_dump_json())
    return out


# ── LLM 결과 캐시 (SQLite) ─────────────────────────────────
# 키는 해시지만 값(역량 근거·업무 근거)에는 경력 서사에서 뽑은 문장이 들어간다 — 민감정보 원칙에 맞춰
# LLM_CACHE_TTL_HOURS 가 지난 항목은 읽지 않고, 연결을 열 때 디스크에서도 지운다.
_cache_lock = threading.Lock()
_cache_conn: sqlite3.Connection | None = None


def _cache_key(schema: type[BaseModel], system: str, user: str) -> str:
    model = settings.ollama_model if settings.llm_provider == "ollama" else settings.anthropic_model
    raw = json.dumps(
        [settings.llm_provider, model, schema.model_json_schema(), system, user],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_db() -> sqlite3.Connection | None:
    global _cache_conn
    if not settings.llm_cache_path:
        return None
    if _cache_conn is None:
        path = Path(settings.llm_cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL DEFAULT 0)"
        )
        # 보존 기한 도입 전의 캐시 파일 — 시각 정보가 없는 옛 항목은 0 으로 채워져 아래에서 지워진다
        if "created_at" not in {row[1] for row in conn.execute("PRAGMA table_info(llm_cache)")}:
            conn.execute("ALTER TABLE llm_cache ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
        conn.execute("DELETE FROM llm_cache WHERE created_at < ?", (_cache_cutoff(),))
        conn.commit()
        _cache_conn = conn
    return _cache_conn


def _cache_cutoff() -> float:
    return time.time() - settings.llm_cache_ttl_hours * 3600


def _cache_get(key: str) -> str | None:
    try:
        with _cache_lock:
            db = _cache_db()
            if db is None:
                return None
            row = db.execute(
                "SELECT value FROM llm_cache WHERE key = ? AND created_at >= ?", (key, _cache_cutoff())
            ).fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        logger.warning("LLM 캐시 읽기 실패: %s", e)
        return None


def _cache_put(key: str, value: str) -> None:
    try:
        with _cache_lock:
            db = _cache_db()
            if db is not None:
                db.execute(
                    "INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
                    (key, value, time.time()),
                )
                db.commit()
    except sqlite3.Error as e:
        logger.warning("LLM 캐시 쓰기 실패: %s", e)


def field_guide(schema: type[BaseModel]) -> str:
    """스키마 필드 설명을 프롬프트용 목록으로 — '- tasks[].automation_score: 0~100점 …'.

    Ollama 의 format(JSON 스키마)은 디코딩 문법으로만 쓰이고 필드 설명은 모델에게 전달되지 않는다.
    그래서 '100점 척도', '영어로 적는 영문 직업명' 같은 지시가 보이지 않아 4b가 영문 직업명을 비우고
    점수를 0~1 비율로 답했다(e2e 실측). 설명을 시스템 프롬프트에 함께 넣어 같은 지시가 닿게 한다."""
    js = schema.model_json_schema()
    defs = js.get("$defs", {})
    lines: list[str] = []

    def walk(props: dict, prefix: str = "") -> None:
        for name, prop in props.items():
            ref = prop.get("$ref") or (prop.get("items") or {}).get("$ref")
            if prop.get("description"):
                lines.append(f"- {prefix}{name}: {prop['description']}")
            if ref:
                walk(defs[ref.split("/")[-1]].get("properties", {}), f"{prefix}{name}[].")

    walk(js.get("properties", {}))
    return "\n\n[출력 JSON 필드 설명 — 각 필드를 이 설명대로 채우세요]\n" + "\n".join(lines) if lines else ""


async def _ollama_parse(schema: type[T], system: str, user: str, num_predict: int) -> T | None:
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system + field_guide(schema)},
            {"role": "user", "content": user},
        ],
        # JSON 스키마를 넘기면 Ollama가 문법(grammar)으로 디코딩을 제한한다 — enum·배열 개수까지 강제됨
        "format": schema.model_json_schema(),
        "stream": False,
        # 사고(thinking) 토큰은 CPU에서 응답 시간만 몇 배로 늘린다
        "think": False,
        # 단계 사이(사용자 검토 시간)에 모델이 내려가면 다음 호출이 재로딩부터 한다
        "keep_alive": settings.ollama_keep_alive,
        "options": {
            "num_predict": num_predict,
            "num_ctx": settings.ollama_num_ctx,
            # 같은 경력이면 같은 역량·판정이 나와야 사용자가 결과를 믿고 비교할 수 있다 — 첫 시도는 결정적으로
            "temperature": 0,
            "seed": 42,
        },
    }
    fallback: T | None = None
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=settings.ollama_timeout_seconds
    ) as client:
        # 문법은 숫자 범위(ge/le)까지는 못 막으므로 검증 실패 시 한 번만 다시 시도한다.
        # 결정적 디코딩은 같은 입력에 같은 실패를 되풀이하므로 재시도만 온도를 올린다
        for attempt in range(2):
            if attempt:
                payload["options"] = {**payload["options"], "temperature": 0.4, "seed": 7}
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("done_reason") == "length":
                # 출력 상한에 잘린 JSON은 다시 돌려도 같은 길이에서 잘린다
                logger.warning("Ollama 출력이 num_predict=%d 에서 잘림 (%s)", num_predict, schema.__name__)
                return fallback
            content = data["message"]["content"]
            try:
                parsed = schema.model_validate_json(content)
            except ValidationError as e:
                logger.warning("Ollama 출력 스키마 검증 실패 %d회차 (%s): %s", attempt + 1, schema.__name__, e)
                continue
            # Qwen 계열은 한국어 출력에 중국어 한자를 섞는다('전환漏斗 분석') — 한 번 다시 생성한다.
            # 4b는 악센트 라틴 문자(스페인어 등)도 섞었다('Estadística 설명적 분석') — 같은 방식으로 다룬다
            if _has_foreign(content) and not _has_foreign(user):
                logger.warning("Ollama 출력에 한자·외국어 혼입 %d회차 (%s)", attempt + 1, schema.__name__)
                fallback = _strip_hanja(schema, parsed)
                continue
            return parsed
    return fallback


def _strip_hanja(schema: type[T], parsed: T) -> T:
    """재생성해도 한자·외국어가 남으면 그 부분만 지운다('프로세스 개선专员' → '프로세스 개선',
    'Estadística 설명적 분석' → '설명적 분석'). 지운 뒤 스키마에 맞지 않으면(예: enum 값) 원래 결과를 쓴다."""
    cleaned = _FOREIGN_WORD.sub("", _HANJA_RUN.sub("", parsed.model_dump_json()))
    try:
        return schema.model_validate_json(cleaned)
    except ValidationError:
        return parsed


_HANJA = re.compile(r"[\u4e00-\u9fff]")
_HANJA_RUN = re.compile(r"[\u4e00-\u9fff]+")
# \uc545\uc13c\ud2b8\uac00 \ubd99\uc740 \ub77c\ud2f4 \ubb38\uc790(\u00c0-\u024f). \uc601\uc5b4 \ub3c4\uad6c\uba85(SQL\u00b7CRM)\uc740 \ud5c8\uc6a9\ud558\ubbc0\ub85c \uc774 \ubc94\uc704\ub9cc \uc678\uad6d\uc5b4\ub85c \ubcf8\ub2e4
_ACCENTED = re.compile(r"[\u00c0-\u024f]")
_FOREIGN_WORD = re.compile(r"[A-Za-z\u00c0-\u024f]*[\u00c0-\u024f][A-Za-z\u00c0-\u024f]*\s?")


def _has_foreign(text: str) -> bool:
    return bool(_HANJA.search(text) or _ACCENTED.search(text))


# 소형 모델은 프롬프트로 금지해도 가끔 '데이터 분석가 (Data Analyst)', 'PMP 자격증 과정'처럼 쓴다.
# 화면에 그대로 나가지 않도록 형식만 정리한다(내용은 바꾸지 않음).
_ENGLISH_PAREN = re.compile(r"\s*[(（][^)）]*[A-Za-z][^)）]*[)）]")
_COURSE_SUFFIX = re.compile(
    r"\s*(전문\s*)?(자격증\s*)?(취득\s*)?(심화\s*)?(과정|부트캠프|교육과정|강좌|프로그램)$"
)
_LIST_PREFIX = re.compile(r"^\s*(\d+[.)]|[-•·])\s*")


# 한국어 직무명 끝에 붙은 영어 직함 — 4b가 '고객 관계 관리 Specialist'처럼 섞어 썼다
_ENGLISH_ROLE = {
    "specialist": "전문가", "expert": "전문가", "manager": "관리자", "analyst": "분석가",
    "designer": "디자이너", "developer": "개발자", "engineer": "엔지니어", "consultant": "컨설턴트",
    "coordinator": "코디네이터", "planner": "기획자", "researcher": "연구원", "administrator": "관리자",
}
_ENGLISH_ROLE_SUFFIX = re.compile(r"\s*\b(" + "|".join(_ENGLISH_ROLE) + r")s?\s*$", re.IGNORECASE)
_HANGUL = re.compile(r"[가-힣]")


def clean_job_title(title: str) -> str:
    """'디지털 마케팅 분석가 (Digital Marketing Analyst)' → '디지털 마케팅 분석가',
    '고객 관계 관리 Specialist' → '고객 관계 관리 전문가'."""
    cleaned = _ENGLISH_PAREN.sub("", title).strip() or title.strip()
    if _HANGUL.search(cleaned):
        cleaned = _ENGLISH_ROLE_SUFFIX.sub(lambda m: " " + _ENGLISH_ROLE[m.group(1).lower()], cleaned).strip()
    return cleaned


def clean_skill_phrase(phrase: str) -> str:
    """'SQL 심화 및 데이터 시각화 전문 과정' → 'SQL 심화 및 데이터 시각화', 'PMP 자격증 과정' → 'PMP'."""
    return _COURSE_SUFFIX.sub("", _LIST_PREFIX.sub("", phrase.strip())).strip()


# 지원 자격·조건은 배울 역량이 아니다. 프롬프트로 금지해도 9b가 '직업상담사 자격증 (필수)'를 적어 사후에 거른다.
# '자격증 대비'처럼 역량을 뜻하는 경우까지 지울 수 있으나, 부족 역량 칸에 자격 조건이 남는 쪽이 더 해롭다.
_CONDITION = re.compile(r"자격증|면허|소지|필수|우대|경력\s*\d|\d+\s*년\s*이상|학력|학위|대졸|고졸|차량")


# 성격·태도는 배울 역량이 아니고 훈련과정도 없다('반복 업무에 대한 인내심', '압박감 견디기' — 간호조무사 실측)
_TRAIT = re.compile(r"인내|압박|견디|버티|성격|태도|성실|열정|마인드")


def is_requirement_condition(phrase: str) -> bool:
    """'직업상담사 자격증', '운전면허 소지', '경력 3년' 같은 지원 조건이나 성격·태도인지."""
    return bool(_CONDITION.search(phrase) or _TRAIT.search(phrase))


def _one_of(names: list[str]):
    """보유 역량명 enum 타입 — 모델이 목록 밖 이름(프롬프트 예시 문구 등)을 쓸 수 없게 한다."""
    return Literal[tuple(dict.fromkeys(names))]


# ── STEP 1. 자동화 위험도 진단 ──────────────────────────────
class _TaskOut(BaseModel):
    name: str = Field(description="사용자가 실제로 수행한 핵심 업무")
    share_percent: float = Field(
        ge=1,
        le=100,
        description="전체 직무에서 이 업무가 차지하는 추정 비중. 모든 업무 합계는 100",
    )
    # 9b는 10점 만점으로 답하는 일이 잦았다(3.5, 6.5 …) — 100점 척도와 기준점을 설명에 못 박는다.
    # 주효과(자동화/보조/사람 중심)는 따로 받지 않고 두 점수로 서버가 정한다 — 점수와 라벨이 어긋났다.
    automation_score: float = Field(
        ge=0,
        le=100,
        description=(
            "사람의 개입 없이 현재 AI·소프트웨어가 끝까지 수행할 가능성. 0~100점(100점 만점, 10점 만점 아님). "
            "기준: 거의 확실히 대체 85, 일부 대체 50, 거의 불가 10"
        ),
    )
    ai_assistance_score: float = Field(
        ge=0,
        le=100,
        description=(
            "사람이 책임을 유지하며 AI로 생산성을 높일 가능성. 0~100점(100점 만점, 10점 만점 아님). "
            "기준: AI가 초안·검색·요약을 크게 도움 80, 약간 도움 40, 거의 무관 10"
        ),
    )
    rationale: str = Field(description="사용자 서사와 판단 기준을 연결한 한 문장 근거")


class _TaskAnalysisOut(BaseModel):
    job_title: str = Field(
        description="현재(최근) 직무명. 부연설명 없이 직무명만 간결하게 (예: '콜센터 상담원')"
    )
    # 노출도 자료(Anthropic Economic Index)가 O*NET 직업명을 쓴다 — 같은 형식으로 받아야 매칭된다
    occupation_title_en: str = Field(
        description=(
            "국제 직업 데이터와 매칭할 O*NET 표준 영어 직업명 한 개, 복수형 "
            "(예: 'Customer Service Representatives', 'Bookkeeping, Accounting, and Auditing Clerks')"
        )
    )
    ncs_search_terms: list[str] = Field(
        min_length=2,
        max_length=8,
        description="NCS 능력단위 검색에 쓸 한국어 과업 핵심어",
    )
    tasks: list[_TaskOut] = Field(
        min_length=3,
        max_length=7,
        description="서사에서 확인되는 핵심 업무. 비중 합계는 반드시 100",
    )
    rationale: str = Field(description="위험 업무와 사람 중심 강점을 함께 설명하는 2~4문장")


async def analyze_automation_tasks(
    career_text: str,
    job_title: str | None = None,
) -> _TaskAnalysisOut:
    """경력 서사를 구조화된 업무 단위로 분해한다.

    최종 점수는 이 함수가 정하지 않는다. 서비스 계층에서 업무 비중과 공개
    직업 노출도 자료를 사용해 재현 가능한 산식으로 계산한다.
    """
    hint = f"\n\n사용자가 직접 입력한 직무명: {job_title}" if job_title else ""
    out = await _parse(
        _TaskAnalysisOut,
        prompts.DIAGNOSIS_SYSTEM,
        f"[경력 서사]\n{career_text}{hint}",
        local_max_tokens=2000,
    )
    if out is None:
        raise RuntimeError("업무별 자동화 분석 결과를 파싱하지 못했습니다.")
    return out


# ── STEP 2. 역량 분해 ──────────────────────────────────────
SkillCategory = Literal["대인·협상", "커뮤니케이션", "분석·데이터", "기획·전략", "실행·운영", "리더십·관리", "도구·기술"]


class _SkillOut(BaseModel):
    name: str = Field(
        description="역량명 — 설명 없이 짧은 명사구 (예: 갈등 완화, 수요 예측·발주 판단)"
    )
    # 자유 문자열이면 소형 모델이 '네거티브 네고세이션' 같은 말을 지어낸다 — 분류는 고정 목록에서 고른다
    category: SkillCategory = Field(description="역량 분류 — 목록 중 하나")
    evidence: str = Field(description="이 역량이 있다고 판단한 근거 — 경력 서사에서 가져온 사실 한두 문장")


class _SkillsOut(BaseModel):
    current_job_title: str = Field(
        description="경력 서사에서 확인되는 현재(최근) 직무명. 부연설명 없이 표준 직무명만 (예: '콜센터 상담원')"
    )
    # 최소 3개 — 4b가 경리 8년차 경력에서 2개만 내 인접 직무 판정에 쓸 역량이 모자랐다(e2e 실측)
    skills: list[_SkillOut] = Field(min_length=3, max_length=6)


async def decompose_skills(career_text: str) -> tuple[list[SkillItem], str | None]:
    """경력 서사 → (전이 가능한 역량 단위 리스트(근거 필수), 현재 직무명).

    현재 직무명은 STEP 3에서 '같은 직무' 후보를 걸러내는 데 쓴다. 이력서·자유 텍스트 경로는
    사용자가 직무명을 따로 입력하지 않으므로 여기서 함께 뽑는다(추가 호출 없이 필드 하나).

    TODO:
      - [ ] 긴 이력서 분할 처리
    """
    out = await _parse(
        _SkillsOut,
        prompts.PROFILE_SYSTEM,
        f"[경력 서사]\n{career_text}",
        local_max_tokens=1200,
    )
    if out is None:
        raise RuntimeError("역량 분해 결과를 파싱하지 못했습니다.")
    skills = [
        SkillItem(name=clean_skill_name(s.name), category=s.category, evidence=s.evidence, confirmed=False)
        for s in out.skills
    ]
    return skills, (out.current_job_title.strip() or None)


# 근거보다 부풀린 꾸밈말 — 프롬프트로 금지해도 4b가 '복합 시스템 운영 관리', '협업 기반 공급망 조정'을 냈다.
# 꾸밈말만 떼고 핵심 명사구는 남긴다. 다 떼면 빈 이름이 되는 경우는 원래 이름을 쓴다
_INFLATED_PREFIX = re.compile(r"^(복합|전략적|종합적|고도화된|협업\s*기반)\s+")


def clean_skill_name(name: str) -> str:
    """'복합 시스템 운영 관리' → '시스템 운영 관리', '협업 기반 공급망 조정' → '공급망 조정'."""
    cleaned = _INFLATED_PREFIX.sub("", name.strip()).strip()
    return cleaned or name.strip()


# ── STEP 3. 직무 매칭 해석 ─────────────────────────────────
# 2026-10-01: 40개 후보를 한 번에 넣던 후보 선별 호출(select_job_candidates)은 제거했다 —
# 입력이 ~8천 토큰이라 로컬 모델 컨텍스트를 넘고, 최종 인접 판정은 어차피 아래 격차 판정이 한다.
# 후보 순서는 벡터 유사도 순위를 그대로 쓰고, 판정은 공고 여러 건을 한 번의 호출로 묶는다.
# 직무마다 같은 직무명 공고 여러 건의 발췌를 붙인다(jobs.py) — 공고 수 × 약 250자
JUDGE_SNIPPET_CHARS = 1200


# 후보 직무별 공개 AI 노출도(ILO·Anthropic, 영문 직업명 기준)를 찾는 데 쓴다 — 현재 직무 진단과 같은 방식.
# 출력 언어 규칙(한국어만)의 예외: 이 필드는 매칭용 키라 영어로 받는다.
_OCCUPATION_EN_DESC = (
    "이 직무에 해당하는 O*NET 표준 영어 직업명 한 개, 복수형 (예: 'Telemarketers', "
    "'Customer Service Representatives', 'Market Research Analysts and Marketing Specialists'). "
    "이 필드만 예외적으로 영어로 적는다"
)


def _judge_batch_schema(skills: list[str], count: int) -> type[BaseModel]:
    """공고 count 건 판정 스키마 — matched_skills 는 보유 역량 enum, 결과 개수는 공고 수와 같게 강제.

    요구역량(required_skills)을 따로 받으면 소형 모델이 보유 역량명을 그대로 복사해 격차가 0으로
    나왔다. 그래서 '이어지는 보유 역량'과 '공고에만 있는 부족 역량'만 받고, 요구역량은 그 합으로 만든다.
    """
    # job_no 를 맨 앞에 둬 모델이 어느 직무를 판정하는지 먼저 밝히게 한다. 순서로 짝지으면 4b가 결과 순서를
    # 바꿔 내 부족 역량이 다른 직무 카드에 붙었다(번역가 페르소나: 콜센터 상담원 ← 웹 기획 역량)
    judge = create_model(
        "_JobJudgeOut",
        job_no=(Literal[tuple(range(count))], Field(description="판정하는 후보 직무의 번호 ([번호])")),
        same_as_current=(bool, Field(description="현재 직무와 사실상 같은 직무인지")),
        matched_skills=(
            list[_one_of(skills)],
            Field(max_length=4, description="이 공고에 실제로 쓰이는 사용자 보유 역량명"),
        ),
        missing_skills=(
            list[str],
            Field(max_length=4, description="여러 공고에 공통으로 요구되지만 사용자 보유 역량에 없는 역량 0~4개 (자격·조건 제외)"),
        ),
        demand_outlook=(Literal["증가", "유지", "감소"], ...),
        transition_difficulty=(Literal["낮음", "보통", "높음"], ...),
        occupation_title_en=(str, Field(description=_OCCUPATION_EN_DESC)),
    )
    return create_model(
        "_JobJudgesOut",
        judgements=(
            list[judge],
            Field(min_length=count, max_length=count, description="후보 직무마다 하나 — job_no 에 그 직무 번호"),
        ),
    )


def _default_judgement() -> dict:
    return {
        "occupation_title_en": "",
        "same_as_current": False,
        "demand_outlook": "유지",
        "transition_difficulty": "보통",
        "matched_skills": [],
        "required_skills": [],
        "missing_skills": [],
    }


async def judge_job_matches(
    skills: list[str], current_job: str | None, postings: list[dict]
) -> list[dict]:
    """검색된 채용공고 여러 건(job_title·snippet)을 한 번에 판정 — 공고 순서대로 결과를 돌려준다.

    결과는 모델이 밝힌 job_no 로 공고와 짝짓는다. 같은 번호가 두 번 오면 앞의 것을 쓰고, 빠진 번호와
    파싱 실패는 기본값(기여 역량 없음)이라 호출부의 인접 판정에서 자연히 탈락한다.
    """
    if not postings or not skills:
        return [_default_judgement() for _ in postings]
    listing = "\n\n".join(
        f"[{i}] 직무명: {p['job_title']}\n공고 발췌:\n{(p.get('snippet') or '')[:JUDGE_SNIPPET_CHARS]}"
        for i, p in enumerate(postings)
    )
    out = await _parse(
        _judge_batch_schema(skills, len(postings)),
        prompts.JOB_JUDGE_SYSTEM,
        f"[현재 직무]\n{current_job or '확인되지 않음'}\n\n"
        f"[사용자 보유 역량]\n{', '.join(skills)}\n\n[후보 직무 {len(postings)}개]\n{listing}",
        max_tokens=4000,
        local_max_tokens=240 * len(postings),
    )
    results = [_default_judgement() for _ in postings]
    if out is None:
        return results

    seen: set[int] = set()
    for judged in out.judgements:
        if judged.job_no in seen:
            logger.info("직무 판정 번호 중복 [%d] — 앞의 판정을 쓴다", judged.job_no)
            continue
        seen.add(judged.job_no)
        # enum 으로 강제했지만 프로바이더가 바뀌어도 안전하도록 보유 목록으로 한 번 더 거른다.
        # 4b는 직무에 필요한 보유 역량을 matched 대신 missing(직무 요구)에 적는 일이 잦았다('웹 개발자'의
        # missing 에 '반응형 웹 구현') — 보유 역량이면서 그 직무에 필요하면 정의상 이어지는 역량이므로 옮긴다
        matched = list(dict.fromkeys(
            [s for s in judged.matched_skills if s in skills] + [s for s in judged.missing_skills if s in skills]
        ))[:4]
        missing = list(dict.fromkeys(
            s for s in judged.missing_skills if s not in skills and not is_requirement_condition(s)
        ))[:4]
        results[judged.job_no] = (
            {
                "same_as_current": judged.same_as_current,
                "demand_outlook": judged.demand_outlook,
                "transition_difficulty": judged.transition_difficulty,
                "occupation_title_en": judged.occupation_title_en.strip(),
                "matched_skills": matched,
                # 요구역량 = 이어지는 보유 역량 + 공고에만 있는 부족 역량
                "required_skills": (matched + missing)[:5],
                "missing_skills": missing,
            }
        )
    if len(seen) < len(postings):
        logger.info("직무 판정 누락 %d건 — 기본값(탈락)으로 둔다", len(postings) - len(seen))
    return results


# ── STEP 3-B. 교육 후 전환형 — 직무 제안(검색 방향 역전) ─────
# 인접 트랙은 '역량 벡터 → 유사 공고'로 찾지만, 교육 후 전환형을 같은 방식으로 하면
# 사용자 역량과 임베딩 거리가 가까운 공고(=사무·상담 계열)만 계속 나온다.
# 개발자·데이터 분석가처럼 표면 직무는 멀어도 밑바탕 역량이 전이되는 직무는 벡터가 못 올린다.
# 그래서 이 트랙만 순서를 뒤집는다: LLM이 먼저 전이 가능한 새 직무를 제안하고,
# 그다음 그 직무에 실제 훈련과정·채용이 있는지 데이터로 확인한다(jobs.py).
def _training_targets_schema(held_names: list[str]) -> type[BaseModel]:
    """직무 제안 스키마 — transferable_skills 를 보유 역량 enum 으로 강제한다.

    로컬 모델은 목록 대신 프롬프트 예시 문구('데이터를 다루는 감각' 등)를 역량명으로 쓰는 경향이
    있어, 사후 필터만으로는 제안이 전부 탈락했다. 스키마에서 선택지 자체를 보유 역량으로 묶는다.
    """
    target = create_model(
        "_TrainingTargetOut",
        job_title=(
            str,
            Field(
                description="전이 가능한 새 직무명. 한국어 표준 직무명만 — 영어 병기·괄호 설명 없이 (예: '데이터 분석가')"
            ),
        ),
        rationale=(
            str,
            Field(description="이 사람의 어떤 밑바탕 역량이 이 직무로 이어지는지, 보유 역량 근거를 들어 2~3문장"),
        ),
        transferable_skills=(
            list[_one_of(held_names)],
            Field(min_length=1, max_length=4, description="이 전이를 뒷받침하는 사용자 보유 역량명 1~4개"),
        ),
        training_needs=(
            list[str],
            Field(
                min_length=1,
                max_length=4,
                description=(
                    "직업교육으로 새로 배워야 하는 핵심 역량 2~4개. 과정·자격증 이름이 아니라 역량명 명사구 "
                    "(예: 'SQL 데이터 추출', '웹 광고 성과 분석'. '부트캠프'·'과정'·'자격증' 같은 단어 금지)"
                ),
            ),
        ),
        demand_outlook=(Literal["증가", "유지", "감소"], Field(description="해당 직종의 일반적 수요 전망")),
        transition_difficulty=(
            Literal["낮음", "보통", "높음"],
            Field(description="교육을 거친 전환의 현실적 난이도"),
        ),
        occupation_title_en=(str, Field(description=_OCCUPATION_EN_DESC)),
    )
    return create_model(
        "_TrainingTargetsOut",
        targets=(
            list[target],
            Field(min_length=1, max_length=5, description="전이 가능한 새 직무 제안 4~5개"),
        ),
    )


# 어느 경력에나 붙는 범용 전환 직무 — 괄호·공백을 뗀 직무명이 이것과 같을 때만 범용으로 본다.
# '금융 데이터 분석가', '물류 데이터 분석가'처럼 도메인을 붙인 제안은 범용이 아니다
_GENERIC_TARGETS = {"데이터분석가", "데이터사이언티스트", "프로젝트관리자", "프로젝트매니저", "비즈니스분석가", "pm"}


def is_generic_target(title: str) -> bool:
    core = re.split(r"[(（\[]", clean_job_title(title))[0].replace(" ", "").casefold()
    return core in _GENERIC_TARGETS


async def suggest_training_targets(
    profile_lines: list[str], current_job: str | None, held_names: list[str]
) -> list[dict]:
    """보유 역량(근거 포함) → 밑바탕이 전이되는 '새 직무군' 후보 제안 (교육 후 전환형).

    반환값은 직무 제안일 뿐 확정이 아니다 — 호출부(jobs.py)가 각 제안에 실제 훈련과정과
    채용이 있는지 데이터로 확인해 근거 없는 제안을 걸러낸다.
    """
    if not profile_lines:
        return []
    # 역량명은 호출부가 프로필에서 그대로 넘긴다. 줄 텍스트에서 ' (' 앞을 잘라 쓰면
    # '고객 서비스 (CS) 대응'처럼 괄호가 든 역량명이 잘려 enum 이 실제 보유 역량과 어긋났다.
    held_names = list(dict.fromkeys(n.strip() for n in held_names if n.strip()))
    out = await _parse(
        _training_targets_schema(held_names),
        prompts.TRAINING_TARGET_SYSTEM,
        f"[현재 직무]\n{current_job or '확인되지 않음'}\n\n"
        f"[보유 역량 (역량 — 근거)]\n" + "\n".join(profile_lines),
        local_max_tokens=1800,
    )
    if out is None:
        return []

    results: list[dict] = []
    generic_kept = False
    for t in out.targets:
        # 지어낸 역량명이 섞이지 않도록 보유 역량 목록으로 거르고, 이어지는 역량이 없으면 버린다
        transferable = list(dict.fromkeys(s for s in t.transferable_skills if s in held_names))
        if not transferable:
            continue
        # 범용 직무는 하나만 — 프롬프트로 제한해도 4b가 15명 중 12명에게 데이터 분석가·PM 을 함께 냈다(e2e 실측)
        if is_generic_target(t.job_title):
            if generic_kept:
                continue
            generic_kept = True
        needs = list(dict.fromkeys(
            n for n in (clean_skill_phrase(x) for x in t.training_needs) if n and not is_requirement_condition(n)
        ))
        results.append(
            {
                "job_title": clean_job_title(t.job_title),
                "rationale": t.rationale,
                "transferable_skills": transferable,
                "training_needs": needs[:4],
                "demand_outlook": t.demand_outlook,
                "transition_difficulty": t.transition_difficulty,
                "occupation_title_en": t.occupation_title_en.strip(),
            }
        )
    return results


# ── STEP 4. 학습 로드맵 (RAG) ──────────────────────────────
class _GapOut(BaseModel):
    gaps: list[str] = Field(
        min_length=1, max_length=4, description="목표 직무에 필요하지만 사용자에게 없는 역량 2~4개"
    )


class _RoadmapItemOut(BaseModel):
    # 과정명은 받지 않는다 — 소형 모델이 course_index 와 다른 과정명을 적거나 목록 한 줄(번호·기관·기간)을
    # 통째로 복사해, 학습 항목과 근거 출처가 어긋났다. 과정명·기간은 서버가 course_index 로 채운다.
    # 격차명은 받지 않고 번호(gap_no)만 받는다. 위치(항목 i = 격차 i)로 맞추면 9b가 항목 순서를 바꿔 내
    # 고른 과정이 전부 '다른 격차의 후보'로 거절됐다(콜센터 로드맵 과정 0/5). 서버는 gap_no 로 맞춘다.
    # 맞는 과정이 없을 때 모델이 지어내던 '일반 학습 항목'은 받지 않는다 — 서버가 '훈련과정 없음'으로 표기한다
    course_index: int = Field(description="이 격차의 후보 과정 중 근거로 고른 과정 번호. 관련 과정이 없으면 -1")
    duration_weeks: int = Field(ge=1, le=52, description="예상 소요 기간(주) — 과정 일정이 없을 때만 사용됨")


def _roadmap_schema(count: int) -> type[BaseModel]:
    # gap_no 를 맨 앞에 둬 모델이 격차를 먼저 밝히고 그 격차의 후보에서 고르게 한다
    fields = {name: (f.annotation, f) for name, f in _RoadmapItemOut.model_fields.items()}
    item = create_model(
        "_RoadmapGapItemOut",
        gap_no=(Literal[tuple(range(1, count + 1))], Field(description="이 항목이 다루는 격차 번호")),
        **fields,
    )
    return create_model(
        "_RoadmapOut",
        items=(
            list[item],
            Field(min_length=count, max_length=count, description="격차 번호 순서대로 격차당 하나"),
        ),
    )


# 로드맵 과정 '근거 약함' 표시 기준(격차↔과정 코사인). 실측: 맞는 과정 — CS 0.59, 물류관리사 0.60,
# Power BI 0.55, 수출입 0.51~0.57 / 약한 연결 — 브랜딩의 힘 0.50, B2B GTM 0.50. 0.52 미만을 약함으로 본다.
# 코사인만으로는 갈리지 않는 경우(요양 과정 0.58)는 NCS 분류를 본 LLM 선택에 맡긴다.
WEAK_GAP_COURSE = 0.52
# 유사도가 이 값 미만이면서 과정명·NCS 분류에 격차의 핵심어가 하나도 없으면 '근거 약함'으로 본다.
# 유사도만으로는 맞는 과정(0.51~0.60)과 엉뚱한 과정이 겹쳤다 — 4b가 고른 'CRM 활용 능력' → '엑셀 함수와
# 매크로 활용능력', '일정 관리 및 조율' → 'R&D 기술로드맵' 은 0.52 이상이라 표시가 붙지 않았다
WEAK_GAP_COURSE_NO_KEYWORD = 0.58
# 어느 역량·과정에나 붙는 말 — 이것만 겹치면 같은 주제로 보지 않는다
_GENERIC_BIGRAMS = {
    "활용", "능력", "관리", "기술", "전략", "방법", "실무", "과정", "기초", "이해", "역량", "업무", "수립",
    "작성", "처리", "향상", "기법", "도구", "분석", "시스", "스템", "운영", "전문", "문가", "양성", "프로",
    "로그", "그램", "실전", "입문", "심화", "개선", "강화", "마스", "스터", "가이", "이드", "비즈", "즈니", "니스",
}
_GENERIC_LATIN = {"up", "with", "and", "the", "for", "a", "to", "of", "in", "ai"}


def _keyword_units(text: str) -> tuple[set[str], set[str]]:
    """(한글 2글자 묶음, 영문·숫자 단어). 띄어쓰기가 제각각이라('활용능력') 한글은 단어마다 2글자씩 자른다."""
    hangul: set[str] = set()
    for word in re.findall(r"[가-힣]+", text):
        hangul.update(word[i : i + 2] for i in range(len(word) - 1))
    latin = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+#]*", text)}
    return hangul - _GENERIC_BIGRAMS, latin - _GENERIC_LATIN


def shares_keyword(gap: str, course_text: str) -> bool:
    """격차와 과정(과정명·NCS 분류)이 핵심어를 하나라도 공유하는지 — '통계적'↔'통계학', 'SQL'↔'MySQL'."""
    gap_h, gap_l = _keyword_units(gap)
    course_h, course_l = _keyword_units(course_text)
    if gap_h & course_h:
        return True
    return any(g in c or c in g for g in gap_l for c in course_l if len(g) >= 2 and len(c) >= 2)


def course_weak_reason(gap: str, course: dict, sim: float) -> str | None:
    """로드맵 과정의 '근거 약함' 이유. 과정은 숨기지 않고 이유만 붙인다 — 판단은 사용자 몫."""
    if sim < WEAK_GAP_COURSE:
        return f"과정과 역량의 연관도가 낮은 편입니다(유사도 {sim:.2f}). 고용24에서 과정 내용을 확인해 주세요."
    course_text = " ".join(str(course.get(k) or "") for k in ("course_name", "ncs_nm", "ncs_units"))
    if sim < WEAK_GAP_COURSE_NO_KEYWORD and not shares_keyword(gap, course_text):
        return (
            f"과정명·NCS 분류에 '{gap}'의 핵심어가 없습니다(유사도 {sim:.2f}). "
            "고용24에서 과정 내용을 확인해 주세요."
        )
    return None
# 맞는 훈련과정이 없을 때 학습 항목에 쓰는 표기
NO_COURSE_LABEL = "훈련과정 없음"


def course_duration_weeks(start: str | None, end: str | None) -> int | None:
    """훈련과정 실제 일정(YYYY-MM-DD 또는 YYYYMMDD)으로 기간(주)을 계산 — 모델 추정치보다 우선한다."""
    def parse(s: str | None) -> date | None:
        digits = re.sub(r"\D", "", s or "")
        if len(digits) != 8:
            return None
        try:
            return date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
        except ValueError:
            return None

    s, e = parse(start), parse(end)
    if not s or not e or e < s:
        return None
    return max(1, min(52, -(-((e - s).days + 1) // 7)))


async def extract_skill_gaps(skills: list[str], target_job: str) -> list[str]:
    """보유 역량 + 목표 직무 → 역량 격차 목록 (STEP 4 검색어)."""
    out = await _parse(
        _GapOut,
        prompts.GAP_SYSTEM,
        f"[보유 역량]\n{', '.join(skills)}\n\n[목표 직무]\n{target_job}",
        max_tokens=2000,
        local_max_tokens=300,
    )
    return [g for g in out.gaps if g not in skills] if out else []


async def build_roadmap_items(
    gaps: list[str], target_job: str, courses_by_gap: dict[str, list[dict]]
) -> list[dict]:
    """역량 격차 + 격차별로 검색된 훈련과정 → 격차당 로드맵 항목 하나 (근거 출처 포함).

    courses_by_gap 은 격차마다 pgvector 로 따로 검색한 실제 훈련과정. 격차를 한 문장으로 합쳐
    검색하면 한 격차 쪽 과정만 올라와 다른 격차에 엉뚱한 과정(예: CRM → 컴활 실기)이 붙었다.
    LLM은 각 격차의 후보 안에서만 고르고, 서버가 그 범위를 다시 확인한다.
    """
    if not gaps:
        return []
    # 같은 과정이 여러 격차 후보로 오면 번호 하나로 합친다 (중복 인용은 아래 used 가 막는다)
    courses: list[dict] = []
    index_of: dict[str, int] = {}
    allowed: list[dict[int, float]] = []  # 격차별 후보 번호 → 그 격차와의 유사도
    blocks = []
    for gi, gap in enumerate(gaps):
        idxs: dict[int, float] = {}
        lines = []
        for c in courses_by_gap.get(gap, []):
            if c["course_name"] not in index_of:
                index_of[c["course_name"]] = len(courses)
                courses.append(c)
            i = index_of[c["course_name"]]
            idxs[i] = c.get("similarity") or 0.0
            # 모델에는 과정명과 NCS 분류만 준다 — 기관·기간은 서버가 붙인다(입력 토큰 절약).
            # 과정명만으로는 내용을 알기 어렵다('브랜딩의 힘'). 원천에 설명이 없어 NCS 세분류·능력단위로 보완한다
            lines.append(
                f"  [{i}] {c['course_name']}"
                + (f"\n      NCS {c['ncs_nm']}" if c.get("ncs_nm") else "")
                + (f" — 능력단위: {c['ncs_units']}" if c.get("ncs_units") else "")
            )
        allowed.append(idxs)
        blocks.append(f"격차 {gi + 1}. {gap}\n" + ("\n".join(lines) or "  (관련 훈련과정 없음)"))

    if not courses:
        # 어느 격차에도 후보 과정이 없으면 고를 것이 없다 — LLM을 부르지 않고 전부 '훈련과정 없음'
        out = _roadmap_schema(len(gaps))(
            items=[{"gap_no": i + 1, "course_index": -1, "duration_weeks": 4} for i in range(len(gaps))]
        )
    else:
        out = await _parse(
            _roadmap_schema(len(gaps)),
            prompts.ROADMAP_SYSTEM,
            f"[목표 직무]\n{target_job}\n\n[격차별 실제 훈련과정 후보]\n" + "\n\n".join(blocks),
            local_max_tokens=150 * len(gaps) + 100,
        )
    if out is None:
        return []
    # 격차 번호로 맞춘다. 같은 번호가 두 번 오면 앞의 것을, 빠진 번호는 과정 없는 일반 항목으로 채운다
    by_gap: dict[int, BaseModel] = {}
    for it in out.items:
        by_gap.setdefault(it.gap_no - 1, it)

    items = []
    used: set[int] = set()
    for gi, (gap, idxs) in enumerate(zip(gaps, allowed)):
        it = by_gap.get(gi) or _RoadmapItemOut(course_index=-1, duration_weeks=4)
        # 그 격차로 검색된 후보 번호일 때만 근거로 인정. 같은 과정 중복 인용도 막는다
        valid = it.course_index in idxs and it.course_index not in used
        course = courses[it.course_index] if valid else None
        if course:
            used.add(it.course_index)
            learning_item = course["course_name"]
            sim = idxs[it.course_index]
            weak_reason = course_weak_reason(gap, course, sim)
            duration_weeks = course_duration_weeks(course.get("start_date"), course.get("end_date")) or it.duration_weeks
            resources = []
            source = (
                f"NCS 능력단위 {course['ncs_cd']}" + (f" ({course['ncs_nm']})" if course.get("ncs_nm") else "")
                if course.get("ncs_cd")
                else f"HRD-Net 훈련과정: {course['course_name']}"
            )
            course_info = {
                "name": course["course_name"],
                "institution": course.get("institution"),
                "start_date": course.get("start_date"),
                "end_date": course.get("end_date"),
                "tuition": course.get("tuition"),
                "address": course.get("address"),
                "url": course.get("source_url"),
            }
        else:
            # DB(고용24)에 맞는 훈련과정이 없으면 학습 항목을 지어내지 않고 그대로 '훈련과정 없음'이라고 밝힌다.
            # 격차명을 붙여 두는 건 항목마다 이름이 겹치지 않게 하기 위해서다(프론트가 이름으로 완료 표시를 관리)
            learning_item = f"{NO_COURSE_LABEL} — {gap}"
            duration_weeks = 0  # 기간을 알 수 없다 — 화면은 '기간 미정'으로 보여준다
            resources = []
            source = "DB에 맞는 고용24 훈련과정 없음"
            course_info = None
            weak_reason = None
        items.append({
            "skill_gap": gap,
            "learning_item": learning_item,
            "duration_weeks": duration_weeks,
            "resources": resources,
            "source": source,
            "course": course_info,
            "weak_reason": weak_reason,
        })
    return items
