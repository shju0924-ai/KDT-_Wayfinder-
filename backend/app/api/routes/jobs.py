"""STEP 3. 다음 직무 탐색 — 두 트랙으로 갈라진다.

인접 직무 전환형(adjacent):
    확정된 역량 프로필 → query 임베딩 → pgvector 유사 공고 top-k → 벡터 순위 상위부터
    TOP_K 건씩 묶어 LLM이 수요 전망·전환 난이도·기여 역량 판정 → 이어지는 역량이 충분한 후보만 남김.
    fit_score 는 역량↔공고 코사인 유사도를 백분율로 환산한 값.

교육 후 직무 전환형(training) — 검색 방향을 뒤집는다:
    역량 벡터로 유사 공고를 찾으면 사용자 역량과 임베딩 거리가 가까운 직무(사무·상담 계열)만
    계속 나온다. 개발자·데이터 분석가처럼 표면 직무는 멀어도 밑바탕 역량이 전이되는 직무는
    벡터가 후보로 올리지 못한다. 그래서 순서를 뒤집는다:
    LLM이 먼저 전이 가능한 새 직무를 제안 → 그 직무에 실제 훈련과정이 있는지(하드 게이트),
    실채용 수요가 있는지 데이터로 확인 → 근거 있는 제안만 카드로 노출.
    fit_score 는 역량 프로필 ↔ '직무 + 요구역량' 문장의 코사인 유사도(인접 트랙과 같은 척도).
"""
import asyncio
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.career import JobMatch, JobSearchTrack, SkillProfile
from app.services import automation, embedding, llm

router = APIRouter()
logger = logging.getLogger(__name__)

TOP_K = 5
# 격차 판정에서 걸러질 후보를 감안해 판정 대상은 넉넉히 둔다 — TOP_K 건씩 묶어 최대 2회 호출.
# 첫 묶음에서 TOP_K 건이 모두 통과하면 두 번째 호출은 하지 않는다(로컬 모델은 호출당 1~2분).
SELECT_K = TOP_K * 2
# 판정 전에 현재 직무·중복 직무를 걸러내므로 검색은 더 넉넉히 한다.
SEARCH_K = TOP_K * 8
# 인접 판정: 공고 업무에 실제로 쓰이는 보유 역량이 이 수 이상이면 인접 직무로 인정한다.
# 이전 기준(부족 역량 비율 ≤ 0.6)은 모델이 부족 역량을 몇 개 적느냐에 좌우됐다 — 4b→9b 교체만으로
# MD 페르소나 결과가 5건→2건이 됐다(2개 이어짐·4개 부족 = 0.67 탈락). 이어지는 역량은 보유 역량
# enum 안에서만 고르므로 모델이 바뀌어도 덜 흔들린다. 보유 역량이 이보다 적으면 전부 이어져야 한다.
ADJACENT_MIN_MATCHED = 2
# 엄격 기준 통과가 이보다 적으면 이어지는 역량 1개짜리 후보로 이만큼까지 채운다('근거 약함' 표시)
ADJACENT_MIN_RESULTS = 3
# 판정 근거로 붙일 같은 직무명 공고 수 (대표 공고 포함 최대 이만큼 + 1)
SAMPLE_PER_TITLE = 4
# 기계 조작·조립·운전·단순 노무 직무명 (KECO 표준명 끝말, 공백 제거 후 비교).
# LLM은 범용 역량('프로젝트 관리')을 이어 붙여 생산관리 → 압연기 조작원을 냈고, 판정 필드로 물으면
# 무역·물류 사무원까지 조작직으로 오판했다. 직무명은 표준 분류명이라 끝말 규칙이 더 정확하다.
# '종사원'은 '기타 사무 지원 종사원'·'일선 관리 종사원'도 있어 '단순 종사원' 등만 잡는다.
_SIMPLE_LABOR = re.compile(r"단순종사원|단순노무")
_MANUAL_TITLE = re.compile(
    r"(조작원|조립원|검사원|운전원|정비원|수리원|청소원|경비원|기능원|단순종사원|적재종사원|기능종사원)$"
)

# 교육 후 전환형: 제안 직무에 이 유사도 이상의 훈련과정이 있어야 '배울 곳이 실재'한다고 본다.
# 실측(제안 직무명 vs training_courses top1): 실제 부트캠프 대상(웹 개발자·데이터 분석가 등)은
# 0.57~0.65, 대조군(심해 잠수부)은 0.46. 0.55가 '배울 곳 없는 제안'을 걸러낸다.
COURSE_MATCH_THRESHOLD = 0.55
# 실채용 수요 확인 — 이 유사도 이상의 공고가 있으면 대표 공고로 첨부(링크). 공고 검색은
# 변별력이 약해(무관한 직무명도 0.57+) 하드 게이트로 쓰지 않고 링크 첨부 여부에만 쓴다.
JOB_DEMAND_THRESHOLD = 0.55

# '근거 약함' 표시 기준 — 걸러내지 않고 카드에 이유만 붙인다.
# 인접 적합도(역량↔공고 코사인×100) 실측: 무관에 가까운 웹 기획자 52.3·경영 기획 사무원 52.6(생산관리) /
# 인접 직무 57~66, 생산·품질관리 사무원 54.3. 54 미만을 약함으로 본다.
WEAK_FIT_SCORE = 54.0

# 현재 직무명과 후보 직무명의 임베딩 유사도가 이 값 이상이면 '같은 직무'로 보고 뺀다.
# 부분 문자열 비교는 '콜센터 상담원' ↔ '고객 상담원'처럼 표기만 다른 같은 직무를 못 잡았다.
# 실측(bge-m3): 같은 직무 — 고객 상담원 0.83, 인바운드 상담원 0.73, 영업 및 마케팅 사무원 0.82,
# 영업지원 사무원 0.87 / 전환 후보 — 영업 및 판매 관련 관리자 0.69, 마케팅·광고 사무원 0.70, 텔레마케터 0.54.
SAME_JOB_SIMILARITY = 0.72

TRACK_SEARCH_CONTEXT: dict[JobSearchTrack, str] = {
    "adjacent_transition": (
        "현재 직무와 동일하지 않지만 사용자가 이미 가진 역량과 업무 수행 방식을 "
        "그대로 이어 쓸 수 있는 인접 직무"
    ),
}


def _base_title(title: str) -> str:
    """괄호 안 부연설명을 떼고 공백을 지운 핵심 직무명.

    공고 제목은 '콜센터 상담원(콜센터·고객센터·CS센터)'처럼, LLM 추출 직무명은
    '콜센터 상담원(강성 민원 처리 담당)'처럼 괄호 표기가 제각각이라 그대로는 매칭되지 않는다.
    """
    return re.split(r"[(（\[]", title)[0].replace(" ", "").strip()


def _is_same_job(candidate: str, current: str) -> bool:
    """현재 직무와 사실상 같은 공고인지 — 표기가 제각각이라 핵심 직무명 부분 일치로 판단."""
    a, b = _base_title(candidate), _base_title(current)
    return bool(a) and bool(b) and (b in a or a in b)


async def _drop_same_jobs(items: list[dict], current: str | None) -> list[dict]:
    """현재 직무와 사실상 같은 직무(item['job_title'])를 뺀다 — 부분 일치 또는 직무명 임베딩 유사도.

    LLM 판정(same_as_current)보다 앞에서 거르면 판정 호출에 같은 직무가 자리를 차지하지 않아,
    로컬 모델에서 두 번째 판정 묶음 호출이 필요한 경우가 줄어든다.
    """
    if not current or not items:
        return items
    items = [it for it in items if not _is_same_job(it["job_title"], current)]
    titles = list(dict.fromkeys(_base_title(it["job_title"]) or it["job_title"] for it in items))
    try:
        sims = dict(zip(titles, await embedding.similarities_to(_base_title(current) or current, titles)))
    except Exception:
        return items  # 임베딩 실패 시 부분 일치 필터 결과만으로 진행
    return [
        it for it in items
        if sims.get(_base_title(it["job_title"]) or it["job_title"], 0.0) < SAME_JOB_SIMILARITY
    ]


def _group_key(row: dict) -> str:
    """괄호 설명만 다른 같은 핵심 직무를 한 후보로 묶는 키."""
    return _base_title(row["job_title"]).casefold() or row["job_title"].casefold()


def _is_manual_title(title: str | None) -> bool:
    """고용직업분류(KECO) 표준 직무명 기준 기계 조작·조립·운전·단순 노무 직무인지."""
    return bool(title) and bool(_MANUAL_TITLE.search(_base_title(title)))


def _drop_manual_downshift(items: list[dict], current: str | None) -> list[dict]:
    """사무·관리·서비스 경력자에게 기계 조작·조립·운전·단순 노무 직무를 인접 후보로 내지 않는다.
    현재 직무도 그런 직무면(예: CNC 선반 조작원) 같은 계열 이동은 그대로 둔다."""
    if _is_manual_title(current):
        # 조작직 경력자도 단순 노무직으로 내리지는 않는다(CNC 10년차 → '기타 제조 관련 단순 종사원' 실측)
        if _SIMPLE_LABOR.search(_base_title(current or "")):
            return items
        kept = [it for it in items if not _SIMPLE_LABOR.search(_base_title(it["job_title"]))]
    else:
        kept = [it for it in items if not _is_manual_title(it["job_title"])]
    if len(kept) < len(items):
        logger.info("인접 후보 제외(조작·노무직) %d건", len(items) - len(kept))
    return kept


def _is_adjacent_match(judged: dict, skill_count: int) -> bool:
    """인접 직무 전환형 판정 — 현재 직무와 다르고, 공고 업무에 실제로 쓰이는 보유 역량이 충분해야 한다.
    표기만 다른 같은 직무는 _is_same_job 이 못 잡아 LLM 판정(same_as_current)으로 거른다."""
    if judged.get("same_as_current"):
        return False
    return len(judged["matched_skills"]) >= min(ADJACENT_MIN_MATCHED, skill_count)


def _skill_search_text(profile: SkillProfile) -> str:
    """역량명만으로는 도메인 맥락이 사라진다 — 근거 문장까지 합쳐 임베딩 입력을 만든다."""
    return "\n".join(
        f"{s.name} ({s.category}) — {s.evidence}" if s.evidence else s.name
        for s in profile.skills
    )


async def _match_adjacent_transition(
    profile: SkillProfile, db: Session, exclude_job: str | None
) -> list[JobMatch]:
    """인접 직무 전환형 — 역량 벡터로 유사 공고를 찾고 격차가 작은 후보만 남긴다."""
    skill_names = [s.name for s in profile.skills]
    try:
        rows = await embedding.search_similar_jobs(
            db,
            _skill_search_text(profile),
            limit=SEARCH_K,
            search_context=TRACK_SEARCH_CONTEXT["adjacent_transition"],
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"직무 검색 실패: {e!r}") from e

    rows = _drop_manual_downshift(await _drop_same_jobs(rows, exclude_job), exclude_job)
    if not rows:
        return []

    # 괄호 설명만 다른 같은 핵심 직무는 하나의 후보로 묶고, 대표 공고와 근거 수를 남긴다.
    grouped_rows: dict[str, list[dict]] = {}
    for row in rows:
        grouped_rows.setdefault(_group_key(row), []).append(row)

    groups = sorted(
        grouped_rows.values(),
        key=lambda group: max(row.get("ranking_similarity", row["similarity"]) for row in group),
        reverse=True,
    )

    representatives: list[dict] = []
    for group in groups:
        representative = max(
            group,
            key=lambda row: row.get("ranking_similarity", row["similarity"]),
        ).copy()
        representative["posting_count"] = len(group)
        # 카드에 보일 근거는 대표 공고 원문 발췌 — 판정용 발췌(여러 공고)는 아래에서 따로 만든다
        representative["snippet_raw"] = representative.get("snippet") or ""
        representatives.append(representative)

    # 대표 공고 한 건만 보면 그 회사만의 요구('영문 이메일', '운전면허')가 부족 역량이 된다.
    # 같은 직무명 공고를 DB에서 몇 건 더 붙여, 판정이 직무 공통 요구를 가려내게 한다.
    candidates = representatives[:SELECT_K]
    group_titles = {id(r): {row["job_title"] for row in grouped_rows[_group_key(r)]} for r in candidates}
    try:
        samples = await embedding.run_db(
            embedding.sample_postings_by_title,
            [t for titles in group_titles.values() for t in titles],
            SAMPLE_PER_TITLE,
        )
    except Exception:
        samples = {}
    for rep in candidates:
        extra = [s for t in group_titles[id(rep)] for s in samples.get(t, [])]
        snippets = [s for s in dict.fromkeys([rep.get("snippet_raw") or "", *extra]) if s.strip()]
        rep["snippet"] = "\n".join(f"- {s}" for s in snippets)[: llm.JUDGE_SNIPPET_CHARS]

    # 벡터 유사도 순위 상위부터 TOP_K 건씩 묶어 판정하고, 인접 판정을 통과한 것만 남긴다.
    matches: list[JobMatch] = []
    titles_en: list[str | None] = []  # 카드별 영문 표준 직업명 — AI 노출도 매칭용
    # 이어지는 역량이 1개뿐인 후보 — 엄격 기준 결과가 너무 적을 때만 '근거 약함'으로 채운다
    fallback: list[tuple[dict, dict]] = []
    for start in range(0, len(candidates), TOP_K):
        if len(matches) == TOP_K:
            break
        batch = candidates[start : start + TOP_K]
        try:
            judgements = await llm.judge_job_matches(skill_names, exclude_job, batch)
        except Exception as e:
            # LLM 자체가 죽은 경우(키 오류·Ollama 미기동 등) '결과 0건'으로 숨기지 않는다
            if matches:
                break
            raise HTTPException(status_code=502, detail=f"직무 판정 실패: {e!r}") from e
        for row, judged in zip(batch, judgements):
            if len(matches) == TOP_K:
                break
            if not _is_adjacent_match(judged, len(skill_names)):
                if not judged.get("same_as_current") and judged["matched_skills"]:
                    fallback.append((row, judged))
                logger.info(
                    "인접 후보 탈락 %s: same=%s matched=%d missing=%d",
                    row["job_title"], judged.get("same_as_current"),
                    len(judged["matched_skills"]), len(judged["missing_skills"]),
                )
                continue
            matches.append(_adjacent_job_match(row, judged))
            titles_en.append(judged.get("occupation_title_en"))

    # 전문 기술 직무(웹 퍼블리셔 등)는 LLM이 공고마다 이어지는 역량을 1개만 고르는 일이 많아
    # 결과가 0건이 됐다. 빈 화면보다, 이유를 밝힌 약한 후보를 보여주고 사용자가 고르게 한다.
    for row, judged in fallback[: max(0, ADJACENT_MIN_RESULTS - len(matches))]:
        match = _adjacent_job_match(row, judged)
        match.weak_reasons.insert(0, "보유 역량 중 이 직무 업무와 이어지는 것이 1개뿐입니다.")
        matches.append(match)
        titles_en.append(judged.get("occupation_title_en"))
    return await _with_ai_exposure(matches, titles_en)


async def _with_ai_exposure(matches: list[JobMatch], titles_en: list[str | None]) -> list[JobMatch]:
    """카드마다 공개 직업 AI 노출도(ILO·Anthropic)를 붙인다 — 점수와 매칭 직업명·출처만, 판정 없음.
    매칭 자료가 없거나 조회가 실패해도 카드는 그대로 낸다(참고 정보)."""
    try:
        exposures = await embedding.run_db(automation.occupation_exposures, titles_en)
    except Exception as e:
        logger.warning("직무별 AI 노출도 조회 실패: %r", e)
        return matches
    for match, (score, sources) in zip(matches, exposures):
        match.ai_exposure_score = score
        match.ai_exposure_sources = sources
    return matches


def _adjacent_weak_reasons(fit: float, judged: dict) -> list[str]:
    """인접 직무 카드의 근거가 약한 이유. 카드는 숨기지 않고 이유만 붙인다 — 판단은 사용자 몫.
    전환 난이도 '높음'은 카드에 이미 태그로 보이므로 이유로 중복하지 않는다(실측 38장 중 13장에 붙어 표시가 무의미해졌다)."""
    reasons = []
    if fit < WEAK_FIT_SCORE:
        reasons.append(f"내 역량과 공고 내용의 유사도가 낮은 편입니다 (적합도 {fit}).")
    return reasons


def _adjacent_job_match(row: dict, judged: dict) -> JobMatch:
    fit = round(row["similarity"] * 100, 1)
    return JobMatch(
        posting_id=row.get("posting_id"),
        posting_count=row["posting_count"],
        job_title=row["job_title"],
        company=row.get("company"),
        region=row.get("region"),
        source_url=row.get("source_url"),
        requirement_excerpt=row.get("snippet_raw") or row.get("snippet"),
        fit_score=fit,
        demand_outlook=judged["demand_outlook"],
        transition_difficulty=judged["transition_difficulty"],
        matched_skills=judged["matched_skills"],
        required_skills=judged["required_skills"],
        missing_skills=judged["missing_skills"],
        weak_reasons=_adjacent_weak_reasons(fit, judged),
    )


async def _match_training_transition(
    profile: SkillProfile, db: Session, exclude_job: str | None
) -> list[JobMatch]:
    """교육 후 직무 전환형 — LLM이 제안한 새 직무를 훈련과정·채용 데이터로 검증한다."""
    profile_lines = [line for line in _skill_search_text(profile).split("\n") if line.strip()]
    try:
        targets = await llm.suggest_training_targets(
            profile_lines, exclude_job, [s.name for s in profile.skills]
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"전환 직무 제안 실패: {e!r}") from e

    # LLM이 현재 직무와 같은 직무를 실수로 제안하면 제외 (이 트랙은 '새 분야'가 핵심)
    targets = await _drop_same_jobs(targets, exclude_job)
    if not targets:
        return []

    async def verify(target: dict) -> tuple[dict, list[dict], list[dict]]:
        courses, jobs = await asyncio.gather(
            embedding.search_similar_courses(db, target["job_title"], limit=3),
            embedding.search_similar_jobs(db, target["job_title"], limit=3),
        )
        return target, courses, jobs

    verified = await asyncio.gather(
        *(verify(t) for t in targets), return_exceptions=True
    )
    # 검증 검색이 전부 실패했다면(DB 연결 끊김 등) '결과 0건'으로 숨기지 않는다
    errors = [r for r in verified if isinstance(r, Exception)]
    if errors and len(errors) == len(verified):
        raise HTTPException(status_code=502, detail=f"훈련과정·채용 검증 실패: {errors[0]}")

    # 적합도: 사용자 역량 프로필과 '직무 + 요구역량' 문장의 벡터 유사도 — 인접 트랙(프로필 ↔ 공고)과 같은 척도.
    # 이전에는 '이어지는 역량 수 / 요구역량 수'였는데, 모델이 거의 항상 2개·3개를 내 모든 카드가 40.0이 됐다.
    ok = [r for r in verified if not isinstance(r, Exception)]
    try:
        vecs = await embedding.embed_queries(
            [_skill_search_text(profile)]
            + [
                f"{t['job_title']}\n요구 역량: {', '.join(t['transferable_skills'] + t['training_needs'])}"
                for t, _, _ in ok
            ]
        )
        fits = [round(sum(a * b for a, b in zip(vecs[0], v)) * 100, 1) for v in vecs[1:]]
    except Exception:
        fits = [None] * len(ok)
    verified = sorted(zip(ok, fits), key=lambda x: x[1] or 0, reverse=True)

    matches: list[JobMatch] = []
    titles_en: list[str | None] = []
    for (target, courses, jobs), fit in verified:
        # 하드 게이트: 실제로 배울 곳(훈련과정)이 있는 직무만 남긴다 — 근거 없는 제안 차단
        if not courses or courses[0]["similarity"] < COURSE_MATCH_THRESHOLD:
            continue
        # 실채용 수요가 확인되면 대표 공고를 첨부(링크). 미달이면 링크 없이 직무만 노출.
        posting = jobs[0] if jobs and jobs[0]["similarity"] >= JOB_DEMAND_THRESHOLD else None

        transferable = target["transferable_skills"]
        training = target["training_needs"]
        # 요구역량 = 이어지는 보유 역량 + 새로 배울 역량 (직무가 필요로 하는 전체)
        required = transferable + [x for x in training if x not in transferable]
        if fit is None:  # 임베딩 실패 시에만 '전이 비율'(이어지는 역량 수 / 요구역량 수)로 대신한다
            fit = round(len(transferable) / (len(required) or 1) * 100, 1)
        weak = []
        if posting is None:
            weak.append("이 직무명으로 확인된 실제 채용공고가 없어 수요가 데이터로 확인되지 않았습니다.")

        matches.append(
            JobMatch(
                posting_id=posting.get("posting_id") if posting else None,
                posting_count=1,
                job_title=target["job_title"],
                company=posting.get("company") if posting else None,
                region=posting.get("region") if posting else None,
                source_url=posting.get("source_url") if posting else None,
                # 이 트랙의 핵심 근거: 왜 이 사람의 밑바탕 역량이 이 직무로 전이되는지
                requirement_excerpt=target["rationale"],
                fit_score=fit,
                demand_outlook=target["demand_outlook"],
                transition_difficulty=target["transition_difficulty"],
                matched_skills=transferable,
                required_skills=required,
                missing_skills=training,
                weak_reasons=weak,
            )
        )
        titles_en.append(target.get("occupation_title_en"))
        if len(matches) == TOP_K:
            break
    return await _with_ai_exposure(matches, titles_en)


@router.post("/match", response_model=list[JobMatch], summary="역량 프로필 → 다음 직무 매핑")
async def match_jobs(
    profile: SkillProfile,
    db: Session = Depends(get_db),
    exclude_job: str | None = None,
    search_track: JobSearchTrack = "adjacent_transition",
) -> list[JobMatch]:
    """선택한 탐색 경로에 맞춰 다음 직무 후보를 찾고 현재 직무는 제외한다.

    exclude_job 이 없으면 STEP 2가 경력 서사에서 뽑은 profile.current_job_title 로 제외한다
    (이력서·자유 텍스트 경로는 사용자가 직무명을 따로 입력하지 않는다)."""
    if not profile.skills:
        return []
    exclude_job = exclude_job or profile.current_job_title
    if search_track == "training_transition":
        return await _match_training_transition(profile, db, exclude_job)
    return await _match_adjacent_transition(profile, db, exclude_job)
