"""STEP 3. 다음 직무 탐색 — 두 트랙으로 갈라진다.

인접 직무 전환형(adjacent):
    확정된 역량 프로필 → query 임베딩 → pgvector 유사 공고 top-k
    → LLM이 공고별 수요 전망·전환 난이도·기여 역량 판정 → 격차 비율로 인접 후보만 남김.
    fit_score 는 역량↔공고 코사인 유사도를 백분율로 환산한 값.

교육 후 직무 전환형(training) — 검색 방향을 뒤집는다:
    역량 벡터로 유사 공고를 찾으면 사용자 역량과 임베딩 거리가 가까운 직무(사무·상담 계열)만
    계속 나온다. 개발자·데이터 분석가처럼 표면 직무는 멀어도 밑바탕 역량이 전이되는 직무는
    벡터가 후보로 올리지 못한다. 그래서 순서를 뒤집는다:
    LLM이 먼저 전이 가능한 새 직무를 제안 → 그 직무에 실제 훈련과정이 있는지(하드 게이트),
    실채용 수요가 있는지 데이터로 확인 → 근거 있는 제안만 카드로 노출.
    fit_score 는 유사도가 아니라 '전이 비율'(요구 중 이미 이어지는 역량 비중)이다.
"""
import asyncio
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.career import JobMatch, JobSearchTrack, SkillProfile
from app.services import embedding, llm

router = APIRouter()

TOP_K = 5
# 최종 트랙 분류(역량 격차 비율)에서 걸러질 후보를 감안해 선별 단계는 넉넉히 가져온다.
SELECT_K = TOP_K * 2
# 후보 선별 전에 현재 직무·중복 직무를 걸러내므로 검색은 더 넉넉히 한다.
SEARCH_K = TOP_K * 8
# 역량 격차 비율(missing/required)이 이 값 이하면 인접 직무로 인정한다.
# 0.4(요구역량 60%+ 보유)는 콜센터 상담원 페르소나 실측 3회 중 2회가 결과 0건이었다 —
# LLM이 뽑는 required_skills가 공고별 세부 항목(예: '한글·엑셀 문서 작성')까지 담아 과증가하는
# 경향이 있어, 실제로 인접한 직무조차 60% 문턱을 넘기 어렵다. 0.6은 3회 모두 1건 이상 통과했다.
ADJACENT_GAP_RATIO = 0.6

# 교육 후 전환형: 제안 직무에 이 유사도 이상의 훈련과정이 있어야 '배울 곳이 실재'한다고 본다.
# 실측(제안 직무명 vs training_courses top1): 실제 부트캠프 대상(웹 개발자·데이터 분석가 등)은
# 0.57~0.65, 대조군(심해 잠수부)은 0.46. 0.55가 '배울 곳 없는 제안'을 걸러낸다.
COURSE_MATCH_THRESHOLD = 0.55
# 실채용 수요 확인 — 이 유사도 이상의 공고가 있으면 대표 공고로 첨부(링크). 공고 검색은
# 변별력이 약해(무관한 직무명도 0.57+) 하드 게이트로 쓰지 않고 링크 첨부 여부에만 쓴다.
JOB_DEMAND_THRESHOLD = 0.55

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


def _gap_ratio(required_skills: list[str], missing_skills: list[str]) -> float:
    """요구역량 대비 부족역량 비율 — 값이 낮을수록 지금 역량으로 바로 갈 수 있는 직무."""
    if not required_skills:
        return 0.0
    return len(missing_skills) / len(required_skills)


def _is_adjacent_match(judged: dict) -> bool:
    """인접 직무 전환형 판정 — 격차가 작고(요구역량 대부분 보유) 실제 이어지는 역량이 있어야 한다."""
    gap_ratio = _gap_ratio(judged["required_skills"], judged["missing_skills"])
    return gap_ratio <= ADJACENT_GAP_RATIO and len(judged["matched_skills"]) > 0


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
        raise HTTPException(status_code=502, detail=f"직무 검색 실패: {e}") from e

    if exclude_job:
        rows = [r for r in rows if not _is_same_job(r["job_title"], exclude_job)]
    if not rows:
        return []

    # 괄호 설명만 다른 같은 핵심 직무는 하나의 후보로 묶고, 대표 공고와 근거 수를 남긴다.
    grouped_rows: dict[str, list[dict]] = {}
    for row in rows:
        group_key = _base_title(row["job_title"]).casefold() or row["job_title"].casefold()
        grouped_rows.setdefault(group_key, []).append(row)

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
        # 묶인 공고 여러 건의 발췌를 합쳐 요구역량 판단 근거를 넓힌다
        representative["snippet"] = "\n".join(
            f"- {row['job_title']}: {row.get('snippet') or ''}" for row in group[:3]
        )
        representatives.append(representative)

    # 넓게 검색한 실제 공고 중 인접 방향에 대체로 맞는 후보를 넉넉히 남긴다.
    # 최종 인접 판정은 역량 격차 판정(judge_job_match) 이후 비율로 가른다.
    try:
        selected_indices = await llm.select_job_candidates(
            skill_names,
            exclude_job,
            representatives,
            "adjacent_transition",
            limit=SELECT_K,
        )
        # structured output이 파싱되지 않아 빈 목록이 오더라도, 실제 검색된 공고까지
        # 사라지게 두지 않는다. 이 경우 벡터 검색 순위의 후보를 그대로 제공한다.
        if selected_indices:
            representatives = [representatives[index] for index in selected_indices]
        else:
            representatives = representatives[:SELECT_K]
    except Exception:
        # 선별 호출만 실패한 경우에도 기존 벡터 검색 결과는 제공한다.
        representatives = representatives[:SELECT_K]

    if not representatives:
        return []

    # 검색된 직무 그룹별 판정은 서로 독립적이므로 동시에 호출
    judgements = await asyncio.gather(
        *(
            llm.judge_job_match(skill_names, r["job_title"], r["snippet"])
            for r in representatives
        ),
        return_exceptions=True,
    )

    matches: list[JobMatch] = []
    for row, judged in zip(representatives, judgements):
        if isinstance(judged, Exception):  # 한 건 실패가 전체를 막지 않도록
            judged = {
                "demand_outlook": "유지",
                "transition_difficulty": "보통",
                "matched_skills": [],
                "required_skills": [],
                "missing_skills": [],
            }
        if not _is_adjacent_match(judged):
            continue
        matches.append(
            JobMatch(
                posting_id=row.get("posting_id"),
                posting_count=row["posting_count"],
                job_title=row["job_title"],
                company=row.get("company"),
                region=row.get("region"),
                source_url=row.get("source_url"),
                requirement_excerpt=row.get("snippet"),
                fit_score=round(row["similarity"] * 100, 1),
                demand_outlook=judged["demand_outlook"],
                transition_difficulty=judged["transition_difficulty"],
                matched_skills=judged["matched_skills"],
                required_skills=judged["required_skills"],
                missing_skills=judged["missing_skills"],
            )
        )
        if len(matches) == TOP_K:
            break
    return matches


async def _match_training_transition(
    profile: SkillProfile, db: Session, exclude_job: str | None
) -> list[JobMatch]:
    """교육 후 직무 전환형 — LLM이 제안한 새 직무를 훈련과정·채용 데이터로 검증한다."""
    profile_lines = [line for line in _skill_search_text(profile).split("\n") if line.strip()]
    try:
        targets = await llm.suggest_training_targets(profile_lines, exclude_job)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"전환 직무 제안 실패: {e}") from e

    # LLM이 현재 직무와 같은 직무군을 실수로 제안하면 제외 (이 트랙은 '새 분야'가 핵심)
    if exclude_job:
        targets = [t for t in targets if not _is_same_job(t["job_title"], exclude_job)]
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

    matches: list[JobMatch] = []
    for result in verified:
        if isinstance(result, Exception):
            continue
        target, courses, jobs = result
        # 하드 게이트: 실제로 배울 곳(훈련과정)이 있는 직무만 남긴다 — 근거 없는 제안 차단
        if not courses or courses[0]["similarity"] < COURSE_MATCH_THRESHOLD:
            continue
        # 실채용 수요가 확인되면 대표 공고를 첨부(링크). 미달이면 링크 없이 직무만 노출.
        posting = jobs[0] if jobs and jobs[0]["similarity"] >= JOB_DEMAND_THRESHOLD else None

        transferable = target["transferable_skills"]
        training = target["training_needs"]
        # 요구역량 = 이어지는 보유 역량 + 새로 배울 역량 (직무가 필요로 하는 전체)
        required = transferable + [x for x in training if x not in transferable]
        # fit_score 를 벡터 유사도로 두면 이 트랙은 태생적으로 유사도가 낮아 카드마다 적합도가
        # 바닥으로 뜨는 모순이 생긴다. 대신 '전이 비율' — 요구역량 중 이미 이어지는 비중 — 을 쓴다.
        fit = round(len(transferable) / (len(required) or 1) * 100, 1)

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
            )
        )
        if len(matches) == TOP_K:
            break
    return matches


@router.post("/match", response_model=list[JobMatch], summary="역량 프로필 → 다음 직무 매핑")
async def match_jobs(
    profile: SkillProfile,
    db: Session = Depends(get_db),
    exclude_job: str | None = None,
    search_track: JobSearchTrack = "adjacent_transition",
) -> list[JobMatch]:
    """선택한 탐색 경로에 맞춰 다음 직무 후보를 찾고 현재 직무는 제외한다."""
    if not profile.skills:
        return []
    if search_track == "training_transition":
        return await _match_training_transition(profile, db, exclude_job)
    return await _match_adjacent_transition(profile, db, exclude_job)
