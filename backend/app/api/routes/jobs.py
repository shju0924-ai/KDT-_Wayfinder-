"""STEP 3. 다음 직무 탐색 — 두 트랙으로 갈라진다.

인접 직무 전환형(adjacent):
    확정된 역량마다 따로 pgvector 유사 공고 검색 → 직무별로 모아 '연결된 역량 수'가 많은 순으로 정렬
    (직무 점수는 상위 공고 몇 건의 평균) → TOP_K 건 단위로 직무마다 LLM이 수요 전망·전환 난이도·기여 역량 판정
    (기여 역량 선택지는 그 직무에 연결된 역량뿐) → 이어지는 역량이 충분한 후보만 남김.
    fit_score 는 역량 프로필 전체↔대표 공고 코사인 유사도를 백분율로 환산한 값.

교육 후 직무 전환형(training) — NCS 직무 후보 안에서만 고른다:
    역량 벡터로 유사 공고를 찾으면 사무·상담 계열만 계속 나오고, LLM에게 직무명을 자유롭게 지어 내게 하면
    범용 직무(데이터 분석가·프로젝트 관리자)가 반복됐다. 그래서 훈련과정이 있는 NCS 세분류를 역량마다
    검색하고 현재 분야(중분류)를 뺀 후보 목록을 만들고(services/ncs_jobs.py), LLM은 그중에서 밑바탕 역량이
    이어지는 직무를 번호로 고른다 → 실채용 공고가 있으면 링크를 붙여 카드로 노출.
    fit_score 는 역량 프로필 ↔ '직무 + 요구역량' 문장의 코사인 유사도(인접 트랙과 같은 척도).
"""
import asyncio
import logging
import re

from fastapi import APIRouter

from app.api.errors import upstream_error
from app.schemas.career import JobMatch, JobSearchTrack, SkillItem, SkillProfile
from app.services import automation, embedding, llm, ncs_jobs

router = APIRouter()
logger = logging.getLogger(__name__)

TOP_K = 5
# 격차 판정에서 걸러질 후보를 감안해 판정 대상은 넉넉히 둔다 — TOP_K 건 단위로 최대 2단위.
# 첫 단위에서 TOP_K 건이 모두 통과하면 두 번째 단위는 판정하지 않는다(로컬 모델은 직무당 30초~1분).
SELECT_K = TOP_K * 2
# 인접 후보 검색 — 역량마다 따로 공고를 찾아 직무별로 모은다(_aggregate_skill_hits).
# 역량 하나당 연결할 직무 수. 판정 전에 현재 직무·조작직을 걸러내므로 판정 대상(SELECT_K)보다 넉넉히 둔다
PER_SKILL_TITLES = 15
# 직무별 점수에 평균낼 공고 수 — 공고 한 건의 특이한 문구에 순위가 흔들리지 않게
TITLE_TOP_POSTINGS = 3
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
    dropped = sorted({t: s for t, s in sims.items() if s >= SAME_JOB_SIMILARITY}.items(), key=lambda x: -x[1])
    if dropped:
        logger.info("같은 직무로 제외(현재 %r): %s", current, ", ".join(f"{t} {s:.2f}" for t, s in dropped))
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
        logger.info(
            "인접 후보 제외(조작·노무직) %s",
            ", ".join(sorted({it["job_title"] for it in items} - {it["job_title"] for it in kept})),
        )
    return kept


def _is_adjacent_match(judged: dict, skill_count: int) -> bool:
    """인접 직무 전환형 판정 — 현재 직무와 다르고, 공고 업무에 실제로 쓰이는 보유 역량이 충분해야 한다.
    표기만 다른 같은 직무는 _is_same_job 이 못 잡아 LLM 판정(same_as_current)으로 거른다."""
    if judged.get("same_as_current"):
        return False
    return len(judged["matched_skills"]) >= min(ADJACENT_MIN_MATCHED, skill_count)


def _skill_line(s: SkillItem) -> str:
    """역량(NCS 능력단위) 한 줄 — 능력단위명만으로는 도메인 맥락이 사라져 경력 근거 구절까지 합친다.

    사용자가 NCS 검색으로 직접 추가한 능력단위는 경력 근거가 없어 능력단위 정의를 쓴다."""
    tail = s.evidence if s.evidence and not s.user_added else s.definition
    return " ".join(f"{s.name} ({s.ncs_job}) — {tail}".split())


def _skill_search_text(profile: SkillProfile) -> str:
    return "\n".join(_skill_line(s) for s in profile.skills)


def _dedupe_targets(targets: list[dict]) -> list[dict]:
    """같은 핵심 직무명 제안은 첫 번째만 남긴다. LLM이 같은 직무를 두 번 내면 카드가 겹치고,
    프론트는 직무명으로 선택 상태를 구분해 둘 다 선택된 것처럼 보였다."""
    seen: set[str] = set()
    out = []
    for t in targets:
        key = _group_key(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _aggregate_skill_hits(skill_names: list[str], per_skill_rows: list[list[dict]]) -> list[dict]:
    """역량별 검색 결과 → 직무 후보(핵심 직무명 단위), 연결된 역량이 많은 순.

    - 역량마다 직무별 점수 = 그 직무에서 역량과 가장 가까운 공고 TITLE_TOP_POSTINGS 건의 평균 유사도.
      공고 한 건의 특이한 문구로 무관한 직무('도료·잉크 생산기계 조작원')가 올라오지 않게 평균을 쓴다.
    - 역량마다 점수 상위 PER_SKILL_TITLES 개 직무에 그 역량이 '연결'된다(linked_skills).
    - 정렬: 연결 역량 수(인접 판정 기준과 같은 축) → 연결 역량 점수 평균.
    대표 공고는 프로필 전체와 가장 가까운 공고 — 카드 적합도(fit_score)가 이전과 같은 척도로 남는다."""
    groups: dict[str, dict] = {}
    for name, rows in zip(skill_names, per_skill_rows):
        by_key: dict[str, list[dict]] = {}
        for row in rows:
            by_key.setdefault(_group_key(row), []).append(row)
        scored = []
        for key, group_rows in by_key.items():
            top = sorted(group_rows, key=lambda r: r["skill_sim"], reverse=True)[:TITLE_TOP_POSTINGS]
            scored.append((key, sum(r["skill_sim"] for r in top) / len(top), group_rows))
        scored.sort(key=lambda x: x[1], reverse=True)
        for key, score, group_rows in scored[:PER_SKILL_TITLES]:
            group = groups.setdefault(key, {"rows": {}, "scores": {}})
            group["scores"][name] = score
            for row in group_rows:
                group["rows"].setdefault(row["posting_id"], row)

    candidates = []
    for group in groups.values():
        rows = list(group["rows"].values())
        rep = max(rows, key=lambda r: r["similarity"]).copy()
        rep["posting_count"] = len(rows)
        rep["group_titles"] = sorted({r["job_title"] for r in rows})
        rep["linked_skills"] = [n for n in skill_names if n in group["scores"]]
        rep["link_score"] = round(sum(group["scores"].values()) / len(group["scores"]), 4)
        candidates.append(rep)
    candidates.sort(key=lambda c: (len(c["linked_skills"]), c["link_score"]), reverse=True)
    return candidates


async def _match_adjacent_transition(profile: SkillProfile, exclude_job: str | None) -> list[JobMatch]:
    """인접 직무 전환형 — 역량마다 유사 공고를 찾아 여러 역량이 이어지는 직무부터 판정한다."""
    skill_names = [s.name for s in profile.skills]
    try:
        per_skill = await embedding.search_jobs_per_skill(
            _skill_lines(profile), _skill_search_text(profile), TITLE_TOP_POSTINGS
        )
    except Exception as e:
        raise upstream_error("직무 검색 실패", e) from e

    candidates = _aggregate_skill_hits(skill_names, per_skill)
    candidates = _drop_manual_downshift(await _drop_same_jobs(candidates, exclude_job), exclude_job)
    if not candidates:
        return []
    candidates = candidates[:SELECT_K]
    logger.info(
        "인접 후보(연결 역량 수·점수): %s",
        ", ".join(f"{c['job_title']} {len(c['linked_skills'])}·{c['link_score']:.3f}" for c in candidates),
    )

    # 대표 공고 한 건만 보면 그 회사만의 요구('영문 이메일', '운전면허')가 부족 역량이 된다.
    # 같은 직무명 공고를 DB에서 몇 건 더 붙여, 판정이 직무 공통 요구를 가려내게 한다.
    try:
        samples = await embedding.run_db(
            embedding.sample_postings_by_title,
            [t for c in candidates for t in c["group_titles"]],
            SAMPLE_PER_TITLE,
        )
    except Exception:
        samples = {}
    for rep in candidates:
        # 카드에 보일 근거는 대표 공고 원문 발췌 — 판정용 발췌(여러 공고)는 따로 만든다
        rep["snippet_raw"] = rep.get("snippet") or ""
        extra = [s for t in rep["group_titles"] for s in samples.get(t, [])]
        snippets = [s for s in dict.fromkeys([rep["snippet_raw"], *extra]) if s.strip()]
        rep["snippet"] = "\n".join(f"- {s}" for s in snippets)[: llm.JUDGE_SNIPPET_CHARS]

    # 연결 역량이 많은 후보부터 TOP_K 건 단위로 직무마다 판정하고, 인접 판정을 통과한 것만 남긴다.
    matches: list[JobMatch] = []
    # 이어지는 역량이 1개뿐인 후보 — 엄격 기준 결과가 너무 적을 때만 '근거 약함'으로 채운다
    fallback: list[tuple[dict, dict]] = []
    for start in range(0, len(candidates), TOP_K):
        if len(matches) == TOP_K:
            break
        batch = candidates[start : start + TOP_K]
        try:
            judgements = await _judge_batch(skill_names, exclude_job, batch)
        except Exception as e:
            # LLM 자체가 죽은 경우(키 오류·Ollama 미기동 등) '결과 0건'으로 숨기지 않는다
            if matches:
                break
            raise upstream_error("직무 판정 실패", e) from e
        for row, judged in zip(batch, judgements):
            if len(matches) == TOP_K:
                break
            if not _is_adjacent_match(judged, len(skill_names)):
                if not judged.get("same_as_current") and judged["matched_skills"]:
                    fallback.append((row, judged))
                logger.info(
                    "인접 후보 탈락 %s: same=%s linked=%d matched=%d missing=%d",
                    row["job_title"], judged.get("same_as_current"), len(row["linked_skills"]),
                    len(judged["matched_skills"]), len(judged["missing_skills"]),
                )
                continue
            matches.append(_adjacent_job_match(row, judged))

    # 전문 기술 직무(웹 퍼블리셔 등)는 LLM이 공고마다 이어지는 역량을 1개만 고르는 일이 많아
    # 결과가 0건이 됐다. 빈 화면보다, 이유를 밝힌 약한 후보를 보여주고 사용자가 고르게 한다.
    for row, judged in fallback[: max(0, ADJACENT_MIN_RESULTS - len(matches))]:
        match = _adjacent_job_match(row, judged)
        match.weak_reasons.insert(0, "보유 역량 중 이 직무 업무와 이어지는 것이 1개뿐입니다.")
        matches.append(match)
    return await _with_ai_exposure(matches)


async def _judge_batch(skill_names: list[str], exclude_job: str | None, batch: list[dict]) -> list[dict]:
    """후보를 직무 하나씩 판정한다. 이어지는 역량의 선택지는 역량별 검색에서 그 직무에 걸린 역량뿐이다.

    묶음 판정(공고 TOP_K 건을 한 번에)은 4b가 판정 내용을 한 칸씩 다음 직무 것으로 밀려 써, 부족 역량과
    영문 직업명이 다른 직무 카드에 붙었다 — 직무 번호(job_no)를 받아도 번호만 순서대로 적고 내용은 밀렸다
    (e2e 실측). 하나씩 부르면 프롬프트가 짧아져 묶음과 걸리는 시간도 비슷했다(로컬 4b, 5명 기준 85분 → 62분)."""
    judgements: list[dict] = []
    for row in batch:
        linked = row.get("linked_skills") or skill_names
        judgements += await llm.judge_job_matches(linked, exclude_job, [row], held_skills=skill_names)
    return judgements


async def _with_ai_exposure(matches: list[JobMatch]) -> list[JobMatch]:
    """카드마다 공개 직업 AI 노출도(ILO·Anthropic)를 붙인다 — 출처별 0~100점의 평균과 매칭 직업명·출처, 판정 없음.
    직무명(한국어)으로 찾는다(automation.exposures_for_titles). 매칭이 없거나 실패해도 카드는 그대로 낸다(참고 정보)."""
    if not matches:
        return matches
    try:
        exposures = await automation.exposures_for_titles([m.job_title for m in matches])
    except Exception as e:
        logger.warning("직무별 AI 노출도 조회 실패: %r", e)
        return matches
    for match, (scores, sources) in zip(matches, exposures):
        match.ai_exposure_score = round(sum(scores) / len(scores), 1) if scores else None
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
        license_required=judged.get("license_required", False),
    )


def _skill_lines(profile: SkillProfile) -> list[str]:
    """역량마다 한 줄 — _skill_search_text 와 같은 형식, 역량 순서와 1:1."""
    return [_skill_line(s) for s in profile.skills]


async def _match_training_transition(profile: SkillProfile, exclude_job: str | None) -> list[JobMatch]:
    """교육 후 직무 전환형 — 훈련과정이 있는 NCS 직무 후보 중 LLM이 밑바탕 역량이 이어지는 것을 고른다."""
    profile_lines = _skill_lines(profile)
    skill_names = [s.name for s in profile.skills]
    try:
        candidates = await embedding.run_db(ncs_jobs.find_candidates, profile_lines, skill_names, exclude_job)
    except Exception as e:
        raise upstream_error("NCS 직무 후보 검색 실패", e) from e
    if not candidates:
        return []
    try:
        targets = await llm.suggest_training_targets(candidates, profile_lines, exclude_job, skill_names)
    except Exception as e:
        raise upstream_error("전환 직무 제안 실패", e) from e

    # 현재 직무와 이름이 사실상 같은 NCS 직무는 제외 (이 트랙은 '새 분야'가 핵심)
    targets = await _drop_same_jobs(_dedupe_targets(targets), exclude_job)
    if not targets:
        return []

    # 실채용 수요 확인 — 대표 공고 링크용. 배울 곳(훈련과정)은 후보 단계에서 이미 보장된다
    found = await asyncio.gather(
        *(embedding.search_similar_jobs(t["job_title"], limit=3) for t in targets), return_exceptions=True
    )
    postings = [r if not isinstance(r, Exception) else [] for r in found]

    # 적합도: 사용자 역량 프로필과 '직무 + 요구역량' 문장의 벡터 유사도 — 인접 트랙(프로필 ↔ 공고)과 같은 척도.
    # 이전에는 '이어지는 역량 수 / 요구역량 수'였는데, 모델이 거의 항상 2개·3개를 내 모든 카드가 40.0이 됐다.
    try:
        vecs = await embedding.embed_queries(
            [_skill_search_text(profile)]
            + [
                f"{t['job_title']}\n요구 역량: {', '.join(t['transferable_skills'] + t['training_needs'])}"
                for t in targets
            ]
        )
        fits = [round(sum(a * b for a, b in zip(vecs[0], v)) * 100, 1) for v in vecs[1:]]
    except Exception:
        fits = [None] * len(targets)
    ranked = sorted(zip(targets, postings, fits), key=lambda x: x[2] or 0, reverse=True)

    matches: list[JobMatch] = []
    for target, jobs_found, fit in ranked[:TOP_K]:
        # 실채용 수요가 확인되면 대표 공고를 첨부(링크). 미달이면 링크 없이 직무만 노출.
        posting = jobs_found[0] if jobs_found and jobs_found[0]["similarity"] >= JOB_DEMAND_THRESHOLD else None

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
                # 이 트랙의 핵심 근거: 어떤 NCS 직무인지 + 왜 이 사람의 밑바탕 역량이 이 직무로 전이되는지
                requirement_excerpt=f"NCS {target['ncs_path']} > {target['job_title']}. {target['rationale']}",
                fit_score=fit,
                demand_outlook=target["demand_outlook"],
                transition_difficulty=target["transition_difficulty"],
                matched_skills=transferable,
                required_skills=required,
                missing_skills=training,
                weak_reasons=weak,
                license_required=target.get("license_required", False),
            )
        )
    return await _with_ai_exposure(matches)


@router.post("/match", response_model=list[JobMatch], summary="역량 프로필 → 다음 직무 매핑")
async def match_jobs(
    profile: SkillProfile,
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
        return await _match_training_transition(profile, exclude_job)
    return await _match_adjacent_transition(profile, exclude_job)
