"""근거 기반 자동화 위험도 계산 서비스 (STEP 1. 진단 — /api/diagnosis/risk).

LLM은 경력 서사를 업무로 구조화하는 역할만 맡는다. 최종 점수는 업무 비중
가중치와 적재된 공개 직업 노출도 자료를 이용한 고정 산식으로 계산한다.
NCS에는 자동화 점수가 없으므로 과업 명칭을 표준화하는 근거로만 사용한다.
"""
from __future__ import annotations

import bisect
import hashlib
import logging
import re
import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import AutomationOccupationScore, NcsUnit
from app.schemas.career import AutomationTask, RiskDiagnosis, RiskSource
from app.services import embedding, llm

logger = logging.getLogger(__name__)

OECD_URL = (
    "https://www.oecd.org/content/dam/oecd/en/publications/reports/2026/05/"
    "the-oecd-ai-exposure-measure_489cfd42/f3da0f0a-en.pdf"
)
KEIS_URL = (
    "https://www.keis.or.kr/keis/ko/proj/113/pblc/"
    "detail.do?categoryIdx=131&pubIdx=11169"
)
NCS_URL = "https://www.data.go.kr/data/15063879/openapi.do"

EXTERNAL_WEIGHT = 0.25
TASK_WEIGHT = 1 - EXTERNAL_WEIGHT
def _round(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


# 업무별 주효과는 두 점수로 정한다 — LLM 라벨은 점수와 어긋났다(자동화 점수 3.5인데 'automation').
AUTOMATION_EFFECT_MIN = 60
AUGMENTATION_EFFECT_MIN = 50


def task_effect(automation_score: float, ai_assistance_score: float) -> str:
    if automation_score >= AUTOMATION_EFFECT_MIN:
        return "automation"
    if ai_assistance_score >= AUGMENTATION_EFFECT_MIN:
        return "augmentation"
    return "human"


def normalize_task_scale(tasks: list) -> float:
    """업무 점수의 척도 배율. 모든 점수가 10 이하이면 모델이 10점 만점으로 답한 것으로 보고 10을 곱하고,
    1 이하이면 0~1 비율로 답한 것으로 보고 100을 곱한다.

    100점 척도에서 모든 업무가 10점 이하일 수도 있지만(사람 중심 직무), 9b 실측에서는 '자동화' 라벨을 단
    업무까지 3.5·6.5점이 나와 10점 척도 오답이 훨씬 흔했다. 4b는 0.4·0.8 처럼 비율로도 답해
    은행 텔러 진단이 0.4점이 됐다. 이 보정은 응답 원문 대신 로그로 남긴다."""
    values = [float(v) for t in tasks for v in (t.automation_score, t.ai_assistance_score)]
    if values and 0 < max(values) <= 1:
        logger.warning("업무 점수가 0~1 비율로 보임 — 100점 척도로 환산 (최댓값 %.2f)", max(values))
        return 100.0
    if values and max(values) <= 10 and max(values) > 1:
        logger.warning("업무 점수가 10점 척도로 보임 — 100점 척도로 환산 (최댓값 %.1f)", max(values))
        return 10.0
    return 1.0


def calculate_risk(
    raw_tasks: list,
    empirical_scores: list[float],
) -> dict[str, float | str]:
    """구조화된 업무와 외부 노출도로 재현 가능한 최종 값을 계산한다."""
    total_share = sum(float(task.share_percent) for task in raw_tasks)
    if total_share <= 0:
        raise ValueError("업무 비중 합계는 0보다 커야 합니다.")

    task_score = 0.0
    effect_shares: defaultdict[str, float] = defaultdict(float)
    for task in raw_tasks:
        normalized_share = float(task.share_percent) / total_share
        task_score += normalized_share * float(task.automation_score)
        effect_shares[task.effect] += normalized_share * 100

    if empirical_scores:
        empirical_score = sum(empirical_scores) / len(empirical_scores)
        final_score = TASK_WEIGHT * task_score + EXTERNAL_WEIGHT * empirical_score
    else:
        empirical_score = task_score
        final_score = task_score

    # 낮음/보통/높음 같은 판정 등급은 내지 않는다 — 점수와 산출 근거만 보여주고 해석은 사용자가 한다
    return {
        "risk_score": _round(final_score),
        "task_based_score": _round(task_score),
        "automation_share": _round(effect_shares["automation"]),
        "augmentation_share": _round(effect_shares["augmentation"]),
        "human_centered_share": _round(effect_shares["human"]),
        "empirical_score": _round(empirical_score),
    }


def _load_occupation_rows(db: Session) -> list[AutomationOccupationScore]:
    try:
        return list(db.scalars(select(AutomationOccupationScore)).all())
    except SQLAlchemyError:
        db.rollback()
        return []


# Anthropic Economic Index 는 Claude 사용 기록에서 관측된 과업 비중이다. 0점(756개 직업 중 411개)은
# '노출도 0'이 아니라 '관측 없음'이라 점수로 쓰지 않는다 — 평균에 넣으면 0.0 표시·반토막 점수가 됐다(e2e 3차)
AEI_SOURCE = "Anthropic Economic Index"


def _has_observation(row) -> bool:
    return not (row.source == AEI_SOURCE and float(row.score) <= 0)


def exposure_percentiles(rows: list) -> dict[tuple[str, str], float]:
    """출처별 원점수 → 직업 간 상대 순위 0~100 (100에 가까울수록 AI로 대체될 가능성이 높은 직업).

    출처마다 척도가 달라(ILO 8~69 잠재 노출도, Anthropic 0~74.5 관측 비중) 원점수를 그대로 평균하면
    의미가 없다. 같은 출처 안의 순위(동점은 평균 순위)로 바꿔야 서로 섞을 수 있다. 관측 없는 행은 뺀다."""
    by_source: defaultdict[str, list] = defaultdict(list)
    for row in rows:
        if _has_observation(row):
            by_source[row.source].append(row)
    out: dict[tuple[str, str], float] = {}
    for source_rows in by_source.values():
        scores = sorted(float(r.score) for r in source_rows)
        n = len(scores)
        for row in source_rows:
            if n == 1:
                out[(row.source, row.occupation_code)] = 50.0
                continue
            s = float(row.score)
            less = bisect.bisect_left(scores, s)
            equal = bisect.bisect_right(scores, s) - less
            out[(row.source, row.occupation_code)] = _round((less + (equal - 1) / 2) / (n - 1) * 100)
    return out


# 직업 매칭 후보 — 출처마다 한국어 직무명과 임베딩이 가까운 직업 이 수만큼을 LLM에 보여 준다
OCCUPATION_SHORTLIST = 5
# 노출도 직업명 벡터 디스크 캐시 (backend/.cache — .gitignore 대상)
OCCUPATION_INDEX_DIR = Path(__file__).resolve().parents[2] / ".cache"

_occupation_index = None  # (rows, np.ndarray) — 프로세스당 한 번 만든다
_occupation_lock = threading.Lock()


def _occupation_rows_index(db: Session):
    """노출도 자료 직업명(약 3,300개) 벡터 — 한국어 직무명과 bge-m3 다국어 임베딩으로 비교한다."""
    global _occupation_index
    if _occupation_index is not None:
        return _occupation_index
    with _occupation_lock:
        if _occupation_index is not None:
            return _occupation_index
        # 순서를 고정해야 아래 캐시 키가 매번 같다
        rows = sorted(_load_occupation_rows(db), key=lambda r: (r.source, r.occupation_code))
        if not rows:
            return None
        texts = [r.occupation_title for r in rows]
        key = hashlib.sha256(
            "\n".join([embedding.EMBEDDING_MODEL, embedding.EMBEDDING_REVISION, *texts]).encode("utf-8")
        ).hexdigest()[:16]
        cache = OCCUPATION_INDEX_DIR / f"occupation_index_{key}.npy"
        if cache.exists():
            _occupation_index = (rows, np.load(cache))
            return _occupation_index
        logger.info("노출도 직업명 색인 생성 — %d건 (첫 실행만)", len(texts))
        arr = np.asarray(embedding.embed_passages(texts), dtype=np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        except OSError as e:
            logger.warning("노출도 직업명 색인 저장 실패: %r", e)
        _occupation_index = (rows, arr)
        return _occupation_index


def warm_occupation_index() -> None:
    """서버 시작 시 백그라운드로 색인을 준비한다 — 첫 요청이 색인 생성을 기다리지 않게."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        _occupation_rows_index(db)
    except Exception as e:
        logger.warning("노출도 직업명 색인 사전 준비 실패: %r", e)
    finally:
        db.close()


def occupation_shortlists(db: Session, titles: list[str]) -> tuple[list[dict[str, list]], dict]:
    """직무명마다 출처별 후보 직업 [(행, 유사도)] OCCUPATION_SHORTLIST 개 + 출처별 0~100 순위표."""
    index = _occupation_rows_index(db)
    if index is None or not titles:
        return [{} for _ in titles], {}
    rows, vecs = index
    q = np.asarray(embedding.embed_passages([t or "" for t in titles]), dtype=np.float32)
    sims = q @ vecs.T
    by_source: defaultdict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[r.source].append(i)
    out: list[dict[str, list]] = []
    for t, row_sims in zip(titles, sims):
        if not (t or "").strip():
            out.append({})
            continue
        out.append({
            src: [(rows[i], float(row_sims[i])) for i in sorted(idx, key=lambda i: -row_sims[i])[:OCCUPATION_SHORTLIST]]
            for src, idx in by_source.items()
        })
    return out, exposure_percentiles(rows)


async def exposures_for_titles(titles: list[str]) -> list[tuple[list[float], list[RiskSource]]]:
    """한국어 직무명마다 출처별 AI 노출도 — (출처별 0~100 점수, 출처 근거). 매칭이 없으면 ([], []).

    임베딩으로 출처별 후보를 추리고(occupation_shortlists), LLM이 같은 직업을 고른다(llm.match_occupations).
    LLM이 영문 직업명을 지어 내고 단어 규칙으로 맞추던 방식은 물류 직무에서 8장 모두 매칭에 실패했다."""
    if not titles:
        return []
    shortlists, percentiles = await embedding.run_db(occupation_shortlists, titles)
    choices = await llm.match_occupations(
        [(t, {src: [r.occupation_title for r, _ in cands] for src, cands in sl.items()}) for t, sl in zip(titles, shortlists)]
    )
    out: list[tuple[list[float], list[RiskSource]]] = []
    for title, sl, chosen in zip(titles, shortlists, choices):
        scores: list[float] = []
        sources: list[RiskSource] = []
        for src, cands in sl.items():
            no = chosen.get(src, -1)
            if no < 0 or no >= len(cands):
                continue
            row, sim = cands[no]
            # 고른 직업이 '관측 없음'이면 다른 후보로 넘어가지 않는다 — 다른 직업 점수가 붙는다
            pct = percentiles.get((row.source, row.occupation_code))
            if pct is None:
                logger.info("노출도 관측 없음 — 제외: %s %s", src, row.occupation_title)
                continue
            # 화면(진단·직무 카드)에는 환산한 0~100점만 낸다 — 원점수는 척도가 출처마다 달라 섞이면 헷갈린다. 로그에만 남긴다
            logger.debug("노출도 %s %s 원점수 %g → %s", src, row.occupation_title, float(row.score), pct)
            scores.append(pct)
            sources.append(
                RiskSource(
                    source=src,
                    label=f"{row.occupation_title} ({row.occupation_code})",
                    score=pct,
                    url=row.source_url,
                    note=(
                        "직업 간 순위로 환산한 0~100점 — 100에 가까울수록 AI로 대체될 가능성이 높은 직업. "
                        f"직업명은 임베딩 후보(유사도 {sim * 100:.0f}%) 중 AI가 고른 것입니다."
                    ),
                )
            )
        logger.info(
            "직무 노출도 %r → %s", title,
            ", ".join(f"{s.source}: {s.label} {s.score}" for s in sources) or "매칭 없음",
        )
        out.append((scores, sources))
    return out


# 이름이 비슷해도 틀린 NCS 라벨은 없는 것보다 해롭다 — 이 유사도 이상만 붙인다
NCS_MATCH_MIN = 0.66
# NCS 능력단위 벡터 디스크 캐시 (backend/.cache — .gitignore 대상)
NCS_INDEX_DIR = Path(__file__).resolve().parents[2] / ".cache"

_ncs_index = None  # (units, np.ndarray) — 프로세스당 한 번 만든다
_ncs_lock = threading.Lock()


def _ncs_unit_index(db: Session):
    """전체 NCS 능력단위('단위명 (세분류)') 벡터 — 프로세스당 한 번 만들어 메모리에 둔다(약 50MB).

    처음 한 번은 로컬 CPU로 1만여 건을 임베딩하느라 수십 분 걸린다 — 서버 시작 시 warm_ncs_index 가
    백그라운드로 만들고 디스크에 저장해, 다음부터는 파일을 읽는다. 만드는 동안 들어온 진단은 잠금을
    기다리지 않고 NCS 매칭 없이 진행한다(_match_ncs 참고)."""
    global _ncs_index
    if _ncs_index is not None:
        return _ncs_index
    with _ncs_lock:
        if _ncs_index is not None:
            return _ncs_index
        try:
            # 순서를 고정해야 아래 캐시 키가 매번 같다 — ORDER BY 가 없으면 행 순서가 바뀌어 색인을 다시 만들었다
            units = [
                u for u in db.scalars(select(NcsUnit).order_by(NcsUnit.ncs_code)).all()
                if "구버전" not in u.unit_name and "삭제" not in u.unit_name
            ]
        except SQLAlchemyError:
            db.rollback()
            return None
        texts = [f"{u.unit_name} ({u.sub_category})" for u in units]
        # 1만여 건 임베딩은 CPU로 수십 분 — 디스크에 저장해 두고 단위 목록·모델이 같으면 다시 쓴다
        key = hashlib.sha256(
            "\n".join([embedding.EMBEDDING_MODEL, embedding.EMBEDDING_REVISION, *texts]).encode("utf-8")
        ).hexdigest()[:16]
        cache = NCS_INDEX_DIR / f"ncs_index_{key}.npy"
        if cache.exists():
            _ncs_index = (units, np.load(cache))
            return _ncs_index
        logger.info("NCS 색인 생성 시작 — 능력단위 %d건 로컬 임베딩", len(texts))
        arr = np.asarray(embedding._encode(texts), dtype=np.float32)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        except OSError as e:
            logger.warning("NCS 색인 저장 실패: %r", e)
        _ncs_index = (units, arr)
        return _ncs_index


def warm_ncs_index() -> None:
    """서버 시작 시 백그라운드로 NCS 색인을 준비한다 — 첫 진단 요청이 색인 생성을 기다리지 않게."""
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        _ncs_unit_index(db)
    except Exception as e:
        logger.warning("NCS 색인 사전 준비 실패: %r", e)
    finally:
        db.close()


def _match_ncs(
    db: Session,
    tasks: list,
    job_title: str | None = None,
) -> tuple[dict[int, NcsUnit], RiskSource | None]:
    """업무명마다 가장 가까운 NCS 능력단위를 임베딩 유사도로 찾는다.

    단어 일치 규칙은 '관리' 같은 흔한 말로 '접수 및 예약 관리' → '항공기 운항 스케줄 관리'를 냈고,
    규칙을 조이면 맞는 것까지 다 놓쳤다. 단어 검색으로 후보를 300건 추리면 맞는 단위가 빠지기도 해서
    전체 단위와 비교한다. 실측(bge-m3): 맞음 — 예약관리(병원안내) 0.715, CNC선반 가공 프로그래밍 0.700,
    고객상담 0.724 / 틀림 — HTML 마크업↔편집 레이아웃 0.653, 신입 교육↔청소년활동 협업 0.606.

    업무명만으로 고르면 글자가 비슷한 다른 분야 단위가 기준을 넘었다(은행 텔러 '정산 검증 및 사고 예방 점검'
    → '정화 검증(토양관리)' 0.700, '고객 입출금 및 송금 처리' → '외환 결제' 0.704). 그래서 고르기는
    '직무명: 업무명'으로 해서 분야를 맞추고(→ '입지급 거래(창구사무)'), 붙일지는 업무명만의 유사도로 정한다
    — 직무명을 붙이면 유사도가 전반적으로 낮아져 같은 기준을 쓸 수 없다."""
    # 색인은 warm_ncs_index 만 만든다 — 진단 요청이 수십 분짜리 첫 생성을 기다리면 안 된다.
    # 아직 없고 만드는 중도 아니면(시작 시 DB가 꺼져 있었던 경우 등) 백그라운드로 다시 시작만 한다
    index = _ncs_index
    if index is None and not _ncs_lock.locked():
        threading.Thread(target=warm_ncs_index, daemon=True).start()
    if index is None or not tasks:
        return {}, None
    units, unit_vecs = index
    names = [t.name for t in tasks]
    job = (job_title or "").strip()
    try:
        vecs = np.asarray(embedding._encode(names + ([f"{job}: {n}" for n in names] if job else [])),
                          dtype=np.float32)
    except Exception as e:  # NCS 는 표기용 근거 — 실패하면 붙이지 않는다
        logger.warning("NCS 매칭 임베딩 실패: %r", e)
        return {}, None
    sims = vecs[: len(names)] @ unit_vecs.T  # 정규화된 벡터라 내적 = 코사인
    pick_sims = vecs[len(names):] @ unit_vecs.T if job else sims
    matches: dict[int, NcsUnit] = {}
    for i, (row, pick_row) in enumerate(zip(sims, pick_sims, strict=True)):
        best = int(pick_row.argmax())
        if row[best] >= NCS_MATCH_MIN:
            matches[i] = units[best]

    if not matches:
        return {}, None
    unique_count = len({unit.ncs_code for unit in matches.values()})
    return matches, RiskSource(
        source="한국산업인력공단 NCS",
        label=f"NCS 능력단위 {unique_count}건 매칭",
        url=NCS_URL,
        note="NCS는 과업 표준화 근거이며 자동화 점수를 제공하지 않습니다.",
    )


# 수치가 든 구체적 사실 — '하루 80콜', '전환율 1.4%', '협력사 20곳', '7년'
_QUANT_FACT = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|퍼센트|명|건|곳|콜|년|개월|주|시간|배|원|만|천|억|회|개|대|단어)")


def evidence_confidence(career_text: str, task_count: int, ncs_matched: bool, exposure_sources: int) -> int:
    """근거 충족도(화면 '근거 충족도') — 진단을 받칠 근거가 입력과 자료에 얼마나 있는지.

    이전 산식(45 + 업무 수 + NCS + 노출도)은 로컬에서 NCS 매칭이 꺼져 있어(원격 임베딩 서버 전용)
    세 줄짜리 이력서와 자세한 경력이 같은 53점이 됐다. 입력 분량과 수치가 든 사실을 함께 본다:
      기본 35 + 서사 분량(150자 미만 0 / 400자 미만 8 / 그 이상 15) + 수치 사실 개당 3(최대 9)
      + 업무 수 개당 2(최대 8) + NCS 매칭 10 + 노출도 출처당 10(최대 20), 상한 90.
    """
    length = len(re.sub(r"\s+", " ", career_text).strip())
    score = 35
    score += 15 if length >= 400 else 8 if length >= 150 else 0
    score += min(9, 3 * len(_QUANT_FACT.findall(career_text)))
    score += min(8, 2 * task_count)
    score += 10 if ncs_matched else 0
    score += min(20, 10 * exposure_sources)
    return min(90, score)


async def diagnose(
    career_text: str,
    job_title: str | None = None,
) -> RiskDiagnosis:
    analysis = await llm.analyze_automation_tasks(career_text, job_title)
    scale = normalize_task_scale(analysis.tasks)
    for task in analysis.tasks:
        task.automation_score = min(100.0, float(task.automation_score) * scale)
        task.ai_assistance_score = min(100.0, float(task.ai_assistance_score) * scale)
    # calculate_risk 는 task.effect 를 쓴다 — 점수에서 정한 주효과를 붙여 넘긴다
    scored_tasks = [
        SimpleNamespace(
            share_percent=t.share_percent,
            automation_score=t.automation_score,
            effect=task_effect(t.automation_score, t.ai_assistance_score),
        )
        for t in analysis.tasks
    ]

    # 동기 쿼리는 스레드에서 — 다른 사용자의 요청을 막지 않게 (embedding.run_db 참고)
    # 노출도는 참고 보정값 — 조회가 실패해도 업무 점수만으로 진단한다
    try:
        [(empirical_scores, sources)] = await exposures_for_titles([job_title or analysis.job_title])
    except Exception as e:
        logger.warning("직업 노출도 조회 실패: %r", e)
        empirical_scores, sources = [], []
    ncs_matches, ncs_source = await embedding.run_db(_match_ncs, analysis.tasks, job_title or analysis.job_title)
    if ncs_source:
        sources.append(ncs_source)

    calculation = calculate_risk(scored_tasks, empirical_scores)
    normalized_total = sum(float(task.share_percent) for task in analysis.tasks)
    tasks = []
    for index, task in enumerate(analysis.tasks):
        ncs = ncs_matches.get(index)
        tasks.append(
            AutomationTask(
                name=task.name,
                share_percent=_round(float(task.share_percent) / normalized_total * 100),
                automation_score=_round(task.automation_score),
                ai_assistance_score=_round(task.ai_assistance_score),
                effect=scored_tasks[index].effect,
                rationale=task.rationale,
                ncs_code=ncs.ncs_code if ncs else None,
                ncs_unit=ncs.unit_name if ncs else None,
                source_quote=task.source_quote,
            )
        )

    sources.extend(
        [
            RiskSource(
                source="OECD AI Exposure Measure",
                label="9개 인간 역량과 AI 역량 격차 기반 방법론 (2026)",
                url=OECD_URL,
                note="업무별 판단 기준에 반영한 방법론 출처",
            ),
            RiskSource(
                source="한국고용정보원",
                label="AI에 의한 화이트칼라 직무 대체 및 변화 (2024)",
                url=KEIS_URL,
                note="국내 직무활동·대체 가능성 해석 기준",
            ),
        ]
    )

    confidence = evidence_confidence(career_text, len(tasks), bool(ncs_matches), len(empirical_scores))
    formula_note = (
        f"업무별 점수 {calculation['task_based_score']}점을 중심으로"
        + (
            f", 직업 AI 노출도(직업 간 순위 0~100) {calculation['empirical_score']}점을 25% 반영해 보정했습니다."
            if empirical_scores
            else " 계산했으며 매칭 가능한 직업 노출도는 보정에 넣지 않았습니다."
        )
    )

    # LLM의 서술형 평가('위험한 부분은…')는 싣지 않는다 — 판정은 사용자 몫이고,
    # 업무별 근거는 tasks[].rationale 에 그대로 있다. 여기엔 점수 산출 방식만 적는다.
    return RiskDiagnosis(
        job_title=analysis.job_title,
        rationale=formula_note,
        confidence=confidence,
        tasks=tasks,
        sources=sources,
        **{
            key: value
            for key, value in calculation.items()
            if key != "empirical_score"
        },
    )
