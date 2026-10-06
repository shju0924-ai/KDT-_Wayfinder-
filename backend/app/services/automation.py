"""근거 기반 자동화 위험도 계산 서비스 (STEP 1. 진단 — /api/diagnosis/risk).

LLM은 경력 서사를 업무로 구조화하는 역할만 맡는다. 최종 점수는 업무 비중
가중치와 적재된 공개 직업 노출도 자료를 이용한 고정 산식으로 계산한다.
NCS에는 자동화 점수가 없으므로 과업 명칭을 표준화하는 근거로만 사용한다.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path
from collections import defaultdict
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
# 단어 겹침 비율. 'nursing assistant' ↔ 'sewing assistant' = 0.5 는 떨어뜨리고 2단어 중 2단어 일치만 통과
MATCH_THRESHOLD = 0.6


def _round(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _title_similarity(left: str, right: str) -> float:
    """영문 직업명 유사도 — 단어 일치 기준.

    문자열 유사도(SequenceMatcher)를 함께 쓰면 'Nursing Assistant' 가 'Sewing assistant' 와 0.85로
    매칭됐다. 직업명은 핵심 단어가 같아야 같은 직업이므로, 복수형 s 를 뗀 단어 집합의 겹침만 본다."""
    def tokens(value: str) -> set[str]:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
        return {re.sub(r"(?<=[a-z]{3})s$", "", t) for t in cleaned.split() if len(t) > 1 and t not in _TITLE_STOPWORDS}

    a, b = tokens(left), tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), len(b))


_TITLE_STOPWORDS = {"and", "of", "the", "other", "all", "workers", "worker"}


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
    """업무 점수의 척도 배율. 모든 점수가 10 이하이면 모델이 10점 만점으로 답한 것으로 보고 10을 곱한다.

    100점 척도에서 모든 업무가 10점 이하일 수도 있지만(사람 중심 직무), 9b 실측에서는 '자동화' 라벨을 단
    업무까지 3.5·6.5점이 나와 10점 척도 오답이 훨씬 흔했다. 이 보정은 응답 원문 대신 로그로 남긴다."""
    values = [float(v) for t in tasks for v in (t.automation_score, t.ai_assistance_score)]
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


def _find_occupation_signals(
    db: Session,
    occupation_title_en: str,
    rows: list[AutomationOccupationScore] | None = None,
) -> tuple[list[float], list[RiskSource]]:
    if rows is None:
        rows = _load_occupation_rows(db)
    if not rows or not occupation_title_en.strip():
        return [], []

    best_by_source: dict[str, tuple[float, AutomationOccupationScore]] = {}
    for row in rows:
        similarity = _title_similarity(occupation_title_en, row.occupation_title)
        current = best_by_source.get(row.source)
        if current is None or similarity > current[0]:
            best_by_source[row.source] = (similarity, row)

    scores: list[float] = []
    sources: list[RiskSource] = []
    for source_name, (similarity, row) in best_by_source.items():
        if similarity < MATCH_THRESHOLD:
            continue
        scores.append(row.score)
        sources.append(
            RiskSource(
                source=source_name,
                label=f"{row.occupation_title} ({row.occupation_code})",
                score=_round(row.score),
                url=row.source_url,
                note=(
                    f"직업명 유사도 {similarity * 100:.0f}%. "
                    "AI 노출도이며 고용 대체 확률이 아닙니다."
                ),
            )
        )
    return scores, sources


def occupation_exposures(
    db: Session, titles_en: list[str | None]
) -> list[tuple[float | None, list[RiskSource]]]:
    """후보 직무(영문 표준 직업명)마다 공개 직업 AI 노출도 — (출처 평균 점수, 출처별 근거).

    STEP 3 카드용. 판정 없이 점수와 매칭된 직업명·출처만 돌려준다. 매칭이 없으면 (None, [])."""
    rows = _load_occupation_rows(db)
    out: list[tuple[float | None, list[RiskSource]]] = []
    for title in titles_en:
        scores, sources = _find_occupation_signals(db, title or "", rows)
        out.append((_round(sum(scores) / len(scores)) if scores else None, sources))
    return out


# 이름이 비슷해도 틀린 NCS 라벨은 없는 것보다 해롭다 — 이 유사도 이상만 붙인다
NCS_MATCH_MIN = 0.66
# NCS 능력단위 벡터 디스크 캐시 (backend/.cache — .gitignore 대상)
NCS_INDEX_DIR = Path(__file__).resolve().parents[2] / ".cache"

_ncs_index = None  # (units, np.ndarray) — 프로세스당 한 번 만든다
_ncs_lock = threading.Lock()


def _ncs_unit_index(db: Session):
    """전체 NCS 능력단위('단위명 (세분류)') 벡터 — 프로세스당 한 번 원격 GPU로 만들어 메모리에 둔다(약 50MB).

    원격 임베딩 서버가 없으면 만들지 않는다 — CPU로 1만여 건은 수십 분이라 진단 응답을 붙잡는다."""
    global _ncs_index
    if _ncs_index is not None:
        return _ncs_index
    with _ncs_lock:
        if _ncs_index is not None:
            return _ncs_index
        if not embedding.settings.embedding_api_url:
            return None
        try:
            # 순서를 고정해야 아래 캐시 키가 매번 같다 — ORDER BY 가 없으면 행 순서가 바뀌어 색인을 다시 만들었다
            units = [
                u for u in db.scalars(select(NcsUnit).order_by(NcsUnit.ncs_code)).all()
                if "구버전" not in u.unit_name
            ]
        except SQLAlchemyError:
            db.rollback()
            return None
        texts = [f"{u.unit_name} ({u.sub_category})" for u in units]
        # 1만여 건 임베딩은 GPU로도 2분 남짓 — 디스크에 저장해 두고 단위 목록·모델이 같으면 다시 쓴다
        key = hashlib.sha256(
            "\n".join([embedding.EMBEDDING_MODEL, embedding.EMBEDDING_REVISION, *texts]).encode("utf-8")
        ).hexdigest()[:16]
        cache = NCS_INDEX_DIR / f"ncs_index_{key}.npy"
        if cache.exists():
            _ncs_index = (units, np.load(cache))
            return _ncs_index
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 500):  # 원격 서버 요청당 상한 512
            part = embedding._encode_remote(texts[i : i + 500])
            if part is None:
                logger.warning("NCS 색인 생성 실패 — 원격 임베딩 서버 응답 없음")
                return None
            vecs.extend(part)
        arr = np.asarray(vecs, dtype=np.float32)
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
) -> tuple[dict[int, NcsUnit], RiskSource | None]:
    """업무명마다 가장 가까운 NCS 능력단위를 임베딩 유사도로 찾는다.

    단어 일치 규칙은 '관리' 같은 흔한 말로 '접수 및 예약 관리' → '항공기 운항 스케줄 관리'를 냈고,
    규칙을 조이면 맞는 것까지 다 놓쳤다. 단어 검색으로 후보를 300건 추리면 맞는 단위가 빠지기도 해서
    전체 단위와 비교한다. 실측(bge-m3): 맞음 — 예약관리(병원안내) 0.715, CNC선반 가공 프로그래밍 0.700,
    고객상담 0.724 / 틀림 — HTML 마크업↔편집 레이아웃 0.653, 신입 교육↔청소년활동 협업 0.606."""
    index = _ncs_unit_index(db)
    if index is None or not tasks:
        return {}, None
    units, unit_vecs = index
    try:
        task_vecs = np.asarray(embedding._encode([t.name for t in tasks]), dtype=np.float32)
    except Exception as e:  # NCS 는 표기용 근거 — 실패하면 붙이지 않는다
        logger.warning("NCS 매칭 임베딩 실패: %r", e)
        return {}, None
    sims = task_vecs @ unit_vecs.T  # 정규화된 벡터라 내적 = 코사인
    matches: dict[int, NcsUnit] = {}
    for i, row in enumerate(sims):
        best = int(row.argmax())
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


async def diagnose(
    db: Session,
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
    empirical_scores, sources = await embedding.run_db(_find_occupation_signals, analysis.occupation_title_en)
    ncs_matches, ncs_source = await embedding.run_db(_match_ncs, analysis.tasks)
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

    confidence = 45 + min(10, len(tasks) * 2)
    confidence += 15 if ncs_matches else 0
    confidence += min(20, len(empirical_scores) * 10)
    formula_note = (
        f"업무별 점수 {calculation['task_based_score']}점을 중심으로"
        + (
            f", 직업 노출도 {calculation['empirical_score']}점을 25% 반영해 보정했습니다."
            if empirical_scores
            else " 계산했으며 매칭 가능한 직업 노출도는 보정에 넣지 않았습니다."
        )
    )

    # LLM의 서술형 평가('위험한 부분은…')는 싣지 않는다 — 판정은 사용자 몫이고,
    # 업무별 근거는 tasks[].rationale 에 그대로 있다. 여기엔 점수 산출 방식만 적는다.
    return RiskDiagnosis(
        job_title=analysis.job_title,
        rationale=formula_note,
        confidence=min(90, confidence),
        tasks=tasks,
        sources=sources,
        **{
            key: value
            for key, value in calculation.items()
            if key != "empirical_score"
        },
    )
