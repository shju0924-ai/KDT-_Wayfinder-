"""근거 기반 자동화 위험도 계산 서비스.

LLM은 경력 서사를 업무로 구조화하는 역할만 맡는다. 최종 점수는 업무 비중
가중치와 적재된 공개 직업 노출도 자료를 이용한 고정 산식으로 계산한다.
NCS에는 자동화 점수가 없으므로 과업 명칭을 표준화하는 근거로만 사용한다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import AutomationOccupationScore, NcsUnit
from app.schemas.career import AutomationTask, RiskDiagnosis, RiskSource
from app.services import llm

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
MATCH_THRESHOLD = 0.50


def _round(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _title_similarity(left: str, right: str) -> float:
    def normalize(value: str) -> tuple[str, set[str]]:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", value.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, {token for token in cleaned.split() if len(token) > 1}

    a, a_tokens = normalize(left)
    b, b_tokens = normalize(right)
    sequence = SequenceMatcher(None, a, b).ratio()
    if not a_tokens or not b_tokens:
        return sequence
    coverage = len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))
    return max(sequence, coverage * 0.9)


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

    final_score = _round(final_score)
    if final_score < 35:
        level = "낮음"
    elif final_score < 65:
        level = "보통"
    else:
        level = "높음"

    return {
        "risk_score": final_score,
        "risk_level": level,
        "task_based_score": _round(task_score),
        "automation_share": _round(effect_shares["automation"]),
        "augmentation_share": _round(effect_shares["augmentation"]),
        "human_centered_share": _round(effect_shares["human"]),
        "empirical_score": _round(empirical_score),
    }


def _find_occupation_signals(
    db: Session,
    occupation_title_en: str,
) -> tuple[list[float], list[RiskSource]]:
    try:
        rows = db.scalars(select(AutomationOccupationScore)).all()
    except SQLAlchemyError:
        db.rollback()
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


def _clean_search_terms(terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    for term in terms:
        value = re.sub(r"[^\w가-힣· ]", " ", term).strip()
        if len(value) >= 2 and value not in cleaned:
            cleaned.append(value)
    return cleaned[:8]


def _match_ncs(
    db: Session,
    tasks: list,
    search_terms: list[str],
) -> tuple[dict[int, NcsUnit], RiskSource | None]:
    terms = _clean_search_terms(search_terms + [task.name for task in tasks])
    if not terms:
        return {}, None

    conditions = []
    for term in terms:
        pattern = f"%{term}%"
        conditions.extend(
            [
                NcsUnit.unit_name.ilike(pattern),
                NcsUnit.sub_category.ilike(pattern),
                NcsUnit.unit_definition.ilike(pattern),
            ]
        )
    try:
        candidates = db.scalars(
            select(NcsUnit).where(or_(*conditions)).limit(300)
        ).all()
    except SQLAlchemyError:
        db.rollback()
        return {}, None

    matches: dict[int, NcsUnit] = {}
    for index, task in enumerate(tasks):
        best: tuple[float, NcsUnit] | None = None
        for candidate in candidates:
            candidate_text = " ".join(
                part
                for part in [
                    candidate.unit_name,
                    candidate.sub_category,
                    candidate.unit_definition,
                ]
                if part
            )
            score = SequenceMatcher(None, task.name, candidate.unit_name).ratio()
            task_tokens = {token for token in task.name.split() if len(token) >= 2}
            if task_tokens:
                hits = sum(token in candidate_text for token in task_tokens)
                score = max(score, hits / len(task_tokens))
            if best is None or score > best[0]:
                best = (score, candidate)
        if best and best[0] >= 0.42:
            matches[index] = best[1]

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

    empirical_scores, sources = _find_occupation_signals(
        db, analysis.occupation_title_en
    )
    ncs_matches, ncs_source = _match_ncs(
        db, analysis.tasks, analysis.ncs_search_terms
    )
    if ncs_source:
        sources.append(ncs_source)

    calculation = calculate_risk(analysis.tasks, empirical_scores)
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
                effect=task.effect,
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

    return RiskDiagnosis(
        job_title=analysis.job_title,
        rationale=f"{analysis.rationale} {formula_note}",
        confidence=min(90, confidence),
        tasks=tasks,
        sources=sources,
        **{
            key: value
            for key, value in calculation.items()
            if key != "empirical_score"
        },
    )
