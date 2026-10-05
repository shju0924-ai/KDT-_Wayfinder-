"""STEP 4. 학습 로드맵 — 역량 격차 분석 + RAG 기반 맞춤 로드맵 생성.

흐름: 목표 직무 요구역량 - 보유 역량 = 격차 → 격차를 query 모델로 임베딩
      → training_courses(HRD-Net 적재분) 유사도 검색(격차마다) → 검색 결과만 근거로 LLM이 로드맵 생성

검색어는 '보유 역량'이 아니라 '부족한 역량'이어야 한다 — 이미 잘하는 것과 비슷한
과정만 추천되는 것을 막기 위함. 각 항목의 source 에는 실제 NCS 코드/과정명이 들어간다.
"""
import asyncio

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.career import LearningRoadmap, RoadmapItem, SkillProfile
from app.services import embedding, llm

router = APIRouter()

# 격차 하나당 LLM에 보여줄 후보 과정 수
COURSE_TOP_K = 4
# 격차 문구 vs 훈련과정 유사도 1차 문턱 — 명백히 무관한 과정만 거르고, 관련성 최종 판단은 LLM이 한다.
# 짧은 문구 유사도는 관련·무관이 0.50 전후에 붙어 있어(관련: 시장조사↔사회조사분석사 0.48, BI↔Power BI 0.55 /
# 무관: B2B GTM 0.50, CRM 정합성 0.52, 심해 잠수 0.45) 문턱 하나로는 가를 수 없다.
# 검색어에 목표 직무를 붙이면 무관한 쪽까지 유사도가 오르고(심해 잠수 0.45→0.57) 직무 쪽 과정으로 쏠려 쓰지 않는다.
GAP_COURSE_THRESHOLD = 0.48


class RoadmapRequest(BaseModel):
    profile: SkillProfile
    target_job: str
    # STEP 3 카드에 이미 표시된 '새로 배워야 할 역량'. 있으면 격차 추출 LLM 호출을 건너뛴다 —
    # 로컬 모델에서 호출 하나(수 분)를 줄이고, 사용자가 본 격차와 로드맵의 격차가 일치하게 된다.
    missing_skills: list[str] = Field(default_factory=list)


@router.post("", response_model=LearningRoadmap, summary="목표 직무 → 맞춤 학습 로드맵 생성")
async def generate_roadmap(req: RoadmapRequest, db: Session = Depends(get_db)) -> LearningRoadmap:
    skill_names = [s.name for s in req.profile.skills]
    given_gaps = [g.strip() for g in req.missing_skills if g.strip() and g.strip() not in skill_names][:4]

    try:
        gaps = given_gaps or await llm.extract_skill_gaps(skill_names, req.target_job)
        if not gaps:
            return LearningRoadmap(target_job=req.target_job, items=[])

        # 격차를 합쳐 한 번에 검색하면 한 격차 쪽 과정만 올라온다 — 격차마다 따로 찾는다
        found = await asyncio.gather(
            *(embedding.search_similar_courses(db, gap, limit=COURSE_TOP_K) for gap in gaps)
        )
        courses_by_gap = {
            gap: [c for c in courses if c["similarity"] >= GAP_COURSE_THRESHOLD]
            for gap, courses in zip(gaps, found)
        }
        items = await llm.build_roadmap_items(gaps, req.target_job, courses_by_gap)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"로드맵 생성 실패: {e!r}") from e

    return LearningRoadmap(target_job=req.target_job, items=[RoadmapItem(**i) for i in items])
