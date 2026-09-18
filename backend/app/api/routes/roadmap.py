"""STEP 4. 학습 로드맵 — 역량 격차 분석 + RAG 기반 맞춤 로드맵 생성.

흐름: 목표 직무 요구역량 - 보유 역량 = 격차 → 격차를 query 모델로 임베딩
      → training_courses(HRD-Net 적재분) 유사도 검색 → 검색 결과만 근거로 LLM이 로드맵 생성

검색어는 '보유 역량'이 아니라 '부족한 역량'이어야 한다 — 이미 잘하는 것과 비슷한
과정만 추천되는 것을 막기 위함. 각 항목의 source 에는 실제 NCS 코드/과정명이 들어간다.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.career import LearningRoadmap, RoadmapItem, SkillProfile
from app.services import embedding, llm

router = APIRouter()

COURSE_TOP_K = 8


class RoadmapRequest(BaseModel):
    profile: SkillProfile
    target_job: str


@router.post("", response_model=LearningRoadmap, summary="목표 직무 → 맞춤 학습 로드맵 생성")
async def generate_roadmap(req: RoadmapRequest, db: Session = Depends(get_db)) -> LearningRoadmap:
    skill_names = [s.name for s in req.profile.skills]

    try:
        gaps = await llm.extract_skill_gaps(skill_names, req.target_job)
        if not gaps:
            return LearningRoadmap(target_job=req.target_job, items=[])

        courses = await embedding.search_similar_courses(db, ", ".join(gaps), limit=COURSE_TOP_K)
        items = await llm.build_roadmap_items(gaps, req.target_job, courses)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"로드맵 생성 실패: {e}") from e

    return LearningRoadmap(target_job=req.target_job, items=[RoadmapItem(**i) for i in items])
