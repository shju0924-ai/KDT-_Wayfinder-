"""STEP 2. 역량 프로필 생성 — LLM 기반 역량 분해 + 사용자 검토·보완.

확정(confirmed) 처리는 프론트엔드에서 사용자가 수행하며, STEP 3 요청 시 확정본이 전달된다.
서버는 초안만 생성한다 — 판정 주체는 AI가 아니라 사용자다.
"""
from fastapi import APIRouter, HTTPException

from app.schemas.career import CareerInput, SkillProfile
from app.services import llm

router = APIRouter()


@router.post("", response_model=SkillProfile, summary="경력 서사 → 전이 가능 역량 분해")
async def generate_profile(career: CareerInput) -> SkillProfile:
    try:
        skills = await llm.decompose_skills(career.raw_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"역량 분해 실패: {e}") from e
    return SkillProfile(skills=skills)
