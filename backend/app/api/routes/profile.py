"""STEP 2. 역량 프로필 생성 — 경력 → NCS 능력단위 연결 + 사용자 검토·보완.

역량은 NCS 능력단위(국가직무능력표준)로 표현하고, 능력단위마다 근거가 된 경력 원문 구절을 붙인다.
확정(confirmed) 처리는 프론트엔드에서 사용자가 수행하며, STEP 3 요청 시 확정본이 전달된다.
서버는 초안만 생성한다 — 판정 주체는 AI가 아니라 사용자다. 빠진 능력단위는 /profile/units 로 찾아 추가한다.
"""
import asyncio

from fastapi import APIRouter, Query

from app.api.errors import upstream_error
from app.schemas.career import CareerInput, SkillItem, SkillProfile
from app.services import ncs_link

router = APIRouter()


@router.post("", response_model=SkillProfile, summary="경력 서사 → NCS 능력단위 연결")
async def generate_profile(career: CareerInput) -> SkillProfile:
    # STEP 1 진단 업무에 붙은 NCS 코드도 후보에 넣는다 — 경력 구절 검색이 놓친 능력단위를 LLM이 고를 수 있게
    extra = [t.ncs_code for t in career.diagnosed_tasks if t.ncs_code]
    try:
        skills = await ncs_link.link_career(career.raw_text, career.current_job_title, extra)
    except Exception as e:
        raise upstream_error("능력단위 연결 실패", e) from e
    # 현재 직무명은 사용자가 적은 값(설문)만 돌려준다 — 이력서 경로는 프론트가 STEP 1 진단의 job_title 을 쓴다
    return SkillProfile(skills=skills, current_job_title=career.current_job_title)


@router.get("/units", response_model=list[SkillItem], summary="NCS 능력단위 검색 — 빠진 역량 직접 추가용")
async def search_units(q: str = Query(..., min_length=1, max_length=50, description="능력단위명 또는 직무명")) -> list[SkillItem]:
    try:
        return await asyncio.to_thread(ncs_link.search_units, q)
    except Exception as e:
        raise upstream_error("능력단위 검색 실패", e) from e
