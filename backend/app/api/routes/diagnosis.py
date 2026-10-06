"""경력 입력 — 이력서 파일 파싱과 설문 응답의 경력 서사 조립.

이력서/설문 어느 경로로 들어와도 '경력 서사(career_text)'를 만들어
STEP 2(역량 분해) 이후 파이프라인에 동일하게 흘려보낸다.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.errors import upstream_error
from app.db.session import get_db
from app.schemas.career import CareerInput, ParsedResume, RiskDiagnosis, SurveyCareer, SurveyInput
from app.services import automation
from app.services import survey as survey_service
from app.services.resume_parser import MAX_FILE_SIZE, ResumeParseError, parse_resume

router = APIRouter()


@router.post(
    "/parse-file",
    response_model=ParsedResume,
    summary="이력서 파일에서 경력 텍스트 추출",
)
async def parse_resume_file(file: UploadFile = File(...)) -> ParsedResume:
    filename = Path(file.filename or "resume").name
    data = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    try:
        text, file_type, warnings = parse_resume(filename, data)
    except ResumeParseError as exc:
        status_code = 413 if len(data) > MAX_FILE_SIZE else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return ParsedResume(
        filename=filename,
        file_type=file_type,
        text=text,
        char_count=len(text),
        warnings=warnings,
    )


@router.post(
    "/survey",
    response_model=SurveyCareer,
    summary="설문 응답 → 경력 서사 조립",
)
def compose_survey(survey: SurveyInput) -> SurveyCareer:
    """이력서가 없는 사용자용 진입점.

    설문을 경력 서사로 조립해 돌려준다(프론트가 STEP 2 요청에 재사용).
    LLM 없이 결정론적으로 조립하므로 즉시 응답한다.
    """
    return SurveyCareer(career_text=survey_service.compose_career_text(survey))


@router.post(
    "/risk",
    response_model=RiskDiagnosis,
    summary="경력 서사 → 업무별 AI 자동화 영향 점수",
)
async def diagnose_risk(career: CareerInput, db: Session = Depends(get_db)) -> RiskDiagnosis:
    """업무를 나눠 업무별 자동화·AI 보조 점수와 공개 직업 노출도로 보정한 점수를 낸다.

    '위험 높음' 같은 판정은 내지 않는다 — 점수·산출 근거·출처만 보여주고 해석은 사용자가 한다.
    경력 원문은 저장하지 않는다.
    """
    try:
        return await automation.diagnose(db, career.raw_text, career.current_job_title)
    except Exception as e:
        raise upstream_error("자동화 영향 진단 실패", e) from e
