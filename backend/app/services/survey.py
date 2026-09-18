"""설문 응답 → 경력 서사 조립.

이력서가 없는 사용자도 진단을 받을 수 있게 6문항 설문을 받고, 그 답변을
이력서 원문과 같은 성격의 '경력 서사' 텍스트로 조립한다. 조립된 서사는
automation.diagnose() 에 그대로 들어가므로 이후 STEP 2~4는 이력서 업로드
경로와 완전히 동일하게 동작한다.

LLM을 쓰지 않고 결정론적으로 조립하는 이유:
  - 진단 단계에서 이미 LLM이 서사를 읽으므로 중간에 한 번 더 요약할 필요가 없다
  - 사용자가 쓴 문장을 그대로 보존해야 진단 근거(evidence)가 사용자 입력과 대응된다
"""
from app.schemas.career import SurveyInput

# 프론트(SurveyForm)의 선택지와 대응 — 자유 입력도 허용하므로 검증에는 쓰지 않는다.
YEAR_OPTIONS = ["1년 미만", "1~3년", "3년 이상", "5년 이상", "10년 이상"]
STRENGTH_OPTIONS = [
    "고객 응대",
    "문제 해결",
    "문서·데이터 관리",
    "협업·조율",
    "운영 개선",
    "현장 대응",
]
CONCERN_OPTIONS = [
    "현재 직무의 전망이 불안해요",
    "내 강점을 잘 모르겠어요",
    "다음 직무를 정하지 못했어요",
    "배워야 할 것이 막막해요",
]


def compose_career_text(survey: SurveyInput) -> str:
    """설문 응답을 진단용 경력 서사로 조립한다.

    사용자가 직접 쓴 문장(experience, aspiration)은 가공하지 않고 그대로 싣는다.
    """
    job = survey.job_title.strip()
    years = survey.years.strip()

    parts: list[str] = [f"저는 {job}으로 {years} 일해왔습니다."]

    experience = survey.experience.strip()
    if experience:
        parts.append(f"가장 자신 있게 설명할 수 있는 경험은 다음과 같습니다.\n{experience}")

    strengths = [s.strip() for s in survey.strengths if s and s.strip()]
    if strengths:
        parts.append(f"업무 중에서 자주 맡았거나 잘했던 일은 {', '.join(strengths)}입니다.")

    concern = survey.concern.strip()
    if concern:
        parts.append(f"지금 커리어에서 가장 고민되는 점은 '{concern}'입니다.")

    aspiration = (survey.aspiration or "").strip()
    if aspiration:
        parts.append(f"앞으로 해보고 싶은 일은 다음과 같습니다.\n{aspiration}")

    return "\n\n".join(parts)
