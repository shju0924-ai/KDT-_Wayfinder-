"""4단계 파이프라인 공통 스키마.

frontend/src/types/api.ts 와 1:1 대응을 유지한다 — 필드를 바꾸면 양쪽 모두 수정할 것.
"""
from typing import Literal

from pydantic import BaseModel, Field

ResumeFileType = Literal["pdf", "docx", "hwp", "hwpx"]
TaskEffect = Literal["automation", "augmentation", "human"]
Region = Literal["seoul", "gg", "incheon"]
DemandOutlook = Literal["증가", "유지", "감소"]
TransitionDifficulty = Literal["낮음", "보통", "높음"]


# ── STEP 1. 경력 입력·진단 ──────────────────────────────────
class AutomationTask(BaseModel):
    """경력 서사에서 추출한 업무와 업무별 AI 영향."""

    name: str
    share_percent: float = Field(..., ge=0, le=100, description="직무에서 차지하는 비중")
    automation_score: float = Field(..., ge=0, le=100)
    ai_assistance_score: float = Field(..., ge=0, le=100)
    effect: TaskEffect = Field(..., description="업무의 주효과 — 자동화 | AI 보조 | 사람 중심")
    rationale: str
    ncs_code: str | None = None
    ncs_unit: str | None = None
    source_quote: str | None = Field(None, description="이 업무의 근거가 된 경력 서사 원문 구절")


class CareerInput(BaseModel):
    """이력서 텍스트 또는 대화형 인터뷰 응답."""
    raw_text: str = Field(..., description="이력서 원문 또는 인터뷰 응답 전문")
    current_job_title: str | None = Field(None, description="현재(최근) 직무명")
    diagnosed_tasks: list[AutomationTask] = Field(
        default_factory=list,
        description="STEP 1 진단 업무 — 업무에 붙은 NCS 코드를 STEP 2 능력단위 후보에 더한다(없으면 건너뜀)",
    )


class ParsedResume(BaseModel):
    """업로드한 이력서에서 추출한 진단용 텍스트."""

    filename: str
    file_type: ResumeFileType
    text: str
    char_count: int = Field(..., ge=0)
    warnings: list[str] = Field(default_factory=list)


class SurveyInput(BaseModel):
    """이력서가 없는 사용자를 위한 6문항 설문 응답.

    services/survey.py 가 이 응답을 경력 서사(raw_text)로 조립하므로,
    이후 STEP 2~4는 이력서 업로드와 완전히 같은 경로를 탄다.
    """

    job_title: str = Field(..., min_length=1, description="Q1. 현재(최근) 직무명")
    years: str = Field(..., min_length=1, description="Q2. 경력 기간 (예: 3년 이상)")
    experience: str = Field(..., min_length=1, description="Q3. 가장 자신 있는 경험 서술")
    strengths: list[str] = Field(
        default_factory=list, description="Q4. 자주 맡았거나 잘했던 일 (복수 선택)"
    )
    concern: str = Field(..., min_length=1, description="Q5. 커리어에서 가장 고민되는 점")
    aspiration: str | None = Field(None, description="Q6. 앞으로 해보고 싶은 일 (선택)")


class RiskSource(BaseModel):
    """위험도 진단에 사용한 데이터 또는 방법론 출처."""

    source: str
    label: str
    url: str
    score: float | None = Field(
        None, ge=0, le=100,
        description="AI 노출도 출처면 직업 간 순위로 환산한 0~100점(100에 가까울수록 AI로 대체될 가능성이 높음)",
    )
    note: str | None = None


class RiskDiagnosis(BaseModel):
    """업무별 분석과 공개 데이터로 보정한 자동화 위험도 진단 결과."""

    job_title: str
    risk_score: float = Field(..., ge=0, le=100, description="자동화 위험도 (0~100). 등급 판정 없이 점수만 제공")
    rationale: str = Field(..., description="점수 산출 방식 설명 (업무별 근거는 tasks[].rationale)")
    task_based_score: float = Field(..., ge=0, le=100)
    automation_share: float = Field(..., ge=0, le=100)
    augmentation_share: float = Field(..., ge=0, le=100)
    human_centered_share: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=100, description="근거 충족도")
    tasks: list[AutomationTask] = Field(default_factory=list)
    sources: list[RiskSource] = Field(default_factory=list)


class SurveyCareer(BaseModel):
    """설문 응답을 조립한 경력 서사.

    career_text 는 프론트가 STEP 2(역량 분해) 요청에 그대로 재사용한다 —
    설문/이력서 어느 경로로 들어와도 이후 파이프라인 입력이 동일해진다.
    """

    career_text: str = Field(..., description="설문 답변으로 조립한 경력 서사")


# ── STEP 2. 역량 프로필 ────────────────────────────────────
class SkillItem(BaseModel):
    """경력에서 확인한 역량 — NCS 능력단위(국가직무능력표준의 직무별 능력)로 표현한다."""
    ncs_code: str = Field(..., description="NCS 능력단위 코드 (예: 0203020101_17v2)")
    name: str = Field(..., description="능력단위명 (예: 전표관리)")
    ncs_job: str = Field(..., description="능력단위가 속한 NCS 직무(세분류) (예: 회계·감사)")
    ncs_path: str = Field(..., description="NCS 분류 경로 (대분류 > 중분류 > 소분류 > 세분류)")
    definition: str = Field(..., description="능력단위 정의(축약)")
    evidence: str | None = Field(
        None, description="이 능력단위의 근거가 된 경력 서사 원문 구절 — 사용자가 직접 추가하면 None"
    )
    user_added: bool = Field(False, description="사용자가 NCS 검색으로 직접 추가한 능력단위인지")
    confirmed: bool = Field(False, description="사용자 검토·확정 여부")


class SkillProfile(BaseModel):
    skills: list[SkillItem]
    current_job_title: str | None = Field(
        None, description="현재(최근) 직무명 — 사용자 입력값(설문). 없으면 프론트가 STEP 1 진단의 job_title 을 쓴다. STEP 3 같은 직무 제외에 사용"
    )


# ── STEP 3. 인접 직무 탐색 ─────────────────────────────────
JobSearchTrack = Literal["adjacent_transition", "training_transition"]


class JobMatch(BaseModel):
    """전환 후보 직무."""
    posting_id: str | None = Field(None, description="원본 공고 식별자 (청크 접미어 제거)")
    posting_count: int = Field(1, ge=1, description="같은 직무로 묶인 공고 수")
    job_title: str
    company: str | None = Field(None, description="공고를 낸 회사명")
    region: Region | None = Field(None, description="공고 지역")
    source_url: str | None = Field(None, description="원본 채용공고 상세 페이지 URL")
    requirement_excerpt: str | None = Field(None, description="요구역량 판단 근거가 된 공고 발췌")
    fit_score: float = Field(..., ge=0, le=100, description="역량 적합도 (벡터 유사도 기반)")
    demand_outlook: DemandOutlook = Field(..., description="수요 전망")
    transition_difficulty: TransitionDifficulty = Field(..., description="전환 난이도")
    matched_skills: list[str] = Field(default_factory=list, description="적합 판정에 기여한 보유 역량")
    required_skills: list[str] = Field(default_factory=list, description="공고에서 확인된 핵심 요구역량")
    missing_skills: list[str] = Field(default_factory=list, description="요구역량 중 아직 없는 역량")
    weak_reasons: list[str] = Field(
        default_factory=list,
        description="추천 근거가 약한 이유 — 숨기지 않고 표시해 사용자가 판단하게 한다. 비어 있으면 근거 충분",
    )
    license_required: bool = Field(
        False, description="국가 면허가 있어야 일할 수 있는 직업인지 — 걸러내지 않고 라벨만 붙인다"
    )
    ai_exposure_score: float | None = Field(
        None, ge=0, le=100,
        description=(
            "이 직무의 AI 노출도 0~100 — 100에 가까울수록 AI로 대체될 가능성이 높은 직무. "
            "공개 자료(ILO·Anthropic)의 직업 간 순위를 출처별로 평균. 매칭 자료가 없으면 None"
        ),
    )
    ai_exposure_sources: list[RiskSource] = Field(
        default_factory=list, description="노출도 출처별 점수와 매칭된 직업명"
    )


# ── STEP 4. 학습 로드맵 ────────────────────────────────────
class HrdCourse(BaseModel):
    """로드맵 근거로 선택된 실제 고용24 훈련과정."""

    name: str
    institution: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    tuition: int | None = None
    address: str | None = None
    url: str | None = None


class RoadmapItem(BaseModel):
    """학습 항목 하나."""
    skill_gap: str = Field(..., description="보완할 역량 격차")
    learning_item: str = Field(..., description="학습 항목명")
    duration_weeks: int = Field(..., ge=0, description="예상 소요 기간(주). 0이면 훈련과정이 없어 기간 미정")
    resources: list[str] = Field(default_factory=list, description="교육 자원 (KDT·HRD-Net 훈련과정 등)")
    source: str = Field(..., description="RAG 근거 출처")
    course: HrdCourse | None = Field(None, description="추천 근거가 된 실제 고용24 훈련과정")
    weak_reason: str | None = Field(None, description="과정과 역량 격차의 연관 근거가 약한 이유. 없으면 None")


class LearningRoadmap(BaseModel):
    target_job: str
    items: list[RoadmapItem]
