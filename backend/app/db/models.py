"""DB 모델 — 공공데이터 적재 테이블 + 벡터 컬럼.

개인정보 원칙(기획서): 이력서 등 민감정보는 세션 종료 시 즉시 파기.
→ 사용자 경력 원문은 DB에 저장하지 않는 것을 기본으로 하고,
  저장이 꼭 필요해지면 팀 회의에서 암호화·파기 정책을 먼저 정한다.

주의: data-pipeline/embedding/common.py 의 DDL과 테이블 구조를 항상 일치시킬 것
     (적재는 data-pipeline이 raw SQL로, 조회는 backend가 이 모델로 수행).

TODO:
  - [ ] 자동화 대체율 참조 테이블 (한국고용정보원)
  - [ ] Alembic 마이그레이션 도입 여부 결정 (해커톤이면 create_all로도 충분)
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.services.embedding import EMBEDDING_DIM


class Base(DeclarativeBase):
    pass


class JobPosting(Base):
    """채용 공고 (서울 일자리포털[서울·경기·인천] + 경기 잡아바 수집분) — STEP 3 인접 직무 탐색 대상."""

    __tablename__ = "job_postings"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(String(80), unique=True, comment="원천 데이터 ID (seoul:/gg: 접두어)")
    job_title: Mapped[str] = mapped_column(String(300), index=True)
    company: Mapped[str | None] = mapped_column(String(200), nullable=True)
    region: Mapped[str | None] = mapped_column(String(20), index=True, comment="seoul | gg | incheon")
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True, comment="원본 채용공고 상세 URL")
    required_skills_text: Mapped[str] = mapped_column(Text, comment="요구역량 서술 (임베딩 입력)")
    deadline: Mapped[str | None] = mapped_column(
        String(10), nullable=True, comment="접수 마감일 YYYY-MM-DD (NULL=상시채용 등 마감일 없음)"
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrainingCourse(Base):
    """훈련과정 (HRD-Net 수집분, NCS 코드 포함) — STEP 4 학습 로드맵 검색 대상."""

    __tablename__ = "training_courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[str] = mapped_column(String(80), unique=True, comment="trprId#회차")
    course_name: Mapped[str] = mapped_column(String(300), index=True)
    institution: Mapped[str | None] = mapped_column(String(200), nullable=True)
    ncs_cd: Mapped[str | None] = mapped_column(String(20), index=True, comment="NCS 코드 — 로드맵 근거(source)")
    ncs_nm: Mapped[str | None] = mapped_column(String(100), nullable=True)
    content_text: Mapped[str] = mapped_column(Text, comment="과정 설명 (임베딩 입력)")
    start_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    tuition: Mapped[int | None] = mapped_column(Integer, nullable=True, comment="수강비(원)")
    address: Mapped[str | None] = mapped_column(String(200), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class NcsUnit(Base):
    """NCS 능력단위. 위험도 점수가 아니라 과업 표준화·근거 표시에 사용."""

    __tablename__ = "ncs_units"

    id: Mapped[int] = mapped_column(primary_key=True)
    ncs_code: Mapped[str] = mapped_column(String(30), unique=True, index=True)
    unit_name: Mapped[str] = mapped_column(String(500), index=True)
    unit_definition: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    large_category: Mapped[str | None] = mapped_column(String(200), nullable=True)
    middle_category: Mapped[str | None] = mapped_column(String(200), nullable=True)
    small_category: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sub_category: Mapped[str | None] = mapped_column(String(200), index=True, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AutomationOccupationScore(Base):
    """직업별 AI 노출도 참고값. 자동화 확률로 직접 해석하지 않는다."""

    __tablename__ = "automation_occupation_scores"
    __table_args__ = (
        UniqueConstraint("source", "occupation_code", name="uq_automation_score_source_code"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(120), index=True)
    occupation_code: Mapped[str] = mapped_column(String(30))
    occupation_title: Mapped[str] = mapped_column(String(500), index=True)
    score: Mapped[float] = mapped_column(Float)
    metric: Mapped[str] = mapped_column(String(100))
    source_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_url: Mapped[str] = mapped_column(String(1000))
    collected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
