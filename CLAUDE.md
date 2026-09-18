# Wayfinder — AI 전환기 커리어 네비게이터

2026 제8회 K-디지털 트레이닝 해커톤 프로젝트 (팀 Wayfinder, 6인).
예선 통과, ~8월 중순 MVP, 9월 초 본선 시연.

## 서비스 한 줄 요약

경력을 전이 가능한 역량 단위로 분해(LLM) → 인접 직무 매핑(pgvector 유사도) → 맞춤 학습 로드맵(RAG). "AI는 지도를 제공하고, 길은 인간이 선택한다" — AI가 판정하지 않고 사용자가 검토·수정·선택하는 UX가 핵심 차별점.

## 구조

- `frontend/` — Vite + React + TS. 4단계 페이지: Diagnosis → SkillProfile → JobExplorer → Roadmap
- `backend/` — FastAPI. `app/api/routes/`에 단계별 라우터, `app/services/`에 LLM·임베딩, `app/db/`에 SQLAlchemy+pgvector
- `data-pipeline/` — 공공데이터(서울·경기 채용정보·HRD-Net) 수집 → 임베딩 → DB 적재 배치
- `db/init/` — pgvector 확장 SQL (docker compose가 자동 실행)

## 핵심 규칙

- **스키마 동기화**: `backend/app/schemas/career.py` ↔ `frontend/src/types/api.ts` 는 1:1 대응. 한쪽을 바꾸면 반드시 다른 쪽도 수정.
- **임베딩 일치**: 모델·차원(Upstage `solar-embedding-2-query`/`-passage`, 1024)은 `backend/app/services/embedding.py` 가 기준. `data-pipeline/embedding/` 과 `db/models.py` Vector 차원이 항상 일치해야 함. 검색 대상(채용공고·NCS·훈련과정)은 passage, 검색어(사용자 역량)는 query 모델로 임베딩할 것 — 섞으면 유사도가 무의미해짐.
- **근거 의무화**: LLM 출력(역량 분해·진단·로드맵)은 항상 evidence/rationale/source 필드를 채워야 함 — 환각 억제가 심사 대응 포인트.
- **민감정보**: 이력서 원문은 DB 저장하지 않는 것이 기본. 저장 필요 시 팀 합의 먼저.
- 실행 방법·팀원별 담당은 `README.md` 참고.

## 환경

- Windows 개발 환경. Node.js 미설치 상태였음(설치 필요). Python은 있음.
- DB는 `docker compose up -d` (pgvector/pgvector:pg17, wayfinder/wayfinder@localhost:5432)
- git 저장소 아직 미초기화 (팀에서 별도 진행 예정)
