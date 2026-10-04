# Wayfinder — AI 전환기 커리어 네비게이터

1인 포트폴리오 프로젝트. 2026 제8회 K-디지털 트레이닝 해커톤 출품작(팀 프로젝트로 시작, 본선 진출 실패)을 개인이 이어서 개발 중. 팀 협업·시연 일정은 없음.

## 서비스 한 줄 요약

경력을 전이 가능한 역량 단위로 분해(LLM) → 인접 직무 매핑(pgvector 유사도) → 맞춤 학습 로드맵(RAG). "AI는 지도를 제공하고, 길은 인간이 선택한다" — AI가 판정하지 않고 사용자가 검토·수정·선택하는 UX가 핵심 차별점.

## 구조

- `frontend/` — Vite + React + TS. 4단계 페이지: Diagnosis → SkillProfile → JobExplorer → Roadmap
- `backend/` — FastAPI. `app/api/routes/`에 단계별 라우터, `app/services/`에 LLM·임베딩, `app/db/`에 SQLAlchemy+pgvector
- `data-pipeline/` — 공공데이터(서울·경기 채용정보·HRD-Net·NCS) 수집 → 임베딩 → DB 적재 배치
- `db/init/` — pgvector 확장 SQL (docker compose가 자동 실행)

## 핵심 규칙

- **스키마 동기화**: `backend/app/schemas/career.py` ↔ `frontend/src/types/api.ts` 는 1:1 대응. 한쪽을 바꾸면 반드시 다른 쪽도 수정.
- **임베딩 일치**: 모델·리비전·차원(로컬 `BAAI/bge-m3` @ `5617a9f`, 1024차원, max_seq_length 512, sentence-transformers)은 `backend/app/services/embedding.py` 와 `data-pipeline/embedding/common.py` 가 항상 같아야 하고, `db/models.py` Vector 차원도 일치해야 함. 검색 대상(채용공고·훈련과정)과 검색어(사용자 역량)는 반드시 같은 모델로 임베딩할 것 — 섞으면 유사도가 무의미해짐. 모델을 바꾸면 DB 벡터 전부 재임베딩 필요.
- **근거 의무화**: LLM 출력(역량 분해·진단·로드맵)은 항상 evidence/rationale/source 필드를 채워야 함 — 환각 억제가 핵심 설계 포인트.
- **민감정보**: 이력서 원문은 DB에 저장하지 않는 것이 기본.
- 실행 방법은 `README.md` 참고.

## 환경

- Windows 10, RAM 8GB. C: 여유 공간이 작아 대용량 캐시·데이터는 D:에 둠 (`HF_HOME=D:\hf-cache`, 페이지파일 D:).
- DB는 `docker compose up -d` (pgvector/pgvector:pg17, wayfinder/wayfinder@localhost:5432). 데이터는 Docker 볼륨 `wayfinder-pgdata` — Windows 바인드 마운트는 대량 적재 중 크래시가 나서 쓰지 않음. Docker Desktop은 부팅 후 수동 실행.
- 적재 현황(2026-10-05): job_postings 51,331 / training_courses 18,459 / ncs_units 15,520 / automation_occupation_scores 3,297. 백업 덤프는 `data-pipeline/data/backup/` (`pg_restore -d wayfinder` 로 복원).
- 전량 임베딩은 로컬 CPU로 하루 이상 걸림 → GPU 서버에서 `python -m embedding.export_vectors` 로 `vectors.npz` 를 만들고, 로컬에서 `EMBED_CACHE=data/vectors.npz` 로 적재 (README 참고).
- 모델은 이미 캐시돼 있으므로 `HF_HUB_OFFLINE=1` 로 실행 (허브 접속 시 멈춤 현상 있었음).
