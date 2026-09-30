# data-pipeline — 공공데이터 수집·전처리·임베딩 배치

## 흐름

```
collectors/  →  data/ (원본·전처리 산출물, git 제외)  →  embedding/  →  PostgreSQL(pgvector)
```

## 데이터 소스

| 소스 | 용도 | API | 상태 |
|---|---|---|---|
| 서울시 일자리포털 | 채용공고(서울·경기·인천) → 인접 직무 매핑 | data.seoul.go.kr OA-23047 `recMntList` (구 `GetJobInfo`는 2026-09-03 종료) | ✅ 40,517건 수집 확인 (2026-09-29) |
| 경기도 잡아바 | 채용공고(경기) → 인접 직무 매핑 | data.gg.go.kr `GGJOBABARECRUSTM` | ✅ 확인됨 |
| HRD-Net | 훈련과정(NCS 코드 포함) → 학습 로드맵 교육 자원 | work24.go.kr `callOpenApiSvcInfo310L01` | ✅ 확인됨 |
| 고용24 (워크넷) | 채용 공고·직업 정보 | https://openapi.work.go.kr | ❌ 개인회원 이용 불가 확인됨 — 서울·경기로 대체 |
| NCS | 능력단위 → 업무 표준화·진단 근거 | CQ-Net `Ncs1info/ncsinfo.do` | ✅ 15,520건 호출·적재 확인 |
| Anthropic Economic Index | 미국 직업별 실제 AI 사용 노출도 | Hugging Face 공개 CSV | ✅ 756개 직업 적재 |
| ILO/NASK 2025 | ISCO-08 직업별 생성형 AI 잠재 노출도 | 공개 XLSX | ✅ 2,541개 직업 적재 |
| OECD | 인간 역량-AI 역량 격차 기반 노출도 방법론 | 2026 보고서 | ✅ 원문 수집·방법론 반영 |
| 한국고용정보원 | 국내 537개 직업·41개 업무활동의 AI 직무대체 연구 | 2024 보고서 | ✅ 원문 수집·해석 기준 반영 |

## 실행

```bash
pip install -r requirements.txt
# data-pipeline/.env 에 API 키·DATABASE_URL 설정 후 (프로젝트 루트 .env.example 참고)
# DB 적재까지 하려면 프로젝트 루트에서 docker compose up -d 먼저

python run_pipeline.py                 # 수집 → 청킹 → 임베딩 → DB 적재 전체 실행
python run_pipeline.py --skip-collect  # 기존 CSV 재사용
python run_pipeline.py --limit 10 --dry-run  # DB 없이 소량 검증
```

개별 단계 실행:

```bash
python -m collectors.seoul_job     # 서울 채용공고 → jobs_seoul.csv
python -m collectors.gg_joba       # 경기 채용공고 → jobs_gg.csv
python -m collectors.hrdnet        # 훈련과정 → courses.csv
python -m collectors.ncs           # NCS 능력단위 → ncs_units.csv
python -m collectors.automation_reference  # Anthropic·ILO 정규화 + OECD·KEIS 원문
python -m embedding.embed_jobs     # 채용공고 임베딩 → job_postings 테이블
python -m embedding.embed_courses  # 훈련과정 임베딩 → training_courses 테이블
python -m embedding.load_reference_data  # NCS·자동화 참고점수 → DB
python -m embedding.backfill_job_urls  # 기존 임베딩은 유지하고 원본 공고 URL만 백필
```

## 마감·개강 필터 (2026-08 반영)

수집기는 이제 **전량 수집 후 만료 항목을 제외**하고 CSV에 저장한다. 기준일은 수집 실행일.

- **채용공고**: 접수 마감일이 지난 공고 제외 (`collectors/filters.py`의 `job_is_open`).
  서울은 `RCEPT_CLOS_NM`, 경기는 `RCPT_END_DE` 기준. 날짜 미표기(상시채용 등)는 유효로 유지.
  `collect(max_records=None)` / `collect(max_pages=None)` 이면 API 총건수까지 전량 수집.
- **훈련과정**: HRD 수집 시 `srchTraStDt=오늘`로 두어 **이미 개강한 과정을 API 단계에서 제외**하고,
  응답에서도 개강일<오늘·정원 마감 과정을 한 번 더 걸러낸 뒤 같은 과정의 회차 중복을 정리한다.
  기본 창은 오늘~+90일. (2026-09-18: 데모 단계의 임베딩 실현성 상한 `max_records=40000`은
  프로젝트 완성 방침에 따라 제거 — 기본값 `None`이면 창 안의 전량을 수집한다.)
- **재적재**: 만료돼 이번 수집에서 빠진 행은 upsert로는 남으므로, 열린 공고/과정만 유지하려면
  `--fresh`로 테이블을 비운 뒤 넣는다: `python -m embedding.embed_jobs --fresh`.
- **런타임 이중 안전장치**: 백엔드 훈련과정 검색은 `end_date >= CURRENT_DATE`로 종료된 과정을 한 번 더 제외한다.
- 임베딩은 로컬 bge-m3로 수행한다(API 호출·rate limit 없음). GPU가 없으면 CPU로 돌아 수만 건에
  수 시간이 걸릴 수 있다. 배치 크기는 `EMBED_BATCH_SIZE`(기본 16)로 조정.

## 적재 테이블 (backend/app/db/models.py 와 동기화)

| 테이블 | 소스 | 용도 |
|---|---|---|
| `job_postings` | 서울(`seoul:` 접두어) + 경기(`gg:`) | STEP 3 인접 직무 유사도 검색 |
| `training_courses` | HRD-Net(`hrdnet:`), ncs_cd 포함 | STEP 4 로드맵 훈련과정 검색·근거 |
| `ncs_units` | 한국산업인력공단 NCS | STEP 1 업무 단위 표준화·출처 |
| `automation_occupation_scores` | Anthropic + ILO/NASK | STEP 1 직업 노출도 보정 |

청킹: 텍스트 500자 초과 시에만 50자 오버랩으로 분할되며(bge-m3 512토큰 상한에 맞춤), 청크 행은
`source_id`에 `#c1`, `#c2` 접미어가 붙는다 (대부분의 공고·과정은 청킹 불필요).

## 규칙

- 원본 응답은 `data/raw/`, 전처리 결과는 `data/processed/` 에 저장 (둘 다 git 제외)
- 임베딩 모델·차원은 `backend/app/services/embedding.py` 와 반드시 일치시킬 것
  (현재: 로컬 `BAAI/bge-m3`, 1024차원, 입력 512토큰 상한 — 검색 대상·검색어 모두 같은 모델)

## 자동화 위험도 산식

1. 경력 서사에서 실제 업무 3~7개와 업무 비중을 구조화한다.
2. 업무별 `automation_score`를 비중 가중 평균해 업무 기반 점수를 만든다.
3. 표준 영어 직업명과 매칭된 Anthropic·ILO 노출도 평균을 25%만 반영한다.
4. 최종 점수 = 업무 기반 점수 75% + 외부 노출도 25%.

외부 데이터의 수치는 **AI 노출도**이지 해고·고용대체 확률이 아니다. NCS도
자동화 점수를 제공하지 않으므로 과업명과 정의를 확인하는 근거로만 사용한다.
