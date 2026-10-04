# Wayfinder Demo

AI 전환기에 일하는 사람이 자신의 경력을 업무와 역량으로 해석하고, 인접 직무와 학습 경로를 탐색할 수 있도록 돕는 데모 서비스입니다.

> AI가 자동화하는 것은 직무의 일부 업무이지, 개인의 역량 전체가 아닙니다. Wayfinder는 점수와 근거를 함께 보여주고 최종 판단은 사용자에게 남깁니다.

2026 제8회 K-디지털 트레이닝 해커톤 출품작으로 시작해, 현재는 1인 포트폴리오 프로젝트로 개발하고 있습니다.

## 제공 기능

- 자유 텍스트 또는 이력서 파일(PDF, DOCX, HWP 5.x, HWPX)에서 경력 텍스트 추출
- 경력 업무 단위 분석과 자동화 위험도 진단
- 전이 가능한 역량 프로필 생성 및 사용자 검토
- 실제 채용공고 기반 인접 직무 탐색
- HRD-Net 훈련과정 근거의 학습 로드맵 생성

이력서 파일은 서버 DB에 저장하지 않으며, 요청 처리 중 메모리에서 텍스트만 추출합니다. 파일당 최대 크기는 10MB입니다. 스캔 이미지로만 구성된 PDF는 OCR이 없어 텍스트를 추출할 수 없습니다.

## 빠른 실행

### 1. 저장소와 환경 변수 준비

```powershell
git clone https://github.com/shju0924-ai/KDT-_Wayfinder-.git
cd KDT-_Wayfinder-

Copy-Item .env.example backend\.env
Copy-Item .env.example data-pipeline\.env
```

`backend/.env`에는 LLM 설정을 채웁니다 — `LLM_PROVIDER=claude`면 `ANTHROPIC_API_KEY`, `LLM_PROVIDER=ollama`면 키 없이 로컬 모델([Ollama](https://ollama.com) 설치 후 `ollama pull qwen3.5:4b`)을 씁니다. 로컬 모델은 GPU 없는 PC에서 단계당 1~4분 걸립니다. 임베딩은 로컬 모델(`BAAI/bge-m3`)이라 키가 필요 없고, 첫 실행 시 모델(약 2GB)을 자동으로 내려받습니다. 공공데이터까지 새로 수집하려면 `data-pipeline/.env`에 아래 키도 넣습니다.

```dotenv
NCS_API_KEY=
SEOUL_JOB_API_KEY=
GG_JOBA_API_KEY=
HRDNET_API_KEY=
```

환경 변수 파일은 Git에 포함하지 않습니다.

### 2. PostgreSQL + pgvector 실행

```powershell
docker compose up -d
```

이 명령은 DB와 `vector` 확장만 생성합니다. 채용공고·훈련과정·NCS·자동화 참고점수는 별도 데이터 적재가 필요합니다.

DB 데이터는 Docker 볼륨 `wayfinder-pgdata`에 저장됩니다. Windows 폴더를 직접 마운트하면 대량 적재 중 PostgreSQL이 비정상 종료되는 문제가 있어 볼륨을 사용합니다. 백업 덤프가 있으면 적재 대신 복원할 수 있습니다.

```powershell
docker cp data-pipeline\data\backup\wayfinder_20261005.dump wayfinder-db:/tmp/
docker exec wayfinder-db pg_restore -U wayfinder -d wayfinder --no-owner /tmp/wayfinder_20261005.dump
```

### 3. 백엔드 실행

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API 문서: http://localhost:8000/docs

### 4. 프론트엔드 실행

새 터미널에서 실행합니다.

```powershell
cd frontend
npm install
npm run dev
```

브라우저: http://localhost:5173

### 5. 데모 데이터를 포함해 4단계 전체를 실행하려면

새 로컬 환경의 DB는 비어 있으므로, STEP 3·4까지 재현하려면 파이프라인을 한 번 실행해야 합니다. 수집 API 키가 준비된 뒤 실행합니다(임베딩은 로컬 모델이라 키 불필요, GPU가 없으면 수 시간 소요).

```powershell
cd data-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_pipeline.py
```

파이프라인은 서울·경기 채용공고, HRD-Net 훈련과정, NCS 능력단위, 자동화 참고자료를 수집하고 임베딩한 뒤 DB에 적재합니다. 원본 및 전처리 데이터는 `data-pipeline/data/`에 생성되며 Git에 포함되지 않습니다.

`python run_pipeline.py --limit 20 --dry-run`으로 DB에 쓰지 않는 소량 점검도 할 수 있습니다.

#### GPU 서버에서 임베딩만 계산하기

전량(약 7만 행) 임베딩은 CPU로 하루 이상 걸립니다. GPU 서버(예: RunPod A40, 약 10분)에서 벡터만 계산해 오고 적재는 로컬에서 할 수 있습니다. GPU 서버에는 DB가 필요 없습니다.

```bash
# GPU 서버: data-pipeline/ 의 embedding/, requirements.txt, data/processed/*.csv 업로드 후
pip install -r requirements.txt
python -m embedding.export_vectors          # → data/vectors.npz
```

```powershell
# 로컬: vectors.npz 를 data-pipeline\data\ 에 두고
$env:EMBED_CACHE = "data\vectors.npz"
python run_pipeline.py --skip-collect
```

`EMBED_CACHE`에 있는 텍스트는 모델을 실행하지 않고 저장된 벡터를 쓰고, 없는 텍스트만 로컬에서 계산합니다. 모델 리비전·입력 길이·정규화가 같아 로컬 계산 결과와 동일한 벡터입니다(코사인 유사도 1.0 확인). 5,000행 이상을 넣을 때는 HNSW 인덱스를 잠시 지우고 적재한 뒤 다시 만듭니다.

## 자동화 위험도 계산 방식

자동화 위험도는 해고 확률이나 개인의 대체 가능성을 뜻하지 않습니다. 현재 입력한 경력에서 어떤 업무가 AI의 영향을 받을 수 있는지를 설명하기 위한 0~100 점수입니다.

1. LLM이 경력 서사를 3~7개의 실제 업무와 업무 비중으로 구조화합니다.
2. 각 업무의 자동화 점수를 업무 비중으로 가중 평균해 `업무 기반 점수`를 계산합니다.
3. 영어 직업명과 매칭된 Anthropic·ILO/NASK 직업별 AI 노출도를 평균해 `외부 노출도`를 만듭니다.
4. 최종 점수는 다음 식으로 계산합니다.

```text
업무 기반 점수 = Σ(업무 비중 × 업무별 자동화 점수)
최종 위험도 = 업무 기반 점수 × 0.75 + 외부 노출도 × 0.25
```

직업명 매칭이 없거나 DB에 참고점수가 없으면 외부 노출도를 억지로 적용하지 않고 업무 기반 점수만 사용합니다.

NCS는 자동화 점수를 제공하지 않습니다. 경력에서 나온 업무를 국내 능력단위와 연결하고, 화면에 업무 근거를 표시하는 데만 사용합니다. OECD와 한국고용정보원 보고서는 업무별 판단을 해석하고 한국 노동시장 맥락을 설명하는 방법론 근거이며, 현재 수식에 직접 점수로 합산하지 않습니다.

## 위험도 수식에 사용한 자료와 원문 링크

### 직접 수치로 적재하는 자료

| 자료 | 서비스에서의 사용 | 원문 링크 |
|---|---|---|
| Anthropic Economic Index `job_exposure.csv` | 실제 Claude 사용을 반영한 직업별 `observed_exposure`를 0~100점으로 변환해 외부 노출도에 사용 | https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/labor_market_impacts |
| Anthropic 다운로드 CSV | 파이프라인이 실제로 내려받는 파일 | https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/labor_market_impacts/job_exposure.csv?download=true |
| ILO–NASK 2025 GenAI 직업 노출도 | ISCO-08 직업별 잠재 생성형 AI 노출도를 0~100점으로 변환해 외부 노출도에 사용 | https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure |
| ILO–NASK 재현용 XLSX | 현재 파이프라인이 읽는 6자리 직업 코드·점수 파일 | https://github.com/pgmyrek/POLAND_2025_GenAI_scores_6digit_occupations |

Anthropic의 `observed_exposure`와 ILO/NASK의 `potential_genai_exposure`은 서로 다른 관점의 **AI 노출도**입니다. 둘 다 곧바로 고용 대체나 해고 확률을 뜻하지 않습니다.

### 업무 표준화와 해석 기준

| 자료 | 서비스에서의 사용 | 원문 링크 |
|---|---|---|
| 한국산업인력공단 NCS 기준정보 API | 경력 업무를 국내 능력단위와 연결하고 과업 근거를 표시 | https://www.data.go.kr/data/15063879/openapi.do |
| OECD AI Exposure Measure (2026) | 언어·추론·사회성·신체 능력 등 AI 역량과 직업 요구 역량의 간극을 해석하는 방법론 기준 | https://www.oecd.org/en/publications/the-oecd-ai-exposure-measure_f3da0f0a-en.html |
| OECD 보고서 PDF | 파이프라인이 방법론 원문으로 내려받는 PDF | https://www.oecd.org/content/dam/oecd/en/publications/reports/2026/05/the-oecd-ai-exposure-measure_489cfd42/f3da0f0a-en.pdf |
| 한국고용정보원, 「인공지능에 의한 화이트칼라의 직무 대체 및 변화」 | 국내 직무활동과 직무 변화·대체 가능성을 해석하는 기준 | https://www.keis.or.kr/keis/ko/proj/113/pblc/detail.do?categoryIdx=131&pubIdx=11169 |

## 데이터 흐름

```text
이력서 텍스트 또는 파일
  → FastAPI: 텍스트 추출·경력 업무 분석
  → NCS 능력단위 매칭 + Anthropic/ILO 직업 노출도 보정
  → 위험도 카드
  → 역량 프로필
  → pgvector 채용공고 유사도 검색
  → HRD-Net 훈련과정 기반 로드맵
```

| DB 테이블 | 적재 데이터 | 사용 단계 |
|---|---|---|
| `ncs_units` | NCS 능력단위 | STEP 1 업무 근거 |
| `automation_occupation_scores` | Anthropic·ILO/NASK 직업별 노출도 | STEP 1 외부 노출도 보정 |
| `job_postings` | 서울·경기 채용공고와 임베딩 | STEP 3 인접 직무 탐색 |
| `training_courses` | HRD-Net 훈련과정과 임베딩 | STEP 4 학습 로드맵 |

## API 요약

| 엔드포인트 | 설명 |
|---|---|
| `POST /api/diagnosis/parse-file` | PDF·DOCX·HWP·HWPX에서 이력서 텍스트 추출 |
| `POST /api/diagnosis` | 경력 텍스트의 자동화 위험도 진단 |
| `POST /api/profile` | 경력 텍스트를 전이 가능한 역량으로 분해 |
| `POST /api/jobs/match` | 역량과 채용공고의 유사도 기반 인접 직무 탐색 |
| `POST /api/roadmap` | 목표 직무의 역량 격차와 훈련과정 기반 로드맵 생성 |

파일 파싱 API는 `multipart/form-data`의 `file` 필드를 받습니다. 지원 확장자는 `.pdf`, `.docx`, `.hwp`, `.hwpx`이며 암호 문서, 위장 확장자, 과도하게 압축된 파일은 거부합니다.

## 개발 검증

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q

cd ..\frontend
npm run build
```

## 디렉터리

```text
frontend/       React + TypeScript UI
backend/        FastAPI API, 파일 파싱, 위험도·LLM 서비스
data-pipeline/  공공데이터 수집, 임베딩, PostgreSQL 적재
db/init/        PostgreSQL 초기화 시 pgvector 확장 생성
docs/           아키텍처 문서·이미지, 개발 세션 기록
```
