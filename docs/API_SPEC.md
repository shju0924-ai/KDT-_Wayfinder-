# API 명세 (v0.1 초안)

> 기준: `backend/app/schemas/career.py` · FE 타입: `frontend/src/types/api.ts`
> 실행 후 http://localhost:8000/docs 에서 자동 문서 확인 가능.

## STEP 1 — POST `/api/diagnosis`

경력 입력 → 자동화 위험도 진단

```jsonc
// 요청
{ "raw_text": "10년간 콜센터 상담원으로 근무하며 …", "current_job_title": "콜센터 상담원" }
// 응답
{ "job_title": "콜센터 상담원", "risk_score": 78.5, "risk_level": "높음", "rationale": "…근거…" }
```

## STEP 2 — POST `/api/profile`

경력 서사 → 전이 가능 역량 분해

```jsonc
// 요청: CareerInput (STEP 1과 동일)
// 응답
{
  "skills": [
    { "name": "갈등 완화", "category": "대인", "evidence": "강성 민원 응대 경험 …", "confirmed": false }
  ]
}
```

## STEP 3 — POST `/api/jobs/match`

역량 프로필 → 인접 직무 매핑 (pgvector 유사도)

```jsonc
// 요청: SkillProfile (사용자 확정본)
// 응답
[
  {
    "job_title": "고객 경험(CX) 매니저",
    "fit_score": 82.1,
    "demand_outlook": "증가",
    "transition_difficulty": "보통",
    "matched_skills": ["갈등 완화", "고객 심리 파악"]
  }
]
```

## STEP 4 — POST `/api/roadmap`

목표 직무 → 맞춤 학습 로드맵 (RAG)

```jsonc
// 요청
{ "profile": { "skills": [/* … */] }, "target_job": "고객 경험(CX) 매니저" }
// 응답
{
  "target_job": "고객 경험(CX) 매니저",
  "items": [
    {
      "skill_gap": "데이터 기반 고객 분석",
      "learning_item": "SQL·데이터 분석 기초",
      "duration_weeks": 6,
      "resources": ["HRD-Net: ○○ 데이터 분석 과정"],
      "source": "NCS 능력단위 0203020105"
    }
  ]
}
```

## 변경 규칙

1. 스키마 변경은 `backend/app/schemas/career.py` 먼저 수정
2. `frontend/src/types/api.ts` 동기화
3. 이 문서 갱신 후 팀 공유
