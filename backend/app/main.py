"""Wayfinder — AI 전환기 커리어 네비게이터 백엔드 진입점."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import diagnosis, profile, jobs, roadmap

app = FastAPI(
    title="Wayfinder API",
    description="AI 전환기 커리어 네비게이터 — 진단·분해·탐색·설계 4단계 파이프라인",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4단계 파이프라인 라우터
app.include_router(diagnosis.router, prefix="/api/diagnosis", tags=["STEP 1. 경력 입력·진단"])
app.include_router(profile.router, prefix="/api/profile", tags=["STEP 2. 역량 프로필"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["STEP 3. 인접 직무 탐색"])
app.include_router(roadmap.router, prefix="/api/roadmap", tags=["STEP 4. 학습 로드맵"])


@app.get("/health", tags=["시스템"])
def health() -> dict:
    return {"status": "ok"}
