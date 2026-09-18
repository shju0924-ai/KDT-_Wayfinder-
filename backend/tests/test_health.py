"""기본 동작 확인 테스트 — backend/ 에서 `pytest` 로 실행.

각 단계 API가 구현되면 시나리오 테스트를 여기에 추가.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_survey_compose():
    resp = client.post(
        "/api/diagnosis/survey",
        json={
            "job_title": "콜센터 상담원",
            "years": "3년 이상",
            "experience": "상담 품질 관리와 신입 교육을 맡았습니다.",
            "strengths": ["고객 응대", "운영 개선"],
            "concern": "내 강점을 잘 모르겠어요",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "career_text" in body
    assert "콜센터 상담원" in body["career_text"]
