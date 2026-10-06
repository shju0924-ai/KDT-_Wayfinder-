from app.api.errors import upstream_error
from app.api.routes.jobs import USER_ADDED_CATEGORY, _dedupe_targets, _skill_search_text
from app.schemas.career import SkillItem, SkillProfile


def test_dedupe_targets_keeps_first_of_same_core_title() -> None:
    targets = [
        {"job_title": "데이터 분석가", "rationale": "first"},
        {"job_title": "데이터분석가(주니어)", "rationale": "dup"},
        {"job_title": "웹 개발자", "rationale": "other"},
    ]

    result = _dedupe_targets(targets)

    assert [t["rationale"] for t in result] == ["first", "other"]


def test_skill_search_text_skips_evidence_for_user_added_skill() -> None:
    profile = SkillProfile(
        skills=[
            SkillItem(name="고객 갈등 완화", category="대인·협상", evidence="강성 민원 진정"),
            SkillItem(name="SQL", category=USER_ADDED_CATEGORY, evidence="사용자가 직접 추가한 역량"),
        ]
    )

    lines = _skill_search_text(profile).split("\n")

    assert lines == ["고객 갈등 완화 (대인·협상) — 강성 민원 진정", "SQL"]


def test_upstream_error_hides_exception_message() -> None:
    exc = RuntimeError("connection to host=db.internal password=secret failed")

    err = upstream_error("직무 검색 실패", exc)

    assert err.status_code == 502
    assert err.detail == "직무 검색 실패 (RuntimeError)"
