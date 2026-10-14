from app.api.errors import upstream_error
from app.api.routes.jobs import _dedupe_targets, _skill_search_text
from app.schemas.career import SkillItem, SkillProfile


def test_dedupe_targets_keeps_first_of_same_core_title() -> None:
    targets = [
        {"job_title": "데이터 분석가", "rationale": "first"},
        {"job_title": "데이터분석가(주니어)", "rationale": "dup"},
        {"job_title": "웹 개발자", "rationale": "other"},
    ]

    result = _dedupe_targets(targets)

    assert [t["rationale"] for t in result] == ["first", "other"]


def test_skill_search_text_uses_definition_for_user_added_unit() -> None:
    def unit(name, job, definition, evidence=None, user_added=False):
        return SkillItem(ncs_code=name, name=name, ncs_job=job, ncs_path=f"경영 > {job}", definition=definition,
                         evidence=evidence, user_added=user_added)

    profile = SkillProfile(
        skills=[
            unit("고객 상담", "보험모집", "고객 요구를 파악해 상담하는", evidence="강성 민원 고객을 진정시켜"),
            unit("채권관리", "회계·감사", "미수금을 회수·관리하는", user_added=True),
        ]
    )

    lines = _skill_search_text(profile).split("\n")

    assert lines == ["고객 상담 (보험모집) — 강성 민원 고객을 진정시켜", "채권관리 (회계·감사) — 미수금을 회수·관리하는"]


def test_upstream_error_hides_exception_message() -> None:
    exc = RuntimeError("connection to host=db.internal password=secret failed")

    err = upstream_error("직무 검색 실패", exc)

    assert err.status_code == 502
    assert err.detail == "직무 검색 실패 (RuntimeError)"
