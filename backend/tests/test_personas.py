"""E2E 페르소나 파일 검증 — 새 페르소나도 설문 선택지 규칙(v2 기준)을 따르는지 CI에서 확인한다."""
import json

import pytest

from scripts import personas


def test_all_personas_pass_validation():
    sets = personas.load()  # 규칙을 어기면 PersonaError 로 전부 나열된다
    assert {"v1", "v2", "resume", "ui"} <= set(sets)
    assert all(sets.values())


def test_select_strips_memo_keys():
    for persona in personas.select("all").values():
        assert not any(k.startswith("_") for k in persona)


def test_select_accepts_several_sets_and_rejects_unknown():
    sets = personas.load()
    chosen = personas.select("resume,ui")
    assert list(chosen) == [*sets["resume"], *sets["ui"]]
    with pytest.raises(personas.PersonaError):
        personas.select("resume,nope")


def test_ui_set_follows_frontend_examples():
    examples = json.loads(personas.UI_EXAMPLES_PATH.read_text(encoding="utf-8"))
    ui = personas.load()["ui"]
    assert [p["raw_text"] for p in ui.values()] == [ex["text"] for ex in examples]


@pytest.mark.parametrize(
    "persona, problem",
    [
        ({"job_title": "MD", "years": "5년 이상", "experience": "x", "strengths": ["데이터 분석"], "concern": "x"},
         "설문 선택지가 아닙니다"),
        ({"job_title": "MD", "years": "6년 이상", "experience": "x", "strengths": ["고객 응대"], "concern": "x"},
         "years"),
        ({"job_title": "MD", "years": "5년 이상", "experience": "x", "strengths": [], "concern": "x"},
         "하나 이상"),
        ({"raw_text": "이력서", "job_title": "MD"}, "raw_text 하나만"),
        ({"raw_text": "  "}, "비어 있습니다"),
    ],
)
def test_validate_rejects_off_survey_values(persona, problem):
    errors = personas.validate("sample", persona)
    assert any(problem in e for e in errors), errors


def test_validate_accepts_memo_keys():
    assert personas.validate("sample", {"_case": "메모", "raw_text": "이력서 본문"}) == []
