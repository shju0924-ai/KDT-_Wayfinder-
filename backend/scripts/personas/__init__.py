"""E2E 회귀용 페르소나 — personas.json 과 화면 예시(frontend/src/mock/examples.json)를 읽어 검증한다.

세트:
  v1      설문 경로 4명 (최초 튜닝 기준)
  v2      설문 3명 + 이력서 1명 — v1 에 맞춘 조정이 다른 경력에서도 통하는지(과적합 점검)
  resume  이력서 원문 경로 — 형식·정보량이 다른 이력서로 직무명 추출과 근거 추출을 점검
  ui      시작 화면 예시 입력 — 화면에 내건 예시가 실제로 끝까지 잘 도는지

페르소나는 두 가지 형태 중 하나다.
  설문형   SurveyInput 필드(job_title·years·experience·strengths·concern·aspiration)
  이력서형 raw_text 하나 — 직무명 입력 없이 STEP 2가 직무명을 추출하는 경로
'_' 로 시작하는 키(예: _case)는 사람이 읽는 메모라 API 로 보내지 않는다.

설문형은 실제 설문 화면(SurveyForm)이 보낼 수 있는 값만 쓰도록 강제한다 — 강점은 STRENGTH_OPTIONS,
경력 기간은 YEAR_OPTIONS 안에서 고른다. 자유 문구('데이터 분석', '6년 이상')는 실제 사용자 입력과
분포가 달라 회귀 결과가 화면 경로를 대표하지 못한다. 새 페르소나도 load() 에서 같은 검사를 받고,
tests/test_personas.py 가 CI 에서 같은 검사를 돌린다.
"""
import json
import re
from pathlib import Path

from pydantic import ValidationError

from app.schemas.career import SurveyInput
from app.services.survey import STRENGTH_OPTIONS, YEAR_OPTIONS

PERSONAS_PATH = Path(__file__).with_name("personas.json")
UI_EXAMPLES_PATH = Path(__file__).resolve().parents[3] / "frontend" / "src" / "mock" / "examples.json"

_KEY = re.compile(r"^[a-z0-9_]+$")
_SURVEY_FIELDS = set(SurveyInput.model_fields)


class PersonaError(ValueError):
    pass


def _public(persona: dict) -> dict:
    """API 로 보낼 필드만 — '_' 메모 키 제외."""
    return {k: v for k, v in persona.items() if not k.startswith("_")}


def validate(key: str, persona: dict) -> list[str]:
    """페르소나 하나의 문제 목록. 비어 있으면 통과."""
    errors: list[str] = []
    if not _KEY.match(key):
        errors.append(f"{key}: 키는 영문 소문자·숫자·_ 만 씁니다")
    data = _public(persona)

    if "raw_text" in data:
        if set(data) != {"raw_text"}:
            errors.append(f"{key}: 이력서형은 raw_text 하나만 둡니다 (설문 필드와 섞지 않음): {sorted(set(data) - {'raw_text'})}")
        if not isinstance(data["raw_text"], str) or not data["raw_text"].strip():
            errors.append(f"{key}: raw_text 가 비어 있습니다")
        return errors

    unknown = set(data) - _SURVEY_FIELDS
    if unknown:
        errors.append(f"{key}: 설문에 없는 필드 {sorted(unknown)}")
    try:
        survey = SurveyInput.model_validate(data)
    except ValidationError as e:
        return [*errors, f"{key}: 설문 형식 오류 — {e.errors()[0]['loc']} {e.errors()[0]['msg']}"]

    if survey.years not in YEAR_OPTIONS:
        errors.append(f"{key}: years '{survey.years}' 는 설문 선택지가 아닙니다 — {YEAR_OPTIONS}")
    if not survey.strengths:
        errors.append(f"{key}: strengths 를 하나 이상 고릅니다 (설문 화면은 빈 선택을 제출하지 않음)")
    off = [s for s in survey.strengths if s not in STRENGTH_OPTIONS]
    if off:
        errors.append(f"{key}: strengths {off} 는 설문 선택지가 아닙니다 — {STRENGTH_OPTIONS}")
    if len(set(survey.strengths)) != len(survey.strengths):
        errors.append(f"{key}: strengths 에 중복이 있습니다")
    return errors


def _ui_set() -> dict[str, dict]:
    """시작 화면 예시 → 이력서형 페르소나. 예시는 자유 입력창에 그대로 들어가므로 raw_text 경로다."""
    examples = json.loads(UI_EXAMPLES_PATH.read_text(encoding="utf-8"))
    return {f"ui_{ex['id']}": {"_case": f"시작 화면 예시: {ex['label']}", "raw_text": ex["text"]} for ex in examples}


def load() -> dict[str, dict[str, dict]]:
    """세트 이름 → {페르소나 키 → 페르소나}. 하나라도 규칙을 어기면 전부 모아 PersonaError."""
    sets: dict[str, dict[str, dict]] = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    sets["ui"] = _ui_set()

    errors: list[str] = []
    seen: dict[str, str] = {}
    for set_name, personas in sets.items():
        for key, persona in personas.items():
            if key in seen:
                errors.append(f"{key}: '{seen[key]}' 세트와 키가 겹칩니다 (--set all 에서 덮어쓰임)")
            seen[key] = set_name
            errors.extend(validate(key, persona))
    if errors:
        raise PersonaError("페르소나 검증 실패:\n  " + "\n  ".join(errors))
    return sets


def select(set_name: str) -> dict[str, dict]:
    """세트(쉼표로 여러 개, 또는 'all')의 페르소나 — API 로 보낼 필드만 남긴다. 예: 'resume,ui'."""
    sets = load()
    names = list(sets) if set_name == "all" else [n.strip() for n in set_name.split(",") if n.strip()]
    unknown = [n for n in names if n not in sets]
    if unknown or not names:
        raise PersonaError(f"알 수 없는 세트 {unknown or set_name!r} — 사용 가능: {', '.join(sets)}, all")
    return {k: _public(v) for n in names for k, v in sets[n].items()}
