"""2026-10-12 페르소나 15명 E2E 검토에서 나온 문제의 회귀 테스트 — 모델 호출 없이 검증한다."""
import asyncio

from app.api.routes import jobs
from app.schemas.career import SurveyInput
from app.services import automation, llm
from app.services.survey import compose_career_text


def _judge(job_no: int, missing: list[str], matched: list[str] | None = None) -> dict:
    return {
        "job_no": job_no, "same_as_current": False,
        "matched_skills": ["고객 응대", "문서 관리"] if matched is None else matched,
        "missing_skills": missing, "demand_outlook": "유지", "transition_difficulty": "보통",
        "license_required": job_no == 2,
    }


# ── 1. 직무 판정 결과가 다른 직무에 붙는 문제 ──────────────────
def test_judgements_align_by_job_no_not_position(monkeypatch):
    """모델이 판정 순서를 바꿔 내도 job_no 로 짝지어 각 직무가 자기 판정을 받는다(번역가 페르소나 실측)."""
    async def fake_parse(schema, system, user, **kw):
        return schema(judgements=[_judge(2, ["무역 서류 처리"]), _judge(0, ["웹 서비스 기획"]), _judge(1, ["콘텐츠 기획"])])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    postings = [{"job_title": t, "snippet": ""} for t in ["웹 기획자", "공연 기획자", "무역 사무원"]]
    out = asyncio.run(llm.judge_job_matches(["고객 응대", "문서 관리"], "번역가", postings))
    assert [o["missing_skills"] for o in out] == [["웹 서비스 기획"], ["콘텐츠 기획"], ["무역 서류 처리"]]
    assert [o["license_required"] for o in out] == [False, False, True]


def test_duplicate_job_no_keeps_first_and_missing_falls_back(monkeypatch):
    async def fake_parse(schema, system, user, **kw):
        return schema(judgements=[_judge(0, ["A 역량"]), _judge(0, ["B 역량"])])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    postings = [{"job_title": "갑", "snippet": ""}, {"job_title": "을", "snippet": ""}]
    first, second = asyncio.run(llm.judge_job_matches(["고객 응대", "문서 관리"], None, postings))
    assert first["missing_skills"] == ["A 역량"]
    assert second["matched_skills"] == [] and second["missing_skills"] == []  # 판정 없음 → 인접 판정에서 탈락


def test_adjacent_candidates_are_judged_one_job_per_call(monkeypatch):
    """묶음 판정은 내용이 다음 직무로 밀려 써졌다 — 직무마다 따로 부른다."""
    calls: list[list[str]] = []

    async def fake_judge(skills, current, postings, held_skills=None):
        calls.append([p["job_title"] for p in postings])
        return [{**_judge(0, [f"{postings[0]['job_title']} 역량"]), "required_skills": []}]

    monkeypatch.setattr(jobs.llm, "judge_job_matches", fake_judge)
    batch = [{"job_title": t} for t in ["영업 기획", "경영 기획", "해외 영업"]]
    out = asyncio.run(jobs._judge_batch(["데이터 분석", "운영 개선"], "MD", batch))
    assert calls == [["영업 기획"], ["경영 기획"], ["해외 영업"]]
    assert [o["missing_skills"] for o in out] == [["영업 기획 역량"], ["경영 기획 역량"], ["해외 영업 역량"]]


def test_held_skill_written_as_missing_counts_as_matched(monkeypatch):
    """4b가 직무에 필요한 보유 역량을 missing 에 적으면 이어지는 역량으로 옮긴다('웹 개발자' ← '반응형 웹 구현')."""
    async def fake_parse(schema, system, user, **kw):
        return schema(judgements=[_judge(0, ["반응형 웹 구현", "React 개발"], matched=[])])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    out = asyncio.run(llm.judge_job_matches(["반응형 웹 구현", "웹 접근성 확보"], "웹 퍼블리셔",
                                            [{"job_title": "웹 개발자", "snippet": ""}]))[0]
    assert out["matched_skills"] == ["반응형 웹 구현"]
    assert out["missing_skills"] == ["React 개발"]


def test_field_guide_carries_nested_descriptions_to_ollama_prompt():
    """Ollama format 은 필드 설명을 모델에 전달하지 않는다 — 프롬프트용 목록에 중첩 필드 설명까지 담긴다."""
    guide = llm.field_guide(llm._judge_batch_schema(["고객 응대"], 1))
    assert "judgements[].license_required" in guide and "국가 면허" in guide
    assert "tasks[].automation_score" in llm.field_guide(llm._TaskAnalysisOut)


# ── 2. AI 노출도 매칭 ──────────────────────────────────────
# ── 3. 로드맵 과정 근거 약함 ─────────────────────────────────
def test_course_without_shared_keyword_is_flagged_weak():
    flagged = [
        ("CRM 활용 능력", "[Office 365 엑셀] 함수와 매크로 활용능력 UP!"),
        ("일정 관리 및 조율", "[R&D전문가] 연구개발 기술로드맵과 프로젝트관리"),
        ("규제 준수 및 컴플라이언스 관리", "[CS경영전문가] 고객만족도(CSI) 조사와 VOC 관리"),
        ("비즈니스 연속성 계획 수립", "ICT 비즈니스 모델 전략"),
    ]
    for gap, name in flagged:
        reason = llm.course_weak_reason(gap, {"course_name": name}, 0.55)
        assert reason and "핵심어" in reason, (gap, name)

    kept = [
        ("SQL 쿼리 작성", "[상위1% 프로그래밍] Hello, MySQL"),
        ("데이터 시각화 도구 활용", "Power BI를 활용한 시각화"),
        ("통계적 분석 기법", "아는 만큼 보인다! 데이터 분석을 위한 기초 통계학"),
        ("고객 관계 관리", "CRM고객관계관리"),
    ]
    for gap, name in kept:
        assert llm.course_weak_reason(gap, {"course_name": name}, 0.55) is None, (gap, name)
    # NCS 분류에 핵심어가 있으면 과정명에 없어도 근거로 본다
    assert llm.course_weak_reason("수출입 서류 관리", {"course_name": "무역 실무 A", "ncs_nm": "수출입 관리"}, 0.55) is None
    # 유사도가 충분히 높으면 핵심어 검사를 하지 않는다
    assert llm.course_weak_reason("CRM 활용 능력", {"course_name": "엑셀 매크로"}, 0.60) is None


# ── 4. 설문 강점이 업무로 둔갑하는 문제 ─────────────────────────
def test_survey_strengths_are_marked_as_self_assessment():
    text = compose_career_text(SurveyInput(
        job_title="온라인 쇼핑몰 MD", years="5년 이상", experience="판매 데이터를 분석했습니다.",
        strengths=["문서·데이터 관리", "협업·조율"], concern="걱정", aspiration=None,
    ))
    assert "자기 평가" in text and "문서·데이터 관리, 협업·조율" in text
    assert "자주 맡았거나 잘했던 일" not in text


# ── 5. 근거 충족도 ──────────────────────────────────────────
def test_evidence_confidence_reflects_input_richness():
    sparse = "의류 매장 판매 직원 5년\n- 매장 판매, 재고 정리, 디스플레이\n- 주말 아르바이트생 근무표 관리"
    rich = (
        "자사몰과 오픈마켓 상품 기획을 맡아 판매 데이터를 엑셀로 분석해 주간 프로모션을 짰고, 협력사 20곳과 "
        "입고 일정·단가를 협상했습니다. 상세페이지 개편으로 전환율을 1.4%에서 2.1%로 올렸습니다. " * 3
    )
    low = automation.evidence_confidence(sparse, 4, False, 0)
    high = automation.evidence_confidence(rich, 4, False, 1)
    assert low <= 50 and high >= 70 and high - low >= 20
    assert automation.evidence_confidence(rich * 5, 7, True, 3) == 90  # 상한


def test_ratio_task_scores_are_rescaled_to_100():
    """4b가 0~1 비율로 답한 점수(은행 텔러 0.4점 진단)는 100점 척도로 환산한다."""
    from types import SimpleNamespace

    ratio = [SimpleNamespace(automation_score=0.4, ai_assistance_score=0.6),
             SimpleNamespace(automation_score=0.8, ai_assistance_score=0.3)]
    assert automation.normalize_task_scale(ratio) == 100.0
    assert automation.normalize_task_scale([SimpleNamespace(automation_score=0, ai_assistance_score=0)]) == 1.0


# ── 7·8. 전환 직무·직무명·외국어 정리 ──────────────────────────────
def test_english_role_suffix_becomes_korean():
    assert llm.clean_job_title("고객 관계 관리 Specialist") == "고객 관계 관리 전문가"
    assert llm.clean_job_title("데이터 Analysts") == "데이터 분석가"
    assert llm.clean_job_title("Data Analyst") == "Data Analyst"  # 한글이 없으면 손대지 않는다


def test_accented_latin_is_foreign_and_stripped():
    from pydantic import BaseModel

    class Out(BaseModel):
        text: str

    assert llm._has_foreign("Estadística 설명적 분석")
    assert not llm._has_foreign("SQL 쿼리 작성, CRM 활용")
    assert llm._strip_hanja(Out, Out(text="Estadística 설명적 분석")).text == "설명적 분석"
