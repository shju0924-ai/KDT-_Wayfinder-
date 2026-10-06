"""LLM 출력 후처리·근거 정합성 — 모델 호출 없이 검증한다."""
import asyncio

from app.api.routes import jobs
from app.core.config import settings
from app.services import llm


def test_clean_job_title_drops_english_parenthesis():
    assert llm.clean_job_title("디지털 마케팅 분석가 (Digital Marketing Analyst)") == "디지털 마케팅 분석가"
    assert llm.clean_job_title("비즈니스 인텔리전스 (BI) 분석가") == "비즈니스 인텔리전스 분석가"
    assert llm.clean_job_title("콜센터 상담원(인바운드)") == "콜센터 상담원(인바운드)"


def test_clean_skill_phrase_drops_course_wording():
    assert llm.clean_skill_phrase("PMP(프로젝트 관리 전문가) 자격증 과정") == "PMP(프로젝트 관리 전문가)"
    assert llm.clean_skill_phrase("Agile/Scrum 방법론 부트캠프") == "Agile/Scrum 방법론"
    assert llm.clean_skill_phrase("3. SQL 데이터 추출") == "SQL 데이터 추출"
    assert llm.clean_skill_phrase("콘텐츠 마케팅") == "콘텐츠 마케팅"


def test_course_duration_weeks_from_schedule():
    assert llm.course_duration_weeks("2026-10-06", "2026-11-05") == 5
    assert llm.course_duration_weeks("20261015", "20261214") == 9
    assert llm.course_duration_weeks(None, "2026-11-05") is None
    assert llm.course_duration_weeks("2026-11-05", "2026-10-06") is None


def test_roadmap_course_name_comes_from_index_not_model(monkeypatch):
    """모델이 적은 과정명이 아니라 course_index 가 가리키는 실제 과정이 학습 항목·출처가 된다."""
    care = {"course_name": "고객 응대와 멘탈 관리", "ncs_cd": "1", "ncs_nm": "고객관리",
            "start_date": "2026-10-06", "end_date": "2026-11-05"}
    sales = {"course_name": "세일즈 전략", "ncs_cd": None, "start_date": None, "end_date": None}
    gaps = ["고객 응대", "설득", "영업", "엑셀"]
    # 고객 응대·설득 후보에 같은 과정이 겹치면 번호 하나(0)로 합쳐진다. 영업 후보는 [1]
    courses_by_gap = {"고객 응대": [care], "설득": [care], "영업": [sales], "엑셀": []}

    async def fake_parse(schema, system, user, **kw):
        assert "격차 3. 영업\n  [1] 세일즈 전략" in user
        return schema(items=[
            {"gap_no": 1, "course_index": 0, "duration_weeks": 8},
            # 같은 과정 중복 인용 → 일반 학습 항목으로 강등
            {"gap_no": 2, "course_index": 0, "duration_weeks": 3},
            {"gap_no": 3, "course_index": 1, "duration_weeks": 4},
            {"gap_no": 4, "course_index": -1, "duration_weeks": 2},
        ])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    items = asyncio.run(llm.build_roadmap_items(gaps, "텔레마케터", courses_by_gap))

    assert [i["skill_gap"] for i in items] == gaps
    assert items[0]["learning_item"] == "고객 응대와 멘탈 관리"
    assert items[0]["course"]["name"] == items[0]["learning_item"]
    assert items[0]["duration_weeks"] == 5  # 실제 일정 기준, 모델 추정 8주 무시
    assert items[1]["course"] is None and items[1]["learning_item"] == "훈련과정 없음 — 설득"
    assert items[1]["duration_weeks"] == 0 and items[1]["source"] == "DB에 맞는 고용24 훈련과정 없음"
    assert items[2]["learning_item"] == "세일즈 전략" and items[2]["duration_weeks"] == 4
    assert items[3]["learning_item"] == "훈련과정 없음 — 엑셀"


def test_roadmap_rejects_course_from_another_gap(monkeypatch):
    """다른 격차로 검색된 과정 번호를 고르면 근거로 인정하지 않는다 (예: CRM 격차에 컴활 과정)."""
    excel = {"course_name": "컴퓨터활용능력 실기", "ncs_cd": None, "start_date": None, "end_date": None}

    async def fake_parse(schema, system, user, **kw):
        return schema(items=[
            {"gap_no": 1, "course_index": 0, "duration_weeks": 4},
            {"gap_no": 2, "course_index": 0, "duration_weeks": 2},
        ])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    items = asyncio.run(llm.build_roadmap_items(
        ["CRM 데이터 정합성 관리", "엑셀 활용"], "영업 기획", {"CRM 데이터 정합성 관리": [], "엑셀 활용": [excel]}
    ))

    assert items[0]["course"] is None and items[0]["learning_item"] == "훈련과정 없음 — CRM 데이터 정합성 관리"
    assert items[1]["course"]["name"] == "컴퓨터활용능력 실기"


def test_ollama_parse_retries_on_hanja(monkeypatch):
    """출력에 한자가 섞이면 한 번 다시 생성하고, 깨끗한 결과를 쓴다."""
    from pydantic import BaseModel

    class Out(BaseModel):
        text: str

    replies = iter(['{"text": "전환漏斗 분석"}', '{"text": "전환 퍼널 분석"}'])
    sent = []

    class FakeResp:
        def __init__(self, content):
            self._c = content

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": self._c}, "done_reason": "stop"}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json):
            sent.append(dict(json["options"]))
            return FakeResp(next(replies))

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    out = asyncio.run(llm._ollama_parse(Out, "sys", "user", 100))

    assert out.text == "전환 퍼널 분석"
    assert sent[0]["temperature"] == 0 and sent[1]["temperature"] > 0


def test_drop_same_jobs_uses_title_similarity(monkeypatch):
    sims = {"고객상담원": 0.83, "텔레마케터": 0.54, "경영기획사무원": 0.43}

    async def fake_similarities(base, titles):
        return [sims[t] for t in titles]

    monkeypatch.setattr(jobs.embedding, "similarities_to", fake_similarities)
    rows = [{"job_title": "고객 상담원"}, {"job_title": "텔레마케터"}, {"job_title": "경영 기획 사무원"},
            {"job_title": "콜센터 상담원(CS센터)"}]
    kept = asyncio.run(jobs._drop_same_jobs(rows, "콜센터 상담원"))
    assert [r["job_title"] for r in kept] == ["텔레마케터", "경영 기획 사무원"]


def test_llm_cache_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "llm_cache_path", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(llm, "_cache_conn", None)
    calls = []

    async def fake_ollama(schema, system, user, num_predict):
        calls.append(user)
        return schema(gaps=["SQL"])

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(llm, "_ollama_parse", fake_ollama)
    first = asyncio.run(llm.extract_skill_gaps(["상담"], "데이터 분석가"))
    second = asyncio.run(llm.extract_skill_gaps(["상담"], "데이터 분석가"))
    assert first == second == ["SQL"]
    assert len(calls) == 1
    llm._cache_conn.close()
    monkeypatch.setattr(llm, "_cache_conn", None)



def test_manual_downshift_uses_keco_title_suffix():
    """사무·관리 경력자에게 조작·조립·운전·단순 노무 직무는 빼고, 사무 '종사원'은 남긴다."""
    items = [{"job_title": t} for t in ["압연기 조작원", "무역 사무원", "기타 제조 관련 단순 종사원",
                                         "기타 사무 지원 종사원", "지게차 운전원", "물류 사무원"]]
    kept = [it["job_title"] for it in jobs._drop_manual_downshift(items, "생산관리 담당자")]
    assert kept == ["무역 사무원", "기타 사무 지원 종사원", "물류 사무원"]
    # 현재 직무가 조작직이면 같은 계열 이동은 막지 않는다 — 단순 노무직으로 내리는 것만 뺀다
    kept_op = [it["job_title"] for it in jobs._drop_manual_downshift(items, "CNC 선반 조작원")]
    assert kept_op == ["압연기 조작원", "무역 사무원", "기타 사무 지원 종사원", "지게차 운전원", "물류 사무원"]


def test_requirement_conditions_are_not_skills():
    for cond in ["직업상담사 자격증", "직업 상담 자격증 (필수)", "운전면허 소지", "경력 3년", "차량 소지자"]:
        assert llm.is_requirement_condition(cond), cond
    for skill in ["물류 시스템 운영", "B2B 영업 프로세스", "고객 서비스 (CS) 대응", "SQL 쿼리 작성"]:
        assert not llm.is_requirement_condition(skill), skill


def test_weak_reasons_flag_low_fit_only():
    assert jobs._adjacent_weak_reasons(60.0, {"transition_difficulty": "보통"}) == []
    reasons = jobs._adjacent_weak_reasons(52.3, {"transition_difficulty": "높음"})
    assert len(reasons) == 1 and "52.3" in reasons[0]  # 난이도는 카드 태그에 이미 있어 중복하지 않는다


def test_roadmap_weak_reason_uses_gap_course_similarity(monkeypatch):
    strong = {"course_name": "고객감동 CS 빌드업", "ncs_cd": None, "similarity": 0.59}
    weak = {"course_name": "브랜딩의 힘", "ncs_cd": None, "similarity": 0.50}

    async def fake_parse(schema, system, user, **kw):
        return schema(items=[
            {"gap_no": 1, "course_index": 0, "duration_weeks": 2},
            {"gap_no": 2, "course_index": 1, "duration_weeks": 2},
            {"gap_no": 3, "course_index": -1, "duration_weeks": 2},
        ])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    items = asyncio.run(llm.build_roadmap_items(
        ["CS 대응", "상품 지식", "단가 비교"], "텔레마케터", {"CS 대응": [strong], "상품 지식": [weak], "단가 비교": []}
    ))
    assert items[0]["weak_reason"] is None
    assert "0.50" in items[1]["weak_reason"]
    assert items[2]["weak_reason"] is None


def test_training_targets_keep_skill_names_with_parentheses(monkeypatch):
    """괄호가 든 역량명('고객 서비스 (CS) 대응')도 잘리지 않고 이어지는 역량으로 남는다."""
    name = "고객 서비스 (CS) 대응"

    async def fake_parse(schema, system, user, **kw):
        return schema(targets=[{
            "job_title": "고객 경험 전략가", "rationale": "CS 경험이 이어짐", "transferable_skills": [name],
            "training_needs": ["고객 여정 분석"], "demand_outlook": "증가", "transition_difficulty": "보통",
            "occupation_title_en": "Customer Experience Managers",
        }])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    out = asyncio.run(llm.suggest_training_targets([f"{name} (대인·협상) — 근거"], "콜센터 상담원", [name]))
    assert out[0]["transferable_skills"] == [name]


def test_roadmap_items_align_by_gap_no_not_position(monkeypatch):
    """모델이 항목 순서를 바꿔 내도 gap_no 로 맞춰, 각 격차가 자기 후보 과정을 받는다."""
    sales = {"course_name": "세일즈 전략", "ncs_cd": None, "similarity": 0.6}
    crm = {"course_name": "영업관리전문가", "ncs_cd": None, "similarity": 0.6}

    async def fake_parse(schema, system, user, **kw):
        return schema(items=[
            {"gap_no": 2, "course_index": 1, "duration_weeks": 2},
            {"gap_no": 1, "course_index": 0, "duration_weeks": 2},
        ])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    items = asyncio.run(llm.build_roadmap_items(
        ["판매 전략", "CRM 활용"], "텔레마케터", {"판매 전략": [sales], "CRM 활용": [crm]}
    ))
    assert [(i["skill_gap"], i["course"]["name"]) for i in items] == [("판매 전략", "세일즈 전략"), ("CRM 활용", "영업관리전문가")]


def test_traits_are_not_skills():
    for trait in ["반복적이고 단조로운 업무에 대한 인내심", "판매 목표 달성 압박감 견디기"]:
        assert llm.is_requirement_condition(trait), trait
    assert not llm.is_requirement_condition("CRM 시스템 활용 능력")


def test_occupation_exposure_scores_match_by_english_title():
    """후보 직무 카드의 AI 노출도 — 영문 직업명으로 출처별 점수를 찾고 평균을 낸다. 없으면 None."""
    from types import SimpleNamespace
    from app.services import automation

    rows = [
        SimpleNamespace(source="ILO", occupation_title="Telemarketer", occupation_code="4227", score=60.0, source_url="u"),
        SimpleNamespace(source="Anthropic", occupation_title="Telemarketers", occupation_code="41-9041", score=28.0, source_url="u"),
    ]
    automation_rows = automation._load_occupation_rows
    try:
        automation._load_occupation_rows = lambda db: rows
        (score, sources), (none_score, none_sources) = automation.occupation_exposures(None, ["Telemarketers", "Deep Sea Divers"])
    finally:
        automation._load_occupation_rows = automation_rows
    assert score == 44.0 and {s.source for s in sources} == {"ILO", "Anthropic"}
    assert none_score is None and none_sources == []


def test_operator_is_not_offered_simple_labor():
    items = [{"job_title": t} for t in ["기타 제조 관련 단순 종사원", "금속가공 기계 조작원", "생산관리 사무원"]]
    kept = [it["job_title"] for it in jobs._drop_manual_downshift(items, "CNC 선반 조작원")]
    assert kept == ["금속가공 기계 조작원", "생산관리 사무원"]


def test_diagnosis_rescales_ten_point_answers_and_derives_effect():
    from types import SimpleNamespace
    from app.services import automation

    ten_point = [SimpleNamespace(automation_score=3.5, ai_assistance_score=6.5),
                 SimpleNamespace(automation_score=1.0, ai_assistance_score=2.0)]
    assert automation.normalize_task_scale(ten_point) == 10.0
    hundred = [SimpleNamespace(automation_score=70, ai_assistance_score=40)]
    assert automation.normalize_task_scale(hundred) == 1.0
    assert automation.task_effect(85, 20) == "automation"
    assert automation.task_effect(35, 65) == "augmentation"
    assert automation.task_effect(10, 10) == "human"


def test_occupation_title_match_needs_core_words():
    from app.services.automation import MATCH_THRESHOLD, _title_similarity

    assert _title_similarity("Nursing Assistant", "Sewing assistant") < MATCH_THRESHOLD
    assert _title_similarity("Telemarketers", "Telemarketer") >= MATCH_THRESHOLD
    assert _title_similarity("Insurance Agent", "Insurance agent") >= MATCH_THRESHOLD


def test_hanja_is_stripped_when_retry_still_has_it(monkeypatch):
    from pydantic import BaseModel

    class Out(BaseModel):
        text: str

    replies = iter(['{"text": "프로세스 개선专员"}', '{"text": "프로세스 개선专员"}'])

    class FakeResp:
        def __init__(self, c):
            self._c = c

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": self._c}, "done_reason": "stop"}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, url, json):
            return FakeResp(next(replies))

    monkeypatch.setattr(llm.httpx, "AsyncClient", FakeClient)
    assert asyncio.run(llm._ollama_parse(Out, "sys", "user", 100)).text == "프로세스 개선"


def test_roadmap_without_any_candidate_skips_llm(monkeypatch):
    """어느 격차에도 후보 과정이 없으면 LLM을 부르지 않고 전부 '훈련과정 없음'으로 낸다."""
    async def boom(*a, **kw):
        raise AssertionError("LLM을 부르면 안 된다")

    monkeypatch.setattr(llm, "_parse", boom)
    items = asyncio.run(llm.build_roadmap_items(["LC/TT 결제 처리", "발주 일정 조율"], "무역 사무원", {}))
    assert [i["learning_item"] for i in items] == ["훈련과정 없음 — LC/TT 결제 처리", "훈련과정 없음 — 발주 일정 조율"]
    assert all(i["course"] is None and i["duration_weeks"] == 0 for i in items)
