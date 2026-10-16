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

    monkeypatch.setattr(llm, "_ollama_parse", fake_ollama)
    first = asyncio.run(llm.extract_skill_gaps(["상담"], "데이터 분석가"))
    second = asyncio.run(llm.extract_skill_gaps(["상담"], "데이터 분석가"))
    assert first == second == ["SQL"]
    assert len(calls) == 1
    llm._cache_conn.close()
    monkeypatch.setattr(llm, "_cache_conn", None)


def test_llm_cache_expires_after_ttl(monkeypatch, tmp_path):
    """캐시 값에는 경력 서사에서 뽑은 근거가 들어간다 — 보존 기한이 지나면 읽지 않고 지운다."""
    monkeypatch.setattr(settings, "llm_cache_path", str(tmp_path / "cache.sqlite"))
    monkeypatch.setattr(settings, "llm_cache_ttl_hours", 1)
    monkeypatch.setattr(llm, "_cache_conn", None)
    now = [1_000_000.0]
    monkeypatch.setattr(llm.time, "time", lambda: now[0])

    llm._cache_put("k", "v")
    assert llm._cache_get("k") == "v"
    now[0] += 3601
    assert llm._cache_get("k") is None

    # 다시 열 때 기한 지난 항목은 디스크에서도 지워진다
    llm._cache_conn.close()
    monkeypatch.setattr(llm, "_cache_conn", None)
    assert llm._cache_db().execute("SELECT count(*) FROM llm_cache").fetchone()[0] == 0
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
            "job_no": 0, "rationale": "CS 경험이 이어짐", "transferable_skills": [name],
            "training_needs": ["고객 여정 분석"], "demand_outlook": "증가", "transition_difficulty": "보통",
            "license_required": False,
        }])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    candidates = [{"ncs_code": "02030201", "job_title": "고객관리", "path": "영업판매 > 마케팅 > 마케팅전략", "units": ""}]
    out = asyncio.run(llm.suggest_training_targets(candidates, [f"{name} (대인·협상) — 근거"], "콜센터 상담원", [name]))
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


def _exposure_rows():
    from types import SimpleNamespace

    def row(source, title, code, score):
        return SimpleNamespace(source=source, occupation_title=title, occupation_code=code, score=score, source_url="u")

    aei = "Anthropic Economic Index"
    return [
        row("ILO", "Telemarketer", "4227", 60.0),
        row("ILO", "Nursing Assistant", "5321", 20.0),
        row("ILO", "Office Clerk", "4110", 40.0),
        row(aei, "Telemarketers", "41-9041", 28.0),
        row(aei, "Customer Service Representatives", "43-4051", 70.0),
        row(aei, "Nursing Assistants", "31-1131", 0.0),
        row(aei, "Tellers", "43-3071", 2.0),
    ]


def test_exposure_is_0_to_100_rank_and_unobserved_aei_is_excluded():
    """노출도는 출처마다 직업 간 순위 0~100(높을수록 대체 가능성 높음). Anthropic 0점은 '관측 없음'이라 뺀다
    (e2e 3차 0.0 오표시 8건)."""
    from app.services import automation

    pct = automation.exposure_percentiles(_exposure_rows())
    assert pct[("ILO", "4227")] == 100.0 and pct[("ILO", "5321")] == 0.0 and pct[("ILO", "4110")] == 50.0
    assert ("Anthropic Economic Index", "31-1131") not in pct
    assert pct[("Anthropic Economic Index", "43-3071")] == 0.0  # 관측된 최저점은 0이 맞다


def test_exposure_uses_llm_choice_among_embedding_shortlist(monkeypatch):
    """임베딩은 후보만 추리고, 같은 직업은 LLM이 고른다. 고른 직업이 '관측 없음'이면 다른 후보로 넘어가지 않는다."""
    from app.services import automation, embedding

    rows = _exposure_rows()
    by = {(r.source, r.occupation_title): r for r in rows}
    aei = "Anthropic Economic Index"
    shortlist = [
        {"ILO": [(by[("ILO", "Telemarketer")], 0.7), (by[("ILO", "Office Clerk")], 0.6)],
         aei: [(by[(aei, "Customer Service Representatives")], 0.7), (by[(aei, "Telemarketers")], 0.69)]},
        {"ILO": [(by[("ILO", "Office Clerk")], 0.6)], aei: [(by[(aei, "Nursing Assistants")], 0.68)]},
        {"ILO": [(by[("ILO", "Office Clerk")], 0.5)], aei: [(by[(aei, "Tellers")], 0.5)]},
    ]

    async def fake_run_db(fn, *args):
        return shortlist, automation.exposure_percentiles(rows)

    async def fake_match(items):
        assert items[0][1][aei] == ["Customer Service Representatives", "Telemarketers"]
        return [{"ILO": 0, aei: 1}, {"ILO": -1, aei: 0}, {"ILO": -1, aei: -1}]

    monkeypatch.setattr(embedding, "run_db", fake_run_db)
    monkeypatch.setattr(automation.llm, "match_occupations", fake_match)
    out = asyncio.run(automation.exposures_for_titles(["텔레마케터", "요양보호사", "심해 잠수부"]))
    # 텔레마케터: ILO 3개 중 1위 = 100, Anthropic 관측 3개 중 2위 = 50 (1순위 후보가 아니라 LLM이 고른 후보)
    assert out[0][0] == [100.0, 50.0] and [s.source for s in out[0][1]] == ["ILO", aei]
    assert all(0 <= s.score <= 100 for s in out[0][1])
    assert out[1] == ([], [])  # LLM이 고른 Nursing Assistants 는 관측 없음(0점) → 0.0으로 표시하지 않는다
    assert out[2] == ([], [])  # 같은 직업 없음

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


def test_snap_quote_restores_original_spacing():
    """화면의 근거는 원문 글자 그대로 — 4b가 '5명'을 '5 명'으로 옮겨도 원문 구절로 되돌린다."""
    src = "강성 민원 고객을 진정시켜 해지 방어율 팀 1위를 했습니다. 신입 상담원 5명의 응대 스크립트 교육도 맡았습니다."
    assert llm.snap_quote("신입 상담원 5 명의 응대 스크립트 교육도 맡았습니다", src) == "신입 상담원 5명의 응대 스크립트 교육도 맡았습니다"
    assert llm.snap_quote("일평균 고객 60 명 응대", "- 입출금, 일평균 고객 60명 응대") == "일평균 고객 60명 응대"
    # 조사·어미를 바꿔 옮긴 인용은 원문에서 찾을 수 없으니 그대로 둔다
    assert llm.snap_quote("스크립트 교육을 맡음", src) == "스크립트 교육을 맡음"
