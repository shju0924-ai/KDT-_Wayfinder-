"""2026-10-13 페르소나 15명 E2E 3차 검토에서 나온 문제의 회귀 테스트 — 모델 호출 없이 검증한다."""
import asyncio
from types import SimpleNamespace

import numpy as np

from app.api.routes import jobs
from app.services import llm, ncs_jobs

RETAIL = "의류 매장 판매 직원 5년\n- 매장 판매, 재고 정리, 디스플레이\n- 주말 아르바이트생 근무표 관리"
BOOKKEEPING = (
    "경리·회계 사무를 8년 했어요. 전표 처리, 급여 정산, 세금계산서 발행이 주 업무예요. 월 마감 때 증빙을 대사해서 "
    "불일치를 찾아 정정하고, 세무사·거래처와 일정을 조율했습니다."
)


# ── 1. 원문에 없는 업무·역량 ───────────────────────────────
def test_quote_must_come_from_source():
    # 원문 그대로, 띄어쓰기·문장부호만 다른 인용, 어미만 바꾼 인용은 인정
    assert llm.is_quoted("매장 판매, 재고 정리", RETAIL)
    assert llm.is_quoted("주말 아르바이트생 근무표관리", RETAIL)
    assert llm.is_quoted("증빙을 대사해서 불일치를 찾아 정정했습니다", BOOKKEEPING)
    # 3차에서 지어낸 업무·역량의 근거로 나올 법한 구절은 거른다
    assert not llm.is_quoted("매장 내 안전 관리 및 응급 대응", RETAIL)
    assert not llm.is_quoted("정산 업무를 엑셀 매크로로 자동화", BOOKKEEPING)
    assert not llm.is_quoted("매장", RETAIL)  # 너무 짧은 인용은 근거가 되지 않는다


def test_ungrounded_items_are_dropped_but_minimum_is_kept():
    def item(name, quote):
        return SimpleNamespace(name=name, source_quote=quote)

    tasks = [
        item("고객 상담 및 의류 판매", "매장 판매"),
        item("재고 정리", "재고 정리"),
        item("근무표 관리", "주말 아르바이트생 근무표 관리"),
        item("매장 안전 관리", "매장 내 안전 관리 및 응급 대응"),
    ]
    assert [t.name for t in llm.keep_quoted(tasks, RETAIL, "업무", min_keep=2)] == ["고객 상담 및 의류 판매", "재고 정리", "근무표 관리"]
    # 걸러서 min_keep 미만이 되면 진단이 성립하지 않으므로 그대로 둔다
    two_fake = [item("a", "없는 구절 하나"), item("b", "없는 구절 둘"), item("c", "재고 정리")]
    assert len(llm.keep_quoted(two_fake, RETAIL, "업무", min_keep=2)) == 3


def test_dangling_connector_is_trimmed():
    assert llm.clean_dangling("매장 재고 정리 및") == "매장 재고 정리"
    assert llm.clean_dangling("전표 처리, ") == "전표 처리"
    assert llm.clean_dangling("고객 상담 및 의류 판매") == "고객 상담 및 의류 판매"


def test_task_outputs_require_source_quote():
    assert "source_quote" in llm._TaskOut.model_fields


# ── 3. 교육 후 전환형 — NCS 직무 후보 ───────────────────────
def _ncs_rows():
    def row(cd, large, middle, sub):
        return {"cd": cd, "large": large, "middle": middle, "small": sub, "sub": sub, "units": f"{sub} 능력단위"}

    return [
        row("02030201", "영업판매", "마케팅", "고객관리"),
        row("02030202", "영업판매", "마케팅", "텔레마케팅"),
        row("20010201", "정보통신", "정보기술", "빅데이터분석"),
        row("02020302", "경영·회계·사무", "회계", "회계·감사"),
        row("13010203", "교육·자연·사회과학", "교육", "직업교육훈련"),
    ]


def test_ncs_candidates_exclude_current_field_and_rank_multi_skill_hits():
    rows = _ncs_rows()
    unit_vecs = np.eye(5, dtype=np.float32)
    # 역량 둘: '민원 응대'는 고객관리·텔레마케팅·직업교육, '상담 기록 분석'은 빅데이터·고객관리 쪽
    skills = np.array([[0.9, 0.8, 0.1, 0.0, 0.5], [0.6, 0.0, 0.7, 0.2, 0.0]], dtype=np.float32)
    current = np.array([0.1, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)  # 현재 직무 = 텔레마케팅 분야(영업판매>마케팅)
    out = ncs_jobs.select_candidates(rows, unit_vecs, skills, ["민원 응대", "상담 기록 분석"], current)
    titles = [c["job_title"] for c in out]
    assert "고객관리" not in titles and "텔레마케팅" not in titles  # 현재 중분류는 '새 분야'가 아니다
    assert titles[0] == "빅데이터분석"  # 유사도 0.7 > 직업교육훈련 0.5
    assert all(c["skills"] for c in out)


def test_training_targets_are_chosen_from_ncs_candidates_by_number(monkeypatch):
    """직무명은 NCS 후보에서만 나온다 — 모델이 자유롭게 지은 범용 직무명은 들어올 자리가 없다."""
    candidates = [
        {"ncs_code": "20010201", "job_title": "빅데이터분석", "path": "정보통신 > 정보기술 > 빅데이터분석", "units": ""},
        {"ncs_code": "13010203", "job_title": "직업교육훈련", "path": "교육·자연·사회과학 > 교육 > 직업교육", "units": ""},
    ]

    def target(no):
        return {"job_no": no, "rationale": "근거", "transferable_skills": ["민원 응대"], "training_needs": ["SQL"],
                "demand_outlook": "유지", "transition_difficulty": "보통", "license_required": False}

    async def fake_parse(schema, system, user, **kw):
        assert "[1] 직업교육훈련" in user
        return schema(targets=[target(1), target(1), target(0)])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    out = asyncio.run(llm.suggest_training_targets(candidates, ["민원 응대 (대인·협상) — 근거"], "콜센터 상담원", ["민원 응대"]))
    assert [(o["job_title"], o["ncs_code"]) for o in out] == [("직업교육훈련", "13010203"), ("빅데이터분석", "20010201")]


# ── 4. 인접 직무 — 역량별 검색 ─────────────────────────────
def _hit(posting_id, title, skill_sim, similarity=0.6):
    return {"posting_id": posting_id, "job_title": title, "company": None, "region": None, "source_url": None,
            "snippet": title, "skill_sim": skill_sim, "similarity": similarity}


def test_adjacent_candidates_rank_by_linked_skill_count():
    """여러 역량에 걸린 직무가 한 역량에만 아주 가까운 직무보다 먼저 판정된다."""
    per_skill = [
        # 역량 '편집 교정': 출판 편집자(2건 평균), 번역가 성격의 무역 사무원
        [_hit("a1", "출판물 편집자", 0.70), _hit("a2", "출판물 편집자", 0.60), _hit("b1", "무역 사무원", 0.62)],
        # 역량 '번역': 무역 사무원, 출판물 편집자, 관광 통역 안내원(한 역량에만 매우 가까움)
        [_hit("b2", "무역 사무원", 0.66), _hit("a1", "출판물 편집자", 0.55), _hit("c1", "관광 통역 안내원", 0.80)],
    ]
    out = jobs._aggregate_skill_hits(["편집 교정", "번역"], per_skill)
    assert [c["job_title"] for c in out][:2] == ["무역 사무원", "출판물 편집자"]  # 둘 다 2개 역량 연결
    assert out[-1]["job_title"] == "관광 통역 안내원" and out[-1]["linked_skills"] == ["번역"]
    editor = next(c for c in out if c["job_title"] == "출판물 편집자")
    assert editor["posting_count"] == 2  # 같은 공고(a1)는 한 번만 센다


def test_title_score_averages_top_postings_not_single_outlier():
    """직무 점수는 상위 공고 몇 건의 평균 — 공고 한 건만 튀는 직무가 앞서지 않는다."""
    rows = [_hit("x1", "도료 생산기계 조작원", 0.75), _hit("x2", "도료 생산기계 조작원", 0.40),
            _hit("x3", "도료 생산기계 조작원", 0.40), _hit("y1", "금속가공 기계 조작원", 0.62),
            _hit("y2", "금속가공 기계 조작원", 0.60)]
    out = jobs._aggregate_skill_hits(["CNC 가공"], [rows])
    assert out[0]["job_title"] == "금속가공 기계 조작원"


def test_judge_choices_are_linked_skills_but_missing_excludes_all_held(monkeypatch):
    seen = {}

    async def fake_parse(schema, system, user, **kw):
        seen["user"] = user
        return schema(judgements=[{
            "job_no": 0, "same_as_current": False, "matched_skills": ["번역"],
            # '용어 관리'는 보유 역량이지만 이 직무에 연결되지 않은 역량 — 부족 역량으로 들어가면 안 된다
            "missing_skills": ["용어 관리", "무역 서류 작성"],
            "demand_outlook": "유지", "transition_difficulty": "보통", "license_required": False,
        }])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    [out] = asyncio.run(llm.judge_job_matches(
        ["번역"], "번역가", [{"job_title": "무역 사무원", "snippet": ""}], held_skills=["번역", "용어 관리"]
    ))
    assert "[사용자 보유 역량]\n번역\n" in seen["user"]
    assert out["matched_skills"] == ["번역"] and out["missing_skills"] == ["무역 서류 작성"]
    schema = llm._judge_batch_schema(["번역"], 1)
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):  # 연결되지 않은 역량은 선택지에 없다
        schema(judgements=[{"job_no": 0, "same_as_current": False, "matched_skills": ["용어 관리"],
                            "missing_skills": [], "demand_outlook": "유지", "transition_difficulty": "보통",
                            "license_required": False}])


def test_occupation_match_keeps_only_valid_choices(monkeypatch):
    aei = "Anthropic Economic Index"
    ilo = "ILO/NASK GenAI Exposure 2025"

    async def fake_parse(schema, system, user, **kw):
        assert "[1] 요양보호사" in user and "(1) Nursing Assistants" in user
        return schema(matches=[{"item_no": 1, "ilo_no": 0, "aei_no": 1}, {"item_no": 0, "ilo_no": 1, "aei_no": -1}])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    items = [("텔레마케터", {ilo: ["Telephone Salesperson"], aei: ["Telemarketers"]}),
             ("요양보호사", {ilo: ["Elderly caregiver"], aei: ["Chiropractors", "Nursing Assistants"]})]
    out = asyncio.run(llm.match_occupations(items))
    # 텔레마케터의 ILO 1번은 후보 범위 밖(후보 1개) → 매칭 없음
    assert out == [{ilo: -1, aei: -1}, {ilo: 0, aei: 1}]


# ── 6. E2E 자동 채점(로컬 LLM) ─────────────────────────────
def test_judge_eval_scores_every_item_and_marks_failures(monkeypatch):
    from scripts import judge_eval

    result = {
        "adjacent": [("무역 사무원", 60, None, ["번역"], [], [], [], False)],
        "training": [("편집", 55, None, ["번역"], ["출판 편집"], [], [], False, "")],
        "roadmap": [("무역 사무원", [("무역 서류", "수출입 실무", None), ("영업", None, None)])],
    }

    async def fake_parse(schema, system, user, **kw):
        if "인접 직무" in user:
            return schema(scores=[{"item_no": 0, "score": 4, "reason": "번역 경험이 무역 서류와 이어짐"}])
        return None  # 채점 실패 → 점수 없음으로 남는다

    monkeypatch.setattr(judge_eval.llm, "_parse", fake_parse)
    out = asyncio.run(judge_eval.judge_persona(result, "번역 5년"))
    assert out["adjacent"][0]["score"] == 4
    assert out["training"][0]["score"] is None and len(out["roadmap"]) == 1  # 과정 없는 격차는 채점 대상 아님
