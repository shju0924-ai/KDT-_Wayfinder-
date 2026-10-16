"""STEP 2 경력 → NCS 능력단위 연결(app/services/ncs_link.py) — DB·LLM·임베딩 없이."""
import asyncio

import numpy as np

from app.schemas.career import SurveyInput
from app.services import embedding, llm, ncs_link
from app.services.survey import compose_career_text, work_evidence_text


def test_survey_evidence_is_experience_only():
    """설문 서사에서 근거는 경험 서술만 — 강점(자기 평가)·고민·희망은 '해 본 일'이 아니다."""
    survey = SurveyInput(job_title="CNC 선반 조작원", years="10년 이상",
                         experience="CNC 선반 프로그램을 작성했습니다.\n\n신입 3명에게 장비 조작을 가르쳤습니다.",
                         strengths=["현장 대응"], concern="자동화", aspiration="설비를 다루는 경험을 살리고 싶어요")
    text = work_evidence_text(compose_career_text(survey))
    assert text == "CNC 선반 프로그램을 작성했습니다.\n\n신입 3명에게 장비 조작을 가르쳤습니다."
    assert work_evidence_text("이력서 원문 그대로") == "이력서 원문 그대로"


def test_clauses_split_on_connectives_and_listed_tasks():
    qs, n = ncs_link.queries_of("매입·매출 전표 입력, 월 결산, 부가세 신고 자료 준비를 했습니다. 거래처 미수금을 관리했고 "
                                "엑셀 매크로로 보고서 시간을 줄였습니다.")
    assert n == 3  # 문장 끝, '…했고' 뒤에서 나뉜다
    assert qs[n:] == ["매입·매출 전표 입력", "부가세 신고 자료 준비를 했습니다."]  # 쉼표 나열 업무는 항목마다(4자 미만 '월 결산' 제외)


def test_clauses_split_after_causative_connective():
    """'…시켜' 뒤에서도 나뉜다 — 서로 다른 일(민원 진정 / 해지 방어)이 한 구절에 묶이면 검색이 한쪽만 찾는다."""
    assert ncs_link.clauses("강성 민원 고객을 진정시켜 해지 방어율 팀 1위를 했습니다.") == [
        "강성 민원 고객을 진정시켜", "해지 방어율 팀 1위를 했습니다."]


def test_standard_terms_are_searched_without_job_prefix(monkeypatch):
    """현장 말은 LLM이 NCS식 표현으로 바꿔 검색어에 더한다 — 바꾼 표현에는 직무 맥락을 붙이지 않는다."""
    ix = _FakeIndex()
    monkeypatch.setattr(ncs_link, "get_index", lambda: ix)
    seen = {}

    async def fake_queries(texts):
        seen["queries"] = texts
        return [[0.0] * 3 for _ in texts]

    async def fake_run_db(fn, qvecs, ks):
        seen["ks"] = ks
        return [("A01", 0.6, 0)]

    async def fake_parse(schema, system, user, **kw):
        if schema is ncs_link._RewriteOut:
            return schema(terms=["고객 이탈 방지와 고객 유지 관리", "고객 이탈 방지와 고객 유지 관리", "짧음"])
        return schema(picks=[])

    monkeypatch.setattr(embedding, "embed_queries", fake_queries)
    monkeypatch.setattr(embedding, "run_db", fake_run_db)
    monkeypatch.setattr(llm, "_parse", fake_parse)
    asyncio.run(ncs_link.link_detail("해지 방어율 팀 1위를 했습니다.", "콜센터 상담원"))
    # 중복·4자 미만은 뺀다
    assert seen["queries"] == ["콜센터 상담원: 해지 방어율 팀 1위를 했습니다.", "고객 이탈 방지와 고객 유지 관리"]
    assert seen["ks"] == [ncs_link.TOP_K, ncs_link.REWRITE_K]


def test_standard_terms_drop_performance_phrases(monkeypatch):
    """실적·순위 표현은 업무가 아니라 검색어에서 뺀다 — 업무를 가리키는 '…률 개선'은 남긴다."""
    async def fake_parse(schema, system, user, **kw):
        return schema(terms=["강성 민원 고객 응대", "해지 방어율 팀 1위 달성", "판매 목표 120% 달성",
                             "매출 순위 관리", "불량률 개선 활동 수행"])

    monkeypatch.setattr(llm, "_parse", fake_parse)
    assert asyncio.run(ncs_link._standard_terms(["구절"])) == ["강성 민원 고객 응대", "불량률 개선 활동 수행"]


class _FakeIndex:
    def __init__(self):
        names = [("A01", "전표관리", "회계·감사"), ("A02", "결산처리", "회계·감사"), ("B01", "출고관리", "보관·하역관리")]
        self.units = [{"code": c, "job": c[0], "job_title": j, "name": n, "def": f"{n} 정의", "path": f"경영 > {j}"}
                      for c, n, j in names]
        self.pos = {u["code"]: i for i, u in enumerate(self.units)}
        self.vecs = np.eye(3, dtype=np.float32)
        self.by_job = {"A": [0, 1], "B": [2]}
        self.jobs = ["A", "B"]
        self.job_title = {"A": "회계·감사", "B": "보관·하역관리"}
        self.job_path = {"A": "경영 > 회계·감사", "B": "경영 > 보관·하역관리"}
        self.weight = np.ones(3)

    item = ncs_link.Index.item


def test_link_keeps_only_quoted_picks_and_adds_diagnosed_codes(monkeypatch):
    ix = _FakeIndex()
    monkeypatch.setattr(ncs_link, "get_index", lambda: ix)

    async def fake_queries(texts):
        return [[0.0] * 3 for _ in texts]

    async def fake_run_db(fn, *args):
        return [("A01", 0.6, 0)]  # 검색 후보 — 진단 업무 코드(A02)는 뒤에 더해진다

    seen = {}

    async def fake_parse(schema, system, user, **kw):
        if schema is ncs_link._RewriteOut:  # 검색어 보강 — 결과는 fake_run_db 가 무시한다
            return schema(terms=["결산 처리"])
        seen["user"] = user
        return schema(picks=[
            {"source_quote": "매입·매출 전표 입력", "unit_no": 0},
            {"source_quote": "매입·매출 전표 입력", "unit_no": 0},     # 같은 능력단위 두 번 — 하나만
            {"source_quote": "원문에 없는 구절입니다", "unit_no": 1},  # 지어낸 인용 — 뺀다
        ])

    monkeypatch.setattr(embedding, "embed_queries", fake_queries)
    monkeypatch.setattr(embedding, "run_db", fake_run_db)
    monkeypatch.setattr(llm, "_parse", fake_parse)
    skills = asyncio.run(ncs_link.link_career("매입·매출 전표 입력과 월 결산을 했습니다.", None, ["A02", "Z99"]))
    assert "[1] 결산처리 (회계·감사)" in seen["user"] and "Z99" not in seen["user"]
    assert [(s.ncs_code, s.name, s.ncs_job, s.evidence) for s in skills] == [("A01", "전표관리", "회계·감사", "매입·매출 전표 입력")]


def test_unpicked_candidates_become_suggestions_with_original_evidence(monkeypatch):
    """AI가 고르지 않은 후보는 유사도 순으로 사용자에게 보여 준다 — 근거는 바꿔 쓴 표현이 아니라 원문 구절."""
    ix = _FakeIndex()
    monkeypatch.setattr(ncs_link, "get_index", lambda: ix)
    career = "매입·매출 전표 입력을 했습니다. 거래처 출고 일정을 조율했습니다."

    async def fake_queries(texts):
        # 구절 2개 + 바꿔 쓴 표현 1개 — 바꾼 표현은 두 번째 구절에 가깝다
        return [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]][: len(texts)]

    async def fake_run_db(fn, qvecs, ks):
        return [("A01", 0.70, 0), ("A02", 0.55, 0), ("B01", 0.65, 2)]

    monkeypatch.setattr(ncs_link, "SUGGEST_PER_CLAUSE", 1)  # 구절당 상한도 함께 확인

    async def fake_parse(schema, system, user, **kw):
        if schema is ncs_link._RewriteOut:
            return schema(terms=["출고 일정 관리"])
        return schema(picks=[{"source_quote": "매입·매출 전표 입력을 했습니다.", "unit_no": 0}])

    monkeypatch.setattr(embedding, "embed_queries", fake_queries)
    monkeypatch.setattr(embedding, "run_db", fake_run_db)
    monkeypatch.setattr(llm, "_parse", fake_parse)
    r = asyncio.run(ncs_link.link_detail(career, None))
    assert [s.ncs_code for s in r["skills"]] == ["A01"]
    assert [(s.ncs_code, s.evidence) for s in r["suggestions"]] == [
        ("B01", "거래처 출고 일정을 조율했습니다."),  # 유사도 높은 순, 바꾼 표현 대신 가장 가까운 원문 구절
        ("A02", "매입·매출 전표 입력을 했습니다."),
    ]
    assert not any(s.user_added for s in r["suggestions"])


def test_rank_jobs_scores_jobs_with_matching_units():
    ix = _FakeIndex()
    rows = ncs_link.rank_jobs(ix, ["A01"])
    assert rows[0]["code"] == "A" and rows[0]["pairs"][0][:2] == ("전표관리", "전표관리")
    assert all(r["code"] != "B" for r in rows)  # 관계없는 직무는 점수 0이라 빠진다
