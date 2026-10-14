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
        return ["A01"]  # 검색 후보 — 진단 업무 코드(A02)는 뒤에 더해진다

    seen = {}

    async def fake_parse(schema, system, user, **kw):
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


def test_rank_jobs_scores_jobs_with_matching_units():
    ix = _FakeIndex()
    rows = ncs_link.rank_jobs(ix, ["A01"])
    assert rows[0]["code"] == "A" and rows[0]["pairs"][0][:2] == ("전표관리", "전표관리")
    assert all(r["code"] != "B" for r in rows)  # 관계없는 직무는 점수 0이라 빠진다
