"""4단계 E2E 회귀 점검 — 여러 페르소나로 전체 흐름을 돌려 품질 지표를 표로 낸다.

LLM·임베딩·DB가 모두 떠 있어야 한다(런팟 터널 또는 로컬). 앱을 프로세스 안에서 ASGI로 호출한다.

    cd backend
    python -m scripts.e2e_eval              # 전체 페르소나
    python -m scripts.e2e_eval --set v2     # 두 번째 페르소나 세트
    python -m scripts.e2e_eval --only md    # 하나만
    python -m scripts.e2e_eval --json out.json

LLM 캐시를 거치면 실제 품질·시간이 아니라 이전 결과를 보게 되므로, 이 스크립트는 새 임시 캐시를 쓴다.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LLM_CACHE_PATH"] = str(Path(tempfile.mkdtemp()) / "eval_cache.sqlite")

import re  # noqa: E402

import httpx  # noqa: E402

from app.main import app  # noqa: E402

PERSONAS: dict[str, dict] = {
    "md": {
        "job_title": "온라인 쇼핑몰 MD",
        "years": "5년 이상",
        "experience": "자사몰과 오픈마켓 상품 기획을 맡아 판매 데이터를 엑셀로 분석해 주간 프로모션을 짰고, "
                      "협력사 20곳과 입고 일정·단가를 협상했습니다. 상세페이지 개편으로 전환율을 1.4%에서 2.1%로 올렸습니다.",
        "strengths": ["데이터 분석", "협상", "기획"],
        "concern": "AI 추천·자동 발주가 늘면서 MD 업무가 줄어들까 걱정됩니다.",
        "aspiration": "데이터를 더 깊게 다루는 일",
    },
    "callcenter": {
        "job_title": "콜센터 상담원",
        "years": "3년 이상",
        "experience": "통신사 인바운드 상담을 하며 하루 80콜 이상 요금·해지 문의를 처리했고, 강성 민원 고객을 진정시켜 "
                      "해지 방어율 팀 1위를 했습니다. 신입 상담원 5명의 응대 스크립트 교육도 맡았습니다.",
        "strengths": ["고객 응대", "갈등 해결", "교육"],
        "concern": "AI 챗봇·음성봇 도입으로 상담 인력이 줄고 있습니다.",
        "aspiration": "사람을 상대하는 강점을 살리고 싶어요",
    },
    "accounting": {
        "job_title": "회계 사무원",
        "years": "4년 이상",
        "experience": "중소 제조업체에서 매입·매출 전표 입력, 월 결산, 부가세 신고 자료 준비를 했습니다. "
                      "더존 ERP를 쓰며 거래처 200곳의 미수금을 관리했고, 엑셀 매크로로 결산 보고서 작성 시간을 절반으로 줄였습니다.",
        "strengths": ["꼼꼼함", "엑셀", "숫자 감각"],
        "concern": "회계 자동화 프로그램 때문에 단순 입력 업무가 사라지고 있습니다.",
        "aspiration": None,
    },
    "production": {
        "job_title": "생산관리 담당자",
        "years": "6년 이상",
        "experience": "자동차 부품 공장에서 주간 생산계획을 세우고 라인별 설비 가동률과 불량률을 관리했습니다. "
                      "협력업체 자재 납기를 조율했고, 공정 개선 TF로 불량률을 3.2%에서 1.8%로 낮췄습니다.",
        "strengths": ["일정 관리", "문제 해결", "협업"],
        "concern": "스마트팩토리 도입으로 관리 업무가 시스템으로 대체될까 걱정됩니다.",
        "aspiration": "데이터 기반 공정 관리",
    },
}


# 두 번째 세트 — 첫 세트에 맞춘 조정이 다른 경력에서도 통하는지 확인한다(과적합 점검).
# 'raw_text' 가 있으면 설문 대신 이력서 텍스트 경로(직무명 입력 없음 → STEP 2가 직무명을 추출)로 돌린다.
PERSONAS_V2: dict[str, dict] = {
    "insurance": {
        "job_title": "보험설계사",
        "years": "5년 이상",
        "experience": "생명보험사 전속 설계사로 고객 300명의 보장 분석과 상품 설계를 했고, 지인 소개 위주로 연간 신규 계약 "
                      "120건을 체결했습니다. 고객 상담 내용을 엑셀로 정리해 갱신 시기를 관리했고 지점 신입 교육도 했습니다.",
        "strengths": ["고객 응대", "협업·조율", "문서·데이터 관리"],
        "concern": "비대면 보험 가입과 AI 상품 추천으로 대면 영업이 줄고 있습니다.",
        "aspiration": "금융 지식을 살려 상담이나 분석 쪽으로 가고 싶어요",
    },
    "nurse_aide": {
        "job_title": "간호조무사",
        "years": "3년 이상",
        "experience": "내과 의원에서 하루 환자 100명의 접수·예약과 활력징후 측정, 주사·검사 보조를 했습니다. "
                      "EMR로 진료 기록과 보험 청구 자료를 입력했고, 고령 환자 보호자 응대와 민원 처리를 맡았습니다.",
        "strengths": ["고객 응대", "현장 대응", "문서·데이터 관리"],
        "concern": "무인 접수기와 AI 문진이 들어오면서 접수 업무가 줄고 있습니다.",
        "aspiration": None,
    },
    "cnc": {
        "job_title": "CNC 선반 조작원",
        "years": "10년 이상",
        "experience": "정밀 기계부품 공장에서 CNC 선반 프로그램을 작성하고 공구 세팅·가공 조건을 조정했습니다. "
                      "도면을 보고 공차를 맞추며 불량 원인을 찾아 개선했고, 신입 작업자 3명에게 장비 조작을 가르쳤습니다.",
        "strengths": ["문제 해결", "현장 대응", "운영 개선"],
        "concern": "자동화 라인이 늘면서 단순 가공 인력이 줄고 있습니다.",
        "aspiration": "설비를 다루는 경험을 살리고 싶어요",
    },
    "web_publisher": {
        "raw_text": "경력 요약\n웹 퍼블리셔 (4년)\n"
                    "- 에이전시에서 쇼핑몰·기업 홈페이지 30여 건의 HTML/CSS 마크업과 반응형 작업 담당\n"
                    "- jQuery로 인터랙션 구현, 웹 접근성 인증 마크 획득 프로젝트 2건 참여\n"
                    "- 디자이너·개발자 사이에서 화면 설계서 검토와 일정 조율\n"
                    "- 최근 노코드 빌더와 AI 코드 생성 도구로 퍼블리싱 의뢰가 줄어 프론트엔드 개발로 전환 고민 중",
    },
}


_HANJA = re.compile(r"[\u4e00-\u9fff]")


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


async def run_persona(c: httpx.AsyncClient, key: str, survey: dict) -> dict:
    t0 = time.perf_counter()
    times: dict[str, float] = {}

    async def timed(name, coro):
        t = time.perf_counter()
        r = await coro
        times[name] = round(time.perf_counter() - t, 1)
        r.raise_for_status()
        return r.json()

    if "raw_text" in survey:  # 이력서 텍스트 경로 — 직무명 없이 STEP 2부터
        career, job_title = survey["raw_text"], None
    else:
        career = (await timed("s1", c.post("/api/diagnosis/survey", json=survey)))["career_text"]
        job_title = survey["job_title"]
    career_input = {"raw_text": career, "current_job_title": job_title}
    # STEP 1 진단과 STEP 2 역량 분해는 같은 경력 서사만 쓰므로 동시에 보낸다
    risk, profile = await asyncio.gather(
        timed("s1_risk", c.post("/api/diagnosis/risk", json=career_input)),
        timed("s2", c.post("/api/profile", json=career_input)),
    )
    # 두 트랙·로드맵은 서로 독립 — 동시에 보내 GPU 병렬 처리(OLLAMA_NUM_PARALLEL)를 쓴다
    adjacent, training = await asyncio.gather(
        timed("s3_adj", c.post("/api/jobs/match?search_track=adjacent_transition", json=profile)),
        timed("s3_trn", c.post("/api/jobs/match?search_track=training_transition", json=profile)),
    )
    roadmaps = await asyncio.gather(*(
        timed(f"s4_{i}", c.post("/api/roadmap", json={"profile": profile, "target_job": job["job_title"],
                                                      "missing_skills": job.get("missing_skills") or []}))
        for i, job in enumerate(adjacent[:1] + training[:1])
    ))
    items = [i for r in roadmaps for i in r["items"]]
    return {
        "persona": key,
        "times": times,
        "total": round(time.perf_counter() - t0, 1),
        "current_job": profile.get("current_job_title"),
        "risk": {
            "score": risk["risk_score"], "task_based": risk["task_based_score"], "confidence": risk["confidence"],
            "shares": (risk["automation_share"], risk["augmentation_share"], risk["human_centered_share"]),
            "tasks": [(t["name"], t["share_percent"], t["automation_score"], t["effect"], t["ncs_unit"]) for t in risk["tasks"]],
            "sources": [s["source"] for s in risk["sources"]],
        },
        "skills": [(s["name"], s["category"]) for s in profile["skills"]],
        "adjacent": [(j["job_title"], j["fit_score"], j["ai_exposure_score"], j["matched_skills"], j["missing_skills"], j["weak_reasons"]) for j in adjacent],
        "training": [(j["job_title"], j["fit_score"], j["ai_exposure_score"], j["missing_skills"], j["weak_reasons"]) for j in training],
        "roadmap": [(r["target_job"], [(i["skill_gap"], i["course"]["name"] if i["course"] else None, i.get("weak_reason")) for i in r["items"]])
                    for r in roadmaps],
        "metrics": {
            "adjacent_n": len(adjacent),
            "training_n": len(training),
            "training_fit_distinct": len({j["fit_score"] for j in training}),
            "roadmap_items": len(items),
            "roadmap_with_course": sum(1 for i in items if i["course"]),
            "jobs_with_exposure": sum(1 for j in adjacent + training if j["ai_exposure_score"] is not None),
            "jobs_total": len(adjacent) + len(training),
            # 화면에 보이는 모든 문자열에서 한자가 남았는지 — 한국어 전용 출력 점검
            "hanja_left": len(_HANJA.findall(json.dumps([profile, adjacent, training, roadmaps, risk], ensure_ascii=False))),
            "weak_jobs": sum(1 for j in adjacent + training if j["weak_reasons"]),
            "weak_courses": sum(1 for i in items if i.get("weak_reason")),
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["v1", "v2", "all"], default="v1", help="페르소나 세트 (all = 8명)")
    ap.add_argument("--only")
    ap.add_argument("--json")
    args = ap.parse_args()

    personas = {"v1": PERSONAS, "v2": PERSONAS_V2, "all": {**PERSONAS, **PERSONAS_V2}}[args.set]
    collector = _Collect()
    logging.getLogger("app").addHandler(collector)
    logging.getLogger("app").setLevel(logging.DEBUG if os.environ.get("EVAL_DEBUG") else logging.INFO)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=1800) as c:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(
            run_persona(c, key, survey) for key, survey in personas.items() if not args.only or key == args.only
        ))
        for res in results:
            print(f"\n##### {res['persona']}  ({res['total']}s) {res['times']}")
            print("현재 직무:", res["current_job"], "| 역량:", res["skills"])
            print("진단:", {k: v for k, v in res["risk"].items() if k != "tasks"})
            for t in res["risk"]["tasks"]:
                print("  업무:", t)
            for a in res["adjacent"]:
                print("  인접:", a)
            for t in res["training"]:
                print("  훈련:", t)
            for r in res["roadmap"]:
                print("  로드맵:", r)
            print("  지표:", res["metrics"])
        # 페르소나를 동시에 돌리므로 로그는 섞여 있다 — 직무명으로 구분한다
        for line in collector.lines:
            print("log:", line)
        print(f"\n전체 {time.perf_counter() - t0:.1f}s")

    print("\n| 페르소나 | 시간(s) | 진단 점수 | 인접 | 훈련 | 로드맵 과정 연결 | 직무 노출도 | 근거 약함(직무/과정) | 한자 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r["metrics"]
        print(f"| {r['persona']} | {r['total']} | {r['risk']['score']} | {m['adjacent_n']} | {m['training_n']} | "
              f"{m['roadmap_with_course']}/{m['roadmap_items']} | {m['jobs_with_exposure']}/{m['jobs_total']} | "
              f"{m['weak_jobs']}/{m['weak_courses']} | {m['hanja_left']} |")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
