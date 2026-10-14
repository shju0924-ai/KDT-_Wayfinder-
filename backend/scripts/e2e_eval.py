"""4단계 E2E 회귀 점검 — 여러 페르소나로 전체 흐름을 돌려 품질 지표를 표로 낸다.

LLM·임베딩·DB가 모두 떠 있어야 한다(런팟 터널 또는 로컬). 앱을 프로세스 안에서 ASGI로 호출한다.

    cd backend
    python -m scripts.e2e_eval              # 첫 번째 페르소나 세트(v1)
    python -m scripts.e2e_eval --set v2     # 두 번째 세트 — v1 과적합 점검
    python -m scripts.e2e_eval --set resume # 이력서 원문 경로
    python -m scripts.e2e_eval --set ui     # 시작 화면 예시 입력
    python -m scripts.e2e_eval --set resume,ui   # 여러 세트
    python -m scripts.e2e_eval --only md    # 하나만
    python -m scripts.e2e_eval --json out.json
    python -m scripts.e2e_eval --set all --sequential   # 로컬 CPU — 한 명씩, 요청도 하나씩

페르소나는 scripts/personas/personas.json(세트·작성 기준은 같은 폴더 README.md)에 있다.

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
from scripts import personas  # noqa: E402

_HANJA = re.compile(r"[\u4e00-\u9fff]")
# \uc545\uc13c\ud2b8 \ub77c\ud2f4 \ubb38\uc790(\uc2a4\ud398\uc778\uc5b4 \ub4f1) \u2014 \uc601\uc5b4 \ub3c4\uad6c\uba85\uc740 \ud5c8\uc6a9\ud558\ubbc0\ub85c \uc774 \ubc94\uc704\ub9cc \uc13c\ub2e4
_ACCENTED = re.compile(r"[\u00c0-\u024f]")


def _llm_texts(profile: dict, adjacent: list, training: list, roadmaps: list, risk: dict) -> str:
    """LLM \uc774 \ub9cc\ub4e0 \ubb38\uc790\uc5f4\ub9cc \ubaa8\uc740\ub2e4 \u2014 \uacf5\uace0 \uc9c1\ubb34\uba85\u00b7\uacfc\uc815\uba85 \uac19\uc740 DB \uc6d0\ubb38\uc744 \ube7c\uc57c \ud55c\uad6d\uc5b4 \ucd9c\ub825 \uc810\uac80\uc774 \uc624\ud0d0\ud558\uc9c0 \uc54a\ub294\ub2e4
    (\uc2e4\uce21: \uc2e4\uc81c \uacfc\uc815\uba85 '\uc138\ubb34\ub97c \uc54c\uba74 \ube44\uc988\ub2c8\uc2a4\uac00 \u901a\ud55c\ub2e4'\uc758 '\u901a'\uc774 \ud55c\uc790 2\uac74\uc73c\ub85c \uc7a1\ud614\ub2e4)."""
    parts: list[str] = [risk.get("job_title") or ""]
    parts += [f"{t['name']} {t['rationale']}" for t in risk["tasks"]]
    parts += [s.get("evidence") or "" for s in profile["skills"]]  # 역량명은 NCS 원문, 근거 인용만 LLM 출력
    parts += [" ".join(j["matched_skills"] + j["missing_skills"]) for j in adjacent]
    parts += [" ".join([j["job_title"], *j["missing_skills"]]) for j in training]  # \uad50\uc721 \ud2b8\ub799 \uc9c1\ubb34\uba85\uc740 LLM \uc81c\uc548
    parts += [i["skill_gap"] for r in roadmaps for i in r["items"]]
    return "\n".join(parts)


# 접속어에서 끊긴 이름 — '매장 재고 정리 및'
_DANGLING = re.compile(r"(및|과|와|또는|그리고|[,·/])\s*$")
# 결과 기본 저장 위치 — 3차는 --json 없이 돌려 원인 분석에 쓸 원자료가 남지 않았다
RESULT_DIR = Path(__file__).resolve().parents[1] / "e2e_results"
# 이 수 이상의 페르소나에게 같은 이름으로 나온 교육 후 전환 직무는 '범용 반복'으로 집계한다
REPEATED_TARGET_MIN = 3


def _dangling_names(risk: dict, profile: dict, adjacent: list, training: list) -> list[str]:
    names = [t["name"] for t in risk["tasks"]] + [s["name"] for s in profile["skills"]]
    names += [x for j in adjacent + training for x in j["missing_skills"]]
    return [n for n in names if _DANGLING.search(n)]


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


async def _gather(sequential: bool, *coros):
    """병렬(GPU) 또는 순차(로컬 CPU) 실행. 로컬 Ollama 는 요청을 하나씩 처리해, 동시에 보내면
    대기열에서 기다린 시간까지 호출 타임아웃(1800초)에 걸려 뒤쪽 요청이 502로 끝난다."""
    if not sequential:
        return await asyncio.gather(*coros)
    return [await co for co in coros]


async def run_persona(c: httpx.AsyncClient, key: str, survey: dict, sequential: bool = False) -> dict:
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
    # 화면과 같은 순서 — STEP 2 능력단위 연결은 STEP 1 진단 업무의 NCS 코드도 후보로 받는다
    risk = await timed("s1_risk", c.post("/api/diagnosis/risk", json=career_input))
    profile = await timed("s2", c.post("/api/profile", json={**career_input, "diagnosed_tasks": risk["tasks"]}))
    # 화면(Chat.tsx)처럼 이력서 경로는 STEP 1 진단의 직무명을 현재 직무로 쓴다 — STEP 3 같은 직무 제외
    profile["current_job_title"] = profile.get("current_job_title") or risk.get("job_title")
    # 두 트랙·로드맵은 서로 독립 — 동시에 보내 GPU 병렬 처리(OLLAMA_NUM_PARALLEL)를 쓴다
    adjacent, training = await _gather(
        sequential,
        timed("s3_adj", c.post("/api/jobs/match?search_track=adjacent_transition", json=profile)),
        timed("s3_trn", c.post("/api/jobs/match?search_track=training_transition", json=profile)),
    )
    roadmaps = await _gather(sequential, *(
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
            "tasks": [(t["name"], t["share_percent"], t["automation_score"], t["effect"], t["ncs_unit"], t.get("source_quote")) for t in risk["tasks"]],
            "sources": [s["source"] for s in risk["sources"]],
        },
        "skills": [(s["name"], s["ncs_job"], s["evidence"]) for s in profile["skills"]],
        "adjacent": [(j["job_title"], j["fit_score"], j["ai_exposure_score"], j["matched_skills"], j["missing_skills"],
                      j["weak_reasons"], [(s["source"], s["label"], s["score"]) for s in j["ai_exposure_sources"]],
                      j["license_required"]) for j in adjacent],
        "training": [(j["job_title"], j["fit_score"], j["ai_exposure_score"], j["matched_skills"], j["missing_skills"],
                      j["weak_reasons"], [(s["source"], s["label"], s["score"]) for s in j["ai_exposure_sources"]],
                      j["license_required"], j.get("requirement_excerpt")) for j in training],
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
            # LLM 이 만든 문자열에 한자·외국어가 남았는지 — 한국어 전용 출력 점검
            "hanja_left": len(_HANJA.findall(llm_text := _llm_texts(profile, adjacent, training, roadmaps, risk))),
            "foreign_left": len(_ACCENTED.findall(llm_text)),
            "weak_jobs": sum(1 for j in adjacent + training if j["weak_reasons"]),
            "weak_courses": sum(1 for i in items if i.get("weak_reason")),
            # 3차까지 사람이 세던 지표 — 노출도 0.0·미연결, 잘린 이름, 면허 직업
            "exposure_zero": sum(1 for j in adjacent + training if j["ai_exposure_score"] == 0),
            "exposure_missing": sum(1 for j in adjacent + training if j["ai_exposure_score"] is None),
            "dangling_names": _dangling_names(risk, profile, adjacent, training),
            "licensed_jobs": [j["job_title"] for j in adjacent + training if j["license_required"]],
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    # 불러오면서 모든 페르소나를 검증한다 — 규칙을 어긴 페르소나가 있으면 실행 전에 멈춘다
    sets = personas.load()
    ap.add_argument("--set", default="v1",
                    help=f"페르소나 세트, 쉼표로 여러 개 또는 all "
                         f"({', '.join(f'{k} {len(v)}명' for k, v in sets.items())})")
    ap.add_argument("--only")
    ap.add_argument("--json", help="결과 JSON 경로(기본 e2e_results/<날짜시각>_<세트>.json) — 페르소나가 끝날 때마다 "
                                   "덮어써 중간에 멈춰도 끝난 결과는 남는다. 같은 이름 .log.txt 에 앱 로그도 남긴다")
    ap.add_argument("--sequential", action="store_true", help="페르소나·요청을 하나씩 실행 (로컬 CPU용)")
    args = ap.parse_args()

    try:
        selected_set = personas.select(args.set)
    except personas.PersonaError as e:
        ap.error(str(e))
    done: list[dict] = []
    if not args.json:
        stamp = time.strftime("%Y%m%d_%H%M")
        only = f"_{args.only}" if args.only else ""
        args.json = str(RESULT_DIR / f"{stamp}_{args.set.replace(',', '+')}{only}.json")
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    print("결과 저장:", json_path, flush=True)
    collector = _Collect()

    def save_progress() -> None:
        json_path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        json_path.with_suffix(".log.txt").write_text("\n".join(collector.lines), encoding="utf-8")

    logging.getLogger("app").addHandler(collector)
    logging.getLogger("app").setLevel(logging.DEBUG if os.environ.get("EVAL_DEBUG") else logging.INFO)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=1800) as c:
        t0 = time.perf_counter()
        selected = [(k, v) for k, v in selected_set.items() if not args.only or k == args.only]

        async def safe_run(key: str, survey: dict) -> dict | None:
            # 한 명이 실패해도 나머지 결과는 남긴다 — 로컬에서는 전체가 몇 시간짜리다
            try:
                res = await run_persona(c, key, survey, args.sequential)
            except Exception as e:
                print(f"[실패] {key}: {e!r}", flush=True)
                return None
            print(f"[완료] {key} {res['total']}s", flush=True)
            done.append(res)
            save_progress()
            return res

        if args.sequential:
            outcomes = [await safe_run(k, v) for k, v in selected]
        else:
            outcomes = await asyncio.gather(*(safe_run(k, v) for k, v in selected))
        results = [r for r in outcomes if r is not None]
        failed = [k for (k, _), r in zip(selected, outcomes) if r is None]
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
        if failed:
            print("실패한 페르소나:", ", ".join(failed))

    print("\n| 페르소나 | 시간(s) | 진단 점수 | 근거 충족도 | 인접 | 훈련 | 로드맵 과정 연결 | 직무 노출도 "
          "| 노출도 0.0 | 근거 약함(직무/과정) | 한자/외국어 | 잘린 이름 | 면허 직업 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r["metrics"]
        print(f"| {r['persona']} | {r['total']} | {r['risk']['score']} | {r['risk']['confidence']} | "
              f"{m['adjacent_n']} | {m['training_n']} | "
              f"{m['roadmap_with_course']}/{m['roadmap_items']} | {m['jobs_with_exposure']}/{m['jobs_total']} | "
              f"{m['exposure_zero']} | {m['weak_jobs']}/{m['weak_courses']} | {m['hanja_left']}/{m['foreign_left']} | "
              f"{', '.join(m['dangling_names']) or '-'} | {', '.join(m['licensed_jobs']) or '-'} |")

    if results:
        linked = sum(r["metrics"]["jobs_with_exposure"] for r in results)
        total = sum(r["metrics"]["jobs_total"] for r in results)
        adjacent = sum(r["metrics"]["adjacent_n"] for r in results)
        print(f"\n노출도 연결 {linked}/{total} ({linked / (total or 1) * 100:.0f}%) · "
              f"노출도 0.0 {sum(r['metrics']['exposure_zero'] for r in results)}건 · "
              f"인접 직무 평균 {adjacent / len(results):.1f}")
        # 여러 페르소나에게 같은 전환 직무가 나오면 경력을 반영하지 못한 범용 제안일 가능성이 크다
        target_personas: dict[str, set[str]] = {}
        for r in results:
            for t in r["training"]:
                target_personas.setdefault(t[0], set()).add(r["persona"])
        repeated = {t: p for t, p in target_personas.items() if len(p) >= REPEATED_TARGET_MIN}
        print(f"{REPEATED_TARGET_MIN}명 이상에게 반복된 전환 직무 {len(repeated)}종:",
              "; ".join(f"{t}({len(p)}명)" for t, p in sorted(repeated.items(), key=lambda x: -len(x[1]))) or "없음")
    done[:] = results  # 병렬 실행은 끝난 순서로 쌓였다 — 마지막에 페르소나 순서로 다시 쓴다
    save_progress()


if __name__ == "__main__":
    asyncio.run(main())
