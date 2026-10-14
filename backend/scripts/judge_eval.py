"""E2E 결과 자동 채점 — 사람이 페르소나 결과를 하나씩 읽던 검토를 LLM 평가로 대신한다.

e2e_eval.py 가 저장한 결과 JSON을 읽어, 페르소나마다 경력 서사와 함께 아래를 1~5점 + 이유로 채점한다.
    - 인접 직무: 지금 경력으로 바로(재교육 없이) 옮겨 갈 수 있는 직무인가
    - 교육 후 전환 직무: 이 사람의 밑바탕 역량이 실제로 이어지는가
    - 로드맵 과정: 과정이 그 역량 격차를 채워 주는가

    cd backend
    python -m scripts.judge_eval e2e_results/20261014_0900_all.json
    python -m scripts.judge_eval e2e_results/20261014_0900_all.json --only translator

설정의 LLM 프로바이더(기본 로컬 Ollama)로 실행한다. 주의: 결과를 만든 모델과 같은 모델이 채점하면
자기 출력에 후한 경향이 있다 — 점수의 절대값보다 회차 간 비교와 '낮은 점수 항목 목록'을 보는 용도로 쓴다.
LLM 캐시는 쓰지 않는다(새 임시 캐시).
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LLM_CACHE_PATH"] = str(Path(tempfile.mkdtemp()) / "judge_cache.sqlite")

from pydantic import BaseModel, Field, create_model  # noqa: E402

from app.schemas.career import SurveyInput  # noqa: E402
from app.services import llm  # noqa: E402
from app.services.survey import compose_career_text  # noqa: E402
from scripts import personas  # noqa: E402

# 이 점수 이하인 항목은 요약에 따로 모은다 — 다음 회차에서 고칠 후보
LOW_SCORE = 2

JUDGE_SYSTEM = """당신은 커리어 상담 품질을 평가하는 검토자입니다.
한 사람의 경력 서사와, 시스템이 그 사람에게 추천한 항목들이 번호와 함께 주어집니다.
항목마다 아래 기준으로 1~5점을 매기고, 이유를 경력 서사의 사실을 들어 한 문장으로 적으세요.

5: 경력 서사의 구체적 경험과 분명히 이어진다
4: 대체로 이어지며 무리가 적다
3: 일부만 이어지거나 근거가 약하다
2: 연결이 억지스럽다
1: 이 사람의 경력과 관계가 없다

- 직무명·과정명이 그럴듯한지가 아니라, 이 사람의 경력과 실제로 이어지는지를 보세요.
- 모르는 부분은 점수를 높이지 말고 3 이하로 두세요.
- 모든 항목에 하나씩, item_no 에 그 항목 번호를 적으세요."""

CRITERIA = {
    "adjacent": "인접 직무 — 지금 경력으로 재교육 없이 옮겨 갈 수 있는 직무인가",
    "training": "교육 후 전환 직무 — 직업교육을 받으면, 이 사람의 밑바탕 역량이 실제로 이어지는 직무인가",
    "roadmap": "학습 과정 — 과정이 적힌 역량 격차를 실제로 채워 주는가",
}


def _schema(count: int) -> type[BaseModel]:
    item = create_model(
        "_ScoreOut",
        item_no=(Literal[tuple(range(count))], Field(description="채점하는 항목의 [번호]")),
        score=(Literal[1, 2, 3, 4, 5], Field(description="1~5점")),
        reason=(str, Field(description="경력 서사의 사실을 든 한 문장 이유")),
    )
    return create_model(
        "_ScoresOut", scores=(list[item], Field(min_length=count, max_length=count, description="항목마다 하나"))
    )


def _career_text(persona: dict) -> str:
    if "raw_text" in persona:
        return persona["raw_text"]
    return compose_career_text(SurveyInput(**{k: v for k, v in persona.items() if not k.startswith("_")}))


def _items(result: dict) -> dict[str, list[str]]:
    """결과 JSON(e2e_eval 형식) → 채점할 항목 문장. 튜플 위치는 e2e_eval.run_persona 와 같다."""
    adjacent = [f"{a[0]} (연결된 역량: {', '.join(a[3]) or '없음'})" for a in result["adjacent"]]
    training = [f"{t[0]} (이어지는 역량: {', '.join(t[3]) or '없음'} / 새로 배울 것: {', '.join(t[4]) or '없음'})"
                for t in result["training"]]
    roadmap = [f"격차 '{gap}' → 과정 '{course}' (목표 직무: {target})"
               for target, items in result["roadmap"] for gap, course, _ in items if course]
    return {"adjacent": adjacent, "training": training, "roadmap": roadmap}


async def judge_persona(result: dict, career: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for kind, items in _items(result).items():
        if not items:
            out[kind] = []
            continue
        listing = "\n".join(f"[{i}] {text}" for i, text in enumerate(items))
        parsed = await llm._parse(
            _schema(len(items)),
            JUDGE_SYSTEM,
            f"[경력 서사]\n{career}\n\n[평가 기준]\n{CRITERIA[kind]}\n\n[추천 항목 {len(items)}개]\n{listing}",
            local_max_tokens=120 * len(items) + 200,
        )
        scored: dict[int, dict] = {}
        for s in (parsed.scores if parsed else []):
            scored.setdefault(s.item_no, {"score": s.score, "reason": s.reason})
        out[kind] = [{"item": text, **scored.get(i, {"score": None, "reason": "채점 실패"})}
                     for i, text in enumerate(items)]
    return out


def _mean(xs: list[int]) -> str:
    return f"{sum(xs) / len(xs):.1f}" if xs else "-"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="e2e_eval.py 결과 JSON")
    ap.add_argument("--only", help="한 페르소나만")
    ap.add_argument("--out", help="채점 결과 JSON 경로(기본: 입력 파일명 + .judge.json)")
    args = ap.parse_args()

    results = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    all_personas = {k: v for s in personas.load().values() for k, v in s.items()}
    out_path = Path(args.out or Path(args.result_json).with_suffix(".judge.json"))
    judged: dict[str, dict] = {}
    for result in results:
        key = result["persona"]
        if args.only and key != args.only:
            continue
        if key not in all_personas:
            print(f"[건너뜀] {key}: 페르소나 정의를 찾지 못함", flush=True)
            continue
        judged[key] = await judge_persona(result, _career_text(all_personas[key]))
        out_path.write_text(json.dumps(judged, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[채점] {key}", flush=True)

    print("\n| 페르소나 | 인접 평균 | 전환 평균 | 과정 평균 | 낮은 점수(≤2) |")
    print("|---|---|---|---|---|")
    low: list[str] = []
    for key, kinds in judged.items():
        means = [_mean([x["score"] for x in kinds[k] if x["score"]]) for k in ("adjacent", "training", "roadmap")]
        lows = [f"{k}: {x['item']} ({x['score']}) — {x['reason']}"
                for k in kinds for x in kinds[k] if x["score"] and x["score"] <= LOW_SCORE]
        low += [f"{key} · {s}" for s in lows]
        print(f"| {key} | {means[0]} | {means[1]} | {means[2]} | {len(lows)} |")
    print(f"\n낮은 점수 항목 {len(low)}건:")
    for line in low:
        print(" -", line)
    print("\n채점 결과:", out_path)


if __name__ == "__main__":
    asyncio.run(main())
