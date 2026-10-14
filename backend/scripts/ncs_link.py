"""STEP 2 경력 → NCS 능력단위 연결 평가 도구 — 페르소나 연결·추천을 만들어 사람이 판정할 정답 데이터를 쌓는다.

연결·점수 계산은 서비스와 같은 코드(app/services/ncs_link.py)·같은 DB 벡터(ncs_units.embedding*)를 쓴다.
능력단위 벡터는 data-pipeline 에서 `python -m embedding.embed_ncs` 로 채운다.

    cd backend
    python -m scripts.ncs_link link                        # 페르소나 15명 연결 → links.json (1명 30~100초)
    python -m scripts.ncs_link link --only cnc,md --force  # 일부만 다시(이미 판정이 붙은 페르소나는 --force 필요)
    python -m scripts.persona_label_server                 # 사람 판정(정답 데이터) — 브라우저 라벨링 도구

결과(data-pipeline/data/eval/ncs_link/ — .gitignore 대상): links.json(연결), labels.json(사람 판정)
ncs_postings.jsonl(직무별 채용 공고 수 — 지금은 지운 이전 평가 도구가 만든 참고 자료, 없으면 0으로 본다)

실험 기록(페르소나 15명, 연결 판정 O=맞음 / D=내용은 맞지만 다른 업종 능력단위 / X=틀림)
  처음엔 Claude 판정으로 방법을 골랐다 — 사람 판정은 Claude보다 너그러웠다(Claude가 D·X로 둔 16개 중 13개가 O).
  기준(분야 없는 임베딩)            연결 44 — O 26 / D 10 / X 8
  ① 분야 경로를 넣은 임베딩(채택)   연결 45 — O 31 / D 9 / X 5, 엉뚱한 분야 연결 사례 5건 모두 해결
                                     → 사람 판정: O 40 / D 4 / X 1 (89%), 추천 TOP5 정확도 76%(전환 쉬움+가능)
  ② 가까운 직무 20개 안에서만 검색  X 10 — 처음 고른 직무가 틀리면 후보가 통째로 쏠림
  ③ 분야 경로를 보이고 업종 먼저 판단  X 7, 시간 1.8배 / ①+③ O 32 / X 9 — 더 넉넉히 고르는 만큼 틀림도 늘었다
  ④ 연결을 LLM이 다시 검증         X 1 이지만 O 도 26 → 18, 2명은 연결이 비었다
  ⑥ 비슷한 능력단위 묶음으로 보이기  변화 없음
  직무 점수에 직무 쪽 가중치(흔한 능력단위 약하게)를 더하자 사람 판정 TOP5 정확도 69% → 76%
  로컬 LLM은 온도 0이어도 다시 돌리면 연결이 달라진다(같은 입력 15명 중 8명) — 판정은 저장된 links.json 기준
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LLM_CACHE_PATH"] = str(Path(tempfile.mkdtemp()) / "ncs_link_cache.sqlite")

from app.schemas.career import SurveyInput  # noqa: E402
from app.services import ncs_link as svc  # noqa: E402
from app.services.survey import compose_career_text  # noqa: E402
from scripts import personas  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "data-pipeline" / "data" / "eval" / "ncs_link"
LINKS_PATH = OUT_DIR / "links.json"
LABELS_PATH = OUT_DIR / "labels.json"
POSTINGS_PATH = OUT_DIR / "ncs_postings.jsonl"
MIN_POSTINGS = 5  # 추천 목록 기본 필터 — 공고가 이 수 미만인 직무는 수요 근거가 약하다


def load_index() -> svc.Index:
    """서비스 색인(DB) + 직무별 채용 공고 수(화면에 수요 참고로 보인다)."""
    ix = svc.get_index()
    ix.postings = {}
    if POSTINGS_PATH.exists():
        for line in POSTINGS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                ix.postings[r["ncs_code"]] = r["assigned_count"]
    return ix


def rank_jobs(ix: svc.Index, codes: list[str]) -> list[dict]:
    """서비스 직무 점수 + 공고 수."""
    return [{**x, "postings": ix.postings.get(x["code"], 0)} for x in svc.rank_jobs(ix, codes)]


def career_input(p: dict) -> tuple[str, str | None]:
    """페르소나 → 화면이 STEP 2에 보내는 것과 같은 (경력 서사, 직무명). 설문형은 서사로 조립하고 직무명을 함께 보낸다."""
    if "raw_text" in p:
        return p["raw_text"], None
    s = SurveyInput.model_validate(p)
    return compose_career_text(s), s.job_title


async def run_link(only: list[str] | None, force: bool) -> None:
    ix = load_index()
    ps = personas.select("all")
    keys = only or list(ps)
    unknown = [k for k in keys if k not in ps]
    if unknown:
        raise SystemExit(f"없는 페르소나: {unknown} — 사용 가능: {', '.join(ps)}")
    saved = json.loads(LINKS_PATH.read_text(encoding="utf-8")) if LINKS_PATH.exists() else {}
    # 사람 판정(labels.json)은 그때의 연결에 붙어 있으므로, 이미 있는 페르소나는 --force 없이는 덮어쓰지 않는다
    labeled = json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else {}
    keep = [k for k in keys if k in saved and k in labeled and not force]
    if keep:
        print(f"이미 판정이 붙은 연결은 건너뜀(다시 만들려면 --force): {', '.join(keep)}", flush=True)
    for key in [k for k in keys if k not in keep]:
        raw, job = career_input(ps[key])
        t0 = time.perf_counter()
        r = await svc.link_detail(raw, job)
        saved[key] = {"career": r["career"], "context": f"직무: {job}" if job else "", "prefix": r["prefix"],
                      "candidates": r["candidates"],
                      "picked": [{"code": s.ncs_code, "quote": s.evidence} for s in r["skills"]],
                      "seconds": round(time.perf_counter() - t0)}
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        LINKS_PATH.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")  # 한 명마다 저장
        top = [j for j in rank_jobs(ix, [s.ncs_code for s in r["skills"]]) if j["postings"] >= MIN_POSTINGS][:5]
        print(f"■ {key} ({saved[key]['seconds']}초) 연결 {len(r['skills'])}개: {', '.join(s.name for s in r['skills'])}\n"
              f"    추천: {' / '.join(j['title'] for j in top)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("link", help="페르소나 경력 → 능력단위 연결 → links.json")
    p.add_argument("--only", help="쉼표로 구분한 페르소나 키")
    p.add_argument("--force", action="store_true", help="이미 사람 판정이 붙은 페르소나도 다시 연결(판정과 어긋날 수 있음)")
    a = ap.parse_args()
    asyncio.run(run_link(a.only.split(",") if a.only else None, a.force))


if __name__ == "__main__":
    main()
