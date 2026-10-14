"""정답 데이터 라벨링 도구(페르소나) — 경력 → NCS 능력단위 연결을 판정하고, 판정한 연결로 다시 계산한 추천 직무를 판정한다.

    cd backend
    python -m scripts.persona_label_server          # http://localhost:8766 (브라우저가 열린다)

능력단위 벡터가 DB에 있어야 하고(data-pipeline: python -m embedding.embed_ncs), ncs_link link 로 links.json 을 만든다.
판정은 누를 때마다 data-pipeline/data/eval/ncs_link/labels.json 에 저장된다.

labels.json — 페르소나 키마다
    links        {능력단위 코드: "O" 맞음 | "D" 업종 다름(일은 맞지만 다른 업종 능력단위) | "X" 틀림}
    added_units  [사람이 추가한 능력단위 코드]
    jobs         {직무(세분류) 코드: "easy" | "possible" | "no"}  — 추천 직무 전환 난이도
    marked_by    {능력단위 코드: "claude"}  — Claude가 미리 표시한 판정(사람이 다시 고르면 지운다)
    claude_notes {능력단위 코드: 의견}, done, note
추천은 '틀림'을 뺀 연결 + 추가한 능력단위로 계산한다(ncs_link.rank_jobs) — 판정을 바꾸면 추천도 바뀐다.
"""
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ncs_link as nl  # noqa: E402

PORT = 8766
TOP_N = 15
SEARCH_N = 30
UI_PATH = Path(__file__).with_name("persona_label_ui.html")
LINK_MARKS = ("O", "D", "X")
JOB_GRADES = ("easy", "possible", "no")  # 전환 쉬움 / 추가 학습하면 가능 / 어려움

_lock = threading.Lock()


def _empty() -> dict:
    return {"links": {}, "added_units": [], "jobs": {}, "marked_by": {}, "claude_notes": {}, "done": False, "note": ""}


class State:
    def __init__(self) -> None:
        self.ix = nl.load_index()
        if not nl.LINKS_PATH.exists():
            raise SystemExit("links.json 이 없습니다 — python -m scripts.ncs_link link 를 먼저 실행하세요")
        self.links: dict = json.loads(nl.LINKS_PATH.read_text(encoding="utf-8"))
        self.labels: dict = json.loads(nl.LABELS_PATH.read_text(encoding="utf-8")) if nl.LABELS_PATH.exists() else {}
        self.all_jobs = [{"code": j, "title": self.ix.job_title[j], "path": self.ix.job_path[j],
                          "postings": self.ix.postings.get(j, 0)} for j in self.ix.jobs]

    def save(self) -> None:
        tmp = nl.LABELS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.labels, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, nl.LABELS_PATH)

    def unit(self, code: str) -> dict:
        u = self.ix.units[self.ix.pos[code]]
        return {"code": code, "name": u["name"], "job_title": u["job_title"], "path": u["path"], "def": u["def"]}


def title_of(r: dict) -> str:
    """설문형은 '직무명 (경력 기간)', 이력서형은 첫 줄 — 길면 자른다('저는 지난 7년 동안 프리랜서로…')."""
    t = r["context"].removeprefix("직무: ") if r["context"] else r["prefix"]
    return t if len(t) <= 24 else t[:23] + "…"


def effective(st: State, key: str) -> list[str]:
    lab = st.labels.get(key, {})
    marks = lab.get("links", {})
    codes = [x["code"] for x in st.links[key]["picked"] if marks.get(x["code"]) != "X"]
    return codes + [c for c in lab.get("added_units", []) if c not in codes]


def persona_list(st: State) -> list[dict]:
    out = []
    for key, r in st.links.items():
        lab = st.labels.get(key, {})
        marks = lab.get("links", {})
        out.append({"key": key, "title": title_of(r), "n_links": len(r["picked"]),
                    "n_marked": sum(1 for x in r["picked"] if x["code"] in marks),
                    "n_jobs": len(lab.get("jobs", {})), "done": bool(lab.get("done"))})
    return out


def persona_detail(st: State, key: str) -> dict:
    r, lab = st.links[key], st.labels.get(key, {})
    picked = {x["code"] for x in r["picked"]}
    return {
        "key": key, "title": title_of(r), "career": r["career"], "context": r["context"],
        "links": [{**st.unit(x["code"]), "quote": x["quote"]} for x in r["picked"] if x["code"] in st.ix.pos],
        "unpicked": [st.unit(c) for c in r["candidates"] if c not in picked and c in st.ix.pos],
        "added": [st.unit(c) for c in lab.get("added_units", []) if c in st.ix.pos],
        "effective": [st.unit(c)["name"] for c in effective(st, key) if c in st.ix.pos],
        "labels": lab,
    }


def recs(st: State, key: str, min_postings: int) -> list[dict]:
    ranked = nl.rank_jobs(st.ix, effective(st, key))
    for n, x in enumerate(ranked, 1):
        x["rank"] = n
    shown = [x for x in ranked if x["postings"] >= min_postings][:TOP_N]
    # 순위 밖으로 밀렸거나 공고 수로 가려졌어도, 이미 판정했거나 직접 추가한 직무는 계속 보여 준다
    marked = st.labels.get(key, {}).get("jobs", {})
    have = {x["code"] for x in shown}
    by_code = {x["code"]: x for x in ranked}
    for c in marked:
        if c not in have:
            x = by_code.get(c) or {"code": c, "title": st.ix.job_title.get(c, c), "path": st.ix.job_path.get(c, ""),
                                   "score": None, "rank": None, "postings": st.ix.postings.get(c, 0), "pairs": []}
            shown.append({**x, "extra": True})
    return shown


def search(st: State, q: str) -> list[dict]:
    q = re.sub(r"\s+", "", q)  # '채권관리'로 '채권 관리'도 찾는다
    if not q:
        return []
    flat = lambda s: re.sub(r"\s+", "", s)  # noqa: E731
    hits = [u for u in st.ix.units if q in flat(u["name"]) or q in flat(u["job_title"])]
    hits.sort(key=lambda u: (q not in flat(u["name"]), len(u["name"])))  # 능력단위명에 있는 것 먼저
    return [{"code": u["code"], "name": u["name"], "job_title": u["job_title"], "path": u["path"], "def": u["def"]}
            for u in hits[:SEARCH_N]]


def apply_label(st: State, body: dict) -> dict:
    key, op = body["key"], body["op"]
    lab = st.labels.setdefault(key, _empty())
    for k, v in _empty().items():
        lab.setdefault(k, v)
    if op == "link":
        lab["marked_by"].pop(body["code"], None)  # 사람이 다시 고르면 Claude 판정 표시를 지운다
        if body.get("mark") in LINK_MARKS:
            lab["links"][body["code"]] = body["mark"]
        else:
            lab["links"].pop(body["code"], None)
    elif op == "add_unit":
        if body["code"] not in st.ix.pos:
            raise ValueError("unknown unit")
        if body["code"] not in lab["added_units"]:
            lab["added_units"].append(body["code"])
    elif op == "remove_unit":
        lab["added_units"] = [c for c in lab["added_units"] if c != body["code"]]
    elif op == "job":
        if body.get("grade") in JOB_GRADES:
            lab["jobs"][body["target"]] = body["grade"]
        else:
            lab["jobs"].pop(body["target"], None)
    elif op == "done":
        lab["done"] = bool(body["value"])
    elif op == "note":
        lab["note"] = body["text"]
    else:
        raise ValueError(op)
    st.save()
    return lab


def make_handler(st: State):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload, ctype="application/json; charset=utf-8") -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/personas":
                    self._send(200, persona_list(st))
                elif u.path == "/api/alljobs":
                    self._send(200, st.all_jobs)
                elif u.path == "/api/persona" and q.get("key") in st.links:
                    self._send(200, persona_detail(st, q["key"]))
                elif u.path == "/api/recs" and q.get("key") in st.links:
                    self._send(200, recs(st, q["key"], int(q.get("min_postings", nl.MIN_POSTINGS))))
                elif u.path == "/api/search":
                    self._send(200, search(st, q.get("q", "")))
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001 — 도구 화면에 오류를 보여 준다
                self._send(500, {"error": str(e)})

        def do_POST(self) -> None:
            if self.path != "/api/label":
                self._send(404, {"error": "not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body.get("key") not in st.links:
                    raise ValueError("unknown persona")
                with _lock:
                    lab = apply_label(st, body)
                self._send(200, lab)
            except Exception as e:  # noqa: BLE001
                self._send(400, {"error": str(e)})

        def log_message(self, *args) -> None:  # 요청마다 찍히는 로그는 끈다
            pass

    return Handler


def main() -> None:
    st = State()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(st))
    url = f"http://localhost:{PORT}"
    print(f"페르소나 {len(st.links)}명 · 능력단위 {len(st.ix.units)}개 · 직무 {len(st.ix.jobs)}개", flush=True)
    print(f"라벨링 도구: {url}  (판정 저장 위치: {nl.LABELS_PATH})  — 끝내려면 Ctrl+C", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
