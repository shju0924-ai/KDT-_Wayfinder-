"""NCS 능력단위 임베딩 → ncs_units.embedding / embedding_domain (STEP 2 경력 → 능력단위 연결용).

흐름: ncs_units 에서 쓸 능력단위만 고른다(구버전·'삭제' 표시 제외, 같은 세분류 안에서 띄어쓰기·버전만 다른
      같은 이름은 첫 코드 하나) → 아직 벡터가 없는 것만 두 가지 문장으로 임베딩 → UPDATE
    embedding         '능력단위명 — 정의'                          (직무 추천 점수 — 분야가 달라도 같은 일이면 가깝게)
    embedding_domain  '대분류 > 중분류 > 소분류 > 세분류 | 능력단위명 — 정의'  (경력 구절로 후보 검색 — 분야를 구분)
분야 경로를 넣은 검색은 페르소나 15명 시험에서 엉뚱한 분야 연결(보험 '상품 설계' → 기계 설계 등)을 줄였다
(backend/scripts/ncs_link.py 실험 기록).

문장 형식·고르는 규칙은 backend/app/services/ncs_link.py 가 읽는 기준과 같아야 한다(정의 축약 _short_def 포함).
CPU로 1만4천 건 × 2 = 약 2시간.

사용:
    python -m embedding.embed_ncs             # 벡터가 없는 단위만
    python -m embedding.embed_ncs --all       # 전부 다시
    python -m embedding.embed_ncs --dry-run   # 대상 수만 확인
"""
import argparse
import re

from embedding.common import embed_passages, ensure_tables, get_connection

DEF_CHARS = 70
LOAD_CHUNK = 500


def _short_def(d: str | None) -> str:
    """'○○이란 …하는 능력이다.' → '…하는' (앞 DEF_CHARS 자). backend/app/services/ncs_link.py 와 같아야 한다."""
    d = re.sub(r"^.*?이란\s*", "", (d or "").strip())
    d = re.sub(r"\s*능력이다\.?$", "", d)
    return d[:DEF_CHARS]


def select_units(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ncs_code, unit_name, unit_definition, large_category, middle_category, small_category,
                   sub_category, embedding IS NOT NULL AND embedding_domain IS NOT NULL
            FROM ncs_units
            WHERE unit_name NOT LIKE '%구버전%' AND unit_name NOT LIKE '%삭제%'
            ORDER BY ncs_code
            """
        )
        rows = cur.fetchall()
    units, seen = [], set()
    for code, name, d, large, middle, small, sub, done in rows:
        key = (code[:8], re.sub(r"\s+", "", name))
        if key in seen:
            continue
        seen.add(key)
        base = f"{name} — {_short_def(d)}"
        units.append({"code": code, "base": base, "dom": f"{large} > {middle} > {small} > {sub} | {base}", "done": done})
    return units


def run(all_: bool = False, dry_run: bool = False) -> None:
    conn = get_connection()
    try:
        ensure_tables(conn)
        units = select_units(conn)
        keep = {u["code"] for u in units}
        todo = [u for u in units if all_ or not u["done"]]
        print(f"쓸 능력단위 {len(units):,}개 중 임베딩 대상 {len(todo):,}개")
        if dry_run:
            return
        # 구버전·삭제·중복으로 빠진 단위에 남은 예전 벡터는 지운다 — 검색 후보에 섞이지 않게
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ncs_units SET embedding = NULL, embedding_domain = NULL "
                "WHERE embedding_domain IS NOT NULL AND NOT (ncs_code = ANY(%s))",
                (list(keep),),
            )
            if cur.rowcount:
                print(f"쓰지 않는 단위의 벡터 {cur.rowcount}개 지움")
        conn.commit()
        n = 0
        for i in range(0, len(todo), LOAD_CHUNK):
            part = todo[i : i + LOAD_CHUNK]
            base = embed_passages([u["base"] for u in part])
            dom = embed_passages([u["dom"] for u in part])
            with conn.cursor() as cur:
                cur.executemany(
                    "UPDATE ncs_units SET embedding = %s::vector, embedding_domain = %s::vector WHERE ncs_code = %s",
                    [(b, d, u["code"]) for u, b, d in zip(part, base, dom, strict=True)],
                )
            conn.commit()
            n += len(part)
            print(f"ncs_units 임베딩 진행 {n:,}/{len(todo):,}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="벡터가 있어도 전부 다시 계산")
    parser.add_argument("--dry-run", action="store_true", help="대상 수만 확인")
    args = parser.parse_args()
    run(all_=args.all, dry_run=args.dry_run)
