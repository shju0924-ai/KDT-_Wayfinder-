"""훈련과정 임베딩 → pgvector 적재 (HRD-Net 국민내일배움카드 과정).

흐름: data/processed/courses.csv 로드 → 과정 텍스트 구성 → (긴 텍스트만) 청킹
      → passage 임베딩 → training_courses upsert

ncsCd가 함께 저장되므로 STEP 4 로드맵의 근거(source) 필드에 바로 사용 가능.

사용:
    python -m embedding.embed_courses             # 전체 실행
    python -m embedding.embed_courses --limit 20  # 소량 테스트
    python -m embedding.embed_courses --dry-run   # DB 적재 없이 임베딩까지만 검증
"""
import argparse
import os

import pandas as pd

from embedding.common import (
    chunk_text,
    embed_passages,
    ensure_tables,
    get_connection,
    truncate_table,
    upsert_training_courses,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _s(v, max_len: int | None = None) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    return s[:max_len] if max_len else s


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def load_courses() -> list[dict]:
    path = os.path.join(PROCESSED_DIR, "courses.csv")
    if not os.path.exists(path):
        print("courses.csv 없음 — 건너뜀 (collectors.hrdnet 먼저 실행)")
        return []
    # dtype=str: ncsCd 같은 코드 컬럼을 숫자로 추론하면 앞자리 0이 사라지고
    # float화("02020302" → "2020302.0")되므로 전 컬럼을 문자열로 읽는다
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        print("courses.csv 비어 있음(수집 실패로 추정) — 건너뜀")
        return []
    rows = []
    for _, r in df.iterrows():
        parts = [
            _s(r.get("title")), _s(r.get("ncsNm")), _s(r.get("trainTarget")),
            _s(r.get("subTitle")), _s(r.get("certificate")),
        ]
        text = "\n".join(p for p in parts if p)
        if not text:
            continue
        rows.append({
            "source_id": f"hrdnet:{r.get('trprId')}#{r.get('trprDegr')}",
            "course_name": _s(r.get("title"), 300) or "",
            "institution": _s(r.get("subTitle"), 200),
            "ncs_cd": _s(r.get("ncsCd"), 20),
            "ncs_nm": _s(r.get("ncsNm"), 100),
            "content_text": text,
            "start_date": _s(r.get("traStartDate"), 10),
            "end_date": _s(r.get("traEndDate"), 10),
            "tuition": _to_int(r.get("courseMan")),
            "address": _s(r.get("address"), 200),
        })
    return rows


# 임베딩·적재를 나눠 수행하는 단위(행)
LOAD_CHUNK = 1000


def run(limit: int | None = None, dry_run: bool = False, fresh: bool = False) -> None:
    rows = load_courses()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("적재할 훈련과정이 없습니다.")
        return

    expanded: list[dict] = []
    for row in rows:
        chunks = chunk_text(row["content_text"])
        if len(chunks) == 1:
            expanded.append(row)
        else:
            for i, chunk in enumerate(chunks, start=1):
                expanded.append({**row, "source_id": f"{row['source_id']}#c{i}", "content_text": chunk})

    print(f"훈련과정 {len(rows)}건 → 청킹 후 {len(expanded)}행, 임베딩 시작")

    if dry_run:
        vectors = embed_passages([r["content_text"] for r in expanded])
        print(f"[dry-run] 임베딩 {len(vectors)}건 완료 (차원 {len(vectors[0])}) — DB 적재 생략")
        return

    conn = get_connection()
    try:
        ensure_tables(conn)
        if fresh:
            truncate_table(conn, "training_courses")
            print("training_courses 테이블 비움 (만료·개강 과정 제거)")
        # CPU 임베딩은 수십 분 걸리므로 구간마다 바로 적재·커밋한다
        # — 중간에 중단돼도 그때까지 처리한 분량은 DB에 남는다
        n = 0
        for i in range(0, len(expanded), LOAD_CHUNK):
            part = expanded[i : i + LOAD_CHUNK]
            for row, vec in zip(part, embed_passages([r["content_text"] for r in part])):
                row["embedding"] = vec
            n += upsert_training_courses(conn, part)
            print(f"training_courses 적재 진행 {n}/{len(expanded)}행")
        print(f"training_courses 테이블에 {n}행 적재 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 과정 수 제한 (테스트용)")
    parser.add_argument("--dry-run", action="store_true", help="DB 적재 없이 임베딩까지만 실행")
    parser.add_argument("--fresh", action="store_true", help="적재 전 테이블 비우기 (만료 과정 제거)")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, fresh=args.fresh)
