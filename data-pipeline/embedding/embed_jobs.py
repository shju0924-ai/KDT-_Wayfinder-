"""채용공고 임베딩 → pgvector 적재 (서울 일자리포털 + 경기 잡아바).

흐름: data/processed/jobs_seoul.csv, jobs_gg.csv 로드
      → 직무 텍스트 구성 → (긴 텍스트만) 청킹 → passage 임베딩 → job_postings upsert

채용공고는 검색 "대상"(passage)이므로 반드시 passage 모델로 임베딩할 것 — 검색어(query)
모델과 섞으면 벡터 공간이 어긋나 유사도가 무의미해짐.

사용:
    python -m embedding.embed_jobs             # 전체 실행
    python -m embedding.embed_jobs --limit 20  # 소량 테스트
    python -m embedding.embed_jobs --dry-run   # DB 적재 없이 임베딩까지만 검증
"""
import argparse
import hashlib
import os

import pandas as pd
from tqdm import tqdm

from embedding.common import (
    chunk_text,
    embed_passages,
    ensure_tables,
    get_connection,
    truncate_table,
    upsert_job_postings,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def _clean(v) -> str | None:
    """NaN·빈 값 → None, 나머지는 공백 정리된 문자열."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


def _first(*values) -> str:
    """앞에서부터 첫 번째 유효한 값 반환. float('nan')은 truthy라 `or` 폴백이
    동작하지 않는 함정이 있어(제목이 "nan"으로 적재됨) 반드시 이 함수를 쓸 것."""
    for v in values:
        s = _clean(v)
        if s:
            return s
    return ""


def _join(parts: list) -> str:
    """NaN·빈 값을 건너뛰고 텍스트 결합."""
    return "\n".join(s for p in parts if (s := _clean(p)))


def _read_csv_safe(path: str) -> pd.DataFrame | None:
    """수집이 실패해 헤더 없이 빈 CSV만 남은 경우에도 파이프라인이 죽지 않도록 처리."""
    try:
        # dtype=str: 코드성 컬럼의 숫자 추론(앞자리 0 소실·float화) 방지
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        print(f"{os.path.basename(path)} 비어 있음(수집 실패로 추정) — 건너뜀")
        return None


def load_seoul() -> list[dict]:
    path = os.path.join(PROCESSED_DIR, "jobs_seoul.csv")
    if not os.path.exists(path):
        print("jobs_seoul.csv 없음 — 건너뜀 (collectors.seoul_job 먼저 실행)")
        return []
    df = _read_csv_safe(path)
    if df is None:
        return []
    rows = []
    for _, r in df.iterrows():
        text = _join([r.get("JOBCODE_NM"), r.get("JO_SJ"), r.get("DTY_CN"), r.get("GUI_LN"), r.get("BSNS_SUMRY_CN")])
        if not text:
            continue
        auth_no = _first(r.get("JO_REGIST_NO"), r.get("JO_REQST_NO"))
        rows.append({
            "source_id": f"seoul:{auth_no}",
            "job_title": _first(r.get("JOBCODE_NM"), r.get("JO_SJ"))[:300],
            "company": (_clean(r.get("CMPNY_NM")) or "")[:200] or None,
            "region": "seoul",
            "source_url": (
                "https://job.seoul.go.kr/hmpg/rmim/ramg/ramgDetail.do"
                f"?wantedAuthNo={auth_no}"
            ) if auth_no else None,
            "required_skills_text": text,
        })
    return rows


def load_gg() -> list[dict]:
    path = os.path.join(PROCESSED_DIR, "jobs_gg.csv")
    if not os.path.exists(path):
        print("jobs_gg.csv 없음 — 건너뜀 (collectors.gg_joba 먼저 실행)")
        return []
    df = _read_csv_safe(path)
    if df is None:
        return []
    rows = []
    for _, r in df.iterrows():
        text = _join([
            r.get("PBANC_CONT"), r.get("RECRUT_FIELD_NM"), r.get("CAREER_DIV"),
            r.get("ACDMCR_DIV"), r.get("PBANC_FORM_DIV"), r.get("WORK_REGION_CONT"),
        ])
        if not text:
            continue
        # 잡아바 응답에는 고유 공고번호가 없어 URL+공고명 해시로 안정적 ID 생성
        uid = hashlib.md5(f"{r.get('URL')}|{r.get('PBANC_CONT')}".encode()).hexdigest()[:16]
        rows.append({
            "source_id": f"gg:{uid}",
            "job_title": _first(r.get("PBANC_CONT"), r.get("RECRUT_FIELD_NM"))[:300],
            "company": (_clean(r.get("ENTRPRS_NM")) or "")[:200] or None,
            "region": "gg",
            "source_url": _clean(r.get("URL")),
            "required_skills_text": text,
        })
    return rows


def run(limit: int | None = None, dry_run: bool = False, fresh: bool = False) -> None:
    rows = load_seoul() + load_gg()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("적재할 채용공고가 없습니다.")
        return

    # 청킹 — 긴 텍스트만 분할되며 청크는 source_id 접미어(#c1…)로 별도 행이 된다
    expanded: list[dict] = []
    for row in rows:
        chunks = chunk_text(row["required_skills_text"])
        if len(chunks) == 1:
            expanded.append(row)
        else:
            for i, chunk in enumerate(chunks, start=1):
                expanded.append({**row, "source_id": f"{row['source_id']}#c{i}", "required_skills_text": chunk})

    print(f"채용공고 {len(rows)}건 → 청킹 후 {len(expanded)}행, 임베딩 시작")
    texts = [r["required_skills_text"] for r in expanded]
    vectors = []
    for i in tqdm(range(0, len(texts), 64), desc="임베딩"):
        vectors.extend(embed_passages(texts[i : i + 64]))
    for row, vec in zip(expanded, vectors):
        row["embedding"] = vec

    if dry_run:
        print(f"[dry-run] 임베딩 {len(vectors)}건 완료 (차원 {len(vectors[0])}) — DB 적재 생략")
        return

    conn = get_connection()
    try:
        ensure_tables(conn)
        if fresh:
            truncate_table(conn, "job_postings")
            print("job_postings 테이블 비움 (만료 공고 제거)")
        n = upsert_job_postings(conn, expanded)
        print(f"job_postings 테이블에 {n}행 적재 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 공고 수 제한 (테스트용)")
    parser.add_argument("--dry-run", action="store_true", help="DB 적재 없이 임베딩까지만 실행")
    parser.add_argument("--fresh", action="store_true", help="적재 전 테이블 비우기 (만료 공고 제거)")
    args = parser.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, fresh=args.fresh)
