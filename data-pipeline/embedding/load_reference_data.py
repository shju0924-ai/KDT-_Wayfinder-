"""NCS 및 자동화 참고 점수 CSV를 PostgreSQL에 적재한다."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from embedding.common import (
    ensure_tables,
    get_connection,
    upsert_automation_scores,
    upsert_ncs_units,
)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _clean(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _load_ncs() -> list[dict]:
    path = PROCESSED_DIR / "ncs_units.csv"
    if not path.exists():
        print("ncs_units.csv 없음 — NCS 적재 건너뜀")
        return []
    frame = pd.read_csv(path, dtype=str)
    return [
        {
            "ncs_code": _clean(row.get("ncs_code")),
            "unit_name": _clean(row.get("unit_name")) or "",
            "unit_definition": _clean(row.get("unit_definition")),
            "unit_level": _clean(row.get("unit_level")),
            "large_category": _clean(row.get("large_category")),
            "middle_category": _clean(row.get("middle_category")),
            "small_category": _clean(row.get("small_category")),
            "sub_category": _clean(row.get("sub_category")),
            "source_url": _clean(row.get("source_url")),
        }
        for _, row in frame.iterrows()
        if _clean(row.get("ncs_code"))
    ]


def _load_scores() -> list[dict]:
    path = PROCESSED_DIR / "automation_occupation_scores.csv"
    if not path.exists():
        print("automation_occupation_scores.csv 없음 — 참고 점수 적재 건너뜀")
        return []
    frame = pd.read_csv(path, dtype={"occupation_code": str})
    return [
        {
            "source": _clean(row.get("source")) or "",
            "occupation_code": _clean(row.get("occupation_code")) or "",
            "occupation_title": _clean(row.get("occupation_title")) or "",
            "score": float(row["score"]),
            "metric": _clean(row.get("metric")) or "",
            "source_version": _clean(row.get("source_version")),
            "source_url": _clean(row.get("source_url")) or "",
        }
        for _, row in frame.iterrows()
        if _clean(row.get("source")) and _clean(row.get("occupation_code"))
    ]


def run(dry_run: bool = False) -> None:
    ncs_rows = _load_ncs()
    score_rows = _load_scores()
    if dry_run:
        print(f"[dry-run] NCS {len(ncs_rows):,}건, 참고 점수 {len(score_rows):,}건")
        return
    if not ncs_rows and not score_rows:
        return

    conn = get_connection()
    try:
        ensure_tables(conn)
        if ncs_rows:
            print(f"ncs_units 테이블에 {upsert_ncs_units(conn, ncs_rows):,}행 적재 완료")
        if score_rows:
            count = upsert_automation_scores(conn, score_rows)
            print(f"automation_occupation_scores 테이블에 {count:,}행 적재 완료")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    run(dry_run=parser.parse_args().dry_run)
