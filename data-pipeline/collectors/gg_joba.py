"""경기도 잡아바 채용정보 수집 — 인접 직무 탐색(STEP 3)의 경기 지역 채용공고.

API: data.gg.go.kr(경기데이터드림) 'GGJOBABARECRUSTM' (경기도일자리재단 잡아바 채용정보)
발급: data.gg.go.kr 로그인 → 인증키발급 → GG_JOBA_API_KEY

서울 일자리포털과 함께 지역 축소(서울·경기) 전략의 채용공고 소스로 채택.
라이선스는 상업적 이용·콘텐츠 변경 허용.
"""
import json
import math
import os
import time

import httpx
import pandas as pd
from dotenv import load_dotenv

from collectors.filters import job_is_open, today

load_dotenv()

API_KEY = os.getenv("GG_JOBA_API_KEY", "")
BASE_URL = "https://openapi.gg.go.kr/GGJOBABARECRUSTM"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PAGE_SIZE = 1000  # API 자체 한도(요청당 최대 1,000건)


def _fetch_page(page_index: int) -> dict:
    params = {"KEY": API_KEY, "Type": "json", "pIndex": page_index, "pSize": PAGE_SIZE}
    r = httpx.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _total_count() -> int | None:
    data = _fetch_page(1)
    body = data.get("GGJOBABARECRUSTM", [])
    head = next((item["head"] for item in body if "head" in item), [])
    for item in head:
        if "list_total_count" in item:
            return int(item["list_total_count"])
    return None


def collect(max_pages: int | None = None, drop_closed: bool = True) -> None:
    """채용정보를 pIndex 페이지네이션으로 수집해 raw/processed 저장.

    max_pages=None 이면 head.list_total_count 기반으로 전량 수집(약 17만 건).
    drop_closed=True 면 접수 마감일(RCPT_END_DE)이 오늘 이전인 공고를 제외한다.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if max_pages is None:
        total = _total_count()
        max_pages = math.ceil(total / PAGE_SIZE) if total else 5

    all_rows: list[dict] = []
    for page in range(1, max_pages + 1):
        data = _fetch_page(page)
        body = data.get("GGJOBABARECRUSTM", [])
        rows = next((item["row"] for item in body if "row" in item), [])
        if not rows:
            break

        with open(os.path.join(RAW_DIR, f"gg_joba_page{page}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.2)

    total_rows = len(all_rows)
    ref = today()
    if drop_closed:
        kept = [r for r in all_rows if job_is_open(r.get("RCPT_END_DE"), ref=ref)]
    else:
        kept = all_rows

    df = pd.DataFrame(kept)
    df.to_csv(os.path.join(PROCESSED_DIR, "jobs_gg.csv"), index=False, encoding="utf-8-sig")
    print(
        f"경기도 잡아바 채용정보 수집 {total_rows}건 중 마감 제외 후 {len(df)}건 저장 "
        f"(기준일 {ref}) → data/processed/jobs_gg.csv"
    )


if __name__ == "__main__":
    collect()
