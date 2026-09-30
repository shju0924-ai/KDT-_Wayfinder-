"""HRD-Net(국민내일배움카드) 훈련과정 수집 — 학습 로드맵의 교육 자원.

API: work24.go.kr OPEN-API '국민내일배움카드 훈련과정' (callOpenApiSvcInfo310L01.do)
발급: work24.go.kr 로그인 → 고객센터 > OPEN-API > 서비스 소개에서 신청 → HRDNET_API_KEY

목록 응답(310L01)에 ncsCd/ncsNm이 이미 포함되어 있어, NCS API 없이도
STEP 4 로드맵의 근거(source)로 바로 쓸 수 있음.
훈련기관 상세(310L02)·취업률 통계(310L03)는 필요해지면 trprId/trprDegr/instCd로 후속 조회.

수집 기준(요구사항):
  - srchTraStDt 를 '오늘'로 두어 이미 개강한 과정은 API 단계에서 제외
  - 응답에서도 개강일<오늘(이미 시작)·정원 마감 과정을 한 번 더 제외
  - 같은 과정의 여러 회차(trprDegr)는 가장 이른 개강 회차만 남긴다
"""
import json
import os
import time
from datetime import timedelta

import httpx
import pandas as pd
from dotenv import load_dotenv

from collectors.filters import course_is_open, parse_date, today

load_dotenv()

API_KEY = os.getenv("HRDNET_API_KEY", "")
LIST_URL = "https://www.work24.go.kr/cm/openApi/call/hr/callOpenApiSvcInfo310L01.do"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PAGE_SIZE = 100  # API 자체 한도(요청당 최대 100건)


def _fetch_page(page_num: int, start_date: str, end_date: str) -> dict:
    params = {
        "authKey": API_KEY,
        "returnType": "JSON",
        "outType": "1",
        "pageNum": page_num,
        "pageSize": PAGE_SIZE,
        "srchTraStDt": start_date,
        "srchTraEndDt": end_date,
    }
    # 전량 수집 중 일시적 연결 오류 한 번에 전체가 중단되지 않도록 재시도한다
    for attempt in range(4):
        try:
            r = httpx.get(LIST_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def _dedupe_by_course(rows: list[dict]) -> list[dict]:
    """같은 trprId(과정)의 여러 회차 중 개강일이 가장 이른 회차만 남긴다."""
    best: dict[str, dict] = {}
    for r in rows:
        key = str(r.get("trprId") or f"{r.get('title')}|{r.get('subTitle')}")
        d = parse_date(r.get("traStartDate"))
        cur = best.get(key)
        if cur is None:
            best[key] = r
        else:
            cd = parse_date(cur.get("traStartDate"))
            if d is not None and (cd is None or d < cd):
                best[key] = r
    return list(best.values())


def collect(
    start_date: str | None = None,
    end_date: str | None = None,
    days_ahead: int = 90,
    max_records: int | None = None,
    drop_closed: bool = True,
) -> None:
    """훈련과정 목록을 페이지네이션으로 수집해 raw/processed 저장.

    start_date=None 이면 오늘, end_date=None 이면 오늘+days_ahead 로 개강일 창을 잡는다.
    → 이미 개강한 과정은 창 밖으로 자연히 빠진다(요구사항: 이미 시작한 교육 제외).

    max_records=None(기본값) 이면 개강일 창(start_date~end_date) 안의 전량을 수집한다
    (2026-09-18: 데모 단계의 임베딩 실현성 상한 4만 건을 프로젝트 완성 방침에 따라 제거—
    필요 시 테스트용으로만 정수를 넘겨 상한을 둘 수 있음).
    drop_closed=True 면 개강일<오늘·정원 마감 과정을 추가 제외하고, 회차 중복을 제거한다.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ref = today()
    start_date = start_date or ref.strftime("%Y%m%d")
    end_date = end_date or (ref + timedelta(days=days_ahead)).strftime("%Y%m%d")
    max_pages = ((max_records + PAGE_SIZE - 1) // PAGE_SIZE) if max_records is not None else None

    all_rows: list[dict] = []
    page = 1
    while max_pages is None or page <= max_pages:
        data = _fetch_page(page, start_date, end_date)
        rows = data.get("srchList", [])
        if not rows:
            break

        with open(os.path.join(RAW_DIR, f"hrdnet_page{page}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        time.sleep(0.2)
        page += 1

    total = len(all_rows)
    if drop_closed:
        kept = [
            r for r in all_rows
            if course_is_open(
                r.get("traStartDate"),
                capacity=r.get("yardMan"),
                enrolled=r.get("regCourseMan"),
                ref=ref,
            )
        ]
        kept = _dedupe_by_course(kept)
    else:
        kept = all_rows

    df = pd.DataFrame(kept)
    df.to_csv(os.path.join(PROCESSED_DIR, "courses.csv"), index=False, encoding="utf-8-sig")
    print(
        f"HRD-Net 훈련과정 수집 {total}건 중 개강전·접수가능·회차중복 정리 후 {len(df)}건 저장 "
        f"(개강일 {start_date}~{end_date}) → data/processed/courses.csv"
    )


if __name__ == "__main__":
    collect()
