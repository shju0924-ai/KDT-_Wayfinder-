"""서울시 일자리포털 채용정보 수집 — 인접 직무 탐색(STEP 3)의 서울 지역 채용공고.

API: data.seoul.go.kr(서울 열린데이터광장) 'GetJobInfo'
발급: data.seoul.go.kr 로그인 → 데이터셋 "서울시 일자리포털 채용 정보" → 인증키 신청 → SEOUL_JOB_API_KEY

고용24 채용정보 API가 개인회원 이용 제한("개인회원은 사용할 수 없는 OPEN-API입니다")에 걸려
지역 축소 대안으로 채택. 라이선스는 공공누리 1유형(상업적 이용·변경 가능).
"""
import json
import os
import time

import httpx
import pandas as pd
from dotenv import load_dotenv

from collectors.filters import job_is_open, today

load_dotenv()

API_KEY = os.getenv("SEOUL_JOB_API_KEY", "")
BASE_URL = "http://openapi.seoul.go.kr:8088"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PAGE_SIZE = 1000  # API 자체 한도(요청당 최대 1,000건)


def _fetch_range(start: int, end: int) -> dict:
    url = f"{BASE_URL}/{API_KEY}/json/GetJobInfo/{start}/{end}/"
    r = httpx.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    # 서울 열린데이터광장 공통 에러 응답은 서비스명("GetJobInfo") 키 없이
    # {"RESULT": {"CODE": "ERROR-xxx", "MESSAGE": "..."}} 형태로만 온다.
    # 예전엔 body = data.get("GetJobInfo", {}) 가 빈 dict를 받아 rows=[]로
    # 조용히 "0건 수집"으로 위장됐음(2026-09-18 ERROR-500 서버 오류로 실제 발견).
    if "GetJobInfo" not in data:
        result = data.get("RESULT", {})
        code = result.get("CODE", "UNKNOWN")
        message = result.get("MESSAGE", "응답에 GetJobInfo가 없습니다.")
        raise RuntimeError(f"서울 일자리포털 API 오류 [{code}]: {message}")

    # 정상 스키마 내부에도 RESULT.CODE가 오는 경우가 있어(INFO-000=정상) 함께 확인.
    inner_result = data["GetJobInfo"].get("RESULT")
    if isinstance(inner_result, dict):
        code = inner_result.get("CODE", "")
        if code and not code.startswith("INFO"):
            message = inner_result.get("MESSAGE", "")
            raise RuntimeError(f"서울 일자리포털 API 오류 [{code}]: {message}")

    return data


def collect(max_records: int | None = None, drop_closed: bool = True) -> None:
    """채용정보를 START_INDEX/END_INDEX 페이지네이션으로 수집해 raw/processed 저장.

    max_records=None 이면 API의 list_total_count 까지 전량 수집(약 2.5만 건).
    drop_closed=True 면 접수 마감일(RCEPT_CLOS_NM)이 오늘 이전인 공고를 제외한다.

    TODO:
      - [ ] 학력코드(ACDMCR_CMMN_CODE_SE)·고용형태코드 등 코드값 → 사람이 읽는 라벨 매핑
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 전량 수집: 첫 호출로 총건수 확인 후 그만큼 수집
    if max_records is None:
        head = _fetch_range(1, 1).get("GetJobInfo", {})
        max_records = int(head.get("list_total_count") or PAGE_SIZE)

    all_rows: list[dict] = []
    start = 1
    while start <= max_records:
        end = min(start + PAGE_SIZE - 1, max_records)
        data = _fetch_range(start, end)
        body = data.get("GetJobInfo", {})
        rows = body.get("row", [])
        if not rows:
            break

        with open(os.path.join(RAW_DIR, f"seoul_job_{start}_{end}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        start = end + 1
        time.sleep(0.2)

    total = len(all_rows)
    ref = today()
    if drop_closed:
        kept = [r for r in all_rows if job_is_open(r.get("RCEPT_CLOS_NM"), ref=ref)]
    else:
        kept = all_rows

    df = pd.DataFrame(kept)
    df.to_csv(os.path.join(PROCESSED_DIR, "jobs_seoul.csv"), index=False, encoding="utf-8-sig")
    print(
        f"서울시 일자리포털 채용정보 수집 {total}건 중 마감 제외 후 {len(df)}건 저장 "
        f"(기준일 {ref}) → data/processed/jobs_seoul.csv"
    )


if __name__ == "__main__":
    collect()
