"""서울시 일자리포털 채용정보 수집 — 인접 직무 탐색(STEP 3)의 서울 지역 채용공고.

API: data.seoul.go.kr(서울 열린데이터광장) OA-23047 'recMntList'
발급: data.seoul.go.kr 로그인 → 데이터셋 "서울시 일자리포털 채용 정보" → 인증키 신청 → SEOUL_JOB_API_KEY

기존 OA-13341 'GetJobInfo'는 원천시스템 연계 방식 변경으로 2026-09-03 종료되고
OA-23047 'recMntList'로 대체됨(열린데이터광장 공지 2026-08-20). 종료된 서비스명은
존재하지 않는 서비스와 똑같이 ERROR-500만 돌려줘 서버 장애처럼 보였음.
새 응답은 공고 고유번호가 없고, 서울 외 지역 공고도 섞여 있으며, 날짜가 YY-MM-DD다.

고용24 채용정보 API가 개인회원 이용 제한("개인회원은 사용할 수 없는 OPEN-API입니다")에 걸려
지역 축소 대안으로 채택. 라이선스는 공공누리 1유형(상업적 이용·변경 가능).
"""
import json
import os
import re
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

SERVICE = "recMntList"
PAGE_SIZE = 1000  # API 자체 한도(요청당 최대 1,000건)

# 담당자 연락처는 추천·검색에 쓰지 않으므로 저장하지 않는다
DROP_COLUMNS = ["EMP_CHARGER_DPT", "CONTACT_TELNO"]

_YY_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{2})")


def _to_iso_date(value) -> str | None:
    """'26-10-04', '채용시까지  26-10-04' → '2026-10-04'. 날짜가 없으면 None.

    공통 필터(filters.parse_date)는 4자리 연도만 인식하므로 여기서 먼저 정규화한다
    — 그대로 두면 마감일이 "미표기"로 판정돼 이미 마감된 공고까지 남는다.
    """
    m = _YY_DATE_RE.search(str(value or ""))
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _fetch_range(start: int, end: int) -> dict:
    url = f"{BASE_URL}/{API_KEY}/json/{SERVICE}/{start}/{end}/"
    r = httpx.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    # 서울 열린데이터광장 공통 에러 응답은 서비스명 키 없이
    # {"RESULT": {"CODE": "ERROR-xxx", "MESSAGE": "..."}} 형태로만 온다.
    # 예전엔 body = data.get(서비스명, {}) 가 빈 dict를 받아 rows=[]로
    # 조용히 "0건 수집"으로 위장됐음(2026-09-18 ERROR-500으로 실제 발견).
    # ERROR-500은 서비스명이 없거나 종료된 경우에도 나온다 — 지속되면 데이터셋 종료 공지부터 확인.
    if SERVICE not in data:
        result = data.get("RESULT", {})
        code = result.get("CODE", "UNKNOWN")
        message = result.get("MESSAGE", f"응답에 {SERVICE}가 없습니다.")
        raise RuntimeError(f"서울 일자리포털 API 오류 [{code}]: {message}")

    # 정상 스키마 내부에도 RESULT.CODE가 오는 경우가 있어(INFO-000=정상) 함께 확인.
    inner_result = data[SERVICE].get("RESULT")
    if isinstance(inner_result, dict):
        code = inner_result.get("CODE", "")
        if code and not code.startswith("INFO"):
            message = inner_result.get("MESSAGE", "")
            raise RuntimeError(f"서울 일자리포털 API 오류 [{code}]: {message}")

    return data


def collect(max_records: int | None = None, drop_closed: bool = True) -> None:
    """채용정보를 START_INDEX/END_INDEX 페이지네이션으로 수집해 raw/processed 저장.

    max_records=None 이면 API의 list_total_count 까지 전량 수집(약 4만 건).
    drop_closed=True 면 접수 마감일(CLOSE_DT)이 오늘 이전인 공고를 제외한다.
    지역 필터는 걸지 않는다 — 서울 외 공고도 원천에 포함돼 있어 적재 단계에서 REGION으로 구분.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 전량 수집: 첫 호출로 총건수 확인 후 그만큼 수집
    if max_records is None:
        head = _fetch_range(1, 1).get(SERVICE, {})
        max_records = int(head.get("list_total_count") or PAGE_SIZE)

    all_rows: list[dict] = []
    start = 1
    while start <= max_records:
        end = min(start + PAGE_SIZE - 1, max_records)
        data = _fetch_range(start, end)
        body = data.get(SERVICE, {})
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
    for r in all_rows:
        r["REG_DATE"] = _to_iso_date(r.get("REG_DT"))
        r["CLOSE_DATE"] = _to_iso_date(r.get("CLOSE_DT"))
        for col in DROP_COLUMNS:
            r.pop(col, None)

    ref = today()
    if drop_closed:
        kept = [r for r in all_rows if job_is_open(r.get("CLOSE_DATE"), ref=ref)]
    else:
        kept = all_rows

    # 페이지 경계에서 같은 공고가 반복 반환되는 경우 대비(경기 잡아바에서 실제 발생)
    df = pd.DataFrame(kept).drop_duplicates()
    df.to_csv(os.path.join(PROCESSED_DIR, "jobs_seoul.csv"), index=False, encoding="utf-8-sig")
    print(
        f"서울시 일자리포털 채용정보 수집 {total}건 중 마감 제외 후 {len(df)}건 저장 "
        f"(기준일 {ref}) → data/processed/jobs_seoul.csv"
    )


if __name__ == "__main__":
    collect()
