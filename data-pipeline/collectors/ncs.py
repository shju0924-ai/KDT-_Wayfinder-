"""한국산업인력공단 NCS 능력단위 수집기.

공공데이터포털의 CQ-Net NCS 관련 정보 API에서 능력단위명, 정의, 수준,
분류체계를 수집한다. NCS는 자동화 점수를 제공하지 않으므로 직무를 실제
과업 단위로 표준화하고 진단 근거를 표시하는 데 사용한다.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NCS_API_KEY", "")
API_URL = "https://c.q-net.or.kr/openapi/Ncs1info/ncsinfo.do"
SOURCE_URL = "https://www.data.go.kr/data/15063879/openapi.do"
PAGE_SIZE = 1000

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw" / "ncs"
PROCESSED_DIR = DATA_DIR / "processed"


def _fetch_page(client: httpx.Client, page_no: int) -> dict:
    response = client.get(
        API_URL,
        params={
            "serviceKey": API_KEY,
            "type": "json",
            "pageNo": page_no,
            "numOfRows": PAGE_SIZE,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if "message" in payload:
        raise RuntimeError(f"NCS API 오류: {payload['message']}")
    root = payload.get("root")
    if not isinstance(root, dict):
        raise RuntimeError("NCS API 응답에 root 객체가 없습니다.")
    return payload


def collect(max_pages: int | None = None) -> Path:
    """NCS 능력단위를 전량(또는 max_pages까지만) 수집해 CSV로 저장한다."""
    if not API_KEY:
        raise RuntimeError("data-pipeline/.env에 NCS_API_KEY를 설정해 주세요.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    page_no = 1
    total_count: int | None = None

    with httpx.Client(timeout=45, follow_redirects=True) as client:
        while max_pages is None or page_no <= max_pages:
            payload = _fetch_page(client, page_no)
            root = payload["root"]
            info = root.get("info") or {}
            rows = root.get("items") or []
            total_count = int(info.get("totalCount") or len(rows))

            (RAW_DIR / f"page_{page_no:03d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            all_rows.extend(rows)

            if not rows or len(all_rows) >= total_count:
                break
            page_no += 1
            time.sleep(0.15)

    columns = {
        "ncsClCd": "ncs_code",
        "compeUnitName": "unit_name",
        "compeUnitDef": "unit_definition",
        "compeUnitLevel": "unit_level",
        "ncsLclasCdnm": "large_category",
        "ncsMclasCdnm": "middle_category",
        "ncsSclasCdnm": "small_category",
        "ncsSubdCdnm": "sub_category",
        "ncsLastLinkDt": "ncs_updated_at",
    }
    frame = pd.DataFrame(all_rows).rename(columns=columns)
    for column in columns.values():
        if column not in frame:
            frame[column] = ""
    frame = frame[list(columns.values())].drop_duplicates(subset=["ncs_code"])
    frame["source_url"] = SOURCE_URL

    output = PROCESSED_DIR / "ncs_units.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"NCS 능력단위 {len(frame):,}건 수집 완료"
        f" (API totalCount={total_count:,}) → {output}"
    )
    return output


if __name__ == "__main__":
    collect()
