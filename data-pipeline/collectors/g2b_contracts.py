"""나라장터 용역 계약정보 수집 — 사업계획서의 공공 조달 단가 근거.

이 스크립트는 서비스 기능이 아니라 **사업계획(TAM-SAM-SOM) 근거 자료**를 만든다.
Wayfinder와 유사한 공공 고용서비스·일자리 플랫폼 사업이 실제로 얼마에 발주됐는지
수집해, "기관당 연 구독료 ○○만원"의 산출 근거로 쓴다.

API: 조달청 나라장터 계약정보서비스 (getCntrctInfoListServcPPSSrch)
     https://www.data.go.kr/data/15129427/openapi.do
발급: data.go.kr 활용신청(자동승인) → 일반 인증키(Decoding) → G2B_API_KEY

사용:
    python -m collectors.g2b_contracts                     # 최근 3년, 기본 키워드
    python -m collectors.g2b_contracts --years 5
    python -m collectors.g2b_contracts --keywords 고용서비스,취업지원
    python -m collectors.g2b_contracts --all-classes       # SW 분류 필터 해제
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import date

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("G2B_API_KEY", "")
BASE_URL = (
    "http://apis.data.go.kr/1230000/ao/CntrctInfoService/getCntrctInfoListServcPPSSrch"
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

PAGE_SIZE = 100  # API 최대 메시지 4000bytes 기준 안전값

# Wayfinder와 비교 가능한 사업을 찾는 검색어 (계약명 부분일치)
DEFAULT_KEYWORDS = [
    "고용서비스",
    "취업지원",
    "일자리",
    "직업훈련",
    "고용정보",
    "진로",
    "직업상담",
    "구직",
]

# 공공조달 분류로 SW 사업만 남긴다 — 청소·경비·인쇄 등 무관한 용역을 걸러내는 핵심 필터
SW_CLASS_KEYWORDS = ["ICT", "SW", "소프트웨어", "정보시스템", "정보화"]

# 공공 SW 유지보수요율 관행(10~15%)의 중간값 — 구축비 → 연 구독료 환산에 사용
MAINTENANCE_RATE = 0.12


def _amount(item: dict) -> int:
    """총계약금액이 0으로 오는 건이 많아 금차계약금액과 큰 값을 취한다."""
    values = []
    for key in ("totCntrctAmt", "thtmCntrctAmt"):
        try:
            values.append(int(float(item.get(key) or 0)))
        except (TypeError, ValueError):
            pass
    return max(values) if values else 0


def _first_of_list(raw: str | None, index: int) -> str:
    """`[1^6460843^전라남도 동부지역본부^지방자치단체^...]` 형태에서 n번째 필드 추출."""
    if not raw:
        return ""
    head = raw.strip().lstrip("[").rstrip("]").split("],")[0]
    parts = head.split("^")
    return parts[index].strip() if len(parts) > index else ""


def _to_date(value: str | None):
    """'2024-04-01' 또는 '20250923' → date. 실패하면 None."""
    if not value:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) != 8:
        return None
    try:
        return date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None


def _fetch(client: httpx.Client, keyword: str, begin: str, end: str, page: int) -> dict:
    params = {
        "ServiceKey": API_KEY,
        "type": "json",
        "inqryDiv": "1",  # 계약체결일자 기준
        "inqryBgnDate": begin,
        "inqryEndDate": end,
        "cntrctNm": keyword,
        "pageNo": page,
        "numOfRows": PAGE_SIZE,
    }
    r = client.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()["response"]
    header = payload.get("header", {})
    if header.get("resultCode") not in ("00", "0"):
        raise RuntimeError(f"나라장터 API 오류: {header.get('resultMsg')}")
    return payload.get("body", {})


def collect(
    keywords: list[str] | None = None,
    years: int = 3,
    sw_only: bool = True,
) -> str:
    if not API_KEY:
        raise RuntimeError("data-pipeline/.env에 G2B_API_KEY를 설정해 주세요.")

    keywords = keywords or DEFAULT_KEYWORDS
    today = date.today()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    rows: dict[str, dict] = {}  # untyCntrctNo 기준 중복 제거
    with httpx.Client() as client:
        # API가 연 단위 조회를 안정적으로 처리하므로 연도별로 나눠 호출한다
        for offset in range(years):
            year = today.year - offset
            begin, end = f"{year}0101", f"{year}1231"
            for kw in keywords:
                page = 1
                while True:
                    body = _fetch(client, kw, begin, end, page)
                    items = body.get("items") or []
                    if isinstance(items, dict):
                        items = [items]
                    for it in items:
                        rows[it.get("untyCntrctNo", "")] = it
                    total = int(body.get("totalCount") or 0)
                    print(f"  {year} · {kw} · {page}p (누적 {len(rows)}건 / 검색 {total}건)")
                    if page * PAGE_SIZE >= total or not items:
                        break
                    page += 1
                    time.sleep(0.15)

    records = []
    for it in rows.values():
        amount = _amount(it)
        if amount <= 0:
            continue

        cls_text = " ".join(
            str(it.get(k) or "")
            for k in ("pubPrcrmntLrgClsfcNm", "pubPrcrmntMidClsfcNm", "pubPrcrmntClsfcNm")
        )
        is_sw = any(k in cls_text for k in SW_CLASS_KEYWORDS) or it.get("infoBizYn") == "Y"
        if sw_only and not is_sw:
            continue

        start = _to_date(it.get("cntrctCnclsDate"))
        finish = _to_date(it.get("ttalScmpltDate")) or _to_date(
            it.get("thtmScmpltDate")
        ) or _to_date(it.get("cntrctPrd"))
        months = None
        if start and finish and finish > start:
            months = round((finish - start).days / 30.44, 1)
        # 12개월 미만 사업도 연 단위 비용으로 환산해 서로 비교 가능하게 만든다
        annual = round(amount * 12 / months) if months and months > 0 else amount

        records.append({
            "계약명": it.get("cntrctNm", ""),
            "수요기관": _first_of_list(it.get("dminsttList"), 2),
            "수주업체": _first_of_list(it.get("corpList"), 3),
            "계약금액": amount,
            "계약체결일": it.get("cntrctCnclsDate", ""),
            "완수일": finish.isoformat() if finish else "",
            "기간(개월)": months,
            "연환산금액": annual,
            f"유지보수추정({int(MAINTENANCE_RATE * 100)}%)": round(annual * MAINTENANCE_RATE),
            "업무구분": it.get("bsnsDivNm", ""),
            "조달분류": it.get("pubPrcrmntClsfcNm", ""),
            "계약방법": it.get("cntrctCnclsMthdNm", ""),
            "상세URL": it.get("cntrctDtlInfoUrl", ""),
        })

    df = pd.DataFrame(records).sort_values("계약금액", ascending=False)
    out = os.path.join(PROCESSED_DIR, "g2b_contracts.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n용역 계약 {len(df)}건 수집 완료 → {out}")
    if not df.empty:
        col = f"유지보수추정({int(MAINTENANCE_RATE * 100)}%)"
        print("\n[단가 산정 근거 요약]")
        print(f"  계약금액   중앙값 {df['계약금액'].median():>15,.0f}원")
        print(f"             평균   {df['계약금액'].mean():>15,.0f}원")
        print(f"  연환산     중앙값 {df['연환산금액'].median():>15,.0f}원")
        print(f"  {col} 중앙값 {df[col].median():>15,.0f}원  ← 기관당 연 구독료 근거")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3, help="최근 N년 (기본 3)")
    parser.add_argument("--keywords", type=str, default=None, help="쉼표로 구분한 계약명 검색어")
    parser.add_argument(
        "--all-classes", action="store_true", help="SW 분류 필터를 끄고 모든 용역 포함"
    )
    args = parser.parse_args()
    collect(
        keywords=[k.strip() for k in args.keywords.split(",")] if args.keywords else None,
        years=args.years,
        sw_only=not args.all_classes,
    )


if __name__ == "__main__":
    main()
