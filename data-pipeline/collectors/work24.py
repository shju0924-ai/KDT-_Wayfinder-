"""고용24(워크넷) 채용 공고·직업 정보 수집.

API 문서: https://openapi.work.go.kr (구인정보 / 직업정보 오픈API)
발급: 워크넷 오픈API 신청 → WORK24_API_KEY 를 .env 에 설정

주의: 실제 호출 테스트 결과 개인회원 키는 이 API를 쓸 수 없음
("개인회원은 사용할 수 없는 OPEN-API입니다" 에러 확인됨).
법인/기관 자격 전환 없이는 사용 불가 — 채용공고는 seoul_job.py·gg_joba.py로 대체.

TODO:
  - [ ] 법인/기관 자격 전환 가능 여부 확인, 또는 개인회원 허용 범위(채용행사·공채속보·
        공채기업정보) 내에서 재검토
  - [ ] 채용 공고 목록·상세 수집 (페이지네이션)
  - [ ] 직무명·요구역량 텍스트 정제 → data/processed/jobs.csv
"""
import os

import httpx  # noqa: F401
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("WORK24_API_KEY", "")
BASE_URL = "https://openapi.work.go.kr/opi/opi/opia/wantedApi.do"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def collect() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    # TODO: httpx 로 목록 조회 → XML/JSON 파싱 → data/raw/ 저장
    raise NotImplementedError("고용24 수집 로직 구현 필요")


if __name__ == "__main__":
    collect()
