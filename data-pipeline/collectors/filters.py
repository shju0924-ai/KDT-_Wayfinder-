"""수집 공통 필터 — 이미 마감된 공고·이미 시작한 훈련과정을 제외한다.

두 가지 규칙만 담는다.
  1. 채용공고: 접수 마감일이 오늘(수집일)보다 이전이면 제외
  2. 훈련과정: 개강일이 오늘보다 이전(이미 시작)이거나 접수 정원이 이미 찬 경우 제외

날짜 표기가 소스마다 달라(YYYYMMDD, YYYY-MM-DD, "마감일 (2026-09-26)" 등)
느슨하게 파싱하고, 파싱 불가·미표기는 "판단 보류"로 남긴다(호출부에서 정책 결정).
"""
from __future__ import annotations

import re
from datetime import date, datetime

_DATE_RE = re.compile(r"(\d{4})[.\-/]?(\d{2})[.\-/]?(\d{2})")


def today() -> date:
    return datetime.now().date()


def parse_date(value) -> date | None:
    """문자열에서 첫 번째 YYYY[?]MM[?]DD 패턴을 날짜로 파싱. 실패 시 None.

    허용 예: '20260926', '2026-09-26', '2026.09.26', '마감일 (2026-09-26)'.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    m = _DATE_RE.search(s)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def job_is_open(deadline_value, *, ref: date | None = None, keep_undated: bool = True) -> bool:
    """접수 마감일이 ref(기본 오늘) 이상이면 열림. 날짜가 없으면 keep_undated 정책을 따른다.

    상시채용·채용시까지 등 날짜 미표기 공고는 기본적으로 유효(열림)로 본다.
    """
    ref = ref or today()
    d = parse_date(deadline_value)
    if d is None:
        return keep_undated
    return d >= ref


def _to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def course_is_open(
    start_value,
    *,
    capacity=None,
    enrolled=None,
    ref: date | None = None,
    keep_undated: bool = False,
) -> bool:
    """훈련과정이 '아직 시작 안 했고 접수 가능'하면 열림.

    - 개강일(start_value)이 ref(기본 오늘) 미만이면 이미 시작 → 제외
    - 정원(capacity)과 신청인원(enrolled)이 모두 있고 신청인원 >= 정원이면 마감 → 제외
    - 개강일 미표기는 keep_undated 정책(기본 제외 — 근거 없는 과정은 로드맵에서 뺀다)
    """
    ref = ref or today()
    d = parse_date(start_value)
    if d is None:
        return keep_undated
    if d < ref:
        return False
    cap, enr = _to_int(capacity), _to_int(enrolled)
    if cap is not None and enr is not None and cap > 0 and enr >= cap:
        return False
    return True
