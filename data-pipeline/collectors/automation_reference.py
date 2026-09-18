"""자동화 위험도 보정에 사용하는 공개 원자료 수집·정규화.

점수로 적재하는 자료:
- Anthropic Economic Index: 미국 직업별 관측 AI 사용 노출도
- ILO/NASK 2025: ISCO-08 6자리 직업별 생성형 AI 노출도

OECD와 한국고용정보원 보고서는 방법론·국내 맥락 근거로 내려받는다.
두 보고서의 수치를 다른 직업분류에 억지로 연결하지 않으며, 신뢰 가능한
직업명 교차표가 확보될 때 별도 매핑한다.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_DIR = DATA_DIR / "raw" / "automation"
PROCESSED_DIR = DATA_DIR / "processed"

ANTHROPIC_DATASET_URL = (
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/resolve/main/"
    "labor_market_impacts/job_exposure.csv?download=true"
)
ANTHROPIC_SOURCE_URL = (
    "https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/"
    "labor_market_impacts"
)
ILO_DATASET_URL = (
    "https://raw.githubusercontent.com/pgmyrek/"
    "POLAND_2025_GenAI_scores_6digit_occupations/main/"
    "POLAND_FINAL_6digit_scores.xlsx"
)
ILO_SOURCE_URL = (
    "https://github.com/pgmyrek/"
    "POLAND_2025_GenAI_scores_6digit_occupations"
)
OECD_REPORT_URL = (
    "https://www.oecd.org/content/dam/oecd/en/publications/reports/2026/05/"
    "the-oecd-ai-exposure-measure_489cfd42/f3da0f0a-en.pdf"
)
KEIS_REPORT_URL = (
    "https://www.keis.or.kr/keis/ko/cmmn/download.do"
    "?dn=20250410093607702.pdf"
    "&fn=%28%EA%B8%B0%EB%B3%B8%EC%97%B0%EA%B5%AC+2024-12%29+"
    "%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5%EC%97%90+%EC%9D%98%ED%95%9C+"
    "%ED%99%94%EC%9D%B4%ED%8A%B8%EC%B9%BC%EB%9D%BC%EC%9D%98+"
    "%EC%A7%81%EB%AC%B4+%EB%8C%80%EC%B2%B4+%EB%B0%8F+%EB%B3%80%ED%99%94.pdf"
    "&fsn=22038&path=pblc&sn=11169&ty=P"
)


def _download(client: httpx.Client, url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with path.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)


def _normalize_anthropic(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"occ_code": str})
    result = pd.DataFrame(
        {
            "source": "Anthropic Economic Index",
            "occupation_code": frame["occ_code"],
            "occupation_title": frame["title"].str.strip(),
            "score": (frame["observed_exposure"] * 100).round(2),
            "metric": "observed_ai_exposure",
            "source_version": "labor_market_impacts",
            "source_url": ANTHROPIC_SOURCE_URL,
        }
    )
    return result.dropna(subset=["occupation_title", "score"])


def _normalize_ilo(path: Path) -> pd.DataFrame:
    task_rows = pd.read_excel(path, dtype={"Kod": str})
    required = {"Kod", "Occupation Title", "mean6d_pl"}
    missing = required.difference(task_rows.columns)
    if missing:
        raise RuntimeError(f"ILO 자료 필수 열 누락: {sorted(missing)}")

    occupation_rows = (
        task_rows[["Kod", "Occupation Title", "mean6d_pl"]]
        .dropna()
        .drop_duplicates(subset=["Kod"])
    )
    result = pd.DataFrame(
        {
            "source": "ILO/NASK GenAI Exposure 2025",
            "occupation_code": occupation_rows["Kod"].astype(str),
            "occupation_title": occupation_rows["Occupation Title"].str.strip(),
            "score": (occupation_rows["mean6d_pl"].astype(float) * 100).round(2),
            "metric": "potential_genai_exposure",
            "source_version": "2025",
            "source_url": ILO_SOURCE_URL,
        }
    )
    return result.dropna(subset=["occupation_title", "score"])


def collect(download_reports: bool = True) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    anthropic_path = RAW_DIR / "anthropic_job_exposure.csv"
    ilo_path = RAW_DIR / "ilo_genai_scores.xlsx"
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        _download(client, ANTHROPIC_DATASET_URL, anthropic_path)
        _download(client, ILO_DATASET_URL, ilo_path)
        if download_reports:
            _download(client, OECD_REPORT_URL, RAW_DIR / "oecd_ai_exposure_2026.pdf")
            _download(client, KEIS_REPORT_URL, RAW_DIR / "keis_white_collar_ai_2024.pdf")

    normalized = pd.concat(
        [_normalize_anthropic(anthropic_path), _normalize_ilo(ilo_path)],
        ignore_index=True,
    )
    normalized["score"] = normalized["score"].clip(0, 100)
    output = PROCESSED_DIR / "automation_occupation_scores.csv"
    normalized.to_csv(output, index=False, encoding="utf-8-sig")
    print(
        f"자동화 참고 점수 {len(normalized):,}건 정규화 완료"
        f" ({normalized['source'].nunique()}개 출처) → {output}"
    )
    return output


if __name__ == "__main__":
    collect()
