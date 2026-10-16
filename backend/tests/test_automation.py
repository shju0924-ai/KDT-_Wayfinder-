from types import SimpleNamespace

from app.services.automation import calculate_risk


def task(share, automation_score, effect):
    return SimpleNamespace(
        share_percent=share,
        automation_score=automation_score,
        effect=effect,
    )


def test_calculate_risk_normalizes_task_shares():
    result = calculate_risk(
        [
            task(3, 80, "automation"),
            task(1, 20, "human"),
        ],
        [],
    )
    assert result["task_based_score"] == 65
    assert result["risk_score"] == 65
    assert "risk_level" not in result  # 등급 판정은 내지 않는다
    assert result["automation_share"] == 75
    assert result["human_centered_share"] == 25


def test_calculate_risk_uses_external_exposure_as_25_percent_calibration():
    result = calculate_risk(
        [
            task(50, 80, "automation"),
            task(50, 40, "augmentation"),
        ],
        [20, 40],
    )
    assert result["task_based_score"] == 60
    assert result["empirical_score"] == 30
    assert result["risk_score"] == 52.5


def test_ncs_match_picks_with_job_context_and_checks_task_only_similarity(monkeypatch):
    """고르기는 '직무명: 업무명'(분야 맞춤), 붙일지는 업무명만의 유사도로 — 글자만 비슷한 다른 분야 단위를 막는다."""
    import numpy as np

    from app.services import automation, embedding

    units = [SimpleNamespace(ncs_code="WRONG", unit_name="정화 검증", sub_category="토양관리"),
             SimpleNamespace(ncs_code="RIGHT", unit_name="입지급 거래", sub_category="창구사무"),
             SimpleNamespace(ncs_code="CTX", unit_name="출납관리", sub_category="창구사무")]
    unit_vecs = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(automation, "_ncs_index", (units, unit_vecs))
    vec = {  # 업무명만 / 직무명을 붙인 것
        "고객 입출금 처리": [0.71, 0.68, 0.0],          # 업무명만이면 WRONG 이 1위
        "은행 창구 텔러: 고객 입출금 처리": [0.5, 0.65, 0.6],  # 맥락을 붙이면 RIGHT 가 1위(업무명만 0.68 ≥ 기준)
        "정산 검증": [0.70, 0.0, 0.62],
        "은행 창구 텔러: 정산 검증": [0.4, 0.0, 0.6],    # 맥락 1위 CTX 의 업무명만 유사도 0.62 < 기준 → 붙이지 않음
    }
    monkeypatch.setattr(embedding, "_encode", lambda texts: [vec[t] for t in texts])
    tasks = [SimpleNamespace(name="고객 입출금 처리"), SimpleNamespace(name="정산 검증")]

    matches, source = automation._match_ncs(None, tasks, "은행 창구 텔러")
    assert {i: u.ncs_code for i, u in matches.items()} == {0: "RIGHT"}
    assert source is not None
    # 직무명이 없으면 예전처럼 업무명만으로 고른다
    matches, _ = automation._match_ncs(None, tasks)
    assert {i: u.ncs_code for i, u in matches.items()} == {0: "WRONG", 1: "WRONG"}
