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
    assert result["risk_level"] == "높음"
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
    assert result["risk_level"] == "보통"
