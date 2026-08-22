import pytest

from train import relative_validation_score


def baseline_metrics() -> dict[str, float]:
    return {
        "EN": 4.8,
        "SD": 32.0,
        "SF": 8.0,
        "SCD": 0.0077,
        "VIF": 0.30,
        "Qabf": 0.49,
    }


def test_scd_does_not_dominate_ranking_score() -> None:
    baseline = baseline_metrics()
    candidate = {
        name: value * (10.0 if name == "SCD" else 1.01)
        for name, value in baseline.items()
    }
    score, minimum_ratio = relative_validation_score(candidate, baseline)
    assert score == pytest.approx(1.01)
    assert minimum_ratio >= 1.0


def test_scd_regression_still_fails_guard() -> None:
    baseline = baseline_metrics()
    candidate = {
        name: (-0.01 if name == "SCD" else value * 1.01)
        for name, value in baseline.items()
    }
    score, minimum_ratio = relative_validation_score(candidate, baseline)
    assert score == pytest.approx(1.01)
    assert minimum_ratio < 1.0
