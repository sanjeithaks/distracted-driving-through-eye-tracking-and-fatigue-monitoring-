"""
Unit tests for PERCLOS, microsleep detection and the Fatigue Risk Index.
"""
import pytest

from src.fatigue_analyzer import FatigueAnalyzer


def test_perclos_and_microsleep_detection():
    analyzer = FatigueAnalyzer(window_frames=10, closure_openness=0.30,
                               microsleep_frames=3, fps=30.0)

    # Establish an open-eye baseline (EAR ~0.40)
    for _ in range(5):
        analyzer.update(ear=0.40, face_detected=True, blinks=0, yawns=0)
    assert analyzer.baseline_open == pytest.approx(0.40, abs=1e-6)

    # Six fully-closed frames -> 60% of the 10-frame window
    out = None
    for _ in range(6):
        out = analyzer.update(ear=0.05, face_detected=True, blinks=0, yawns=0)

    assert out["perclos"] == pytest.approx(0.60, abs=1e-6)
    assert out["microsleeps"] == 1
    assert out["longest_closure_sec"] == pytest.approx(6 / 30.0, abs=1e-6)
    # A long closure must not drag the open-eye baseline down
    assert analyzer.baseline_open == pytest.approx(0.40, abs=1e-6)

    # Reopening ends the microsleep; a second closure episode increments again
    analyzer.update(ear=0.40, face_detected=True, blinks=1, yawns=0)
    for _ in range(3):
        out = analyzer.update(ear=0.05, face_detected=True, blinks=1, yawns=0)
    assert out["microsleeps"] == 2


def test_risk_levels():
    assert FatigueAnalyzer.risk_level_for(10) == "LOW"
    assert FatigueAnalyzer.risk_level_for(30) == "MODERATE"
    assert FatigueAnalyzer.risk_level_for(60) == "HIGH"
    assert FatigueAnalyzer.risk_level_for(90) == "CRITICAL"


def test_fatigue_risk_factors_sum_to_index():
    analyzer = FatigueAnalyzer(window_frames=20, microsleep_frames=2, fps=30.0)
    out = None
    for _ in range(5):
        analyzer.update(ear=0.40, face_detected=True, blinks=2, yawns=1)
    for _ in range(10):
        out = analyzer.update(ear=0.02, face_detected=True, blinks=2, yawns=3)

    assert 0.0 <= out["fatigue_risk"] <= 100.0
    assert out["fatigue_risk"] == pytest.approx(
        sum(out["risk_factors"].values()), abs=1e-6
    )
    assert out["risk_level"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_no_face_does_not_trigger_false_fatigue():
    analyzer = FatigueAnalyzer(window_frames=10, microsleep_frames=3, fps=30.0)
    for _ in range(10):
        out = analyzer.update(ear=0.0, face_detected=False, blinks=0, yawns=0)
    assert out["perclos"] == 0.0
    assert out["microsleeps"] == 0
    assert out["fatigue_risk"] == 0.0
