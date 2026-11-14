"""
Unit tests for iris-based eye-gaze tracking.
"""
import pytest

from src.gaze_tracker import compute_gaze_ratios, GazeTracker


# 6-point eye contour used by the detector (image coordinates: y grows downward)
EYE = [(0, 0), (3, -3), (7, -3), (10, 0), (7, 3), (3, 3)]


def _update(tracker, iris_x=5, iris_y=0, eyes_open=True):
    iris = (iris_x, iris_y)
    return tracker.update(EYE, EYE, iris, iris, eyes_open=eyes_open)


def test_compute_gaze_ratios_centered():
    gx, gy = compute_gaze_ratios(EYE, EYE, (5, 0), (5, 0))
    assert gx == pytest.approx(0.5, abs=1e-6)
    assert gy == pytest.approx(0.5, abs=1e-6)


def test_compute_gaze_ratios_shifted_right():
    gx, gy = compute_gaze_ratios(EYE, EYE, (8, 0), (8, 0))
    assert gx == pytest.approx(0.8, abs=1e-6)


def test_gaze_tracker_calibrates_and_detects_off_road():
    tracker = GazeTracker(horizontal_delta=0.10, vertical_delta=0.10,
                          consec_frames=3, calibration_frames=3,
                          center_band=0.02)

    # Calibration frames: iris centred -> neutral gaze
    for _ in range(3):
        label, _, _, distracted, _ = _update(tracker, iris_x=5)
        assert label == "CENTER"
        assert not distracted

    # Eyes shift to the right for 3 frames -> sustained gaze alert
    for _ in range(2):
        _, _, _, distracted, _ = _update(tracker, iris_x=8)
        assert not distracted

    label, _, _, distracted, total = _update(tracker, iris_x=8)
    assert distracted
    assert label == "RIGHT"      # iris shifted toward image-right
    assert total == 1


def test_gaze_tracker_ignores_closed_eyes():
    tracker = GazeTracker(consec_frames=2, calibration_frames=1)
    _update(tracker, iris_x=5)

    for _ in range(5):
        label, _, _, distracted, total = _update(tracker, iris_x=9, eyes_open=False)
        assert label == "CLOSED"
        assert not distracted
        assert total == 0
