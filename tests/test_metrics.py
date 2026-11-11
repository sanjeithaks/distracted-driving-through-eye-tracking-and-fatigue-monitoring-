"""
Unit tests for facial feature metrics (EAR, MOR) and tracking state machines.
"""
import pytest
import numpy as np
from src.facial_features import (
    euclidean_distance,
    calculate_ear,
    calculate_mor,
    EyeTracker,
    YawnTracker
)


def test_euclidean_distance():
    p1 = (0, 0)
    p2 = (3, 4)
    assert euclidean_distance(p1, p2) == 5.0


def test_calculate_ear_open_eye():
    # Synthetic open eye coordinates:
    # p1: outer corner (0, 0), p4: inner corner (10, 0) -> dist = 10
    # p2: (3, 3), p6: (3, -3) -> dist = 6
    # p3: (7, 3), p5: (7, -3) -> dist = 6
    # EAR = (6 + 6) / (2 * 10) = 12 / 20 = 0.60
    open_eye = [(0, 0), (3, 3), (7, 3), (10, 0), (7, -3), (3, -3)]
    ear = calculate_ear(open_eye)
    assert np.isclose(ear, 0.60, atol=1e-3)


def test_calculate_ear_closed_eye():
    # Synthetic closed eye coordinates: vertical distance is 0.5
    # EAR = (0.5 + 0.5) / 20 = 0.05
    closed_eye = [(0, 0), (3, 0.25), (7, 0.25), (10, 0), (7, -0.25), (3, -0.25)]
    ear = calculate_ear(closed_eye)
    assert ear < 0.10


def test_calculate_mor():
    # Normal closed lips:
    # Horizontal width = 20, vertical gap = 2
    # MOR = (2 + 2 + 2) / (3 * 20) = 6 / 60 = 0.10
    normal_mouth = [
        (-10, 0), (-5, 1), (0, 1), (5, 1),
        (10, 0), (5, -1), (0, -1), (-5, -1)
    ]
    mor_normal = calculate_mor(normal_mouth)
    assert mor_normal < 0.25

    # Wide open mouth (Yawn):
    # Horizontal width = 20, vertical gap = 14
    # MOR = (14 + 14 + 14) / (3 * 20) = 42 / 60 = 0.70
    yawn_mouth = [
        (-10, 0), (-5, 7), (0, 7), (5, 7),
        (10, 0), (5, -7), (0, -7), (-5, -7)
    ]
    mor_yawn = calculate_mor(yawn_mouth)
    assert mor_yawn > 0.60


def test_eye_tracker_drowsiness():
    tracker = EyeTracker(ear_thresh=0.21, consec_frames=5, blink_min_frames=2)

    # 4 frames below threshold -> not yet drowsy
    for _ in range(4):
        is_drowsy, _, _ = tracker.update(0.15, 0.15)
        assert not is_drowsy

    # 5th frame below threshold -> triggers drowsiness
    is_drowsy, _, _ = tracker.update(0.15, 0.15)
    assert is_drowsy

    # Recovery: eyes open
    is_drowsy, _, _ = tracker.update(0.30, 0.30)
    assert not is_drowsy


def test_eye_tracker_blink_counter():
    tracker = EyeTracker(ear_thresh=0.21, consec_frames=10, blink_min_frames=2)

    # Normal open eyes
    tracker.update(0.30, 0.30)
    assert tracker.total_blinks == 0

    # Quick blink: closed for 3 frames (< 10 consec frames)
    tracker.update(0.15, 0.15)
    tracker.update(0.15, 0.15)
    tracker.update(0.15, 0.15)
    # Reopened
    _, is_blinking, _ = tracker.update(0.30, 0.30)
    assert is_blinking
    assert tracker.total_blinks == 1


def test_yawn_tracker():
    tracker = YawnTracker(mor_thresh=0.58, consec_frames=3, cooldown_frames=5)

    # 2 frames open -> not yet registered
    tracker.update(0.65)
    tracker.update(0.65)
    assert not tracker.is_yawning
    assert tracker.total_yawns == 0

    # 3rd frame -> yawn registered!
    is_yawning, total_yawns, _ = tracker.update(0.65)
    assert is_yawning
    assert total_yawns == 1
