"""
End-to-end unit tests for FaceMeshDetector, HeadPoseEstimator, and DriverMonitoringEngine.
"""
import os
import cv2
import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))

from src.head_pose import HeadPoseEstimator
from src.face_mesh_detector import FaceMeshDetector
from src.detector_engine import DriverMonitoringEngine
from generate_test_media import create_driver_frame


def test_head_pose_estimation_straight():
    estimator = HeadPoseEstimator()
    detector = FaceMeshDetector()
    frame = create_driver_frame(head_x=320, head_y=240, turn=0.0, is_looking_down=False)
    res = detector.process(frame)
    assert res['detected']
    yaw, pitch, roll, rvec, tvec, valid = estimator.estimate_pose(res['head_pose_pts'], frame.shape)
    assert valid
    # With frontal symmetry, yaw should be near 0
    assert abs(yaw) < 10.0


def test_head_pose_distraction_tracker():
    estimator = HeadPoseEstimator(yaw_left_thresh=20.0, consec_frames=3)

    # 2 frames looking left
    estimator.update(yaw=30.0, pitch=0.0, roll=0.0)
    estimator.update(yaw=30.0, pitch=0.0, roll=0.0)
    assert not estimator.is_distracted

    # 3rd frame looking left -> alert triggered
    is_distracted, reason, total = estimator.update(yaw=30.0, pitch=0.0, roll=0.0)
    assert is_distracted
    assert "LOOKING LEFT" in reason
    assert total == 1


def test_face_mesh_detector_on_synthetic_frame():
    detector = FaceMeshDetector()
    frame = create_driver_frame(eye_h=15, mouth_h=10)
    result = detector.process(frame)
    assert result['detected']
    assert len(result['left_eye_pts']) == 6
    assert len(result['right_eye_pts']) == 6
    assert len(result['mouth_pts']) == 8
    assert len(result['head_pose_pts']) == 6


def test_driver_monitoring_engine_frame():
    engine = DriverMonitoringEngine(enable_serial=False, enable_audio=False)
    frame = create_driver_frame(eye_h=15, mouth_h=8)
    annotated, telemetry = engine.process_frame(frame)

    assert annotated.shape == frame.shape
    assert telemetry['face_detected']
    assert telemetry['ear'] > 0.15
    assert not telemetry['is_drowsy']
    assert 0.0 <= telemetry['perclos_pct'] <= 100.0
    assert 0.0 <= telemetry['fatigue_risk'] <= 100.0
    assert telemetry['risk_level'] in {'LOW', 'MODERATE', 'HIGH', 'CRITICAL'}
    assert telemetry['microsleeps'] == 0
    engine.close()
