"""
Unit tests for the session logger, summary aggregation and HTML report.
"""
import json
import os

import numpy as np
import pytest

from src.report import generate_html_report
from src.session_logger import SessionLogger


def _telemetry(**overrides):
    base = {
        'face_detected': True,
        'ear': 0.30,
        'mor': 0.10,
        'yaw': 0.0,
        'pitch': 0.0,
        'roll': 0.0,
        'is_drowsy': False,
        'is_yawning': False,
        'is_distracted': False,
        'distraction_reason': 'NORMAL',
        'gaze_label': 'CENTER',
        'gaze_x': 0.5,
        'gaze_y': 0.5,
        'is_gaze_distracted': False,
        'blinks': 0,
        'yawns': 0,
        'distractions': 0,
        'gaze_distractions': 0,
        'perclos_pct': 5.0,
        'fatigue_risk': 12.0,
        'risk_level': 'LOW',
        'microsleeps': 0,
        'longest_closure_sec': 0.2,
        'blink_rate': 15.0,
        'yawn_rate': 1.0,
        'risk_factors': {'perclos': 6.0, 'microsleep': 0.0,
                         'yawning': 5.0, 'closure': 1.0},
        'fps': 30.0,
    }
    base.update(overrides)
    return base


def test_session_logger_writes_csv_and_summary(tmp_path):
    logger = SessionLogger(output_dir=str(tmp_path), session_name="test")
    logger.log_frame(_telemetry(), frame_index=1)
    logger.log_frame(_telemetry(ear=0.10, is_drowsy=True,
                                perclos_pct=40.0, fatigue_risk=62.0,
                                risk_level='HIGH'), frame_index=2)
    logger.log_frame(_telemetry(gaze_label='LEFT', is_distracted=True,
                                distraction_reason='EYES OFF ROAD / GAZE LEFT',
                                distractions=1, gaze_distractions=1,
                                blinks=2, yawns=1), frame_index=3)

    summary = logger.close()

    assert os.path.exists(logger.csv_path)
    assert os.path.exists(logger.summary_path)
    assert summary['frames'] == 3
    assert summary['face_detection_rate_pct'] == 100.0
    assert summary['drowsy_events'] == 1
    assert summary['distraction_events'] == 1
    assert summary['min_ear'] == 0.10
    assert summary['gaze_direction_counts']['LEFT'] == 1
    assert summary['avg_perclos'] == pytest.approx((5 + 40 + 5) / 3, abs=0.01)
    assert summary['peak_fatigue_risk'] == 62.0

    with open(logger.summary_path, encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored['frames'] == 3

    with open(logger.csv_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    assert len(lines) == 4  # header + 3 frames
    assert 'perclos' in lines[0] and 'fatigue_risk' in lines[0]


def test_alert_evidence_snapshot(tmp_path):
    snapshot_dir = tmp_path / "snapshots"
    logger = SessionLogger(output_dir=str(tmp_path), session_name="snap",
                           snapshot_dir=str(snapshot_dir))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    logger.log_frame(_telemetry(is_drowsy=True), frame_index=7, frame=frame)
    summary = logger.close()

    snapshots = list(snapshot_dir.glob("snapshot_*.jpg"))
    assert len(snapshots) == 1
    assert summary['events'][0].get('snapshot') == snapshots[0].name


def test_html_report_generation(tmp_path):
    summary = {
        'session': 'unit-test',
        'duration_sec': 12.5,
        'frames': 100,
        'face_detection_rate_pct': 98.0,
        'avg_ear': 0.3,
        'min_ear': 0.1,
        'avg_mor': 0.12,
        'max_mor': 0.5,
        'avg_fps': 29.5,
        'state_frames': {'NORMAL': 80, 'DROWSY': 10, 'DISTRACTED': 5, 'YAWNING': 5},
        'gaze_direction_counts': {'CENTER': 90, 'LEFT': 10},
        'blinks': 4, 'yawns': 2, 'distractions': 1, 'gaze_distractions': 1,
        'drowsy_events': 1, 'distraction_events': 1, 'yawn_events': 1,
        'events': [{'timestamp': 3.0, 'frame': 20, 'type': 'DROWSY',
                    'message': 'Drowsiness detected', 'reason': ''}],
    }

    out = tmp_path / "report.html"
    path = generate_html_report(summary, str(out))

    assert os.path.exists(path)
    html = out.read_text(encoding="utf-8")
    assert "Session overview" in html
    assert "Driver state distribution" in html
    assert "Iris gaze distribution" in html
    assert "DROWSY" in html
