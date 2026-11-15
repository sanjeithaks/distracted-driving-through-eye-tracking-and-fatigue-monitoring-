"""
Unit tests for the immersive driving simulator renderer.
"""
import numpy as np

from src.driving_simulator import DrivingSimulator


def _telemetry(**overrides):
    base = {
        'fatigue_risk': 10.0, 'risk_level': 'LOW', 'microsleeps': 0,
        'is_drowsy': False, 'is_distracted': False, 'is_yawning': False,
        'perclos_pct': 2.0, 'ear': 0.3, 'gaze_label': 'CENTER',
        'distraction_reason': 'NORMAL',
    }
    base.update(overrides)
    return base


def test_driving_simulator_renders_and_reacts_to_state():
    sim = DrivingSimulator(960, 540, fps=30)
    monitor = np.zeros((480, 640, 3), dtype=np.uint8)

    sim.update(_telemetry(), 0.05)
    scene = sim.render(monitor, _telemetry())
    assert scene.shape == (540, 960, 3)
    assert sim.speed > 0

    sim.update(_telemetry(is_drowsy=True, fatigue_risk=82.0,
                          risk_level='CRITICAL', microsleeps=2), 0.05)
    drowsy_scene = sim.render(monitor, _telemetry(is_drowsy=True, fatigue_risk=82.0,
                                                  risk_level='CRITICAL'))
    assert drowsy_scene.shape == (540, 960, 3)
    assert sim.state == "DROWSY"
    # Drowsiness should reduce the target speed
    assert sim.target_speed <= 32.0

    sim.update(_telemetry(is_distracted=True, distraction_reason='LOOKING LEFT / PASSENGER'),
               0.05)
    assert sim.state == "DISTRACTED"
    sim.render(monitor, _telemetry(is_distracted=True))
