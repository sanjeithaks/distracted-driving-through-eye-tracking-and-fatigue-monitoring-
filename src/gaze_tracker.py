"""
Iris-Based Eye-Gaze Tracking Module
-----------------------------------
Implements true eye tracking (not only eyelid openness) for the IEEE project:
"Detect distracted driving through eye tracking and fatigue monitoring using
embedded system".

The iris centre is located with MediaPipe's 478-point face mesh and normalised
inside each eye using the eye corners / eyelids:

    gx = (iris_x - eye_left_x)  / eye_width     -> ~0.5 when looking straight
    gy = (iris_y - eyelid_top_y) / eye_height   -> ~0.5 when looking straight

Deviations from the calibrated neutral position identify horizontal / vertical
gaze away from the road.  A sustained off-road gaze (default 15 frames) raises
the "EYES OFF ROAD" distraction state, which is forwarded to the same audio
and embedded seatbelt actuator used by the head-pose detector.
"""
from collections import deque

import numpy as np

from .config import (
    GAZE_HORIZONTAL_DELTA, GAZE_VERTICAL_DELTA,
    GAZE_CONSEC_FRAMES, GAZE_CALIBRATION_FRAMES,
    GAZE_SMOOTHING, GAZE_CENTER_BAND,
)


def _eye_gaze_ratio(eye_points, iris_center):
    """
    Normalise the iris position inside a 6-point eye contour.

    eye_points order (as produced by FaceMeshDetector):
        p1 outer corner, p2 top-outer, p3 top-inner,
        p4 inner corner, p5 bottom-inner, p6 bottom-outer
    Returns (gx, gy) or None when geometry is degenerate.
    """
    if iris_center is None or len(eye_points) < 6:
        return None

    p1, p2, p3, p4, p5, p6 = eye_points[:6]
    x_min = min(p1[0], p4[0])
    x_max = max(p1[0], p4[0])
    top_y = min(p2[1], p3[1])
    bottom_y = max(p5[1], p6[1])

    width = max(1.0, float(x_max - x_min))
    height = max(1.0, float(bottom_y - top_y))

    gx = (float(iris_center[0]) - x_min) / width
    gy = (float(iris_center[1]) - top_y) / height
    return gx, gy


def compute_gaze_ratios(right_eye_pts, left_eye_pts,
                        right_iris_center, left_iris_center):
    """
    Compute the average horizontal / vertical gaze ratio of both eyes.
    Returns (gx, gy) or None if the iris landmarks are unavailable.
    """
    right = _eye_gaze_ratio(right_eye_pts, right_iris_center)
    left = _eye_gaze_ratio(left_eye_pts, left_iris_center)

    ratios = [r for r in (right, left) if r is not None]
    if not ratios:
        return None

    gx = float(np.mean([r[0] for r in ratios]))
    gy = float(np.mean([r[1] for r in ratios]))
    return gx, gy


class GazeTracker:
    """
    Tracks iris position over time, calibrates the neutral gaze, smooths the
    signal and raises a sustained off-road gaze alert.
    """

    def __init__(self,
                 horizontal_delta=GAZE_HORIZONTAL_DELTA,
                 vertical_delta=GAZE_VERTICAL_DELTA,
                 consec_frames=GAZE_CONSEC_FRAMES,
                 calibration_frames=GAZE_CALIBRATION_FRAMES,
                 smoothing=GAZE_SMOOTHING,
                 center_band=GAZE_CENTER_BAND):
        self.horizontal_delta = horizontal_delta
        self.vertical_delta = vertical_delta
        self.consec_frames = consec_frames
        self.calibration_frames = calibration_frames
        self.smoothing = smoothing
        self.center_band = center_band

        # Neutral gaze estimated from the first frames (single exponential mean)
        self.baseline_x = None
        self.baseline_y = None
        self.calibration_count = 0

        self.smoothed_x = None
        self.smoothed_y = None
        self.current_gx = 0.5
        self.current_gy = 0.5

        self.gaze_label = "CENTER"
        self.off_road_counter = 0
        self.is_gaze_distracted = False
        self.total_gaze_distractions = 0

        self._history = deque(maxlen=100)

    def _calibrate(self, gx, gy):
        """Estimate the neutral gaze from the first frames."""
        if self.baseline_x is None:
            self.baseline_x, self.baseline_y = gx, gy
        else:
            self.baseline_x = 0.8 * self.baseline_x + 0.2 * gx
            self.baseline_y = 0.8 * self.baseline_y + 0.2 * gy
        self.calibration_count += 1

    def _classify(self, dx, dy):
        """Convert gaze deviation into a human-readable direction label."""
        if abs(dx) <= self.center_band and abs(dy) <= self.center_band:
            return "CENTER"

        if abs(dx) >= abs(dy):
            if dx > self.horizontal_delta:
                return "RIGHT"
            if dx < -self.horizontal_delta:
                return "LEFT"
        else:
            if dy > self.vertical_delta:
                return "DOWN"
            if dy < -self.vertical_delta:
                return "UP"
        return "CENTER"

    def update(self, right_eye_pts, left_eye_pts,
               right_iris_center, left_iris_center, eyes_open=True):
        """
        Update the gaze state for one video frame.

        Returns:
            (gaze_label, gx, gy, is_gaze_distracted, total_gaze_distractions)

        ``eyes_open`` should be False when EAR indicates closed eyes; the gaze
        signal is then unreliable and is ignored (no false distraction alerts).
        """
        ratios = None
        if eyes_open:
            ratios = compute_gaze_ratios(
                right_eye_pts, left_eye_pts,
                right_iris_center, left_iris_center
            )

        if ratios is None:
            # Eyes closed, iris missing or unreliable -> hold neutral state.
            self.off_road_counter = 0
            self.is_gaze_distracted = False
            self.gaze_label = "CLOSED" if not eyes_open else "N/A"
            return self.gaze_label, self.current_gx, self.current_gy, False, self.total_gaze_distractions

        gx, gy = ratios
        self._history.append((gx, gy))

        # Smooth the noisy per-frame iris signal.
        if self.smoothed_x is None:
            self.smoothed_x, self.smoothed_y = gx, gy
        else:
            a = self.smoothing
            self.smoothed_x = a * gx + (1.0 - a) * self.smoothed_x
            self.smoothed_y = a * gy + (1.0 - a) * self.smoothed_y

        self.current_gx, self.current_gy = self.smoothed_x, self.smoothed_y

        # Learn the neutral position during the calibration window.
        if self.calibration_count < self.calibration_frames:
            self._calibrate(self.smoothed_x, self.smoothed_y)
            self.gaze_label = "CENTER"
            self.off_road_counter = 0
            self.is_gaze_distracted = False
            return self.gaze_label, self.current_gx, self.current_gy, False, self.total_gaze_distractions

        dx = self.smoothed_x - self.baseline_x
        dy = self.smoothed_y - self.baseline_y
        label = self._classify(dx, dy)

        # Slow baseline adaptation while looking at the road keeps calibration
        # valid when the driver shifts slightly in the seat.
        if label == "CENTER":
            self.baseline_x = 0.995 * self.baseline_x + 0.005 * self.smoothed_x
            self.baseline_y = 0.995 * self.baseline_y + 0.005 * self.smoothed_y

        if label != "CENTER":
            self.off_road_counter += 1
            if self.off_road_counter >= self.consec_frames:
                if not self.is_gaze_distracted:
                    self.total_gaze_distractions += 1
                self.is_gaze_distracted = True
                self.gaze_label = label
        else:
            self.off_road_counter = 0
            self.is_gaze_distracted = False
            self.gaze_label = "CENTER"

        return self.gaze_label, self.current_gx, self.current_gy, \
            self.is_gaze_distracted, self.total_gaze_distractions

    def reset(self):
        self.baseline_x = None
        self.baseline_y = None
        self.calibration_count = 0
        self.smoothed_x = None
        self.smoothed_y = None
        self.current_gx = 0.5
        self.current_gy = 0.5
        self.gaze_label = "CENTER"
        self.off_road_counter = 0
        self.is_gaze_distracted = False
        self.total_gaze_distractions = 0
        self._history.clear()
