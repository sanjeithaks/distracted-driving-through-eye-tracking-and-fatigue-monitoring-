"""
Head Pose Estimation & Distracted Driving Analysis Module
----------------------------------------------------------
Computes 3D head pose angles (Yaw, Pitch, Roll) using cv2.solvePnP and detects
driver distractions:
- Peeping through the window / looking left
- Talking to a passenger / looking right
- Looking down into a mobile phone / texting

Research-grade false-positive control
-------------------------------------
* **Per-driver neutral calibration** - the median yaw/pitch of the first stable
  frames is stored as the neutral pose.  Driver-specific thresholds are required
  for meaningful head-yaw based attention estimation (Talamonti et al., 2014;
  Chalmers, 2021).
* **Large sustained deviations only** - alerts require a deviation of ~25 deg
  for ~1 s (30 frames at 30 FPS).  NHTSA's distraction guidelines highlight
  glances of 2 s and driver monitoring systems typically act on sustained
  attention shifts, not micro-glances.
* **Hysteresis** - once alerting, the pose must return well inside the dead-band
  for several frames before the alert clears, preventing warning chatter.
* **Adaptive neutral** - extremely slow baseline adaptation while centred keeps
  the calibration valid if the driver shifts posture.
"""
import cv2
import numpy as np

from .config import (
    YAW_LEFT_THRESHOLD, YAW_RIGHT_THRESHOLD,
    PITCH_DOWN_THRESHOLD, PITCH_UP_THRESHOLD,
    HEAD_POSE_CONSEC_FRAMES, HEAD_POSE_RELEASE_FRAMES, HEAD_POSE_EXIT_MARGIN,
    HEAD_CALIBRATION_FRAMES, HEAD_CALIBRATION_MAX_FRAMES, HEAD_POSE_SMOOTHING,
)


class HeadPoseEstimator:
    """
    Estimates 3D head orientation and detects sustained looking-away events.
    """

    def __init__(self,
                 yaw_left_thresh=YAW_LEFT_THRESHOLD,
                 yaw_right_thresh=YAW_RIGHT_THRESHOLD,
                 pitch_down_thresh=PITCH_DOWN_THRESHOLD,
                 pitch_up_thresh=PITCH_UP_THRESHOLD,
                 consec_frames=HEAD_POSE_CONSEC_FRAMES,
                 calibration_frames=HEAD_CALIBRATION_FRAMES,
                 calibration_max_frames=HEAD_CALIBRATION_MAX_FRAMES,
                 exit_margin=HEAD_POSE_EXIT_MARGIN,
                 release_frames=HEAD_POSE_RELEASE_FRAMES,
                 smoothing=HEAD_POSE_SMOOTHING):
        self.yaw_threshold = max(abs(float(yaw_left_thresh)),
                                 abs(float(yaw_right_thresh)))
        self.pitch_down_thresh = float(pitch_down_thresh)
        self.pitch_up_thresh = float(pitch_up_thresh)
        self.consec_frames = int(consec_frames)
        self.calibration_frames = int(calibration_frames)
        self.calibration_max_frames = max(int(calibration_max_frames),
                                          self.calibration_frames)
        self.exit_margin = float(exit_margin)
        self.release_frames = int(release_frames)
        self.smoothing = float(smoothing)

        # Generic 3D facial model points (millimetres)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Neutral-pose calibration
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.is_calibrated = self.calibration_frames <= 0
        self._calibration_samples = []

        # Smoothing
        self.smoothed_yaw = None
        self.smoothed_pitch = None

        # Deviations from neutral (degrees)
        self.yaw_deviation = 0.0
        self.pitch_deviation = 0.0

        # Distraction state
        self.distracted_frame_counter = 0
        self.release_counter = 0
        self.total_distractions = 0
        self.is_distracted = False
        self.distraction_reason = "NORMAL"

    # ------------------------------------------------------------------ #
    @property
    def calibration_progress(self):
        if self.is_calibrated:
            return 1.0
        if self.calibration_frames <= 0:
            return 1.0
        return min(1.0, len(self._calibration_samples) / float(self.calibration_frames))

    # ------------------------------------------------------------------ #
    def estimate_pose(self, image_points_2d, frame_shape):
        """
        Estimate Euler angles from 6 standard 2D facial landmarks:
        [nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner,
         right_mouth_corner]

        Returns (yaw, pitch, roll, rvec, tvec, is_valid).
        """
        if len(image_points_2d) < 6:
            return 0.0, 0.0, 0.0, None, None, False

        height, width = frame_shape[:2]
        image_points = np.array(image_points_2d, dtype=np.float64)

        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0, None, None, False

        rotation_mat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

        return float(angles[1]), float(angles[0]), float(angles[2]), \
            rvec, tvec, True

    # ------------------------------------------------------------------ #
    def _calibrate(self, yaw, pitch):
        """Collect neutral-pose samples and finish when stable."""
        if not self.is_calibrated:
            self._calibration_samples.append((yaw, pitch))
            n = len(self._calibration_samples)

            if n >= self.calibration_frames:
                yaws = np.array([s[0] for s in self._calibration_samples])
                pitches = np.array([s[1] for s in self._calibration_samples])
                # Wait for a stable neutral pose (unless the max window is reached)
                stable = (np.std(yaws) <= 6.0 and np.std(pitches) <= 6.0)
                if stable or n >= self.calibration_max_frames:
                    self.baseline_yaw = float(np.median(yaws))
                    self.baseline_pitch = float(np.median(pitches))
                    self.is_calibrated = True
                    self._calibration_samples = []
            return

    # ------------------------------------------------------------------ #
    def update(self, yaw, pitch, roll):
        """
        Update the distraction state with the latest head pose.

        Returns (is_distracted, distraction_reason, total_distractions).
        During the calibration window it always returns the CALIBRATING state.
        """
        # Smooth noisy solvePnP output
        if self.smoothed_yaw is None:
            self.smoothed_yaw, self.smoothed_pitch = float(yaw), float(pitch)
        else:
            a = self.smoothing
            self.smoothed_yaw = a * yaw + (1.0 - a) * self.smoothed_yaw
            self.smoothed_pitch = a * pitch + (1.0 - a) * self.smoothed_pitch

        if not self.is_calibrated:
            self._calibrate(self.smoothed_yaw, self.smoothed_pitch)
            self.yaw_deviation = 0.0
            self.pitch_deviation = 0.0
            return False, "CALIBRATING", self.total_distractions

        self.yaw_deviation = self.smoothed_yaw - self.baseline_yaw
        self.pitch_deviation = self.smoothed_pitch - self.baseline_pitch

        # Candidate distraction based on deviation from the neutral pose
        candidate = False
        reason = "NORMAL"
        if self.yaw_deviation > self.yaw_threshold:
            candidate = True
            reason = "LOOKING LEFT / PASSENGER"
        elif self.yaw_deviation < -self.yaw_threshold:
            candidate = True
            reason = "LOOKING RIGHT / WINDOW"
        elif self.pitch_deviation < self.pitch_down_thresh:
            candidate = True
            reason = "LOOKING DOWN / PHONE"
        elif self.pitch_deviation > self.pitch_up_thresh:
            candidate = True
            reason = "LOOKING UP"

        # Hysteresis: require a sustained offset to enter, and a good return
        # inside the dead-band before clearing.
        if self.is_distracted:
            still_away = (
                abs(self.yaw_deviation) > (self.yaw_threshold - self.exit_margin)
                or self.pitch_deviation < (self.pitch_down_thresh + self.exit_margin)
                or self.pitch_deviation > (self.pitch_up_thresh - self.exit_margin)
            )
            if still_away:
                self.release_counter = 0
            else:
                self.release_counter += 1
                if self.release_counter >= self.release_frames:
                    self.is_distracted = False
                    self.distraction_reason = "NORMAL"
                    self.distracted_frame_counter = 0
                    self.release_counter = 0
        else:
            if candidate:
                self.distracted_frame_counter += 1
                if self.distracted_frame_counter >= self.consec_frames:
                    self.is_distracted = True
                    self.total_distractions += 1
                    self.distraction_reason = reason
                    self.release_counter = 0
            else:
                self.distracted_frame_counter = 0

        # Very slow adaptation while centred (keeps calibration valid if the
        # driver changes posture, without drifting during a real distraction).
        if not candidate and not self.is_distracted:
            self.baseline_yaw = 0.998 * self.baseline_yaw + 0.002 * self.smoothed_yaw
            self.baseline_pitch = 0.998 * self.baseline_pitch + 0.002 * self.smoothed_pitch

        return self.is_distracted, self.distraction_reason, self.total_distractions

    # ------------------------------------------------------------------ #
    def get_projected_axis(self, rvec, tvec, frame_shape):
        """Project the nose direction axis for visualisation."""
        if rvec is None or tvec is None:
            return None, None

        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        axis_3d = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 450.0)], dtype=np.float64)
        projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        p1 = (int(projected[0][0][0]), int(projected[0][0][1]))
        p2 = (int(projected[1][0][0]), int(projected[1][0][1]))
        return p1, p2

    # ------------------------------------------------------------------ #
    def reset_calibration(self):
        """Discard the neutral pose and recalibrate on the next frames."""
        self.is_calibrated = self.calibration_frames <= 0
        self._calibration_samples = []
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.smoothed_yaw = None
        self.smoothed_pitch = None
        self.yaw_deviation = 0.0
        self.pitch_deviation = 0.0

    def reset(self):
        self.reset_calibration()
        self.distracted_frame_counter = 0
        self.release_counter = 0
        self.total_distractions = 0
        self.is_distracted = False
        self.distraction_reason = "NORMAL"
