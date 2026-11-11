"""
Head Pose Estimation & Distracted Driving Analysis Module
Computes 3D head pose angles (Yaw, Pitch, Roll) using cv2.solvePnP
and detects driver distractions:
- Peeping through window / looking left
- Talking to passenger / looking right
- Looking down into mobile phone / texting
as documented in Section III and Figures 7, 8, 9 of the IEEE Paper.
"""
import cv2
import numpy as np


class HeadPoseEstimator:
    """
    Estimates 3D head orientation and detects looking away / distracted driving.
    """
    def __init__(self,
                 yaw_left_thresh=20.0,
                 yaw_right_thresh=-20.0,
                 pitch_down_thresh=-14.0,
                 pitch_up_thresh=22.0,
                 consec_frames=18):
        self.yaw_left_thresh = yaw_left_thresh
        self.yaw_right_thresh = yaw_right_thresh
        self.pitch_down_thresh = pitch_down_thresh
        self.pitch_up_thresh = pitch_up_thresh
        self.consec_frames = consec_frames

        # Generic 3D facial model points (in world coordinates millimeters)
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        # Distraction tracking state
        self.distracted_frame_counter = 0
        self.total_distractions = 0
        self.is_distracted = False
        self.distraction_reason = "NORMAL"

    def estimate_pose(self, image_points_2d, frame_shape):
        """
        Estimate Euler angles from 6 standard 2D facial landmarks:
        [nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth_corner, right_mouth_corner]
        
        Returns:
            (yaw, pitch, roll, rvec, tvec, is_valid)
        """
        if len(image_points_2d) < 6:
            return 0.0, 0.0, 0.0, None, None, False

        height, width = frame_shape[:2]
        image_points = np.array(image_points_2d, dtype=np.float64)

        # Approximate camera focal length and optical center
        focal_length = width
        center = (width / 2.0, height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rvec, tvec = cv2.solvePnP(
            self.model_points_3d,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, None, None, False

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles
        # RQDecomp3x3 returns angles in degrees: [pitch, yaw, roll]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        return float(yaw), float(pitch), float(roll), rvec, tvec, True

    def update(self, yaw, pitch, roll):
        """
        Evaluate distraction rules based on Euler angles.
        Returns:
            (is_distracted, distraction_reason, total_distractions)
        """
        distracted_current = False
        reason = "NORMAL"

        if yaw > self.yaw_left_thresh:
            distracted_current = True
            reason = "LOOKING LEFT / PASSENGER"
        elif yaw < self.yaw_right_thresh:
            distracted_current = True
            reason = "LOOKING RIGHT / WINDOW"
        elif pitch < self.pitch_down_thresh:
            distracted_current = True
            reason = "LOOKING DOWN / PHONE"
        elif pitch > self.pitch_up_thresh:
            distracted_current = True
            reason = "LOOKING UP"

        if distracted_current:
            self.distracted_frame_counter += 1
            if self.distracted_frame_counter >= self.consec_frames:
                if not self.is_distracted:
                    self.total_distractions += 1
                self.is_distracted = True
                self.distraction_reason = reason
        else:
            self.distracted_frame_counter = 0
            self.is_distracted = False
            self.distraction_reason = "NORMAL"

        return self.is_distracted, self.distraction_reason, self.total_distractions

    def get_projected_axis(self, rvec, tvec, frame_shape):
        """
        Project 3D pose direction vector forward from the nose for visualization.
        """
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

        # 3D line from nose (0, 0, 0) extending out along Z-axis (0, 0, 500)
        axis_3d = np.array([
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 450.0)
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
        p1 = (int(projected[0][0][0]), int(projected[0][0][1]))
        p2 = (int(projected[1][0][0]), int(projected[1][0][1]))
        return p1, p2

    def reset(self):
        self.distracted_frame_counter = 0
        self.total_distractions = 0
        self.is_distracted = False
        self.distraction_reason = "NORMAL"
