"""
Driver Fatigue & Distraction Detection Pipeline Engine
Integrates:
1. Face Detection & Kazemi-Sullivan/MediaPipe Landmark Localization
2. Eye Aspect Ratio (EAR) computation and Drowsiness tracking
3. Mouth Opening Ratio (MOR) computation and Yawn tracking
4. 3D Head Pose Estimation (solvePnP) and Looking Away detection
5. Audio Alert dispatch (AlertSystem)
6. Embedded System Serial signal dispatch (EmbeddedSerialBridge)
7. On-screen visualization HUD (Visualizer)
"""
import time
import cv2
from .config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, BLINK_MIN_FRAMES,
    MOR_THRESHOLD, MOR_CONSEC_FRAMES, YAWN_COOLDOWN_FRAMES,
    YAW_LEFT_THRESHOLD, YAW_RIGHT_THRESHOLD,
    PITCH_DOWN_THRESHOLD, PITCH_UP_THRESHOLD, HEAD_POSE_CONSEC_FRAMES,
    SERIAL_PORT, SERIAL_BAUD_RATE, ENABLE_EMBEDDED_SERIAL,
    ENABLE_AUDIO_ALARM, SHOW_LANDMARKS, SHOW_BOUNDING_BOX,
    SHOW_HEAD_AXIS, SHOW_METRICS_HUD
)
from .face_mesh_detector import FaceMeshDetector
from .facial_features import calculate_ear, calculate_mor, EyeTracker, YawnTracker
from .head_pose import HeadPoseEstimator
from .gaze_tracker import GazeTracker
from .alert_system import AlertSystem
from .embedded_serial import EmbeddedSerialBridge
from .visualizer import Visualizer


class DriverMonitoringEngine:
    """
    Main pipeline engine orchestrating all modules.
    """
    def __init__(self,
                 ear_thresh=EAR_THRESHOLD,
                 ear_consec_frames=EAR_CONSEC_FRAMES,
                 mor_thresh=MOR_THRESHOLD,
                 mor_consec_frames=MOR_CONSEC_FRAMES,
                 enable_serial=ENABLE_EMBEDDED_SERIAL,
                 enable_audio=ENABLE_AUDIO_ALARM,
                 serial_port=SERIAL_PORT):
        # 1. Detector
        self.detector = FaceMeshDetector()

        # 2. Trackers
        self.eye_tracker = EyeTracker(
            ear_thresh=ear_thresh,
            consec_frames=ear_consec_frames,
            blink_min_frames=BLINK_MIN_FRAMES
        )
        self.yawn_tracker = YawnTracker(
            mor_thresh=mor_thresh,
            consec_frames=mor_consec_frames,
            cooldown_frames=YAWN_COOLDOWN_FRAMES
        )
        self.head_pose = HeadPoseEstimator(
            yaw_left_thresh=YAW_LEFT_THRESHOLD,
            yaw_right_thresh=YAW_RIGHT_THRESHOLD,
            pitch_down_thresh=PITCH_DOWN_THRESHOLD,
            pitch_up_thresh=PITCH_UP_THRESHOLD,
            consec_frames=HEAD_POSE_CONSEC_FRAMES
        )
        # Iris-based eye tracking (gaze direction independent of head pose)
        self.gaze_tracker = GazeTracker()

        # 3. Audio Alert & Embedded Hardware Serial
        self.alert_system = AlertSystem(enabled=enable_audio)
        self.serial_bridge = EmbeddedSerialBridge(
            port=serial_port,
            baud_rate=SERIAL_BAUD_RATE,
            enabled=enable_serial
        )

        # 4. Visualizer
        self.visualizer = Visualizer(
            show_landmarks=SHOW_LANDMARKS,
            show_bbox=SHOW_BOUNDING_BOX,
            show_axis=SHOW_HEAD_AXIS,
            show_hud=SHOW_METRICS_HUD
        )

        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(self, frame_bgr):
        """
        Process a single video frame through the pipeline.
        Returns:
            annotated_frame: np.ndarray
            telemetry: dict containing all metrics and statuses
        """
        now = time.time()
        dt = now - self.prev_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = now

        # Run face & landmark detector
        detection = self.detector.process(frame_bgr)

        ear = 0.0
        mor = 0.0
        yaw, pitch, roll = 0.0, 0.0, 0.0
        axis_pts = (None, None)
        is_drowsy = False
        is_yawning = False
        is_distracted = False
        distraction_reason = "NORMAL"
        gaze_label = "N/A"
        gaze_x, gaze_y = 0.5, 0.5
        is_gaze_distracted = False

        if detection['detected']:
            # 1. EAR (Eye Aspect Ratio)
            left_ear = calculate_ear(detection['left_eye_pts'])
            right_ear = calculate_ear(detection['right_eye_pts'])
            is_drowsy, is_blinking, ear = self.eye_tracker.update(left_ear, right_ear)

            # 1b. Iris / gaze tracking (only meaningful while eyes are open)
            gaze_label, gaze_x, gaze_y, is_gaze_distracted, _ = self.gaze_tracker.update(
                detection['right_eye_pts'],
                detection['left_eye_pts'],
                detection.get('right_iris_center'),
                detection.get('left_iris_center'),
                eyes_open=(ear >= self.eye_tracker.ear_thresh)
            )

            # 2. MOR (Mouth Opening Ratio)
            mor_val = calculate_mor(detection['mouth_pts'])
            is_yawning, total_yawns, mor = self.yawn_tracker.update(mor_val)

            # 3. Head Pose & Looking Away
            yaw, pitch, roll, rvec, tvec, valid_pose = self.head_pose.estimate_pose(
                detection['head_pose_pts'], frame_bgr.shape
            )
            if valid_pose:
                is_distracted, distraction_reason, _ = self.head_pose.update(yaw, pitch, roll)
                axis_pts = self.head_pose.get_projected_axis(rvec, tvec, frame_bgr.shape)

                # Gaze distraction is used when the head itself is still, e.g.
                # the driver turns only their eyes away from the road.
                if not is_distracted and is_gaze_distracted:
                    is_distracted = True
                    distraction_reason = f"EYES OFF ROAD / GAZE {gaze_label}"

            # 4. Sound & Hardware Alert Dispatching
            # Priority: Drowsy > Distracted > Yawning > Normal
            if is_drowsy:
                self.alert_system.trigger_drowsy_alarm()
                self.serial_bridge.send_state('D')
            elif is_distracted:
                self.alert_system.trigger_distraction_alert()
                self.serial_bridge.send_state('L')
            elif is_yawning:
                self.alert_system.trigger_yawn_alert()
                self.serial_bridge.send_state('Y')
            else:
                self.serial_bridge.send_state('N')
        else:
            # Face lost - reset short-term counters
            self.eye_tracker.closed_frame_counter = 0
            self.eye_tracker.is_drowsy = False
            self.yawn_tracker.open_frame_counter = 0
            self.yawn_tracker.is_yawning = False
            self.head_pose.distracted_frame_counter = 0
            self.head_pose.is_distracted = False
            self.gaze_tracker.off_road_counter = 0
            self.gaze_tracker.is_gaze_distracted = False
            self.serial_bridge.send_state('N')

        total_distractions = (self.head_pose.total_distractions +
                              self.gaze_tracker.total_gaze_distractions)

        # 5. Render HUD & Overlays
        annotated_frame = self.visualizer.render(
            frame=frame_bgr,
            detection=detection,
            ear=ear,
            mor=mor,
            is_drowsy=is_drowsy,
            is_yawning=is_yawning,
            is_distracted=is_distracted,
            distraction_reason=distraction_reason,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            axis_pts=axis_pts,
            blinks=self.eye_tracker.total_blinks,
            yawns=self.yawn_tracker.total_yawns,
            distractions=total_distractions,
            fps=self.fps,
            serial_connected=self.serial_bridge.is_connected,
            gaze_label=gaze_label,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            ear_thresh=self.eye_tracker.ear_thresh,
            mor_thresh=self.yawn_tracker.mor_thresh
        )

        telemetry = {
            'face_detected': detection['detected'],
            'ear': ear,
            'mor': mor,
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'is_drowsy': is_drowsy,
            'is_yawning': is_yawning,
            'is_distracted': is_distracted,
            'distraction_reason': distraction_reason,
            'gaze_label': gaze_label,
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'is_gaze_distracted': is_gaze_distracted,
            'blinks': self.eye_tracker.total_blinks,
            'yawns': self.yawn_tracker.total_yawns,
            'distractions': total_distractions,
            'gaze_distractions': self.gaze_tracker.total_gaze_distractions,
            'fps': self.fps,
            'embedded_connected': self.serial_bridge.is_connected
        }

        return annotated_frame, telemetry

    def close(self):
        """Clean up resources."""
        self.serial_bridge.close()
