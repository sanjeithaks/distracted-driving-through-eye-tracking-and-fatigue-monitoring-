"""
High-Precision Face and Landmark Detector using MediaPipe FaceLandmarker
Maps high-density facial mesh to the 68-landmark Kazemi-Sullivan conventions
as described in the IEEE paper.
"""
import os
import cv2
import numpy as np
import mediapipe.tasks as mt
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp


class FaceMeshDetector:
    """
    Detects faces and extracts key landmark points for EAR, MOR, and Head Pose.
    Uses singleton detector reuse to prevent C++ thread handle conflicts on Windows.
    """
    _cached_detector = None

    def __init__(self, model_path=None, min_detection_confidence=0.5):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        if not os.path.exists(model_path):
            print(f"[INFO] Downloading pre-trained face landmarker model to: {model_path}...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("[INFO] Model downloaded successfully.")
            except Exception as e:
                raise FileNotFoundError(f"Failed to auto-download model from {model_url}: {e}")

        if FaceMeshDetector._cached_detector is None:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True
            )
            FaceMeshDetector._cached_detector = FaceLandmarker.create_from_options(options)

        self.detector = FaceMeshDetector._cached_detector

        # Canonical landmark indices in MediaPipe 468 mesh
        # Right Eye (image left from camera view)
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        # Left Eye (image right from camera view)
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

        # Inner & Outer Lip landmarks for Mouth Opening Ratio (MOR)
        self.MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178]

        # Key points for 3D Head Pose (PnP)
        # [Nose Tip, Chin, Left Eye Outer, Right Eye Outer, Mouth Left, Mouth Right]
        self.HEAD_POSE_INDICES = [1, 152, 263, 33, 291, 61]

        # Contours for full eye rendering
        self.RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

        # Iris landmarks (MediaPipe FaceLandmarker returns 478 points when iris is available).
        # 468 = right iris center, 473 = left iris center (image coordinates).
        # The 4 surrounding points are used to draw the iris ring.
        self.RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def process(self, frame_bgr):
        """
        Process a BGR video frame.
        Returns a dict with:
            - 'detected': bool
            - 'bbox': (x, y, w, h)
            - 'left_eye_pts': list of (x, y)
            - 'right_eye_pts': list of (x, y)
            - 'mouth_pts': list of (x, y)
            - 'head_pose_pts': list of (x, y)
            - 'left_eye_contour': list of (x, y)
            - 'right_eye_contour': list of (x, y)
            - 'lips_contour': list of (x, y)
            - 'all_landmarks_2d': list of (x, y)
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return {'detected': False}

        landmarks = detection_result.face_landmarks[0]
        pts_2d = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        # Calculate bounding box
        xs = [p[0] for p in pts_2d]
        ys = [p[1] for p in pts_2d]
        min_x, max_x = max(0, min(xs)), min(w, max(xs))
        min_y, max_y = max(0, min(ys)), min(h, max(ys))
        bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Extract required feature points
        left_eye_pts = [pts_2d[i] for i in self.LEFT_EYE_INDICES]
        right_eye_pts = [pts_2d[i] for i in self.RIGHT_EYE_INDICES]
        mouth_pts = [pts_2d[i] for i in self.MOUTH_INDICES]
        head_pose_pts = [pts_2d[i] for i in self.HEAD_POSE_INDICES]

        left_eye_contour = [pts_2d[i] for i in self.LEFT_EYE_CONTOUR]
        right_eye_contour = [pts_2d[i] for i in self.RIGHT_EYE_CONTOUR]
        lips_contour = [pts_2d[i] for i in self.LIPS_CONTOUR]

        # Iris centers (available when the 478-point mesh is produced)
        right_iris = None
        left_iris = None
        if len(pts_2d) >= 478:
            right_iris = pts_2d[self.RIGHT_IRIS_INDICES[0]]
            left_iris = pts_2d[self.LEFT_IRIS_INDICES[0]]

        return {
            'detected': True,
            'bbox': bbox,
            'left_eye_pts': left_eye_pts,
            'right_eye_pts': right_eye_pts,
            'mouth_pts': mouth_pts,
            'head_pose_pts': head_pose_pts,
            'left_eye_contour': left_eye_contour,
            'right_eye_contour': right_eye_contour,
            'lips_contour': lips_contour,
            'right_iris_center': right_iris,
            'left_iris_center': left_iris,
            'all_landmarks_2d': pts_2d
        }
