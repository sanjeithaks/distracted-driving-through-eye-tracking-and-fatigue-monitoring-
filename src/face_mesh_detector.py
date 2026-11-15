"""
High-Precision Face and Landmark Detector with Dual Backends
------------------------------------------------------------
Maps a dense facial mesh to the 68-landmark Kazemi-Sullivan conventions used
by the IEEE paper (EAR, MOR, head pose and iris gaze).

Two interchangeable backends are provided:

1. **MediaPipe FaceLandmarker** (primary, 478 points with irises).  Requires the
   MediaPipe native runtime to be loadable.
2. **OpenCV-DNN ONNX fallback** (secondary, 478 points with irises).  Uses two
   Apache-2.0 ONNX models running entirely inside OpenCV's ``dnn`` module:
   BlazeFace short-range detection + MediaPipe Face Landmarker.  This keeps the
   project working on locked-down Windows machines (e.g. Smart App Control)
   where unsigned native DLLs such as ``libmediapipe.dll`` are blocked.

Both backends expose exactly the same ``process(frame_bgr)`` dictionary, so the
rest of the pipeline is backend-agnostic.

ONNX model attribution
----------------------
* ``face_detection_short_range.onnx`` and ``face_landmarker_Nx3x256x256.onnx``
  are exported by https://github.com/yakhyo/mediapipe-face-mesh-onnx
  (Apache-2.0), from Google MediaPipe weights (Apache-2.0).
* Inference/post-processing re-implemented here with OpenCV DNN, following the
  reference preprocessing (letterbox [-1, 1] for BlazeFace; rotated square ROI,
  RGB [0, 1] for the 256x256 landmarker).
"""
import os
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core.base_options import BaseOptions
    MEDIAPIPE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the host security policy
    MEDIAPIPE_AVAILABLE = False


# Canonical landmark indices in the MediaPipe 478-point mesh
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 81, 13, 311, 291, 402, 14, 178]
HEAD_POSE_INDICES = [1, 152, 263, 33, 291, 61]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]

ONNX_DETECTOR_URL = ("https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/"
                     "download/weights/face_detection_short_range.onnx")
ONNX_LANDMARKER_URL = ("https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/"
                       "download/weights/face_landmarker_Nx3x256x256.onnx")


def package_landmarks(pts_2d, width, height):
    """Convert 478 mesh points into the pipeline's feature dictionary."""
    bbox_min_x = max(0, min(p[0] for p in pts_2d))
    bbox_max_x = min(width, max(p[0] for p in pts_2d))
    bbox_min_y = max(0, min(p[1] for p in pts_2d))
    bbox_max_y = min(height, max(p[1] for p in pts_2d))
    bbox = (bbox_min_x, bbox_min_y,
            bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y)

    right_iris = pts_2d[RIGHT_IRIS_INDICES[0]] if len(pts_2d) >= 478 else None
    left_iris = pts_2d[LEFT_IRIS_INDICES[0]] if len(pts_2d) >= 478 else None

    return {
        'detected': True,
        'bbox': bbox,
        'left_eye_pts': [pts_2d[i] for i in LEFT_EYE_INDICES],
        'right_eye_pts': [pts_2d[i] for i in RIGHT_EYE_INDICES],
        'mouth_pts': [pts_2d[i] for i in MOUTH_INDICES],
        'head_pose_pts': [pts_2d[i] for i in HEAD_POSE_INDICES],
        'left_eye_contour': [pts_2d[i] for i in LEFT_EYE_CONTOUR],
        'right_eye_contour': [pts_2d[i] for i in RIGHT_EYE_CONTOUR],
        'lips_contour': [pts_2d[i] for i in LIPS_CONTOUR],
        'right_iris_center': right_iris,
        'left_iris_center': left_iris,
        'all_landmarks_2d': pts_2d,
    }


# --------------------------------------------------------------------------- #
# Backend 1: MediaPipe FaceLandmarker
# --------------------------------------------------------------------------- #
class _MediaPipeBackend:
    _cached_detector = None

    def __init__(self, model_path, min_detection_confidence=0.5):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe is not importable")

        if _MediaPipeBackend._cached_detector is None:
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            _MediaPipeBackend._cached_detector = FaceLandmarker.create_from_options(options)

        self.detector = _MediaPipeBackend._cached_detector

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return {'detected': False}

        pts_2d = [(int(lm.x * w), int(lm.y * h)) for lm in result.face_landmarks[0]]
        return package_landmarks(pts_2d, w, h)


# --------------------------------------------------------------------------- #
# Backend 2: OpenCV DNN ONNX (BlazeFace + Face Landmarker)
# --------------------------------------------------------------------------- #
def _generate_face_anchors():
    """SSD anchor centres for BlazeFace short-range at 128x128."""
    anchors = []
    strides = [8, 16, 16, 16]
    idx = 0
    while idx < len(strides):
        last = idx
        while last < len(strides) and strides[last] == strides[idx]:
            last += 1
        repeats = 2 * (last - idx)
        cells = 128 // strides[idx]
        for y in range(cells):
            for x in range(cells):
                anchors.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * repeats)
        idx = last
    return np.array(anchors, dtype=np.float32)


def _weighted_nms(detections, iou_threshold=0.3):
    """MediaPipe's weighted non-maximum suppression (boxes + keypoints)."""
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1, ty1 = top[1] - top[3] / 2, top[2] - top[4] / 2
        tx2, ty2 = top[1] + top[3] / 2, top[2] + top[4] / 2
        inter = (np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1)) *
                 np.maximum(0, np.minimum(y2, ty2) - np.maximum(y1, ty1)))
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter
        iou = inter / np.maximum(union, 1e-9)

        overlapping = remaining[iou > iou_threshold]
        weights = overlapping[:, :1]
        blended = top.copy()
        blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[iou <= iou_threshold]
    return np.array(output)


class _OnnxBackend:
    INPUT_SIZE = 256
    FACE_INPUT_SIZE = 128

    def __init__(self, base_dir, min_detection_confidence=0.5):
        model_dir = os.path.join(base_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        detector_path = os.path.join(model_dir, "face_detection_short_range.onnx")
        landmarker_path = os.path.join(model_dir, "face_landmarker_Nx3x256x256.onnx")

        self._ensure_model(detector_path, ONNX_DETECTOR_URL)
        self._ensure_model(landmarker_path, ONNX_LANDMARKER_URL)

        self.detector_net = cv2.dnn.readNetFromONNX(detector_path)
        self.landmarker_net = cv2.dnn.readNetFromONNX(landmarker_path)
        self.anchors = _generate_face_anchors()
        self.threshold = min_detection_confidence

    @staticmethod
    def _ensure_model(path, url):
        if os.path.exists(path) and os.path.getsize(path) > 100000:
            return
        print(f"[INFO] Downloading ONNX fallback model to: {path}...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            raise FileNotFoundError(
                f"ONNX model missing and download failed ({url}): {exc}"
            )

    # -- detection ------------------------------------------------------- #
    def _detect(self, image):
        h, w = image.shape[:2]
        scale = self.FACE_INPUT_SIZE / max(h, w)
        pad_x = (self.FACE_INPUT_SIZE - w * scale) / 2.0
        pad_y = (self.FACE_INPUT_SIZE - h * scale) / 2.0
        matrix = np.array([[scale, 0.0, pad_x], [0.0, scale, pad_y]], dtype=np.float32)
        canvas = cv2.warpAffine(image, matrix, (self.FACE_INPUT_SIZE, self.FACE_INPUT_SIZE))
        rgb = canvas[:, :, ::-1].astype(np.float32)
        blob = np.transpose((rgb - 127.5) / 127.5, (2, 0, 1))[np.newaxis]

        self.detector_net.setInput(blob)
        regressors, logits = self.detector_net.forward(['regressors', 'scores'])

        scores = 1.0 / (1.0 + np.exp(-logits[0].ravel().astype(np.float64)))
        keep = scores >= self.threshold
        if not keep.any():
            return None

        reg = regressors[0][keep].astype(np.float64)
        anchor = self.anchors[keep]
        rows = np.empty((int(keep.sum()), 5 + 12))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = reg[:, 0:2] / self.FACE_INPUT_SIZE + anchor
        rows[:, 3:5] = reg[:, 2:4] / self.FACE_INPUT_SIZE
        for k in range(6):
            rows[:, 5 + 2 * k:7 + 2 * k] = reg[:, 4 + 2 * k:6 + 2 * k] / self.FACE_INPUT_SIZE + anchor
        rows = _weighted_nms(rows)

        def unletterbox(xy):
            return (xy * self.FACE_INPUT_SIZE - (pad_x, pad_y)) / scale

        centers = unletterbox(rows[:, 1:3])
        halves = rows[:, 3:5] * self.FACE_INPUT_SIZE / scale / 2.0
        bboxes = np.concatenate([centers - halves, centers + halves], axis=1)
        keypoints = unletterbox(rows[:, 5:].reshape(-1, 6, 2))
        return bboxes.astype(np.float32), keypoints.astype(np.float32), rows[:, 0], w, h

    # -- mesh ------------------------------------------------------------ #
    def _mesh(self, image, bbox, keypoints):
        """Warp a rotated square ROI and run the 478-point landmarker."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        side = 1.5 * max(x2 - x1, y2 - y1)

        dx = float(keypoints[1][0] - keypoints[0][0])
        dy = float(keypoints[1][1] - keypoints[0][1])
        angle = float(np.degrees(np.arctan2(dy, dx)))

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, self.INPUT_SIZE / side)
        matrix[0, 2] += self.INPUT_SIZE / 2.0 - cx
        matrix[1, 2] += self.INPUT_SIZE / 2.0 - cy

        crop = cv2.warpAffine(image, matrix, (self.INPUT_SIZE, self.INPUT_SIZE))
        inverse = cv2.invertAffineTransform(matrix)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis]

        self.landmarker_net.setInput(blob)
        landmarks, _score = self.landmarker_net.forward(['landmarks', 'score'])
        landmarks = landmarks[0].astype(np.float64)

        xy = landmarks[:, :2] @ inverse[:, :2].T + inverse[:, 2]
        z = landmarks[:, 2] * side / self.INPUT_SIZE
        return xy, z

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        detection = self._detect(frame_bgr)
        if detection is None:
            return {'detected': False}

        bboxes, keypoints, scores, _, _ = detection
        best = int(np.argmax(scores))
        xy, _z = self._mesh(frame_bgr, bboxes[best], keypoints[best])

        pts_2d = [(int(round(p[0])), int(round(p[1]))) for p in xy]
        return package_landmarks(pts_2d, w, h)


# --------------------------------------------------------------------------- #
# Public facade
# --------------------------------------------------------------------------- #
class FaceMeshDetector:
    """
    Detects faces and extracts landmarks for EAR, MOR, head pose and gaze.

    Selects MediaPipe automatically when its native runtime loads; otherwise it
    transparently falls back to the OpenCV-DNN ONNX backend.
    """

    _mediapipe_broken = False

    def __init__(self, model_path=None, min_detection_confidence=0.5, backend="auto"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        self.backend = None
        self._impl = None

        if (backend in ("auto", "mediapipe") and MEDIAPIPE_AVAILABLE
                and not FaceMeshDetector._mediapipe_broken):
            try:
                self._impl = _MediaPipeBackend(model_path, min_detection_confidence)
                self.backend = "mediapipe"
            except Exception as exc:
                FaceMeshDetector._mediapipe_broken = True
                if backend == "mediapipe":
                    raise
                print(f"[WARN] MediaPipe runtime unavailable ({exc}). "
                      f"Switching to OpenCV-DNN ONNX backend.")

        if self._impl is None:
            if backend == "mediapipe":
                raise RuntimeError("MediaPipe backend requested but unavailable")
            self._impl = _OnnxBackend(base_dir, min_detection_confidence)
            self.backend = "opencv-dnn"

        # Expose the landmark index tables for backwards compatibility
        for name in ("RIGHT_EYE_INDICES", "LEFT_EYE_INDICES", "MOUTH_INDICES",
                     "HEAD_POSE_INDICES", "RIGHT_EYE_CONTOUR", "LEFT_EYE_CONTOUR",
                     "LIPS_CONTOUR", "RIGHT_IRIS_INDICES", "LEFT_IRIS_INDICES"):
            setattr(self, name, globals()[name])

    def process(self, frame_bgr):
        return self._impl.process(frame_bgr)
