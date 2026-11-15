# Face-landmark models

| File | Purpose | Source | License |
|---|---|---|---|
| `face_landmarker.task` | Primary MediaPipe FaceLandmarker (478 points with irises) | Google MediaPipe | Apache-2.0 |
| `face_detection_short_range.onnx` | Fallback face detector (BlazeFace short-range, 128×128) | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) | Apache-2.0 |
| `face_landmarker_Nx3x256x256.onnx` | Fallback 478-point face mesh (256×256, includes 10 iris points) | [yakhyo/mediapipe-face-mesh-onnx](https://github.com/yakhyo/mediapipe-face-mesh-onnx) | Apache-2.0 |

The ONNX fallback runs through **OpenCV DNN only** — no ONNX Runtime or
MediaPipe native DLL is required. This matters on Windows machines protected by
Smart App Control / WDAC, which may block MediaPipe's unsigned
`libmediapipe.dll`.

## Attribution

The ONNX exports are by Yakhyokhuja Valikhujaev
(https://github.com/yakhyo/mediapipe-face-mesh-onnx, Apache-2.0), derived from
Google MediaPipe's Face Mesh / Face Landmarker weights (Apache-2.0).
Inference preprocessing and post-processing in this project follow that
reference implementation:

- BlazeFace: 128×128 letterboxed RGB input in `[-1, 1]`, 896 SSD anchors,
  weighted NMS, 6 keypoints per detection.
- Face Landmarker: rotated square ROI (`1.5×` long side, eye-line roll
  normalised) resampled to 256×256 RGB in `[0, 1]`; landmarks are mapped back
  through the inverse affine transform.

If a model file is missing, `src/face_mesh_detector.py` downloads it
automatically from the release URLs.
