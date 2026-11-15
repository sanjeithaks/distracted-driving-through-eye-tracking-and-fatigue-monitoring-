"""
Configuration and Hyperparameters for Distracted Driving & Fatigue Monitoring System
Based on IEEE Research Paper:
"Detect distracted driving through eye tracking and fatigue monitoring using embedded system"
"""
import os

# --- Model Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FACE_LANDMARKER_MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SOUNDS_DIR = os.path.join(ASSETS_DIR, "sounds")

# --- Eye Aspect Ratio (EAR) Thresholds ---
# Open eye EAR is typically ~0.25 - 0.35. Falls below 0.21 during closure/blinks.
EAR_THRESHOLD = 0.21

# Number of consecutive frames eyes must be closed to trigger drowsiness alert
# At 30 FPS, 25 frames ~ 0.83 seconds
EAR_CONSEC_FRAMES = 25

# Normal blink minimum frames (to avoid counting noise as blinks)
BLINK_MIN_FRAMES = 2

# --- Mouth Opening Ratio (MOR) Thresholds ---
# Normal mouth MOR is ~0.06 - 0.18. When yawning, MOR exceeds 0.38 - 0.55.
MOR_THRESHOLD = 0.38

# Consecutive frames mouth must be open wide to register as a full yawn
MOR_CONSEC_FRAMES = 12

# Cooldown frames between consecutive yawns
YAWN_COOLDOWN_FRAMES = 30

# --- Head Pose / Distraction Thresholds (in degrees, deviation from neutral) ---
# Calibrated per-driver; small movements inside the dead-band are ignored.
# 25 deg sustained ~1 s is used for a clear head turn; a brief glance does not alert.
YAW_LEFT_THRESHOLD = 25.0     # deviation to the left / passenger
YAW_RIGHT_THRESHOLD = -25.0   # deviation to the right / window

# Pitch: looking down (phone) or up. Deviation from the calibrated neutral pose.
PITCH_DOWN_THRESHOLD = -22.0
PITCH_UP_THRESHOLD = 25.0

# Consecutive frames (at 30 FPS) for a sustained head-pose distraction
HEAD_POSE_CONSEC_FRAMES = 30
# Frames below the release threshold needed to clear the alert (hysteresis)
HEAD_POSE_RELEASE_FRAMES = 10
HEAD_POSE_EXIT_MARGIN = 6.0

# Personalized calibration: median neutral yaw/pitch from the first stable frames
HEAD_CALIBRATION_FRAMES = 45
HEAD_CALIBRATION_MAX_FRAMES = 120
HEAD_POSE_SMOOTHING = 0.5

# --- Iris / Eye-Gaze Tracking (true eye tracking) ---
# Normalised iris position is ~0.5 when looking straight ahead.
# A deviation larger than these deltas is treated as an off-road gaze.
GAZE_HORIZONTAL_DELTA = 0.10
GAZE_VERTICAL_DELTA = 0.20
GAZE_CENTER_BAND = 0.03        # dead-band around the neutral position
GAZE_CONSEC_FRAMES = 30        # sustained off-road gaze (~1 s) before alert
GAZE_CALIBRATION_FRAMES = 30   # frames used to learn the neutral gaze
GAZE_SMOOTHING = 0.35          # exponential smoothing factor (0-1)

# --- Fatigue Analytics: PERCLOS, Microsleeps & Composite Risk Index ---
# PERCLOS (Wierwille et al., 1994; FHWA/NHTSA) = proportion of time the eyes
# are more than ~80% closed, evaluated over a rolling window.
PERCLOS_WINDOW_FRAMES = 900          # ~30 s at 30 FPS
PERCLOS_CLOSURE_OPENNESS = 0.30      # eye openness below 30% counts as closed
PERCLOS_BASELINE_ADAPT = 0.05        # slow adaptation of the open-eye baseline
MICROSLEEP_MIN_FRAMES = 15           # >= 0.5 s continuous closure = microsleep
SOURCE_FPS = 30.0                    # nominal source frame rate for time metrics

# Weighted contributions (sum = 100 points)
FATIGUE_WEIGHT_PERCLOS = 50.0        # PERCLOS contributes up to 50 points
FATIGUE_WEIGHT_MICROSLEEP = 20.0     # microsleep episodes up to 20 points
FATIGUE_WEIGHT_YAWN = 20.0           # yawn rate up to 20 points
FATIGUE_WEIGHT_CLOSURE = 10.0        # longest closure up to 10 points
FATIGUE_RISK_MODERATE = 25.0
FATIGUE_RISK_HIGH = 50.0
FATIGUE_RISK_CRITICAL = 75.0

# --- Alert Evidence Snapshots ---
SAVE_ALERT_SNAPSHOTS = True
ALERT_SNAPSHOT_DIR = "snapshots"     # relative to the session log directory

# --- Embedded System (Arduino / Raspberry Pi) ---
SERIAL_PORT = "COM3"           # Default port (auto-detect fallback available)
SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.1
ENABLE_EMBEDDED_SERIAL = False  # PC simulation by default; enable with --serial

# Embedded Vibration Frequencies (as specified in IEEE paper: 100 Hz to 300 Hz)
SEATBELT_VIBRATION_FREQ_HZ = 200

# --- Audio Warning Settings ---
ENABLE_AUDIO_ALARM = True
ALARM_BEEP_FREQ_HZ = 1200      # 1.2 kHz piercing alert tone
ALARM_DURATION_SEC = 0.35

# --- Camera & Display Settings ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DISPLAY_FPS = True
SHOW_LANDMARKS = True
SHOW_BOUNDING_BOX = True
SHOW_HEAD_AXIS = True
SHOW_METRICS_HUD = True
