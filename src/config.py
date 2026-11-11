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

# --- Head Pose / Distraction Thresholds (in degrees) ---
# Yaw: Looking left/right (peeping window, talking to passenger beside driver)
YAW_LEFT_THRESHOLD = 12.0     # Looking left / passenger from camera's perspective
YAW_RIGHT_THRESHOLD = -12.0   # Looking right / window from camera's perspective

# Pitch: Looking down (looking into phone / texting) or looking up
PITCH_DOWN_THRESHOLD = -14.0  # Head tilted down looking at mobile phone
PITCH_UP_THRESHOLD = 20.0     # Head tilted up

# Consecutive frames for head distraction alert
HEAD_POSE_CONSEC_FRAMES = 10

# --- Embedded System (Arduino / Raspberry Pi) ---
SERIAL_PORT = "COM3"           # Default port (auto-detect fallback available)
SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.1
ENABLE_EMBEDDED_SERIAL = True  # Set False if running pure software mode

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
