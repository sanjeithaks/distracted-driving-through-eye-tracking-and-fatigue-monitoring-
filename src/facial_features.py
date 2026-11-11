"""
Facial Feature Extraction and Metrics Computation Module
Implements Eye Aspect Ratio (EAR) and Mouth Opening Ratio (MOR)
as defined in the IEEE Research Paper:
"Detect distracted driving through eye tracking and fatigue monitoring using embedded system"
"""
import numpy as np


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two 2D points (x, y)."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio (EAR) according to the paper formula:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    eye_landmarks: list or array of 6 (x, y) coordinates ordered:
      p1: outer corner
      p2: top-outer
      p3: top-inner
      p4: inner corner
      p5: bottom-inner
      p6: bottom-outer
    """
    if len(eye_landmarks) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]

    # Vertical eye distances
    dist_vertical_1 = euclidean_distance(p2, p6)
    dist_vertical_2 = euclidean_distance(p3, p5)

    # Horizontal eye distance
    dist_horizontal = euclidean_distance(p1, p4)

    if dist_horizontal == 0:
        return 0.0

    ear = (dist_vertical_1 + dist_vertical_2) / (2.0 * dist_horizontal)
    return float(ear)


def calculate_mor(mouth_landmarks):
    """
    Compute Mouth Opening Ratio (MOR) according to the paper formula:
    MOR = (||p15 - p23|| + ||p16 - p22|| + ||p17 - p21||) / (3 * ||p19 - p13||)
    
    where p19, p13 are horizontal corners, and vertical pairs are across lips.
    Or, given standard landmarks [corner_left, top_lip_left, top_lip_center, top_lip_right,
                                  corner_right, bot_lip_right, bot_lip_center, bot_lip_left]:
    Computes ratio of average vertical opening to horizontal mouth width.
    """
    if len(mouth_landmarks) < 8:
        # Fallback if given 4 points: [left_corner, right_corner, top_lip, bottom_lip]
        if len(mouth_landmarks) >= 4:
            left, right, top, bottom = mouth_landmarks[:4]
            horiz = euclidean_distance(left, right)
            vert = euclidean_distance(top, bottom)
            return float(vert / horiz) if horiz > 0 else 0.0
        return 0.0

    # For 8-point lip landmarks:
    # 0: left corner (p13)
    # 1: top-left (p15)
    # 2: top-center (p16)
    # 3: top-right (p17)
    # 4: right corner (p19)
    # 5: bottom-right (p21)
    # 6: bottom-center (p22)
    # 7: bottom-left (p23)
    p13 = mouth_landmarks[0]
    p15 = mouth_landmarks[1]
    p16 = mouth_landmarks[2]
    p17 = mouth_landmarks[3]
    p19 = mouth_landmarks[4]
    p21 = mouth_landmarks[5]
    p22 = mouth_landmarks[6]
    p23 = mouth_landmarks[7]

    vert_1 = euclidean_distance(p15, p23)
    vert_2 = euclidean_distance(p16, p22)
    vert_3 = euclidean_distance(p17, p21)
    horiz = euclidean_distance(p19, p13)

    if horiz == 0:
        return 0.0

    mor = (vert_1 + vert_2 + vert_3) / (3.0 * horiz)
    return float(mor)


class EyeTracker:
    """
    Tracks eye closure duration, blink counts, and drowsiness state.
    """
    def __init__(self, ear_thresh=0.21, consec_frames=25, blink_min_frames=2):
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.blink_min_frames = blink_min_frames

        self.closed_frame_counter = 0
        self.total_blinks = 0
        self.is_drowsy = False
        self.current_ear = 0.0
        self.ear_history = []

    def update(self, left_ear, right_ear):
        """
        Update state with the latest EAR measurements for left and right eyes.
        Returns (is_drowsy, is_blinking, avg_ear)
        """
        avg_ear = (left_ear + right_ear) / 2.0
        self.current_ear = avg_ear
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 100:
            self.ear_history.pop(0)

        is_blinking = False

        # Check if eyes are closed
        if avg_ear < self.ear_thresh:
            self.closed_frame_counter += 1

            if self.closed_frame_counter >= self.consec_frames:
                self.is_drowsy = True
        else:
            # Eyes just reopened
            if self.blink_min_frames <= self.closed_frame_counter < self.consec_frames:
                self.total_blinks += 1
                is_blinking = True

            self.closed_frame_counter = 0
            self.is_drowsy = False

        return self.is_drowsy, is_blinking, avg_ear

    def reset(self):
        self.closed_frame_counter = 0
        self.total_blinks = 0
        self.is_drowsy = False
        self.ear_history.clear()


class YawnTracker:
    """
    Tracks mouth opening, yawning events, and cumulative yawns.
    """
    def __init__(self, mor_thresh=0.58, consec_frames=15, cooldown_frames=30):
        self.mor_thresh = mor_thresh
        self.consec_frames = consec_frames
        self.cooldown_frames = cooldown_frames

        self.open_frame_counter = 0
        self.total_yawns = 0
        self.is_yawning = False
        self.cooldown_counter = 0
        self.current_mor = 0.0

    def update(self, mor):
        """
        Update state with latest MOR measurement.
        Returns (is_yawning, total_yawns, mor)
        """
        self.current_mor = mor

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        if mor > self.mor_thresh:
            self.open_frame_counter += 1

            if self.open_frame_counter >= self.consec_frames:
                self.is_yawning = True
                if self.cooldown_counter == 0:
                    self.total_yawns += 1
                    self.cooldown_counter = self.cooldown_frames
        else:
            self.open_frame_counter = 0
            self.is_yawning = False

        return self.is_yawning, self.total_yawns, mor

    def reset(self):
        self.open_frame_counter = 0
        self.total_yawns = 0
        self.is_yawning = False
        self.cooldown_counter = 0
