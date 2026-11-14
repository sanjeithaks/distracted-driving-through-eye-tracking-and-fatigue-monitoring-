"""
Test Video & Dataset Generator
Generates realistic simulated driver video clips demonstrating all monitored states:
1. Normal driving (Eyes open, looking ahead)
2. Prolonged Eye Closure (Drowsiness Alert)
3. Yawning (Mouth Opening Ratio spike)
4. Distraction: Looking left / window (Head yaw)
5. Distraction: Looking down / texting on phone (Head pitch)
6. Eye-only glance (Iris gaze left / right with the head still)

Used for automated testing, offline evaluation, and visual demonstration.
"""
import os
import cv2
import numpy as np

# Suppress matplotlib warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))


def create_driver_frame(
    width=640,
    height=480,
    head_x=320,
    head_y=240,
    eye_h=15,
    mouth_h=8,
    turn=0.0,         # Yaw turning: -1.5 to 1.5
    is_looking_down=False,
    gaze_shift=0.0    # Iris offset inside the eye: -1.0 (left) .. +1.0 (right)
):
    """
    Synthesize a realistic driver face frame with adjustable
    facial features, eye aspect, mouth aperture, and 3D pose offsets.
    """
    frame = np.full((height, width, 3), 45, dtype=np.uint8)

    # Windshield / road background horizon
    cv2.rectangle(frame, (0, 0), (width, 220), (120, 140, 160), -1)
    cv2.line(frame, (0, 220), (width, 220), (80, 90, 100), 2)
    # Road lanes
    cv2.line(frame, (260, 220), (180, 120), (220, 220, 220), 3)
    cv2.line(frame, (380, 220), (460, 120), (220, 220, 220), 3)

    cx = int(head_x)
    cy = int(head_y)

    # Head oval (skin tone)
    skin_color = (165, 195, 225)
    face_w = 125
    face_h = 165
    cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, skin_color, -1)
    cv2.ellipse(frame, (cx, cy), (face_w, face_h), 0, 0, 360, (130, 160, 190), 2)

    # Hair on top
    hair_color = (30, 30, 35)
    cv2.ellipse(frame, (cx, cy - 85), (face_w - 5, 80), 0, 180, 360, hair_color, -1)

    if is_looking_down:
        # Looking down posture (chin tucked, eyes lower, nose/mouth compressed)
        eb_y = cy - 20
        eye_y = cy - 5
        nose_y = cy + 25
        mouth_y = cy + 55

        # Eyebrows
        cv2.line(frame, (cx - 65, eb_y), (cx - 20, eb_y - 6), hair_color, 4)
        cv2.line(frame, (cx + 20, eb_y - 6), (cx + 65, eb_y), hair_color, 4)

        # Eyes (glancing down)
        left_eye_x = cx - 45
        right_eye_x = cx + 45
        cv2.ellipse(frame, (left_eye_x, eye_y), (22, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (right_eye_x, eye_y), (22, 10), 0, 0, 360, (255, 255, 255), -1)
        # Pupils towards bottom of eyes
        cv2.circle(frame, (left_eye_x, eye_y + 3), 6, (40, 40, 40), -1)
        cv2.circle(frame, (right_eye_x, eye_y + 3), 6, (40, 40, 40), -1)

        # Nose
        cv2.circle(frame, (cx, nose_y), 9, (140, 170, 200), -1)

        # Mouth
        cv2.ellipse(frame, (cx, mouth_y), (35, 6), 0, 0, 360, (60, 70, 160), -1)

        # Smartphone in driver's hands at bottom
        phone_x, phone_y = cx + 40, cy + 130
        cv2.rectangle(frame, (phone_x - 35, phone_y - 50), (phone_x + 35, phone_y + 50), (30, 30, 30), -1)
        cv2.rectangle(frame, (phone_x - 30, phone_y - 45), (phone_x + 30, phone_y + 40), (180, 220, 255), -1)
        cv2.putText(frame, "SMS", (phone_x - 20, phone_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    else:
        # Standard or Turning posture
        eb_y = cy - 45
        cv2.line(frame, (cx - 65, eb_y), (cx - 20, eb_y - 8), hair_color, 4)
        cv2.line(frame, (cx + 20, eb_y - 8), (cx + 65, eb_y), hair_color, 4)

        eye_y = cy - 25
        left_eye_x = cx - 45 + int(turn * 25)
        right_eye_x = cx + 45 + int(turn * 25)

        left_eye_w = int(22 * (1.0 - max(0, turn) * 0.35))
        right_eye_w = int(22 * (1.0 - max(0, -turn) * 0.35))

        if eye_h > 3:
            cv2.ellipse(frame, (left_eye_x, eye_y), (left_eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y), (right_eye_w, eye_h), 0, 0, 360, (255, 255, 255), -1)

            pupil_r = min(7, eye_h)
            # Shift the pupil to simulate the eye (iris) glancing sideways
            pupil_dx_left = int(gaze_shift * max(0, left_eye_w - pupil_r))
            pupil_dx_right = int(gaze_shift * max(0, right_eye_w - pupil_r))
            cv2.circle(frame, (left_eye_x + pupil_dx_left, eye_y), pupil_r, (40, 40, 40), -1)
            cv2.circle(frame, (right_eye_x + pupil_dx_right, eye_y), pupil_r, (40, 40, 40), -1)

            cv2.ellipse(frame, (left_eye_x, eye_y), (left_eye_w, eye_h), 0, 0, 360, (70, 70, 70), 1)
            cv2.ellipse(frame, (right_eye_x, eye_y), (right_eye_w, eye_h), 0, 0, 360, (70, 70, 70), 1)
        else:
            # Closed eyes
            cv2.line(frame, (left_eye_x - left_eye_w, eye_y), (left_eye_x + left_eye_w, eye_y), (70, 70, 80), 3)
            cv2.line(frame, (right_eye_x - right_eye_w, eye_y), (right_eye_x + right_eye_w, eye_y), (70, 70, 80), 3)

        # Nose
        nose_x = cx + int(turn * 38)
        nose_y = cy + 15
        cv2.circle(frame, (nose_x, nose_y), 9, (140, 170, 200), -1)
        cv2.line(frame, (nose_x, nose_y - 25), (nose_x, nose_y), (130, 160, 190), 2)

        # Mouth
        mouth_x = cx + int(turn * 26)
        mouth_y = cy + 70
        mouth_w = 40

        if mouth_h > 20:
            # Yawn
            cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, mouth_h), 0, 0, 360, (20, 20, 60), -1)
            cv2.ellipse(frame, (mouth_x, mouth_y + mouth_h - 12), (mouth_w - 12, 14), 0, 0, 180, (70, 70, 170), -1)
            cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, mouth_h), 0, 0, 360, (60, 80, 160), 2)
        else:
            # Normal
            cv2.ellipse(frame, (mouth_x, mouth_y), (mouth_w, max(mouth_h, 5)), 0, 0, 360, (80, 90, 170), -1)
            cv2.line(frame, (mouth_x - mouth_w, mouth_y), (mouth_x + mouth_w, mouth_y), (50, 60, 140), 2)

    # Steering wheel overlay at bottom
    cv2.ellipse(frame, (320, 520), (190, 140), 0, 180, 360, (30, 30, 30), 18)
    cv2.line(frame, (170, 460), (320, 520), (40, 40, 40), 10)
    cv2.line(frame, (470, 460), (320, 520), (40, 40, 40), 10)

    return frame


def generate_scenario_video(output_path="demo_sample_driving.mp4", total_frames=540, fps=30):
    """
    Generate synthetic test driving video cycling through all IEEE paper scenarios:
    - 0-70:    Normal Driving (Fig 4)
    - 71-150:  Prolonged Eye Closure (Fig 5 - Drowsiness Alert)
    - 151-210: Recovery / Normal
    - 211-280: Yawning Detected (Fig 6 - High MOR)
    - 281-350: Distraction - Head turned (Fig 7 & 8 - Window / Passenger)
    - 351-420: Distraction - Head pitched down (Fig 9 - Looking into phone)
    - 421-480: Eye-only glance right (Fig 10 - iris gaze tracking)
    - 481-540: Eye-only glance left  (Fig 10 - iris gaze tracking)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (640, 480))

    print(f"[Generator] Creating scenario test video: {output_path} ({total_frames} frames)...")

    for i in range(total_frames):
        eye_h = 15
        mouth_h = 8
        turn = 0.0
        gaze_shift = 0.0
        is_looking_down = False

        # Scenario 1: Normal alert driving (0-70)
        if i <= 70:
            if 30 <= i <= 33:
                eye_h = 2  # Brief natural blink
            else:
                eye_h = 15
            mouth_h = 8

        # Scenario 2: Prolonged Eye Closure / Drowsiness (71-150)
        elif 71 < i <= 150:
            eye_h = 2  # Closed eyes
            mouth_h = 8

        # Scenario 3: Normal recovery (151-210)
        elif 151 < i <= 210:
            eye_h = 15
            mouth_h = 8

        # Scenario 4: Yawning (211-280)
        elif 211 < i <= 280:
            eye_h = 11
            t = (i - 211) / 70.0
            mouth_h = int(8 + 50 * np.sin(np.pi * t))

        # Scenario 5: Looking Left / Passenger / Window (281-350)
        elif 281 < i <= 350:
            eye_h = 14
            mouth_h = 8
            turn = 1.5  # Full head turn

        # Scenario 6: Looking Down / Mobile Phone (351-420)
        elif 351 < i <= 420:
            is_looking_down = True

        # Scenario 7: Head still, eyes glance right (421-480) -> iris gaze alert
        elif 421 < i <= 480:
            eye_h = 15
            gaze_shift = 1.0

        # Scenario 8: Head still, eyes glance left (481-540) -> iris gaze alert
        elif i > 480:
            eye_h = 15
            gaze_shift = -1.0

        frame = create_driver_frame(
            eye_h=eye_h,
            mouth_h=mouth_h,
            turn=turn,
            is_looking_down=is_looking_down,
            gaze_shift=gaze_shift
        )
        writer.write(frame)

    writer.release()
    print(f"[Generator] Video successfully generated: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_scenario_video("demo_sample_driving.mp4", total_frames=540)
