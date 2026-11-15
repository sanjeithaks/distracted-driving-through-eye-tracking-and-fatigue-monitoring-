"""
Visual HUD & Overlay Module
Renders visual feedback exactly matching the figures from the IEEE paper:
- Fig 4: FACE DETECTED (Green text), Bounding Box, Eye & Lip contours
- Fig 5: DROWSINESS ALERT! (Bold Red Banner)
- Fig 6: YAWNING DETECTED! (Bold Magenta/Purple Banner)
- Fig 7, 8, 9: LOOKING AWAY! (Bold Red/Orange Banner)
Also renders a real-time telemetry dashboard (EAR, MOR, Head Pose, Counters, FPS).
"""
import cv2
import numpy as np


class Visualizer:
    """
    Renders high-visibility banners, face bounding boxes, facial landmarks,
    and driver telemetry HUD.
    """
    def __init__(self,
                 show_landmarks=True,
                 show_bbox=True,
                 show_axis=True,
                 show_hud=True,
                 show_framing_hint=False):
        self.show_landmarks = show_landmarks
        self.show_bbox = show_bbox
        self.show_axis = show_axis
        self.show_hud = show_hud
        self.show_framing_hint = show_framing_hint

        # Colors (BGR)
        self.COLOR_GREEN = (0, 255, 0)
        self.COLOR_RED = (0, 0, 255)
        self.COLOR_PURPLE = (255, 0, 180)     # Magenta/Purple for Yawning
        self.COLOR_YELLOW = (0, 230, 255)     # Yellow for Face Bounding Box
        self.COLOR_CYAN = (255, 230, 0)       # Cyan for contours
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_DARK_BG = (20, 20, 20)

    def draw_text_with_outline(self, frame, text, origin, font_scale=1.0, color=(0, 255, 0), thickness=2, outline_thickness=5):
        """Draw text with high-contrast black outline for crystal clarity."""
        x, y = origin
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_dashed_rect(frame, p1, p2, color, thickness=2, dash=14):
        """Draw a dashed rectangle (camera framing guide)."""
        x1, y1 = p1
        x2, y2 = p2
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)

    def render(self, frame, detection, ear, mor, is_drowsy, is_yawning, is_distracted,
               distraction_reason, yaw, pitch, roll, axis_pts, blinks, yawns, distractions,
               fps, serial_connected, gaze_label="N/A", gaze_x=0.5, gaze_y=0.5,
               ear_thresh=0.21, mor_thresh=0.38, perclos=0.0, fatigue_risk=0.0,
               risk_level="LOW", microsleeps=0, head_calibrated=True,
               calibration_progress=1.0):
        """
        Render all visual overlays onto the frame.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. If Face is detected, draw bounding box & landmarks
        if detection.get('detected', False):
            bbox = detection['bbox']
            x, y, bw, bh = bbox

            # Bounding box around face (matching Fig 4)
            if self.show_bbox:
                cv2.rectangle(output, (x, y), (x + bw, y + bh), self.COLOR_YELLOW, 2)

            # Draw Eye Contours and Landmarks
            if self.show_landmarks:
                # Left eye contour
                left_contour = np.array(detection['left_eye_contour'], dtype=np.int32)
                cv2.polylines(output, [left_contour], True, self.COLOR_GREEN, 1, cv2.LINE_AA)

                # Right eye contour
                right_contour = np.array(detection['right_eye_contour'], dtype=np.int32)
                cv2.polylines(output, [right_contour], True, self.COLOR_GREEN, 1, cv2.LINE_AA)

                # Lips contour
                lips_contour = np.array(detection['lips_contour'], dtype=np.int32)
                cv2.polylines(output, [lips_contour], True, self.COLOR_GREEN, 1, cv2.LINE_AA)

                # Mark key landmark points (small dots)
                for pt in detection['left_eye_pts'] + detection['right_eye_pts']:
                    cv2.circle(output, pt, 2, self.COLOR_CYAN, -1, cv2.LINE_AA)
                for pt in detection['mouth_pts']:
                    cv2.circle(output, pt, 2, self.COLOR_YELLOW, -1, cv2.LINE_AA)

                # Iris centres (true eye-tracking cue)
                for key in ('right_iris_center', 'left_iris_center'):
                    iris_pt = detection.get(key)
                    if iris_pt:
                        cv2.circle(output, iris_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)

            # Draw 3D Head Orientation Axis (Nose vector)
            if self.show_axis and axis_pts[0] is not None and axis_pts[1] is not None:
                p1, p2 = axis_pts
                cv2.arrowedLine(output, p1, p2, (0, 165, 255), 2, cv2.LINE_AA, tipLength=0.25)

            # Draw Status Banners matching IEEE Paper Figures:
            # Banner order priority: Drowsiness > Looking Away > Yawning > Normal
            banner_y = 50

            if is_drowsy:
                # Fig 5: DROWSINESS ALERT!
                self.draw_text_with_outline(output, "DROWSINESS ALERT!", (30, banner_y),
                                            font_scale=1.1, color=self.COLOR_RED, thickness=3, outline_thickness=6)
                banner_y += 45
            elif is_distracted:
                # Fig 7, 8, 9: LOOKING AWAY!
                self.draw_text_with_outline(output, "LOOKING AWAY!", (30, banner_y),
                                            font_scale=1.1, color=self.COLOR_RED, thickness=3, outline_thickness=6)
                # Show specific reason under banner
                if distraction_reason != "NORMAL":
                    self.draw_text_with_outline(output, f"[{distraction_reason}]", (30, banner_y + 32),
                                                font_scale=0.65, color=(0, 140, 255), thickness=2, outline_thickness=4)
                    banner_y += 68
                else:
                    banner_y += 45
            elif is_yawning:
                # Fig 6: YAWNING DETECTED!
                self.draw_text_with_outline(output, "YAWNING DETECTED!", (30, banner_y),
                                            font_scale=1.1, color=self.COLOR_PURPLE, thickness=3, outline_thickness=6)
                banner_y += 45

            # Green "FACE DETECTED" tag below any alert banner (as seen in Figs 4, 5, 6, 7, 8, 9)
            self.draw_text_with_outline(output, "FACE DETECTED", (30, banner_y),
                                        font_scale=0.9, color=self.COLOR_GREEN, thickness=2, outline_thickness=5)

            # Live-camera framing/calibration hints (only for real camera input,
            # not the synthetic demo clip).
            if self.show_framing_hint:
                if not head_calibrated:
                    pct = int(max(0.0, min(1.0, calibration_progress)) * 100)
                    self.draw_text_with_outline(
                        output,
                        f"CALIBRATING NEUTRAL POSE {pct}% - LOOK STRAIGHT",
                        (w - 470, 30), font_scale=0.50, color=(0, 200, 255),
                        thickness=2, outline_thickness=4)
                elif y + bh > h - 105:
                    self.draw_text_with_outline(output, "MOVE BACK - MOUTH MUST BE VISIBLE", (w - 380, 30),
                                                font_scale=0.50, color=(0, 165, 255),
                                                thickness=2, outline_thickness=4)
                elif bw < w * 0.12:
                    self.draw_text_with_outline(output, "MOVE CLOSER TO CAMERA", (w - 300, 30),
                                                font_scale=0.55, color=(0, 165, 255),
                                                thickness=2, outline_thickness=4)

        else:
            # No face detected - show a framing box so the user can position
            # their whole face (including the mouth, needed for yawn detection).
            self.draw_text_with_outline(output, "NO FACE DETECTED", (30, 50),
                                        font_scale=1.0, color=(0, 165, 255), thickness=2, outline_thickness=5)
            if self.show_framing_hint:
                gx1, gy1 = int(w * 0.22), 95
                gx2, gy2 = int(w * 0.78), h - 150
                self._draw_dashed_rect(output, (gx1, gy1), (gx2, gy2), (0, 165, 255), 2)
                self.draw_text_with_outline(output, "PLACE WHOLE FACE IN BOX", (gx1 + 10, gy1 - 12),
                                            font_scale=0.55, color=(0, 165, 255), thickness=2, outline_thickness=4)
                self.draw_text_with_outline(output, "INCLUDE YOUR MOUTH", (gx1 + 10, gy2 + 28),
                                            font_scale=0.52, color=(0, 165, 255), thickness=1, outline_thickness=3)

        # 2. Telemetry HUD Bar at Bottom
        if self.show_hud:
            self._render_hud_panel(output, ear, mor, yaw, pitch, roll,
                                   blinks, yawns, distractions, fps, serial_connected,
                                   gaze_label, ear_thresh, mor_thresh,
                                   perclos, fatigue_risk, risk_level, microsleeps)

        return output

    def _render_hud_panel(self, frame, ear, mor, yaw, pitch, roll,
                          blinks, yawns, distractions, fps, serial_connected,
                          gaze_label="N/A", ear_thresh=0.21, mor_thresh=0.38,
                          perclos=0.0, fatigue_risk=0.0, risk_level="LOW",
                          microsleeps=0):
        """Render modern semi-transparent HUD dashboard at bottom of frame."""
        h, w = frame.shape[:2]
        hud_h = 96
        y_start = h - hud_h

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Top border line for HUD
        cv2.line(frame, (0, y_start), (w, y_start), (70, 70, 70), 1)

        # Row 1: Metrics
        font = cv2.FONT_HERSHEY_SIMPLEX
        ear_color = (0, 0, 255) if ear < ear_thresh else (0, 255, 0)
        mor_color = (255, 0, 200) if mor > mor_thresh else (0, 255, 0)
        gaze_color = {
            "CENTER": (0, 255, 0),
            "CLOSED": (130, 130, 130),
            "N/A": (130, 130, 130),
        }.get(gaze_label, (0, 165, 255))

        r1_y = y_start + 23
        cv2.putText(frame, f"EAR: {ear:.2f}", (15, r1_y), font, 0.52, ear_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"MOR: {mor:.2f}", (120, r1_y), font, 0.52, mor_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw: {yaw:+.1f}", (230, r1_y), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch: {pitch:+.1f}", (330, r1_y), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gaze: {gaze_label}", (435, r1_y), font, 0.48, gaze_color, 2, cv2.LINE_AA)

        # Row 2: Event Counters and System Telemetry
        r2_y = y_start + 49
        cv2.putText(frame, f"Blinks: {blinks}", (15, r2_y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yawns: {yawns}", (110, r2_y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distractions: {distractions}", (205, r2_y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Microsleeps: {microsleeps}", (355, r2_y), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS & Hardware connection status
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, r2_y), font, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        hw_text = "HW: On" if serial_connected else "HW: SIM"
        hw_color = (0, 255, 0) if serial_connected else (0, 200, 255)
        cv2.putText(frame, hw_text, (w - 68, r2_y), font, 0.42, hw_color, 1, cv2.LINE_AA)

        # Row 3: PERCLOS & explainable Fatigue Risk Index
        r3_y = y_start + 78
        risk_color = {
            "LOW": (0, 255, 0),
            "MODERATE": (0, 230, 255),
            "HIGH": (0, 140, 255),
            "CRITICAL": (0, 0, 255),
        }.get(risk_level, (200, 200, 200))
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (15, r3_y), font, 0.52,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FATIGUE RISK: {fatigue_risk:.0f}/100", (185, r3_y), font, 0.52,
                    risk_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"({risk_level})", (410, r3_y), font, 0.46, risk_color, 1, cv2.LINE_AA)

        # Risk bar
        bar_x, bar_w = 520, 108
        cv2.rectangle(frame, (bar_x, r3_y - 10), (bar_x + bar_w, r3_y + 2), (60, 60, 60), -1)
        fill = int(bar_w * max(0.0, min(100.0, fatigue_risk)) / 100.0)
        cv2.rectangle(frame, (bar_x, r3_y - 10), (bar_x + fill, r3_y + 2), risk_color, -1)
