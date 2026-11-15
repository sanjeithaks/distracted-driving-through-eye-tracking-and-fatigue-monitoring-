"""
Immersive Driving Simulator
---------------------------
Renders a realistic procedural driving cockpit around the live driver-monitoring
feed:

* animated perspective road with moving lane markings, roadside poles and hills
* speedometer, RPM bar, gear indicator and a rotating steering wheel
* driver-camera inset with the live annotated face mesh
* risk-adaptive behaviour: the car slows when the driver is drowsy and drifts
  toward the lane edge when the driver is distracted
* flashing alert border + seatbelt-vibration indicator driven by the same
  Fatigue Risk Index used by the embedded subsystem

The simulator is deliberately dependency-free (OpenCV + NumPy only) so it runs
on any laptop in PC-simulation mode.
"""
import math

import cv2
import numpy as np

from .config import SEATBELT_VIBRATION_FREQ_HZ


class DrivingSimulator:
    """Procedural driving scene composited with the monitoring frame."""

    def __init__(self, width=1280, height=720, fps=30):
        self.w = int(width)
        self.h = int(height)
        self.fps = fps
        self.horizon = int(self.h * 0.42)

        self.speed = 0.0
        self.target_speed = 62.0
        self.steering = 0.0          # -1 .. +1
        self.lane_offset = 0.0       # pixels at the bottom of the road
        self.distance = 0.0
        self.time = 0.0
        self.flash = 0.0
        self.vibration_phase = 0.0
        self.state = "NORMAL"
        self.risk = 0.0
        self.microsleeps = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        self.speed = 0.0
        self.steering = 0.0
        self.lane_offset = 0.0
        self.distance = 0.0
        self.flash = 0.0
        self.state = "NORMAL"

    # ------------------------------------------------------------------ #
    def update(self, telemetry, dt, steer_input=0.0, throttle=0.0):
        """Advance the driving simulation based on driver state and controls."""
        dt = max(0.0, min(0.1, dt))
        self.time += dt
        self.risk = float(telemetry.get('fatigue_risk', 0.0))
        self.microsleeps = int(telemetry.get('microsleeps', 0))

        if telemetry.get('is_drowsy'):
            self.state = "DROWSY"
        elif telemetry.get('is_distracted'):
            self.state = "DISTRACTED"
        elif telemetry.get('is_yawning'):
            self.state = "YAWNING"
        else:
            self.state = "NORMAL"

        # Target speed: driver controls, autonomous slowdown when drowsy
        target = 62.0 + throttle * 45.0
        if self.state == "DROWSY":
            target = min(target, 32.0)
        elif self.state == "DISTRACTED":
            target = min(target, 48.0)
        self.target_speed = float(np.clip(target, 0.0, 140.0))
        self.speed += (self.target_speed - self.speed) * min(1.0, dt * 1.6)
        self.speed = float(np.clip(self.speed, 0.0, 140.0))
        self.distance += self.speed * dt * 3.2

        # Steering: keyboard input, self-centering, plus a risk-induced drift
        self.steering += steer_input * dt * 2.4
        if steer_input == 0.0:
            self.steering *= max(0.0, 1.0 - dt * 1.8)
        self.steering = float(np.clip(self.steering, -1.0, 1.0))

        drift_target = 0.0
        if self.state == "DROWSY":
            drift_target = self.w * 0.10 * math.sin(self.time * 0.8)
        elif self.state == "DISTRACTED":
            drift_target = self.w * 0.045
        desired_offset = self.steering * self.w * 0.14 + drift_target
        self.lane_offset += (desired_offset - self.lane_offset) * min(1.0, dt * 2.2)
        self.lane_offset = float(np.clip(self.lane_offset, -self.w * 0.25, self.w * 0.25))

        # Alert flash intensity
        if self.state == "NORMAL":
            self.flash = max(0.0, self.flash - dt * 3.0)
        else:
            self.flash = min(1.0, self.flash + dt * 3.0)
        self.vibration_phase += dt * 12.0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _x_at(y, offset_frac, vp_x, horizon, w, h, lane_offset):
        t = max(0.0, min(1.0, (y - horizon) / float(h - horizon)))
        half = 32.0 + (w * 0.62 - 32.0) * t
        return vp_x + offset_frac * half + lane_offset * t

    # ------------------------------------------------------------------ #
    def _draw_sky(self, canvas):
        top = np.array([70, 110, 165], dtype=np.float32)     # BGR dusk blue
        bottom = np.array([170, 205, 235], dtype=np.float32)
        for y in range(self.horizon):
            a = y / max(1, self.horizon - 1)
            canvas[y, :, :] = (1 - a) * top + a * bottom

        # Sun
        sun = (int(self.w * 0.52), int(self.horizon * 0.34))
        cv2.circle(canvas, sun, 34, (120, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, sun, 46, (90, 190, 235), 2, cv2.LINE_AA)

        # Distant hills
        pts = np.array([
            (0, self.horizon), (int(self.w * 0.12), int(self.horizon * 0.72)),
            (int(self.w * 0.26), self.horizon), (int(self.w * 0.40), int(self.horizon * 0.62)),
            (int(self.w * 0.55), self.horizon), (int(self.w * 0.70), int(self.horizon * 0.70)),
            (int(self.w * 0.85), self.horizon), (self.w, int(self.horizon * 0.66)),
            (self.w, self.horizon),
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [pts], (95, 120, 95), cv2.LINE_AA)

        # Buildings silhouette near the horizon
        for i in range(14):
            bx = int((i / 14.0) * self.w)
            bw = int(self.w * 0.035)
            bh = int(30 + 55 * (0.5 + 0.5 * math.sin(i * 1.7)))
            cv2.rectangle(canvas, (bx, self.horizon - bh), (bx + bw, self.horizon),
                          (70, 78, 88), -1)

    # ------------------------------------------------------------------ #
    def _draw_road(self, canvas):
        w, h, horizon = self.w, self.h, self.horizon
        vp_x = int(w / 2 + self.lane_offset * 0.25)

        # Road surface
        road = np.array([
            (vp_x - 30, horizon), (vp_x + 30, horizon),
            (w + 200, h), (-200, h)
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [road], (58, 58, 64), cv2.LINE_AA)

        # Grass verge
        left_verge = np.array([(0, horizon), (vp_x - 30, horizon), (-200, h), (0, h)], np.int32)
        right_verge = np.array([(vp_x + 30, horizon), (w, horizon), (w, h), (w + 200, h)], np.int32)
        cv2.fillPoly(canvas, [left_verge], (60, 105, 70), cv2.LINE_AA)
        cv2.fillPoly(canvas, [right_verge], (60, 105, 70), cv2.LINE_AA)

        # Solid edge lines
        for off in (-1.0, 1.0):
            pts = []
            for y in range(horizon, h + 10, 6):
                pts.append((int(self._x_at(y, off, vp_x, horizon, w, h, self.lane_offset)), y))
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (220, 220, 220), 2, cv2.LINE_AA)

        # Moving dashed lane markings (3 lanes)
        for lane_off in (-0.34, 0.34):
            for i in range(18):
                t = ((self.distance * 0.03 + i / 18.0) % 1.0)
                y1 = int(horizon + (h - horizon) * (t ** 2.1))
                y2 = int(horizon + (h - horizon) * (min(1.0, t + 0.028) ** 2.1))
                if y2 <= y1:
                    continue
                x1 = int(self._x_at(y1, lane_off, vp_x, horizon, w, h, self.lane_offset))
                x2 = int(self._x_at(y2, lane_off, vp_x, horizon, w, h, self.lane_offset))
                thickness = max(1, int(1 + 7 * t))
                cv2.line(canvas, (x1, y1), (x2, y2), (245, 245, 235), thickness, cv2.LINE_AA)

        # Roadside poles / trees with perspective
        for side in (-1.55, 1.55):
            for i in range(10):
                t = ((self.distance * 0.012 + i / 10.0) % 1.0)
                y = int(horizon + (h - horizon) * (t ** 2.0))
                x = int(self._x_at(y, side, vp_x, horizon, w, h, self.lane_offset))
                height = int(18 + 120 * t)
                thickness = max(1, int(1 + 5 * t))
                cv2.line(canvas, (x, y), (x, y - height), (40, 45, 50), thickness, cv2.LINE_AA)
                if t > 0.12:
                    cv2.circle(canvas, (x, y - height), max(3, int(6 + 20 * t)),
                               (35, 90, 45), -1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_windshield(self, canvas):
        """Dark windshield pillars framing the cockpit view."""
        w, h = self.w, self.h
        overlay = canvas.copy()
        left = np.array([(0, 0), (int(w * 0.045), 0),
                         (int(w * 0.10), h), (0, h)], np.int32)
        right = np.array([(w, 0), (int(w * 0.955), 0),
                          (int(w * 0.90), h), (w, h)], np.int32)
        cv2.fillPoly(overlay, [left, right], (16, 16, 20))
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.line(canvas, (int(w * 0.045), 0), (int(w * 0.10), h), (60, 60, 68), 2)
        cv2.line(canvas, (int(w * 0.955), 0), (int(w * 0.90), h), (60, 60, 68), 2)

    # ------------------------------------------------------------------ #
    def _draw_dashboard(self, canvas, telemetry):
        w, h = self.w, self.h
        dash_top = int(h * 0.80)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, dash_top), (w, h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)

        # Speedometer (left)
        cx, cy, radius = 135, dash_top + 48, 58
        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 135, 405, (60, 60, 70), 8, cv2.LINE_AA)
        for tick in range(0, 141, 20):
            ang = math.radians(135 + (tick / 140.0) * 270)
            x1 = int(cx + (radius - 10) * math.cos(ang))
            y1 = int(cy + (radius - 10) * math.sin(ang))
            x2 = int(cx + radius * math.cos(ang))
            y2 = int(cy + radius * math.sin(ang))
            cv2.line(canvas, (x1, y1), (x2, y2), (150, 150, 160), 2, cv2.LINE_AA)

        needle_ang = math.radians(135 + (self.speed / 140.0) * 270)
        nx = int(cx + (radius - 16) * math.cos(needle_ang))
        ny = int(cy + (radius - 16) * math.sin(needle_ang))
        cv2.line(canvas, (cx, cy), (nx, ny), (60, 200, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, (230, 230, 230), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(self.speed)}", (cx - 26, cy + 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "km/h", (cx + 22, cy + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 160), 1, cv2.LINE_AA)

        # RPM / risk bars
        bar_x, bar_y, bar_w = 245, dash_top + 26, 190
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (55, 55, 65), -1)
        rpm_frac = min(1.0, self.speed / 140.0 * 0.85 + 0.12)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * rpm_frac), bar_y + 10),
                      (60, 200, 255), -1)
        cv2.putText(canvas, "RPM", (bar_x, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (170, 170, 180), 1, cv2.LINE_AA)

        risk_color = {"LOW": (0, 220, 90), "MODERATE": (0, 220, 255),
                      "HIGH": (0, 140, 255), "CRITICAL": (0, 0, 255)}.get(
            telemetry.get('risk_level', 'LOW'), (200, 200, 200))
        cv2.rectangle(canvas, (bar_x, bar_y + 22), (bar_x + bar_w, bar_y + 32), (55, 55, 65), -1)
        cv2.rectangle(canvas, (bar_x, bar_y + 22),
                      (bar_x + int(bar_w * min(1.0, self.risk / 100.0)), bar_y + 32),
                      risk_color, -1)
        cv2.putText(canvas, f"FATIGUE RISK {int(self.risk)}", (bar_x, bar_y + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, risk_color, 1, cv2.LINE_AA)

        # Gear / indicators (right of the steering wheel)
        cv2.putText(canvas, "D", (890, dash_top + 50), cv2.FONT_HERSHEY_DUPLEX,
                    0.9, (60, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"PERCLOS {telemetry.get('perclos_pct', 0):.0f}%",
                    (890, dash_top + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 190), 1, cv2.LINE_AA)

        # Steering wheel (bottom centre)
        center = (int(w / 2), h - 12)
        steer_deg = -self.steering * 38.0
        cv2.ellipse(canvas, center, (178, 178), steer_deg, 0, 360, (38, 38, 45), 22, cv2.LINE_AA)
        cv2.ellipse(canvas, center, (138, 138), steer_deg, 0, 360, (62, 62, 74), 4, cv2.LINE_AA)
        for base_ang in (90, 210, 330):
            ang = math.radians(base_ang + steer_deg)
            x = int(center[0] + 132 * math.cos(ang))
            y = int(center[1] + 132 * math.sin(ang))
            cv2.line(canvas, center, (x, y), (46, 46, 54), 14, cv2.LINE_AA)
        cv2.circle(canvas, center, 38, (28, 28, 34), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 33, (72, 72, 84), 2, cv2.LINE_AA)
        cv2.putText(canvas, "DM", (center[0] - 17, center[1] + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (140, 200, 255), 1, cv2.LINE_AA)

        # Control hints (top-left, clear of the steering wheel)
        cv2.putText(canvas, "A/D or arrows: steer    W/S: speed    R: reset    Q: quit",
                    (24, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (190, 190, 200), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_monitor_inset(self, canvas, monitor_frame, telemetry):
        if monitor_frame is None:
            return
        w = self.w
        inset_w, inset_h = 300, 225
        x, y = w - inset_w - 22, 22
        frame = cv2.resize(monitor_frame, (inset_w, inset_h))

        state = telemetry.get('status_state', self.state)
        border = {
            "DROWSY": (0, 0, 255),
            "DISTRACTED": (0, 140, 255),
            "YAWNING": (255, 0, 180),
            "NORMAL": (0, 220, 90),
        }.get(state, (0, 220, 90))

        panel = canvas.copy()
        cv2.rectangle(panel, (x - 5, y - 5), (x + inset_w + 5, y + inset_h + 26), (15, 15, 20), -1)
        cv2.addWeighted(panel, 0.85, canvas, 0.15, 0, canvas)
        canvas[y:y + inset_h, x:x + inset_w] = frame
        cv2.rectangle(canvas, (x - 3, y - 3), (x + inset_w + 3, y + inset_h + 3), border, 3, cv2.LINE_AA)
        cv2.putText(canvas, "DRIVER CAMERA", (x, y + inset_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, border, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def _draw_alerts(self, canvas, telemetry):
        w, h = self.w, self.h
        if self.flash > 0.02 and self.state != "NORMAL":
            color = {
                "DROWSY": (0, 0, 255),
                "DISTRACTED": (0, 140, 255),
                "YAWNING": (255, 0, 180),
            }.get(self.state, (0, 220, 90))
            alpha = 0.22 + 0.25 * abs(math.sin(self.vibration_phase))
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, 18)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Top status banner
        messages = {
            "DROWSY": "DROWSINESS ALERT - PULL OVER SAFELY",
            "DISTRACTED": "DISTRACTION DETECTED - LOOK AT THE ROAD",
            "YAWNING": "FATIGUE WARNING - YAWNING DETECTED",
        }
        if self.state in messages:
            text = messages[self.state]
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cx = (w - tw) // 2
            cv2.rectangle(canvas, (cx - 20, 18), (cx + tw + 20, 18 + th + 24), (10, 10, 14), -1)
            cv2.putText(canvas, text, (cx, 18 + th + 8), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (0, 0, 255) if self.state == "DROWSY" else (0, 170, 255),
                        2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "DRIVING MODE  |  ATTENTIVE", (24, 42),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 90), 1, cv2.LINE_AA)

        # Virtual seatbelt vibration indicator
        if self.state == "DROWSY":
            if int(self.vibration_phase) % 2 == 0:
                freq = 100 + int(2 * self.risk)
                cv2.putText(canvas, f"SEATBELT VIBRATION {freq} Hz",
                            (24, h - 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 255), 1, cv2.LINE_AA)

        # Speed + risk panel top-left
        cv2.putText(canvas, f"{int(self.speed)} km/h", (24, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FRI {int(self.risk)}/100  {telemetry.get('risk_level', 'LOW')}",
                    (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 255) if self.risk < 50 else (0, 140, 255), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def render(self, monitor_frame, telemetry):
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_sky(canvas)
        self._draw_road(canvas)
        self._draw_windshield(canvas)
        self._draw_dashboard(canvas, telemetry)
        self._draw_monitor_inset(canvas, monitor_frame, telemetry)
        self._draw_alerts(canvas, telemetry)
        return canvas
