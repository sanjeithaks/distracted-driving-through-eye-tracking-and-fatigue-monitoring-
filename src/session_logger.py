"""
Session Logging Module
-----------------------
Records per-frame telemetry (including PERCLOS, microsleeps and the composite
Fatigue Risk Index) plus alert events, and produces a machine-readable summary
(JSON) used by the HTML report generator.

Generated files (inside the chosen output directory):
    session_<timestamp>.csv          - per-frame metrics
    session_<timestamp>_summary.json - aggregated statistics + events
    snapshots/                       - evidence frame saved when an alert starts

The logger is intentionally lightweight for long sessions: only running
statistics and a down-sampled risk series are retained in memory.
"""
import csv
import json
import os
import time
from datetime import datetime

import cv2

STATE_PRIORITY = ("DROWSY", "DISTRACTED", "YAWNING", "NORMAL")

STATE_MESSAGES = {
    "DROWSY": "Drowsiness detected - prolonged eye closure",
    "DISTRACTED": "Driver distracted - looking away from the road",
    "YAWNING": "Yawning detected - driver fatigue warning",
}


def _state_of(telemetry):
    """Reduce a telemetry dict to a single priority state label."""
    if telemetry.get('is_drowsy'):
        return "DROWSY"
    if telemetry.get('is_distracted'):
        return "DISTRACTED"
    if telemetry.get('is_yawning'):
        return "YAWNING"
    return "NORMAL"


class SessionLogger:
    """
    Streams telemetry to a CSV file and aggregates a session summary.

    Parameters
    ----------
    output_dir : str
        Directory where the CSV / JSON files are written.
    session_name : str, optional
        Human-readable session label (defaults to a timestamp).
    enabled : bool
        When False all logging calls are no-ops.
    snapshot_dir : str, optional
        Directory for alert evidence images. When omitted, snapshots are not saved.
    """

    def __init__(self, output_dir="logs", session_name=None, enabled=True,
                 snapshot_dir=None):
        self.enabled = enabled
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir
        self.csv_path = None
        self.summary_path = None
        self.snapshot_dir = snapshot_dir

        self._file = None
        self._writer = None
        self._start_time = time.time()
        self._closed = False

        # Aggregates (running statistics, no per-frame lists)
        self.frames = 0
        self.face_frames = 0
        self.state_frames = {s: 0 for s in STATE_PRIORITY}
        self.gaze_direction_counts = {}
        self.events = []
        self._prev_state = "NORMAL"
        self.last_telemetry = {}

        self._ear_sum = 0.0
        self._ear_min = None
        self._ear_max = None
        self._mor_sum = 0.0
        self._mor_max = 0.0
        self._fps_sum = 0.0
        self._fps_count = 0
        self._perclos_sum = 0.0
        self._perclos_max = 0.0
        self._risk_sum = 0.0
        self._risk_max = 0.0

        # Down-sampled risk timeline for the HTML report
        self.risk_series = []

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                base = f"session_{self.session_name}"
                self.csv_path = os.path.join(self.output_dir, f"{base}.csv")
                self.summary_path = os.path.join(self.output_dir, f"{base}_summary.json")
                if self.snapshot_dir:
                    os.makedirs(self.snapshot_dir, exist_ok=True)

                self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=[
                    "frame", "timestamp", "state", "face_detected",
                    "ear", "mor", "yaw", "pitch", "roll",
                    "gaze", "gaze_x", "gaze_y",
                    "perclos", "fatigue_risk", "risk_level",
                    "microsleeps", "blink_rate", "yawn_rate",
                    "blinks", "yawns", "distractions", "fps",
                    "distraction_reason"
                ])
                self._writer.writeheader()
            except OSError:
                # Logging must never crash the safety pipeline.
                self.enabled = False
                self._file = None
                self._writer = None

    # ------------------------------------------------------------------ #
    def _save_snapshot(self, frame, frame_index, state):
        """Save an evidence image when an alert state begins."""
        if frame is None or not self.snapshot_dir:
            return None
        try:
            filename = f"snapshot_{int(frame_index):06d}_{state.lower()}.jpg"
            cv2.imwrite(os.path.join(self.snapshot_dir, filename), frame)
            return filename
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def log_frame(self, telemetry, frame_index=None, frame=None):
        """Record one processed frame and detect alert state transitions."""
        if not self.enabled:
            return

        self.frames += 1
        self.last_telemetry = telemetry
        if telemetry.get('face_detected'):
            self.face_frames += 1

        state = _state_of(telemetry)
        self.state_frames[state] += 1

        ear = float(telemetry.get('ear', 0.0))
        mor = float(telemetry.get('mor', 0.0))
        fps = float(telemetry.get('fps', 0.0))
        perclos = float(telemetry.get('perclos_pct', 0.0))
        risk = float(telemetry.get('fatigue_risk', 0.0))

        self._ear_sum += ear
        self._ear_min = ear if self._ear_min is None else min(self._ear_min, ear)
        self._ear_max = ear if self._ear_max is None else max(self._ear_max, ear)
        self._mor_sum += mor
        self._mor_max = max(self._mor_max, mor)
        self._perclos_sum += perclos
        self._perclos_max = max(self._perclos_max, perclos)
        self._risk_sum += risk
        self._risk_max = max(self._risk_max, risk)
        if fps > 0:
            self._fps_sum += fps
            self._fps_count += 1

        gaze = telemetry.get('gaze_label', "N/A")
        self.gaze_direction_counts[gaze] = self.gaze_direction_counts.get(gaze, 0) + 1

        # Down-sample the risk timeline (max 600 points)
        if self.frames % 5 == 0 and len(self.risk_series) < 600:
            self.risk_series.append({
                "frame": frame_index if frame_index is not None else self.frames,
                "risk": round(risk, 1),
            })

        # Record state transitions as labelled events (+ evidence snapshot)
        if state != self._prev_state:
            if state in STATE_MESSAGES:
                event = {
                    "timestamp": round(time.time() - self._start_time, 2),
                    "frame": frame_index if frame_index is not None else self.frames,
                    "type": state,
                    "message": STATE_MESSAGES[state],
                    "reason": telemetry.get('distraction_reason', ''),
                    "fatigue_risk": round(risk, 1),
                }
                snapshot = self._save_snapshot(frame, event["frame"], state)
                if snapshot:
                    event["snapshot"] = snapshot
                self.events.append(event)
            self._prev_state = state

        if self._writer is not None:
            try:
                self._writer.writerow({
                    "frame": frame_index if frame_index is not None else self.frames,
                    "timestamp": round(time.time() - self._start_time, 3),
                    "state": state,
                    "face_detected": int(bool(telemetry.get('face_detected', False))),
                    "ear": round(ear, 4),
                    "mor": round(mor, 4),
                    "yaw": round(float(telemetry.get('yaw', 0.0)), 2),
                    "pitch": round(float(telemetry.get('pitch', 0.0)), 2),
                    "roll": round(float(telemetry.get('roll', 0.0)), 2),
                    "gaze": gaze,
                    "gaze_x": round(float(telemetry.get('gaze_x', 0.5)), 4),
                    "gaze_y": round(float(telemetry.get('gaze_y', 0.5)), 4),
                    "perclos": round(perclos, 2),
                    "fatigue_risk": round(risk, 2),
                    "risk_level": telemetry.get('risk_level', 'LOW'),
                    "microsleeps": int(telemetry.get('microsleeps', 0)),
                    "blink_rate": round(float(telemetry.get('blink_rate', 0.0)), 2),
                    "yawn_rate": round(float(telemetry.get('yawn_rate', 0.0)), 2),
                    "blinks": int(telemetry.get('blinks', 0)),
                    "yawns": int(telemetry.get('yawns', 0)),
                    "distractions": int(telemetry.get('distractions', 0)),
                    "fps": round(fps, 2),
                    "distraction_reason": telemetry.get('distraction_reason', ''),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def summary(self):
        """Build the aggregated session summary dictionary."""
        duration = max(0.0, time.time() - self._start_time)
        last = self.last_telemetry
        frames = self.frames or 1

        risk_series = list(self.risk_series)
        final_risk = float(last.get('fatigue_risk', 0.0))
        if not risk_series or risk_series[-1]["risk"] != round(final_risk, 1):
            risk_series.append({"frame": self.frames, "risk": round(final_risk, 1)})

        return {
            "session": self.session_name,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(timespec="seconds"),
            "duration_sec": round(duration, 2),
            "frames": self.frames,
            "face_detection_rate_pct": round(100.0 * self.face_frames / frames, 1),
            "avg_ear": round(self._ear_sum / frames, 4),
            "min_ear": round(self._ear_min if self._ear_min is not None else 0.0, 4),
            "max_ear": round(self._ear_max if self._ear_max is not None else 0.0, 4),
            "avg_mor": round(self._mor_sum / frames, 4),
            "max_mor": round(self._mor_max, 4),
            "avg_fps": round(self._fps_sum / self._fps_count, 2) if self._fps_count else 0.0,
            "avg_perclos": round(self._perclos_sum / frames, 2),
            "max_perclos": round(self._perclos_max, 2),
            "avg_fatigue_risk": round(self._risk_sum / frames, 1),
            "peak_fatigue_risk": round(self._risk_max, 1),
            "final_fatigue_risk": round(final_risk, 1),
            "final_risk_level": last.get('risk_level', 'LOW'),
            "final_risk_factors": last.get('risk_factors', {}),
            "microsleeps": int(last.get('microsleeps', 0)),
            "longest_closure_sec": round(float(last.get('longest_closure_sec', 0.0)), 2),
            "blink_rate": round(float(last.get('blink_rate', 0.0)), 1),
            "yawn_rate": round(float(last.get('yawn_rate', 0.0)), 1),
            "state_frames": dict(self.state_frames),
            "gaze_direction_counts": dict(self.gaze_direction_counts),
            "blinks": int(last.get('blinks', 0)),
            "yawns": int(last.get('yawns', 0)),
            "distractions": int(last.get('distractions', 0)),
            "gaze_distractions": int(last.get('gaze_distractions', 0)),
            "drowsy_events": sum(1 for e in self.events if e['type'] == 'DROWSY'),
            "distraction_events": sum(1 for e in self.events if e['type'] == 'DISTRACTED'),
            "yawn_events": sum(1 for e in self.events if e['type'] == 'YAWNING'),
            "risk_series": risk_series,
            "events": list(self.events),
        }

    def close(self):
        """Flush files and write the JSON summary. Returns the summary dict."""
        if self._closed:
            return self.summary()
        self._closed = True

        summary = self.summary()

        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

        if self.enabled and self.summary_path:
            try:
                with open(self.summary_path, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
            except OSError:
                pass

        return summary
