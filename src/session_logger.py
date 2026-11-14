"""
Session Logging Module
-----------------------
Records per-frame telemetry and alert events for every monitoring session and
produces a machine-readable summary (JSON) that can be used for driver
behaviour analysis, IEEE paper evaluation tables or the HTML report generator.

Generated files (inside the chosen output directory):
    session_<timestamp>.csv          - per-frame metrics
    session_<timestamp>_summary.json - aggregated session statistics + events
"""
import csv
import json
import os
import time
from datetime import datetime

import numpy as np


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
        When False all logging calls are no-ops (useful in unit tests).
    """

    def __init__(self, output_dir="logs", session_name=None, enabled=True):
        self.enabled = enabled
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir
        self.csv_path = None
        self.summary_path = None

        self._file = None
        self._writer = None
        self._start_time = time.time()
        self._closed = False

        # Aggregates
        self.frames = 0
        self.face_frames = 0
        self.state_frames = {s: 0 for s in STATE_PRIORITY}
        self.gaze_direction_counts = {}
        self.ear_values = []
        self.mor_values = []
        self.fps_values = []
        self.events = []
        self._prev_state = "NORMAL"
        self.last_telemetry = {}

        if self.enabled:
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                base = f"session_{self.session_name}"
                self.csv_path = os.path.join(self.output_dir, f"{base}.csv")
                self.summary_path = os.path.join(self.output_dir, f"{base}_summary.json")
                self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
                self._writer = csv.DictWriter(self._file, fieldnames=[
                    "frame", "timestamp", "state", "face_detected",
                    "ear", "mor", "yaw", "pitch", "roll",
                    "gaze", "gaze_x", "gaze_y",
                    "blinks", "yawns", "distractions", "fps", "distraction_reason"
                ])
                self._writer.writeheader()
            except OSError:
                # Logging must never crash the safety pipeline.
                self.enabled = False
                self._file = None
                self._writer = None

    # ------------------------------------------------------------------ #
    def log_frame(self, telemetry, frame_index=None):
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
        self.ear_values.append(ear)
        self.mor_values.append(mor)
        if fps > 0:
            self.fps_values.append(fps)

        gaze = telemetry.get('gaze_label', "N/A")
        self.gaze_direction_counts[gaze] = self.gaze_direction_counts.get(gaze, 0) + 1

        # Record state transitions as labelled events
        if state != self._prev_state:
            if state in STATE_MESSAGES:
                self.events.append({
                    "timestamp": round(time.time() - self._start_time, 2),
                    "frame": frame_index if frame_index is not None else self.frames,
                    "type": state,
                    "message": STATE_MESSAGES[state],
                    "reason": telemetry.get('distraction_reason', ''),
                })
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

        summary = {
            "session": self.session_name,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(timespec="seconds"),
            "duration_sec": round(duration, 2),
            "frames": self.frames,
            "face_detection_rate_pct": round(100.0 * self.face_frames / self.frames, 1) if self.frames else 0.0,
            "avg_ear": round(float(np.mean(self.ear_values)), 4) if self.ear_values else 0.0,
            "min_ear": round(float(np.min(self.ear_values)), 4) if self.ear_values else 0.0,
            "avg_mor": round(float(np.mean(self.mor_values)), 4) if self.mor_values else 0.0,
            "max_mor": round(float(np.max(self.mor_values)), 4) if self.mor_values else 0.0,
            "avg_fps": round(float(np.mean(self.fps_values)), 2) if self.fps_values else 0.0,
            "state_frames": dict(self.state_frames),
            "gaze_direction_counts": dict(self.gaze_direction_counts),
            "blinks": int(last.get('blinks', 0)),
            "yawns": int(last.get('yawns', 0)),
            "distractions": int(last.get('distractions', 0)),
            "gaze_distractions": int(last.get('gaze_distractions', 0)),
            "drowsy_events": sum(1 for e in self.events if e['type'] == 'DROWSY'),
            "distraction_events": sum(1 for e in self.events if e['type'] == 'DISTRACTED'),
            "yawn_events": sum(1 for e in self.events if e['type'] == 'YAWNING'),
            "events": list(self.events),
        }
        return summary

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
