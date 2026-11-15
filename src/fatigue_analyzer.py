"""
Fatigue Analytics: PERCLOS, Microsleep Detection & Composite Fatigue Risk Index
-------------------------------------------------------------------------------
Research basis
--------------
PERCLOS ("percentage of eyelid closure over time") was established by
Wierwille et al. (1994) and is considered by FHWA/NHTSA to be one of the most
reliable real-time measures of driver alertness.  The classical definition is
the proportion of time in a rolling window during which the eyes are at least
~80% closed.  This implementation estimates eyelid openness from the Eye Aspect
Ratio (EAR) normalised by a slowly-adapted open-eye baseline:

    openness = EAR / EAR_open_baseline
    PERCLOS  = mean(openness < PERCLOS_CLOSURE_OPENNESS) over a 30 s window

The project also detects *microsleeps* (continuous eye closure >= 0.5 s) and
combines four behavioural indicators into an explainable 0-100
**Fatigue Risk Index (FRI)**:

    FRI = 50 * min(1, PERCLOS / 0.40)
        + 20 * min(1, microsleeps / 5)
        + 20 * min(1, yawn_rate_per_min / 4)
        + 10 * min(1, longest_closure_sec / 3)

Risk bands: LOW (<25), MODERATE (25-49), HIGH (50-74), CRITICAL (>=75).
"""
from collections import deque

import numpy as np

from .config import (
    PERCLOS_WINDOW_FRAMES, PERCLOS_CLOSURE_OPENNESS, PERCLOS_BASELINE_ADAPT,
    MICROSLEEP_MIN_FRAMES, SOURCE_FPS,
    FATIGUE_WEIGHT_PERCLOS, FATIGUE_WEIGHT_MICROSLEEP,
    FATIGUE_WEIGHT_YAWN, FATIGUE_WEIGHT_CLOSURE,
    FATIGUE_RISK_MODERATE, FATIGUE_RISK_HIGH, FATIGUE_RISK_CRITICAL,
)


class FatigueAnalyzer:
    """Temporal fatigue analysis over a sliding PERCLOS window."""

    def __init__(self,
                 window_frames=PERCLOS_WINDOW_FRAMES,
                 closure_openness=PERCLOS_CLOSURE_OPENNESS,
                 baseline_adapt=PERCLOS_BASELINE_ADAPT,
                 microsleep_frames=MICROSLEEP_MIN_FRAMES,
                 fps=SOURCE_FPS):
        self.window_frames = window_frames
        self.closure_openness = closure_openness
        self.baseline_adapt = baseline_adapt
        self.microsleep_frames = microsleep_frames
        self.fps = fps

        self.frames = 0
        self.window = deque(maxlen=window_frames)

        # Open-eye EAR baseline learned online
        self.baseline_open = None

        # Eye-closure / microsleep state
        self.closed_run = 0
        self.longest_closed_run = 0
        self.microsleeps = 0
        self.microsleep_active = False

        # Latest values
        self.perclos = 0.0
        self.openness = None
        self.blink_rate = 0.0
        self.yawn_rate = 0.0
        self.fatigue_risk = 0.0
        self.risk_level = "LOW"
        self.risk_factors = {
            "perclos": 0.0, "microsleep": 0.0, "yawning": 0.0, "closure": 0.0
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def risk_level_for(risk):
        if risk >= FATIGUE_RISK_CRITICAL:
            return "CRITICAL"
        if risk >= FATIGUE_RISK_HIGH:
            return "HIGH"
        if risk >= FATIGUE_RISK_MODERATE:
            return "MODERATE"
        return "LOW"

    # ------------------------------------------------------------------ #
    def update(self, ear, face_detected, blinks, yawns):
        """
        Update fatigue analytics with one frame.

        Parameters
        ----------
        ear : float          current average Eye Aspect Ratio
        face_detected : bool
        blinks : int         cumulative blink count
        yawns : int          cumulative yawn count

        Returns a dictionary with PERCLOS, microsleep, rates and risk factors.
        """
        self.frames += 1
        closed_for_perclos = False
        self.openness = None

        if face_detected and ear > 0.0:
            if self.baseline_open is None:
                self.baseline_open = max(0.18, ear)
            else:
                # Only adapt the baseline while the eyes are clearly open,
                # so a long closure cannot drag the baseline down.
                if ear / self.baseline_open >= 0.5:
                    self.baseline_open = ((1.0 - self.baseline_adapt) * self.baseline_open
                                          + self.baseline_adapt * ear)

            self.openness = float(np.clip(ear / self.baseline_open, 0.0, 1.5))
            closed_for_perclos = self.openness < self.closure_openness

        self.window.append(1 if closed_for_perclos else 0)
        self.perclos = float(np.mean(self.window)) if self.window else 0.0

        # Microsleep episodes: continuous closure >= MICROSLEEP_MIN_FRAMES
        if closed_for_perclos:
            self.closed_run += 1
            self.longest_closed_run = max(self.longest_closed_run, self.closed_run)
            if (self.closed_run >= self.microsleep_frames
                    and not self.microsleep_active):
                self.microsleeps += 1
                self.microsleep_active = True
        else:
            self.closed_run = 0
            self.microsleep_active = False

        # Rates use at least 10 s of observation to avoid startup spikes
        video_seconds = self.frames / self.fps
        rate_minutes = max(video_seconds, 10.0) / 60.0
        self.blink_rate = float(blinks) / rate_minutes
        self.yawn_rate = float(yawns) / rate_minutes

        longest_closure_sec = self.longest_closed_run / self.fps

        # Explainable weighted risk factors (points out of 100)
        self.risk_factors = {
            "perclos": FATIGUE_WEIGHT_PERCLOS * min(1.0, self.perclos / 0.40),
            "microsleep": FATIGUE_WEIGHT_MICROSLEEP * min(1.0, self.microsleeps / 5.0),
            "yawning": FATIGUE_WEIGHT_YAWN * min(1.0, self.yawn_rate / 4.0),
            "closure": FATIGUE_WEIGHT_CLOSURE * min(1.0, longest_closure_sec / 3.0),
        }
        self.fatigue_risk = float(np.clip(sum(self.risk_factors.values()), 0.0, 100.0))
        self.risk_level = self.risk_level_for(self.fatigue_risk)

        return {
            "perclos": self.perclos,
            "perclos_pct": 100.0 * self.perclos,
            "openness": self.openness,
            "microsleeps": self.microsleeps,
            "longest_closure_sec": longest_closure_sec,
            "blink_rate": self.blink_rate,
            "yawn_rate": self.yawn_rate,
            "fatigue_risk": self.fatigue_risk,
            "risk_level": self.risk_level,
            "risk_factors": dict(self.risk_factors),
        }

    def reset(self):
        self.frames = 0
        self.window.clear()
        self.baseline_open = None
        self.closed_run = 0
        self.longest_closed_run = 0
        self.microsleeps = 0
        self.microsleep_active = False
        self.perclos = 0.0
        self.openness = None
        self.blink_rate = 0.0
        self.yawn_rate = 0.0
        self.fatigue_risk = 0.0
        self.risk_level = "LOW"
        self.risk_factors = {
            "perclos": 0.0, "microsleep": 0.0, "yawning": 0.0, "closure": 0.0
        }
