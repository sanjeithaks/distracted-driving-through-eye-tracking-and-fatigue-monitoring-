"""
Audio Warning Subsystem
Provides non-blocking threaded audio alerts for:
- Drowsiness (urgent alarm)
- Yawning (fatigue alert)
- Distraction / Looking Away (caution tone)
As specified in Section IV of the IEEE Paper.
"""
import threading
import time
import os
import numpy as np


class SoundSynthesizer:
    """
    Generates standard in-memory alert wave tones so the project
    runs without requiring external audio files.
    """
    @staticmethod
    def generate_tone_wav(filename, freq=1200, duration=0.3, sample_rate=44100):
        """Generate a clean sine-wave WAV file."""
        import wave
        import struct

        n_samples = int(sample_rate * duration)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with wave.open(filename, 'w') as wav:
            wav.setnchannels(1)        # Mono
            wav.setsampwidth(2)        # 16-bit
            wav.setframerate(sample_rate)

            # Generate envelope: smooth attack & decay
            t = np.linspace(0, duration, n_samples, endpoint=False)
            envelope = np.ones(n_samples)
            fade = int(sample_rate * 0.02)
            if fade > 0 and len(envelope) > 2 * fade:
                envelope[:fade] = np.linspace(0, 1, fade)
                envelope[-fade:] = np.linspace(1, 0, fade)

            waveform = np.sin(2.0 * np.pi * freq * t) * envelope
            waveform = (waveform * 32767 * 0.8).astype(np.int16)

            data = struct.pack(f'<{len(waveform)}h', *waveform)
            wav.writeframes(data)


class AlertSystem:
    """
    Thread-safe alert system managing audio warnings and cooldowns.
    """
    def __init__(self, sounds_dir=None, enabled=True):
        self.enabled = enabled
        if sounds_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sounds_dir = os.path.join(base_dir, "assets", "sounds")

        self.sounds_dir = sounds_dir
        self.drowsy_wav = os.path.join(sounds_dir, "drowsy_alarm.wav")
        self.yawn_wav = os.path.join(sounds_dir, "yawn_alert.wav")
        self.distracted_wav = os.path.join(sounds_dir, "distracted_alert.wav")

        self._ensure_sounds()

        # Initialize pygame mixer
        self.mixer_ready = False
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sound_drowsy = pygame.mixer.Sound(self.drowsy_wav)
            self.sound_yawn = pygame.mixer.Sound(self.yawn_wav)
            self.sound_distracted = pygame.mixer.Sound(self.distracted_wav)
            self.mixer_ready = True
        except Exception as e:
            # Fallback to winsound on Windows if mixer fails
            self.mixer_ready = False

        self._is_playing_drowsy = False
        self._lock = threading.Lock()
        self._last_alert_time = {}

    def _ensure_sounds(self):
        """Generate audio wave files if not present."""
        if not os.path.exists(self.drowsy_wav):
            SoundSynthesizer.generate_tone_wav(self.drowsy_wav, freq=1200, duration=0.35)
        if not os.path.exists(self.yawn_wav):
            SoundSynthesizer.generate_tone_wav(self.yawn_wav, freq=850, duration=0.30)
        if not os.path.exists(self.distracted_wav):
            SoundSynthesizer.generate_tone_wav(self.distracted_wav, freq=1050, duration=0.25)

    def trigger_drowsy_alarm(self):
        """Trigger urgent drowsiness alarm (continuous / rapid)."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if now - self._last_alert_time.get('drowsy', 0) < 0.4:
                return
            self._last_alert_time['drowsy'] = now

        threading.Thread(target=self._play_sound, args=('drowsy',), daemon=True).start()

    def trigger_yawn_alert(self):
        """Trigger yawn warning tone."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if now - self._last_alert_time.get('yawn', 0) < 2.0:
                return
            self._last_alert_time['yawn'] = now

        threading.Thread(target=self._play_sound, args=('yawn',), daemon=True).start()

    def trigger_distraction_alert(self):
        """Trigger distraction / looking away tone."""
        if not self.enabled:
            return

        now = time.time()
        with self._lock:
            if now - self._last_alert_time.get('distracted', 0) < 1.0:
                return
            self._last_alert_time['distracted'] = now

        threading.Thread(target=self._play_sound, args=('distracted',), daemon=True).start()

    def _play_sound(self, sound_type):
        """Execute sound playback in background thread."""
        try:
            if self.mixer_ready:
                if sound_type == 'drowsy':
                    self.sound_drowsy.play()
                elif sound_type == 'yawn':
                    self.sound_yawn.play()
                elif sound_type == 'distracted':
                    self.sound_distracted.play()
            else:
                # Windows winsound fallback
                import winsound
                freq = 1200 if sound_type == 'drowsy' else (850 if sound_type == 'yawn' else 1050)
                winsound.Beep(freq, 250)
        except Exception:
            pass
