"""
Embedded System Serial Interface
Handles communication between the Computer Vision pipeline and the physical
hardware microcontroller (Arduino / Raspberry Pi / ESP32).
Controls:
- Seatbelt Vibration Motor (100 Hz - 300 Hz PWM modulation)
- Physical Hardware Buzzer
- Vehicle Status LEDs (Normal / Drowsy / Yawn / Distracted)
As described in Section IV of the IEEE Paper.
"""
import time
import threading
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class EmbeddedSerialBridge:
    """
    Manages serial connection to embedded hardware (Arduino/Raspberry Pi).
    """
    def __init__(self, port="COM3", baud_rate=9600, timeout=0.1, enabled=True):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.enabled = enabled

        self.ser = None
        self.is_connected = False
        self.last_sent_state = None
        self.last_sent_risk = None
        self.last_send_time = 0
        self._lock = threading.Lock()

        if self.enabled and SERIAL_AVAILABLE:
            self._connect()

    def _auto_detect_port(self):
        """Scan for connected microcontroller serial ports."""
        if not SERIAL_AVAILABLE:
            return None
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            # Common Arduino / CH340 / CP210x / FTDI identifiers
            desc = p.description.lower()
            if any(k in desc for k in ['arduino', 'ch340', 'usb serial', 'cp210', 'ftdi']):
                return p.device
        return None

    def _connect(self):
        """Attempt connection to serial device."""
        try:
            target_port = self.port
            # If target port fails, try auto-detection
            try:
                self.ser = serial.Serial(target_port, self.baud_rate, timeout=self.timeout)
                self.is_connected = True
                time.sleep(1.5)  # Arduino auto-reset wait
                return
            except Exception:
                detected = self._auto_detect_port()
                if detected and detected != target_port:
                    self.ser = serial.Serial(detected, self.baud_rate, timeout=self.timeout)
                    self.port = detected
                    self.is_connected = True
                    time.sleep(1.5)
                    return
        except Exception:
            self.is_connected = False
            self.ser = None

    def send_state(self, state_char, fatigue_risk=None):
        """
        Send state character (and optional fatigue risk 0-100) to the
        embedded hardware:

        - 'D' -> Drowsiness (seatbelt vibration + buzzer)
        - 'Y' -> Yawn (fatigue LED pulse)
        - 'L' -> Looking away / Distracted (intermittent vibration pulse)
        - 'N' -> Normal (all alarms off, green status)

        Wire format: ``"<state>,<risk>\\n"`` e.g. ``"D,82\\n"``. Firmware maps the
        risk score to the seatbelt vibration frequency in the paper's
        100-300 Hz range. A legacy ``"<state>\\n"`` message is still supported.
        """
        if not self.enabled or not self.is_connected or self.ser is None:
            return

        risk = None
        if fatigue_risk is not None:
            try:
                risk = int(max(0.0, min(100.0, float(fatigue_risk))))
            except (TypeError, ValueError):
                risk = None

        now = time.time()
        with self._lock:
            # Avoid flooding the serial line (rate limit to 10 Hz), but still
            # send when the risk score changes meaningfully.
            risk_changed = (risk is not None and self.last_sent_risk is not None
                            and abs(risk - self.last_sent_risk) >= 3)
            if (state_char == self.last_sent_state and not risk_changed
                    and (now - self.last_send_time < 0.25)):
                return

            try:
                message = f"{state_char},{risk}\n" if risk is not None else f"{state_char}\n"
                self.ser.write(message.encode('utf-8'))
                self.ser.flush()
                self.last_sent_state = state_char
                self.last_sent_risk = risk
                self.last_send_time = now
            except Exception:
                self.is_connected = False

    def close(self):
        """Safely close serial connection."""
        if self.ser and self.is_connected:
            try:
                self.send_state('N')
                self.ser.close()
            except Exception:
                pass
            self.is_connected = False
