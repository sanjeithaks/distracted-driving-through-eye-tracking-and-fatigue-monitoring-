"""
Raspberry Pi GPIO Hardware Controller
Provides GPIO hardware actuator support for Raspberry Pi deployment.
Interfaces with vibration motor (via PWM) and buzzer.
As referenced in Section II of the IEEE Paper.
"""
import time
import sys

try:
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False


class RaspberryPiHardware:
    """
    Controls physical vibration motor, buzzer, and LEDs on Raspberry Pi GPIO.
    """
    PIN_VIBRATION = 18  # Hardware PWM (BCM 18)
    PIN_BUZZER = 23
    PIN_LED_GREEN = 24
    PIN_LED_RED = 25
    PIN_LED_YELLOW = 12
    PIN_LED_BLUE = 16

    def __init__(self, vibration_freq=200):
        self.vibration_freq = vibration_freq
        self.pwm_vibration = None

        if RPI_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Output pins
            for pin in [self.PIN_BUZZER, self.PIN_LED_GREEN, self.PIN_LED_RED,
                        self.PIN_LED_YELLOW, self.PIN_LED_BLUE]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)

            GPIO.setup(self.PIN_VIBRATION, GPIO.OUT)
            # Setup PWM on vibration motor pin at desired frequency (100 - 300 Hz)
            self.pwm_vibration = GPIO.PWM(self.PIN_VIBRATION, self.vibration_freq)
            self.pwm_vibration.start(0)

            # Normal startup LED
            GPIO.output(self.PIN_LED_GREEN, GPIO.HIGH)
            print(f"[RPI Hardware] Initialized successfully at {self.vibration_freq} Hz PWM.")
        else:
            print("[RPI Hardware] RPi.GPIO not detected. Running in mock simulation mode.")

    def set_state(self, state, fatigue_risk=None):
        """
        Actuate hardware based on state and optional Fatigue Risk Index (0-100):
        'D': Drowsy -> Red LED + risk-scaled PWM vibration (100-300 Hz) + Buzzer
        'L': Looking Away -> Blue LED + pulsed vibration
        'Y': Yawning -> Yellow LED
        'N': Normal -> Green LED, actuators off
        """
        if not RPI_AVAILABLE:
            return

        risk = 0
        if fatigue_risk is not None:
            try:
                risk = max(0, min(100, int(float(fatigue_risk))))
            except (TypeError, ValueError):
                risk = 0

        # Risk maps the seatbelt vibration into the paper's 100-300 Hz range
        target_freq = 100 + risk * 2
        duty = 45 + (risk * 55) // 100

        if self.pwm_vibration and target_freq != self.vibration_freq:
            try:
                self.pwm_vibration.ChangeFrequency(target_freq)
                self.vibration_freq = target_freq
            except (AttributeError, RuntimeError):
                pass

        if state == 'D':
            GPIO.output(self.PIN_LED_RED, GPIO.HIGH)
            GPIO.output(self.PIN_LED_GREEN, GPIO.LOW)
            GPIO.output(self.PIN_LED_YELLOW, GPIO.LOW)
            GPIO.output(self.PIN_LED_BLUE, GPIO.LOW)
            GPIO.output(self.PIN_BUZZER, GPIO.HIGH)
            if self.pwm_vibration:
                self.pwm_vibration.ChangeDutyCycle(duty)

        elif state == 'L':
            GPIO.output(self.PIN_LED_BLUE, GPIO.HIGH)
            GPIO.output(self.PIN_LED_GREEN, GPIO.LOW)
            GPIO.output(self.PIN_LED_RED, GPIO.LOW)
            GPIO.output(self.PIN_LED_YELLOW, GPIO.LOW)
            GPIO.output(self.PIN_BUZZER, GPIO.LOW)
            if self.pwm_vibration:
                self.pwm_vibration.ChangeDutyCycle(max(30, duty - 20))

        elif state == 'Y':
            GPIO.output(self.PIN_LED_YELLOW, GPIO.HIGH)
            GPIO.output(self.PIN_LED_GREEN, GPIO.LOW)
            GPIO.output(self.PIN_LED_RED, GPIO.LOW)
            GPIO.output(self.PIN_LED_BLUE, GPIO.LOW)
            GPIO.output(self.PIN_BUZZER, GPIO.LOW)
            if self.pwm_vibration:
                self.pwm_vibration.ChangeDutyCycle(0)

        else:  # Normal 'N'
            GPIO.output(self.PIN_LED_GREEN, GPIO.HIGH)
            GPIO.output(self.PIN_LED_RED, GPIO.LOW)
            GPIO.output(self.PIN_LED_YELLOW, GPIO.LOW)
            GPIO.output(self.PIN_LED_BLUE, GPIO.LOW)
            GPIO.output(self.PIN_BUZZER, GPIO.LOW)
            if self.pwm_vibration:
                self.pwm_vibration.ChangeDutyCycle(0)

    def cleanup(self):
        if RPI_AVAILABLE:
            if self.pwm_vibration:
                self.pwm_vibration.stop()
            GPIO.cleanup()


if __name__ == "__main__":
    print("Testing Raspberry Pi Hardware Interface...")
    hw = RaspberryPiHardware(vibration_freq=200)
    for s in ['N', 'Y', 'L', 'D', 'N']:
        print(f"Testing state: {s}")
        hw.set_state(s)
        time.sleep(1)
    hw.cleanup()
    print("Test finished.")
