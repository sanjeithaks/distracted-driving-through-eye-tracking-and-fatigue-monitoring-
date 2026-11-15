/*
 * Distracted Driving Through Eye Tracking and Fatigue Monitoring
 * Embedded System - Arduino Hardware Controller
 *
 * Target: Arduino Uno / Nano / Mega (AVR). ESP32 also works, with analogWrite
 * falling back to the default PWM implementation.
 *
 * Hardware Connections:
 * - Pin 9:  Seatbelt Vibration Motor (PWM, 100 Hz - 300 Hz)
 * - Pin 8:  Piezo Buzzer (Audio Warning)
 * - Pin 5:  Normal Status LED (Green)
 * - Pin 6:  Yawn / Fatigue LED (Yellow)
 * - Pin 7:  Distraction LED (Blue)
 * - Pin 10: Drowsiness Alert LED (Red)
 *
 * Serial Protocol (9600 Baud):
 *   "<state>,<fatigue_risk>\n"   e.g.  "D,82\n"
 *   - state: D = drowsy, L = looking away, Y = yawning, N = normal
 *   - fatigue_risk: 0-100 from the computer-vision Fatigue Risk Index (FRI)
 *
 * The risk score dynamically maps the seatbelt vibration frequency into the
 * paper's 100-300 Hz range and the vibration intensity into 45-100% duty:
 *   frequency = 100 + risk * 2      (100 Hz at FRI 0, 300 Hz at FRI 100)
 *
 * Reference:
 * "Detect distracted driving through eye tracking and fatigue monitoring
 *  using embedded system", Section IV: "seat belt vibration whose frequency
 *  will vary between 100 to 300 Hz".
 */

const int PIN_VIBRATION_MOTOR = 9;   // Timer1 PWM pin for vibration motor
const int PIN_BUZZER          = 8;   // Hardware audio buzzer
const int PIN_LED_GREEN       = 5;   // Normal
const int PIN_LED_YELLOW      = 6;   // Yawning
const int PIN_LED_BLUE        = 7;   // Looking away
const int PIN_LED_RED         = 10;  // Drowsy alert

char currentState = 'N';
int currentRisk = 0;
int appliedFrequency = 0;
unsigned long lastHeartbeatTime = 0;
const unsigned long TIMEOUT_MS = 2500; // Auto-reset if serial heartbeat lost

void setup() {
  Serial.begin(9600);

  pinMode(PIN_VIBRATION_MOTOR, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_YELLOW, OUTPUT);
  pinMode(PIN_LED_BLUE, OUTPUT);
  pinMode(PIN_LED_RED, OUTPUT);

  setVibration(100, 0);

  // Initial self-test sequence
  setAllLeds(HIGH);
  tone(PIN_BUZZER, 1000, 150);
  delay(200);
  setAllLeds(LOW);
  digitalWrite(PIN_LED_GREEN, HIGH);
}

void loop() {
  // Read incoming "<state>,<risk>\n" commands from the Python CV pipeline
  while (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) continue;

    char cmd = line.charAt(0);
    int comma = line.indexOf(',');
    if (comma > 0) {
      currentRisk = constrain(line.substring(comma + 1).toInt(), 0, 100);
    }

    if (cmd == 'D' || cmd == 'Y' || cmd == 'L' || cmd == 'N') {
      currentState = cmd;
      lastHeartbeatTime = millis();
    }
  }

  // Safety fallback: if no command is received for > 2.5 s, go to safe state
  if (millis() - lastHeartbeatTime > TIMEOUT_MS) {
    currentState = 'N';
    currentRisk = 0;
  }

  // Risk-driven seatbelt vibration frequency: 100 Hz (low) -> 300 Hz (critical)
  int vibrationFrequency = 100 + (currentRisk * 2);
  // Intensity: 45% at low risk -> 100% at critical risk
  int vibrationDuty = 45 + ((currentRisk * 55) / 100);

  switch (currentState) {
    case 'D': // DROWSINESS ALERT!
      digitalWrite(PIN_LED_RED, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_YELLOW, LOW);
      digitalWrite(PIN_LED_BLUE, LOW);

      setVibration(vibrationFrequency, vibrationDuty);
      tone(PIN_BUZZER, 1500);   // Piercing buzzer tone
      break;

    case 'L': // LOOKING AWAY / DISTRACTION!
      digitalWrite(PIN_LED_BLUE, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_RED, LOW);
      digitalWrite(PIN_LED_YELLOW, LOW);

      // Pulsing vibration burst to reorient the driver
      if ((millis() / 250) % 2 == 0) {
        setVibration(vibrationFrequency, vibrationDuty);
        tone(PIN_BUZZER, 1100);
      } else {
        setVibration(100, 0);
        noTone(PIN_BUZZER);
      }
      break;

    case 'Y': // YAWNING DETECTED!
      digitalWrite(PIN_LED_YELLOW, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_RED, LOW);
      digitalWrite(PIN_LED_BLUE, LOW);

      setVibration(100, 0);
      if ((millis() / 400) % 3 == 0) {
        tone(PIN_BUZZER, 800);  // Gentle reminder beep
      } else {
        noTone(PIN_BUZZER);
      }
      break;

    case 'N': // NORMAL DRIVING
    default:
      digitalWrite(PIN_LED_GREEN, HIGH);
      digitalWrite(PIN_LED_RED, LOW);
      digitalWrite(PIN_LED_YELLOW, LOW);
      digitalWrite(PIN_LED_BLUE, LOW);
      setVibration(100, 0);
      noTone(PIN_BUZZER);
      break;
  }
}

/**
 * Configure Timer1 fast PWM on pin 9 with a variable frequency and duty cycle.
 * On AVR, TOP = ICR1 so frequencies from 100 Hz to 300 Hz are accurate.
 */
void setVibration(int frequencyHz, int dutyPercent) {
  frequencyHz = constrain(frequencyHz, 100, 300);
  dutyPercent = constrain(dutyPercent, 0, 100);

#ifdef __AVR__
  if (frequencyHz != appliedFrequency) {
    // Fast PWM, TOP = ICR1, prescaler = 8
    uint32_t top = (16000000UL / (8UL * (uint32_t)frequencyHz)) - 1UL;
    TCCR1A = _BV(COM1A1) | _BV(WGM11);
    TCCR1B = _BV(WGM13) | _BV(WGM12) | _BV(CS11);
    ICR1 = (uint16_t)top;
    appliedFrequency = frequencyHz;
  }
  OCR1A = (uint32_t)ICR1 * dutyPercent / 100UL;
#else
  analogWrite(PIN_VIBRATION_MOTOR, dutyPercent * 255 / 100);
#endif
}

void setAllLeds(int state) {
  digitalWrite(PIN_LED_GREEN, state);
  digitalWrite(PIN_LED_YELLOW, state);
  digitalWrite(PIN_LED_BLUE, state);
  digitalWrite(PIN_LED_RED, state);
}
