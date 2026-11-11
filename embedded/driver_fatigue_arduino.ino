/*
 * Distracted Driving Through Eye Tracking and Fatigue Monitoring
 * Embedded System - Arduino Hardware Controller
 * 
 * Target: Arduino Uno / Nano / Mega / ESP32
 * 
 * Hardware Connections:
 * - Pin 9:  Seatbelt Vibration Motor (PWM controlled, 100 Hz - 300 Hz)
 * - Pin 8:  Piezo Buzzer (Audio Warning)
 * - Pin 5:  Normal Status LED (Green)
 * - Pin 6:  Yawn / Fatigue LED (Yellow)
 * - Pin 7:  Distraction LED (Blue)
 * - Pin 10: Drowsiness Alert LED (Red)
 * 
 * Serial Protocol (9600 Baud):
 * - 'D': Drowsiness Alert (High-intensity seatbelt vibration + continuous buzzer alarm)
 * - 'Y': Yawning Detected (Intermittent warning beep + yellow LED)
 * - 'L': Looking Away Distraction (Pulsed seatbelt vibration + blue LED)
 * - 'N': Normal Driving (Green LED ON, actuators OFF)
 * 
 * Reference:
 * "Detect distracted driving through eye tracking and fatigue monitoring using embedded system"
 * Section IV: "seat belt vibration whose frequency will vary between 100 to 300 Hz"
 */

const int PIN_VIBRATION_MOTOR = 9;  // PWM pin for vibration motor
const int PIN_BUZZER          = 8;  // Hardware audio buzzer
const int PIN_LED_GREEN       = 5;  // Normal
const int PIN_LED_YELLOW      = 6;  // Yawning
const int PIN_LED_BLUE        = 7;  // Looking away
const int PIN_LED_RED         = 10; // Drowsy alert

char currentState = 'N';
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

  // Initial self-test sequence
  setAllLeds(HIGH);
  tone(PIN_BUZZER, 1000, 150);
  delay(200);
  setAllLeds(LOW);
  digitalWrite(PIN_LED_GREEN, HIGH);
}

void loop() {
  // Read incoming serial commands from Python CV pipeline
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'D' || cmd == 'Y' || cmd == 'L' || cmd == 'N') {
      currentState = cmd;
      lastHeartbeatTime = millis();
    }
  }

  // Safety fallback: if no command received for > 2.5 seconds, reset to safe normal state
  if (millis() - lastHeartbeatTime > TIMEOUT_MS) {
    currentState = 'N';
  }

  // Actuator handling based on current state
  switch (currentState) {
    case 'D': // DROWSINESS ALERT!
      // Continuous urgent alarm + 200 Hz seatbelt vibration
      digitalWrite(PIN_LED_RED, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_YELLOW, LOW);
      digitalWrite(PIN_LED_BLUE, LOW);

      // Seatbelt vibration at ~200 Hz PWM duty cycle (approx 80% duty)
      analogWrite(PIN_VIBRATION_MOTOR, 205);
      
      // Piercing buzzer tone
      tone(PIN_BUZZER, 1500);
      break;

    case 'L': // LOOKING AWAY / DISTRACTION!
      digitalWrite(PIN_LED_BLUE, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_RED, LOW);
      digitalWrite(PIN_LED_YELLOW, LOW);

      // Pulsing vibration burst to reorient driver
      if ((millis() / 250) % 2 == 0) {
        analogWrite(PIN_VIBRATION_MOTOR, 180);
        tone(PIN_BUZZER, 1100);
      } else {
        analogWrite(PIN_VIBRATION_MOTOR, 0);
        noTone(PIN_BUZZER);
      }
      break;

    case 'Y': // YAWNING DETECTED!
      digitalWrite(PIN_LED_YELLOW, HIGH);
      digitalWrite(PIN_LED_GREEN, LOW);
      digitalWrite(PIN_LED_RED, LOW);
      digitalWrite(PIN_LED_BLUE, LOW);

      analogWrite(PIN_VIBRATION_MOTOR, 0);
      // Gentle reminder beep
      if ((millis() / 400) % 3 == 0) {
        tone(PIN_BUZZER, 800);
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
      analogWrite(PIN_VIBRATION_MOTOR, 0);
      noTone(PIN_BUZZER);
      break;
  }
}

void setAllLeds(int state) {
  digitalWrite(PIN_LED_GREEN, state);
  digitalWrite(PIN_LED_YELLOW, state);
  digitalWrite(PIN_LED_BLUE, state);
  digitalWrite(PIN_LED_RED, state);
}
