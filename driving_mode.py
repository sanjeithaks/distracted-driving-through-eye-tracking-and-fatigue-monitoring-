"""
Immersive Driving Mode
======================
Full-screen driving cockpit around the live driver-monitoring pipeline.

Run:
    python driving_mode.py                  # live webcam, full-screen cockpit
    python driving_mode.py --windowed       # windowed (1280 x 720)
    python driving_mode.py --source clip.mp4 --headless --save out.mp4

Controls:
    A / D  or  Left / Right : steer
    W / S  or  Up / Down    : accelerate / brake
    R : reset counters and recalibrate
    F : toggle full-screen
    Q / Esc : quit

The road, speedometer, steering wheel and risk-driven lane behaviour are
procedurally rendered with OpenCV, so no game engine or extra dependency is
required.  Driver state directly affects the simulation: drowsiness reduces
speed and pulses the virtual seatbelt motor, while distraction makes the car
drift toward the lane edge.
"""
import argparse
import os
import sys
import time

import cv2

from src.detector_engine import DriverMonitoringEngine
from src.driving_simulator import DrivingSimulator
from src.session_logger import SessionLogger
from src.report import generate_html_report
from src.config import ENABLE_AUDIO_ALARM


ARROW_LEFT = 2424832
ARROW_RIGHT = 2555904
ARROW_UP = 2490368
ARROW_DOWN = 2621440


def parse_args():
    p = argparse.ArgumentParser(description="Immersive driving mode with driver monitoring")
    p.add_argument("--source", default="0", help="Camera index or video file path")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--duration", type=float, default=0.0, help="Stop after N seconds (0 = unlimited)")
    p.add_argument("--save", default=None, help="Save the driving view to an MP4 file")
    p.add_argument("--headless", action="store_true", help="No window (batch rendering)")
    p.add_argument("--windowed", action="store_true", help="Do not go full-screen")
    p.add_argument("--no-audio", action="store_true", help="Disable alert sounds")
    p.add_argument("--log-dir", default=os.path.join("logs", "driving"))
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source
    is_camera = isinstance(source, int)

    print("=" * 70)
    print(" IMMERSIVE DRIVING MODE - Driver Fatigue & Distraction Monitoring")
    print("=" * 70)
    print(" A/D or arrows: steer   W/S: speed   R: reset   F: fullscreen   Q: quit")
    print("=" * 70)

    engine = DriverMonitoringEngine(
        enable_serial=False,
        enable_audio=(not args.no_audio) and ENABLE_AUDIO_ALARM,
        show_framing_hint=is_camera,
    )
    logger = SessionLogger(output_dir=args.log_dir, session_name="driving",
                           snapshot_dir=os.path.join(args.log_dir, "snapshots"))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        if is_camera:
            print("        Close other camera apps (Teams/Zoom/Camera) and retry.")
        engine.close()
        logger.close()
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_in if 5.0 <= fps_in <= 120.0 else 30.0
    frame_dt = 1.0 / fps if not is_camera else None

    sim = DrivingSimulator(args.width, args.height, fps=int(round(fps)))
    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (args.width, args.height))

    win = "Driving Mode - Driver Monitoring"
    fullscreen = not args.windowed
    if not args.headless:
        if fullscreen:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            # AUTOSIZE shows the cockpit 1:1 (no DPI scaling crop)
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(win, 80, 50)

    steer_input, throttle = 0.0, 0.0
    prev = time.time()
    start = time.time()
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("\n[INFO] End of source.")
                break

            frame_idx += 1
            annotated, telemetry = engine.process_frame(frame)

            now = time.time()
            dt = frame_dt if frame_dt is not None else (now - prev)
            prev = now

            if not args.headless:
                key = cv2.waitKeyEx(1)
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('r'), ord('R')):
                    engine.eye_tracker.reset()
                    engine.yawn_tracker.reset()
                    engine.head_pose.reset()
                    engine.gaze_tracker.reset()
                    engine.fatigue_analyzer.reset()
                    sim.reset()
                    print("[INFO] Counters reset; head pose recalibrating.")
                elif key in (ord('f'), ord('F')):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        win, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                    )
                    if not fullscreen:
                        cv2.resizeWindow(win, args.width, args.height)
                        cv2.moveWindow(win, 80, 50)

                steer_input = 0.0
                throttle = 0.0
                if key in (ord('a'), ord('A'), ARROW_LEFT):
                    steer_input = -1.0
                elif key in (ord('d'), ord('D'), ARROW_RIGHT):
                    steer_input = 1.0
                elif key in (ord('w'), ord('W'), ARROW_UP):
                    throttle = 1.0
                elif key in (ord('s'), ord('S'), ARROW_DOWN):
                    throttle = -1.0

            sim.update(telemetry, dt, steer_input, throttle)
            scene = sim.render(annotated, telemetry)

            if writer is not None:
                writer.write(scene)
            if (not args.headless) and frame_idx == 10:
                os.makedirs("docs", exist_ok=True)
                cv2.imwrite(os.path.join("docs", "driving_live_frame.jpg"), scene)
            logger.log_frame(telemetry, frame_idx, frame=scene)

            if not args.headless:
                cv2.imshow(win, scene)

            if frame_idx % 30 == 0:
                status = telemetry.get('risk_level', 'LOW')
                print(f"\rFrame {frame_idx:05d} | {int(sim.speed):3d} km/h | "
                      f"EAR {telemetry['ear']:.2f} | Gaze {telemetry['gaze_label']:<6} | "
                      f"FRI {telemetry['fatigue_risk']:3.0f} ({status:<8}) | "
                      f"State {sim.state:<10}", end="", flush=True)

            if args.duration > 0 and (time.time() - start) >= args.duration:
                print(f"\n[INFO] Duration ({args.duration:.0f} s) reached.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n[SUCCESS] Driving video saved to: {args.save}")
        if not args.headless:
            cv2.destroyAllWindows()
        engine.close()
        summary = logger.close()
        report_path = os.path.join(args.log_dir, "driving_report.html")
        try:
            generate_html_report(summary, report_path)
            print(f"[INFO] Session report: {report_path}")
        except Exception:
            pass
        print(f"[INFO] Frames: {summary['frames']} | Distractions: {summary['distractions']} "
              f"| Microsleeps: {summary['microsleeps']} | Peak FRI: {summary['peak_fatigue_risk']}")


if __name__ == "__main__":
    main()
