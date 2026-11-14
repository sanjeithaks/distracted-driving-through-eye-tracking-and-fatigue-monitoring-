"""
Main Execution Script for Distracted Driving & Fatigue Monitoring System
Based on the IEEE Research Paper:
"Detect distracted driving through eye tracking and fatigue monitoring using embedded system"

Usage Examples:
  # Run live with default webcam
  python main.py

  # Run with a recorded test video file and save output
  python main.py --source driving_test.mp4 --save result.mp4

  # Run headless (no GUI window, for server / batch processing)
  python main.py --source driving_test.mp4 --save result.mp4 --headless
"""
import argparse
import sys
import os
import cv2

# Set MPLCONFIGDIR to avoid matplotlib cache warnings
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))

from src.detector_engine import DriverMonitoringEngine
from src.session_logger import SessionLogger
from src.report import generate_html_report
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    EAR_THRESHOLD, MOR_THRESHOLD, SERIAL_PORT
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect distracted driving through eye tracking and fatigue monitoring using embedded system"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(CAMERA_INDEX),
        help="Video source: camera index (e.g., '0') or path to video file (e.g., 'test.mp4')"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output video file path to save annotated results (e.g., 'output.mp4')"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (useful for headless servers / batch jobs)"
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio sound alarm"
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Disable embedded microcontroller serial communication"
    )
    parser.add_argument(
        "--port",
        type=str,
        default=SERIAL_PORT,
        help=f"Serial port for embedded system (default: {SERIAL_PORT})"
    )
    parser.add_argument(
        "--ear-thresh",
        type=float,
        default=EAR_THRESHOLD,
        help=f"Eye Aspect Ratio threshold (default: {EAR_THRESHOLD})"
    )
    parser.add_argument(
        "--mor-thresh",
        type=float,
        default=MOR_THRESHOLD,
        help=f"Mouth Opening Ratio threshold (default: {MOR_THRESHOLD})"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for session CSV / JSON logs (default: logs)"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable session telemetry logging (CSV / JSON / HTML report)"
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path for the generated HTML session report"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input source
    source = args.source
    if source.isdigit():
        source = int(source)

    print("=" * 70)
    print(" DISTRACTED DRIVING & FATIGUE MONITORING SYSTEM")
    print(" Based on IEEE Research Paper (CIT, Bengaluru)")
    print("=" * 70)
    print(f" Source:           {source}")
    print(f" EAR Threshold:    {args.ear_thresh}")
    print(f" MOR Threshold:    {args.mor_thresh}")
    print(f" Audio Warning:    {'OFF' if args.no_audio else 'ON'}")
    print(f" Embedded Serial:  {'OFF' if args.no_serial else f'ON (Port: {args.port})'}")
    print(f" Headless Mode:    {'YES' if args.headless else 'NO'}")
    print(f" Session Logging:  {'OFF' if args.no_log else f'ON ({args.log_dir})'}")
    print("=" * 70)

    # Initialize monitoring engine
    engine = DriverMonitoringEngine(
        ear_thresh=args.ear_thresh,
        mor_thresh=args.mor_thresh,
        enable_serial=not args.no_serial,
        enable_audio=not args.no_audio,
        serial_port=args.port
    )

    logger = SessionLogger(output_dir=args.log_dir, enabled=not args.no_log)
    if logger.enabled:
        print(f"[INFO] Session log: {logger.csv_path}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {source}")
        if isinstance(source, int):
            print("Tip: If you do not have a physical webcam connected, you can run the demo simulator:")
            print("     python demo.py")
        sys.exit(1)

    # Prepare video writer if saving output
    writer = None
    if args.save:
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in <= 0 or fps_in is None:
            fps_in = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            width, height = FRAME_WIDTH, FRAME_HEIGHT

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))
        print(f"[INFO] Saving processed video to: {args.save}")

    window_name = "Driver Fatigue & Distraction Monitoring System"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached.")
                break

            frame_count += 1

            # Process frame through the complete pipeline
            annotated_frame, telemetry = engine.process_frame(frame)

            # Write to output file if requested
            if writer:
                writer.write(annotated_frame)

            # Record telemetry for the session report
            logger.log_frame(telemetry, frame_count)

            # Print telemetry update in terminal every 30 frames
            if frame_count % 30 == 0:
                status_str = "NORMAL"
                if telemetry['is_drowsy']:
                    status_str = "DROWSINESS ALERT!"
                elif telemetry['is_distracted']:
                    status_str = f"LOOKING AWAY ({telemetry['distraction_reason']})"
                elif telemetry['is_yawning']:
                    status_str = "YAWNING DETECTED!"

                sys.stdout.write(
                    f"\rFrame {frame_count:05d} | EAR: {telemetry['ear']:.2f} | "
                    f"MOR: {telemetry['mor']:.2f} | Yaw: {telemetry['yaw']:+5.1f} | "
                    f"Pitch: {telemetry['pitch']:+5.1f} | Gaze: {telemetry['gaze_label']:<6} | "
                    f"Blinks: {telemetry['blinks']} | "
                    f"Yawns: {telemetry['yawns']} | Distr: {telemetry['distractions']} | "
                    f"Status: {status_str:<38}"
                )
                sys.stdout.flush()

            # Display GUI window
            if not args.headless:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or Esc
                    print("\n[INFO] User quit.")
                    break
                elif key == ord('r'):  # Reset counters
                    engine.eye_tracker.reset()
                    engine.yawn_tracker.reset()
                    engine.head_pose.reset()
                    engine.gaze_tracker.reset()
                    print("\n[INFO] Counters reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\n[SUCCESS] Output video successfully saved to: {args.save}")
        if not args.headless:
            cv2.destroyAllWindows()
        engine.close()

        # Finalize the session log and generate the HTML report
        summary = logger.close()
        if logger.enabled:
            print("\n" + "=" * 70)
            print(" SESSION SUMMARY")
            print("=" * 70)
            print(f" Frames processed:      {summary['frames']}")
            print(f" Duration:              {summary['duration_sec']} s")
            print(f" Face detection rate:   {summary['face_detection_rate_pct']} %")
            print(f" Blinks / Yawns:        {summary['blinks']} / {summary['yawns']}")
            print(f" Distractions (gaze):   {summary['distractions']} "
                  f"({summary['gaze_distractions']} gaze alerts)")
            print(f" Drowsy alert events:   {summary['drowsy_events']}")
            print(f" Session CSV:           {logger.csv_path}")
            print(f" Session summary JSON:  {logger.summary_path}")

            report_path = args.report or os.path.join(
                args.log_dir, f"session_{logger.session_name}_report.html"
            )
            try:
                generate_html_report(summary, report_path)
                print(f" HTML report:           {report_path}")
            except Exception as exc:
                print(f"[WARN] Could not generate HTML report: {exc}")
            print("=" * 70)


if __name__ == "__main__":
    main()
