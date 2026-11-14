"""
All-in-One Demonstration Script
Executes full system verification:
1. Generates test driving video simulating all paper scenarios
2. Processes video through DriverMonitoringEngine
3. Generates annotated video with HUD overlays (demo_output_annotated.mp4)
4. Saves screenshot stills matching Figures 4-9 from the IEEE Paper
5. Prints full verification and performance analytics report
"""
import os
import sys
import cv2

# Set MPLCONFIGDIR
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))

from generate_test_media import generate_scenario_video
from src.detector_engine import DriverMonitoringEngine
from src.session_logger import SessionLogger
from src.report import generate_html_report


def run_demo(input_video="demo_sample_driving.mp4",
             output_video="demo_output_annotated.mp4",
             screenshots_dir="docs/screenshots",
             force_regenerate=True):
    print("=" * 75)
    print(" DISTRACTED DRIVING & FATIGUE MONITORING SYSTEM - COMPLETE DEMO")
    print(" IEEE Paper Implementation & Verification")
    print("=" * 75)

    if force_regenerate and os.path.exists(input_video):
        try:
            os.remove(input_video)
        except Exception:
            pass

    if not os.path.exists(input_video):
        generate_scenario_video(input_video, total_frames=540)

    os.makedirs(screenshots_dir, exist_ok=True)

    # Initialize monitoring engine (software demo mode)
    engine = DriverMonitoringEngine(
        enable_serial=False,
        enable_audio=False
    )
    logger = SessionLogger(output_dir="docs", session_name="demo")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[ERROR] Could not open {input_video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"\n[DEMO] Processing {input_video} -> {output_video}...")

    frame_idx = 0
    captured_shots = {
        'fig4': False,
        'fig5': False,
        'fig6': False,
        'fig7': False,
        'fig9': False,
        'fig10': False
    }

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        annotated_frame, telemetry = engine.process_frame(frame)
        writer.write(annotated_frame)
        logger.log_frame(telemetry, frame_idx)

        # Snapshot 1: Normal Face Detected (Fig 4)
        if 20 <= frame_idx <= 40 and not captured_shots['fig4'] and telemetry['face_detected']:
            shot_path = os.path.join(screenshots_dir, "fig4_face_detected.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig4'] = True
            print(f" -> Captured Figure 4 (Face Detected): {shot_path}")

        # Snapshot 2: Drowsiness Alert (Fig 5)
        if telemetry['is_drowsy'] and not captured_shots['fig5']:
            shot_path = os.path.join(screenshots_dir, "fig5_drowsiness_alert.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig5'] = True
            print(f" -> Captured Figure 5 (Drowsiness Alert): {shot_path}")

        # Snapshot 3: Yawning Detected (Fig 6)
        if telemetry['is_yawning'] and not captured_shots['fig6']:
            shot_path = os.path.join(screenshots_dir, "fig6_yawning_detected.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig6'] = True
            print(f" -> Captured Figure 6 (Yawning Detected): {shot_path}")

        # Snapshot 4: Looking Away - Window / Passenger (Fig 7 & 8)
        if telemetry['is_distracted'] and ("LOOKING LEFT" in telemetry['distraction_reason'] or "LOOKING RIGHT" in telemetry['distraction_reason']) and not captured_shots['fig7']:
            shot_path = os.path.join(screenshots_dir, "fig7_looking_away.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig7'] = True
            print(f" -> Captured Figure 7/8 (Looking Away - Window/Passenger): {shot_path}")

        # Snapshot 5: Looking Down - Mobile Phone (Fig 9)
        if telemetry['is_distracted'] and "LOOKING DOWN" in telemetry['distraction_reason'] and not captured_shots['fig9']:
            shot_path = os.path.join(screenshots_dir, "fig9_phone_distraction.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig9'] = True
            print(f" -> Captured Figure 9 (Phone Distraction): {shot_path}")

        # Snapshot 6: Eye-only gaze away from road (Fig 10 - iris tracking)
        if telemetry['is_distracted'] and "GAZE" in telemetry['distraction_reason'] and not captured_shots['fig10']:
            shot_path = os.path.join(screenshots_dir, "fig10_gaze_tracking.jpg")
            cv2.imwrite(shot_path, annotated_frame)
            captured_shots['fig10'] = True
            print(f" -> Captured Figure 10 (Iris Gaze Tracking): {shot_path}")

        if frame_idx % 40 == 0:
            print(f"   Processed frame {frame_idx:03d}/{total_frames or 540} | EAR: {telemetry['ear']:.2f} | MOR: {telemetry['mor']:.2f} | Yaw: {telemetry['yaw']:+5.1f} | Pitch: {telemetry['pitch']:+5.1f} | Gaze: {telemetry['gaze_label']}")

    cap.release()
    writer.release()
    engine.close()

    summary = logger.close()
    report_path = os.path.join(screenshots_dir, "..", "demo_report.html")
    generate_html_report(summary, report_path)

    print("\n" + "=" * 75)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print(f" Annotated Video Output:   {output_video}")
    print(f" Total Frames Processed:   {frame_idx}")
    print(f" Total Blinks Counted:     {summary['blinks']}")
    print(f" Total Yawns Counted:      {summary['yawns']}")
    print(f" Total Distractions:       {summary['distractions']}")
    print(f" Gaze Alerts:              {summary['gaze_distractions']}")
    print(f" Gaze Distribution:        {summary['gaze_direction_counts']}")
    print(f" Screenshots Directory:    {screenshots_dir}")
    print(f" Session CSV:              {logger.csv_path}")
    print(f" Session Summary JSON:     {logger.summary_path}")
    print(f" HTML Report:              {os.path.normpath(report_path)}")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
