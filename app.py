"""
Streamlit Web Dashboard for Distracted Driving & Fatigue Monitoring System
Provides an interactive web interface with:
- Built-in demo clip, uploaded video file or live webcam input
- Live telemetry HUD (EAR, MOR, iris gaze, head pose, alert state)
- Interactive threshold tuning
- Real-time time-series plots for EAR and MOR
- Session CSV / JSON logging and downloadable HTML report
"""
import os
import tempfile
import time

import cv2
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Driver Fatigue & Distraction Monitor",
    page_icon="🚗",
    layout="wide"
)

# Set matplotlib cache dir
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))

from src.detector_engine import DriverMonitoringEngine
from src.session_logger import SessionLogger
from src.report import generate_html_report
from src.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    MOR_THRESHOLD, MOR_CONSEC_FRAMES,
    SERIAL_PORT
)

LOGS_DIR = "logs"


def main():
    st.title("🚗 Distracted Driving Through Eye Tracking & Fatigue Monitoring")
    st.caption(
        "IEEE paper implementation (CIT Bengaluru, ECE Dept) — EAR drowsiness, "
        "MOR yawning, solvePnP head pose and iris-based eye tracking."
    )

    # ------------------------------------------------------------------ #
    # Sidebar controls
    # ------------------------------------------------------------------ #
    st.sidebar.header("⚙️ Monitoring Parameters")
    ear_thresh = st.sidebar.slider(
        "Eye Aspect Ratio (EAR) Threshold",
        min_value=0.15, max_value=0.35, value=EAR_THRESHOLD, step=0.01,
        help="Values below this threshold indicate closed eyes."
    )
    ear_consec = st.sidebar.slider(
        "Drowsiness Frame Threshold",
        min_value=10, max_value=60, value=EAR_CONSEC_FRAMES, step=5,
        help="Consecutive frames below EAR threshold before triggering drowsiness alert."
    )
    mor_thresh = st.sidebar.slider(
        "Mouth Opening Ratio (MOR) Threshold",
        min_value=0.20, max_value=0.80, value=MOR_THRESHOLD, step=0.02,
        help="Values above this threshold indicate mouth open / yawning."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Embedded System Interface")
    enable_serial = st.sidebar.checkbox("Enable Serial Hardware Bridge", value=False)
    serial_port = st.sidebar.text_input("COM Port", value=SERIAL_PORT)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎥 Video Input")
    input_mode = st.sidebar.radio(
        "Input mode",
        options=["Run Built-in Demo Clip", "Upload Driving Video File", "Live Webcam Stream"]
    )

    uploaded_file = None
    if input_mode == "Upload Driving Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])

    webcam_seconds = 20
    if input_mode == "Live Webcam Stream":
        webcam_seconds = st.sidebar.slider(
            "Webcam capture duration (seconds)", min_value=5, max_value=120, value=20, step=5
        )
        st.sidebar.caption("The dashboard shows highlights while the clip is processed.")

    # ------------------------------------------------------------------ #
    # Dashboard layout
    # ------------------------------------------------------------------ #
    col_video, col_metrics = st.columns([3, 2])

    with col_metrics:
        st.subheader("📊 Live Telemetry HUD")
        m1, m2, m3 = st.columns(3)
        metric_ear = m1.metric("EAR", "0.00")
        metric_mor = m2.metric("MOR", "0.00")
        metric_gaze = m3.metric("Gaze", "N/A")

        m4, m5 = st.columns(2)
        metric_yaw = m4.metric("Yaw", "0.0°")
        metric_pitch = m5.metric("Pitch", "0.0°")

        c1, c2, c3 = st.columns(3)
        metric_blinks = c1.metric("Blinks", "0")
        metric_yawns = c2.metric("Yawns", "0")
        metric_distractions = c3.metric("Distractions", "0")

        status_box = st.empty()
        status_box.success("STATUS: SYSTEM READY")

        st.subheader("📈 Real-time Trend")
        chart_holder = st.empty()

    with col_video:
        st.subheader("📹 Video Stream & Facial Feature Tracking")
        video_placeholder = st.empty()
        video_placeholder.info("Select an input mode in the sidebar and press Start Monitoring.")
        start_btn = st.button("🚀 Start Monitoring", key="start_monitoring", type="primary")

    if not start_btn:
        return

    # ------------------------------------------------------------------ #
    # Resolve input source
    # ------------------------------------------------------------------ #
    temp_input_file = None
    if input_mode == "Run Built-in Demo Clip":
        if not os.path.exists("demo_sample_driving.mp4"):
            with st.spinner("Generating synthetic driving clip..."):
                from generate_test_media import generate_scenario_video
                generate_scenario_video("demo_sample_driving.mp4", total_frames=540)
        source = "demo_sample_driving.mp4"

    elif input_mode == "Upload Driving Video File":
        if uploaded_file is None:
            st.warning("Please upload a video file in the sidebar first.")
            return
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        temp_input_file = tfile.name
        source = temp_input_file

    else:
        source = 0  # Default webcam

    engine = DriverMonitoringEngine(
        ear_thresh=ear_thresh,
        ear_consec_frames=ear_consec,
        mor_thresh=mor_thresh,
        enable_serial=enable_serial,
        enable_audio=False,
        serial_port=serial_port
    )
    logger = SessionLogger(output_dir=LOGS_DIR, session_name="dashboard")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error(f"Could not open video source: {source}")
        engine.close()
        logger.close()
        return

    history = []
    frame_num = 0
    start_time = time.time()
    last_display = 0.0

    if input_mode == "Run Built-in Demo Clip" or input_mode == "Upload Driving Video File":
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        progress = st.progress(0.0)
    else:
        total_frames = 0
        progress = None

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_num += 1
            annotated_frame, telemetry = engine.process_frame(frame)
            logger.log_frame(telemetry, frame_num)

            # Update the video preview at ~15 FPS to keep the UI responsive
            now = time.time()
            if now - last_display > 1.0 / 15.0:
                last_display = now
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                metric_ear.metric("EAR", f"{telemetry['ear']:.2f}")
                metric_mor.metric("MOR", f"{telemetry['mor']:.2f}")
                metric_gaze.metric("Gaze", telemetry['gaze_label'])
                metric_yaw.metric("Yaw", f"{telemetry['yaw']:+.1f}°")
                metric_pitch.metric("Pitch", f"{telemetry['pitch']:+.1f}°")
                metric_blinks.metric("Blinks", str(telemetry['blinks']))
                metric_yawns.metric("Yawns", str(telemetry['yawns']))
                metric_distractions.metric("Distractions", str(telemetry['distractions']))

                if telemetry['is_drowsy']:
                    status_box.error("🚨 DROWSINESS ALERT! Seatbelt vibration + buzzer triggered")
                elif telemetry['is_distracted']:
                    status_box.warning(f"⚠️ LOOKING AWAY! ({telemetry['distraction_reason']})")
                elif telemetry['is_yawning']:
                    status_box.info("🥱 YAWNING DETECTED! Fatigue warning")
                elif telemetry['face_detected']:
                    status_box.success("✅ DRIVER ATTENTIVE — FACE DETECTED")
                else:
                    status_box.warning("👁️ NO FACE DETECTED")

            history.append({'Frame': frame_num,
                            'EAR': telemetry['ear'],
                            'MOR': telemetry['mor']})
            if len(history) > 150:
                history.pop(0)
            if frame_num % 5 == 0 and history:
                chart_holder.line_chart(pd.DataFrame(history).set_index('Frame')[['EAR', 'MOR']])

            if progress is not None and total_frames > 0 and frame_num % 10 == 0:
                progress.progress(min(1.0, frame_num / total_frames))

            # Webcam sessions are time limited so the Streamlit page stays usable
            if input_mode == "Live Webcam Stream" and (time.time() - start_time) >= webcam_seconds:
                break

    except Exception as exc:  # keep the dashboard alive on unexpected input
        st.error(f"Monitoring stopped: {exc}")
    finally:
        cap.release()
        engine.close()
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.remove(temp_input_file)
            except OSError:
                pass

    if progress is not None:
        progress.progress(1.0)

    summary = logger.close()

    st.success(f"Monitoring session completed — {summary['frames']} frames processed.")
    st.subheader("🧾 Session Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Frames", summary['frames'])
    s2.metric("Drowsy events", summary['drowsy_events'])
    s3.metric("Yawn events", summary['yawn_events'])
    s4.metric("Gaze alerts", summary['gaze_distractions'])

    if summary['events']:
        st.dataframe(pd.DataFrame(summary['events']), use_container_width=True)

    download_col1, download_col2 = st.columns(2)
    if logger.csv_path and os.path.exists(logger.csv_path):
        with open(logger.csv_path, "rb") as fh:
            download_col1.download_button(
                "⬇️ Download session CSV",
                fh.read(),
                file_name=os.path.basename(logger.csv_path),
                mime="text/csv"
            )

    report_path = os.path.join(LOGS_DIR, "dashboard_report.html")
    try:
        generate_html_report(summary, report_path)
        with open(report_path, "rb") as fh:
            download_col2.download_button(
                "⬇️ Download HTML report",
                fh.read(),
                file_name="dashboard_report.html",
                mime="text/html"
            )
    except OSError:
        pass


if __name__ == "__main__":
    main()
