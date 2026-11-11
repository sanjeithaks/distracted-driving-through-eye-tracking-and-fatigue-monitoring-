"""
Streamlit Web Dashboard for Distracted Driving & Fatigue Monitoring System
Provides an interactive web interface with:
- Video file upload or webcam stream
- Live telemetry HUD (EAR, MOR, Head Pose, Alert state)
- Interactive threshold tuning
- Real-time time-series plots for EAR and MOR
- Downloadable processed video report
"""
import os
import tempfile
import cv2
import numpy as np
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
from src.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    MOR_THRESHOLD, MOR_CONSEC_FRAMES,
    SERIAL_PORT
)


def main():
    st.title("🚗 Distracted Driving Through Eye Tracking & Fatigue Monitoring")
    st.caption("Implementation of IEEE Research Paper (CIT Bengaluru, ECE Dept)")

    # Sidebar controls
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
        min_value=0.40, max_value=0.80, value=MOR_THRESHOLD, step=0.02,
        help="Values above this threshold indicate mouth open / yawning."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 Embedded System Interface")
    enable_serial = st.sidebar.checkbox("Enable Serial Hardware Bridge", value=False)
    serial_port = st.sidebar.text_input("COM Port", value=SERIAL_PORT)

    st.sidebar.markdown("---")
    input_mode = st.sidebar.radio(
        "Video Input Mode",
        options=["Run Built-in Demo Clip", "Upload Driving Video File", "Live Webcam Stream"]
    )

    # Main dashboard columns
    col1, col2 = st.columns([3, 2])

    with col2:
        st.subheader("📊 Live Telemetry HUD")
        metric_col1, metric_col2 = st.columns(2)
        metric_ear = metric_col1.metric("EAR (Eye Aspect)", "0.00")
        metric_mor = metric_col2.metric("MOR (Mouth Opening)", "0.00")

        pose_col1, pose_col2 = st.columns(2)
        metric_yaw = pose_col1.metric("Yaw (Heading)", "0.0 deg")
        metric_pitch = pose_col2.metric("Pitch (Tilt)", "0.0 deg")

        count_col1, count_col2, count_col3 = st.columns(3)
        metric_blinks = count_col1.metric("Blinks", "0")
        metric_yawns = count_col2.metric("Yawns", "0")
        metric_distractions = count_col3.metric("Distractions", "0")

        status_box = st.empty()
        status_box.success("STATUS: SYSTEM READY")

        st.subheader("📈 Real-time Trend")
        chart_holder = st.empty()

    with col1:
        st.subheader("📹 Video Stream & Facial Feature Tracking")
        video_placeholder = st.empty()
        start_btn = st.button("🚀 Start Monitoring", key="start_monitoring")

    if start_btn:
        # Determine source
        temp_input_file = None
        if input_mode == "Run Built-in Demo Clip":
            if not os.path.exists("demo_sample_driving.mp4"):
                from generate_test_media import generate_scenario_video
                generate_scenario_video("demo_sample_driving.mp4", total_frames=420)
            source = "demo_sample_driving.mp4"

        elif input_mode == "Upload Driving Video File":
            uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI Video", type=["mp4", "avi", "mov"])
            if uploaded_file is None:
                st.warning("Please upload a video file in the sidebar.")
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

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error(f"Could not open video source: {source}")
            engine.close()
            return

        history_records = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_num += 1
            annotated_frame, telemetry = engine.process_frame(frame)

            # Update video display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update metrics
            metric_ear.metric("EAR", f"{telemetry['ear']:.2f}")
            metric_mor.metric("MOR", f"{telemetry['mor']:.2f}")
            metric_yaw.metric("Yaw", f"{telemetry['yaw']:+.1f}°")
            metric_pitch.metric("Pitch", f"{telemetry['pitch']:+.1f}°")
            metric_blinks.metric("Blinks", str(telemetry['blinks']))
            metric_yawns.metric("Yawns", str(telemetry['yawns']))
            metric_distractions.metric("Distractions", str(telemetry['distractions']))

            # Update status box
            if telemetry['is_drowsy']:
                status_box.error("🚨 DROWSINESS ALERT! SEATBELT VIBRATION TRIGGERED")
            elif telemetry['is_distracted']:
                status_box.warning(f"⚠️ LOOKING AWAY! ({telemetry['distraction_reason']})")
            elif telemetry['is_yawning']:
                status_box.info("🥱 YAWNING DETECTED! FATIGUE WARNING")
            elif telemetry['face_detected']:
                status_box.success("✅ DRIVER ATTENTIVE - FACE DETECTED")
            else:
                status_box.warning("👁️ NO FACE DETECTED")

            # Append to history and update plot
            history_records.append({
                'Frame': frame_num,
                'EAR': telemetry['ear'],
                'MOR': telemetry['mor']
            })
            if len(history_records) > 100:
                history_records.pop(0)

            if frame_num % 5 == 0:
                df = pd.DataFrame(history_records).set_index('Frame')
                chart_holder.line_chart(df[['EAR', 'MOR']])

        cap.release()
        engine.close()
        if temp_input_file and os.path.exists(temp_input_file):
            try:
                os.remove(temp_input_file)
            except Exception:
                pass
        st.success("Monitoring session completed!")


if __name__ == "__main__":
    main()
