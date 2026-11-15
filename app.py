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
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Driver Fatigue & Distraction Monitor",
    page_icon="🚗",
    layout="wide"
)

# Set matplotlib cache dir
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".matplotlib_cache"))

from src.session_logger import SessionLogger
from src.report import generate_html_report
from src.config import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES,
    MOR_THRESHOLD, MOR_CONSEC_FRAMES,
    SERIAL_PORT
)

LOGS_DIR = "logs"


def _sparkline_svg(history, width=640, height=130):
    """Lightweight SVG chart (avoids pandas/pyarrow entirely)."""
    if len(history) < 2:
        return "<div style='color:#64748b;font-size:13px'>Waiting for data…</div>"

    def polyline(key, color):
        pts = []
        n = len(history)
        for i, row in enumerate(history):
            x = 10 + (i / max(1, n - 1)) * (width - 20)
            y = (height - 12) - max(0.0, min(1.0, row[key])) * (height - 24)
            pts.append(f"{x:.1f},{y:.1f}")
        return (f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')

    return f"""
    <div style="background:#111c30;border:1px solid #1e293b;border-radius:10px;padding:10px">
      <div style="font-size:12px;color:#94a3b8;margin-bottom:4px">
        <span style="color:#22c55e">● EAR</span>&nbsp;&nbsp;
        <span style="color:#a855f7">● MOR</span>
      </div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="auto">
        <line x1="10" y1="{height-12}" x2="{width-10}" y2="{height-12}" stroke="#334155"/>
        {polyline('EAR', '#22c55e')}
        {polyline('MOR', '#a855f7')}
      </svg>
    </div>"""


def _draw_framing_box(frame):
    """Draw the camera positioning guide on a preview still."""
    out = frame.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = int(w * 0.22), 95, int(w * 0.78), h - 150
    color = (0, 165, 255)
    dash = 14
    for x in range(x1, x2, dash * 2):
        cv2.line(out, (x, y1), (min(x + dash, x2), y1), color, 2)
        cv2.line(out, (x, y2), (min(x + dash, x2), y2), color, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(out, (x1, y), (x1, min(y + dash, y2)), color, 2)
        cv2.line(out, (x2, y), (x2, min(y + dash, y2)), color, 2)
    cv2.putText(out, "PLACE WHOLE FACE IN BOX", (x1 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, "PLACE WHOLE FACE IN BOX", (x1 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(out, "INCLUDE YOUR MOUTH", (x1 + 10, y2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, "INCLUDE YOUR MOUTH", (x1 + 10, y2 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out


def _events_markdown(events):
    """Render alert events as a plain Markdown table (no pandas needed)."""
    if not events:
        return "_No alert events recorded._"
    lines = [
        "| Time (s) | Frame | Type | Message | Reason | FRI | Snapshot |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in events:
        cells = [
            e.get('timestamp', ''), e.get('frame', ''), e.get('type', ''),
            e.get('message', ''), e.get('reason', ''),
            e.get('fatigue_risk', ''), e.get('snapshot', ''),
        ]
        lines.append("| " + " | ".join(str(c).replace('|', '\\|') for c in cells) + " |")
    return "\n".join(lines)


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
    st.sidebar.subheader("🔌 Embedded Output")
    output_mode = st.sidebar.radio(
        "Output mode",
        options=["PC simulation (no hardware)", "Arduino via serial"],
        index=0,
        help="Simulation shows the virtual vibration motor, buzzer and LEDs on screen."
    )
    enable_serial = output_mode.startswith("Arduino")
    serial_port = SERIAL_PORT
    if enable_serial:
        serial_port = st.sidebar.text_input("COM Port", value=SERIAL_PORT)
        st.sidebar.caption("Connect an Arduino to enable physical output.")
    else:
        st.sidebar.caption("No hardware needed — all actuators are simulated on screen.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎥 Video Input")
    input_mode = st.sidebar.radio(
        "Input mode",
        options=["Run Built-in Demo Clip", "Upload Driving Video File", "Live Webcam Stream"],
        index=2,
        help="Live Webcam monitors your actual camera in real time."
    )

    uploaded_file = None
    if input_mode == "Upload Driving Video File":
        uploaded_file = st.sidebar.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])

    webcam_seconds = 120
    if input_mode == "Live Webcam Stream":
        webcam_seconds = st.sidebar.slider(
            "Live monitoring duration (seconds)", min_value=30, max_value=600,
            value=120, step=30
        )
        st.sidebar.info(
            "**Live monitoring**\n\n"
            "Keep your **whole face including your mouth** visible in the "
            "camera. Try closing your eyes, yawning, or turning your head."
        )
        if st.sidebar.button("📷 Test camera framing", width="stretch"):
            test_cap = cv2.VideoCapture(0)
            frame_ok = False
            for _ in range(12):          # allow auto-exposure to settle
                frame_ok, preview_frame = test_cap.read()
            test_cap.release()
            if frame_ok and preview_frame is not None:
                st.session_state["camera_preview"] = preview_frame
            else:
                st.sidebar.error("Could not open the webcam. Close other camera apps and retry.")
    else:
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

        f1, f2, f3 = st.columns(3)
        metric_perclos = f1.metric("PERCLOS", "0.0%")
        metric_risk = f2.metric("Fatigue Risk", "0/100")
        metric_microsleeps = f3.metric("Microsleeps", "0")
        risk_bar = st.progress(0)
        risk_caption = st.caption("Risk level: LOW")

        status_box = st.empty()
        status_box.success("STATUS: SYSTEM READY")

        with st.container(border=True):
            st.markdown("**Simulated embedded output**")
            st.caption("PC simulation — no Arduino or Raspberry Pi required")
            e1, e2, e3 = st.columns(3)
            metric_actuator = e1.metric("Actuator state", "NORMAL")
            metric_motor = e2.metric("Seatbelt motor", "Idle")
            metric_buzzer = e3.metric("Buzzer", "OFF")
            led_line = st.caption("LEDs: 🟢 green (normal)")

        st.subheader("📈 Real-time Trend")
        chart_holder = st.empty()

    with col_video:
        st.subheader("📹 Video Stream & Facial Feature Tracking")
        video_placeholder = st.empty()
        preview = st.session_state.get("camera_preview")
        if input_mode == "Live Webcam Stream" and preview is not None:
            preview_rgb = cv2.cvtColor(_draw_framing_box(preview), cv2.COLOR_BGR2RGB)
            video_placeholder.image(preview_rgb, channels="RGB", width="stretch")
            st.caption("Move back until your whole face — **including your mouth** — "
                       "is inside the dashed box, then press Start Monitoring.")
        else:
            video_placeholder.info("Choose an input mode and press Start Monitoring.")
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

    engine = None
    with st.spinner("Loading eye-tracking and fatigue models..."):
        from src.detector_engine import DriverMonitoringEngine
        engine = DriverMonitoringEngine(
            ear_thresh=ear_thresh,
            ear_consec_frames=ear_consec,
            mor_thresh=mor_thresh,
            enable_serial=enable_serial,
            enable_audio=False,
            serial_port=serial_port
        )
    st.caption(f"Face landmark backend: `{engine.detector.backend}`")
    logger = SessionLogger(output_dir=LOGS_DIR, session_name="dashboard",
                           snapshot_dir=os.path.join(LOGS_DIR, "snapshots"))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if source == 0:
            st.error(
                "Could not open the webcam. Close other apps using the camera "
                "(Teams/Zoom/Camera), then allow camera access in "
                "Windows Settings → Privacy & security → Camera."
            )
        else:
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
            logger.log_frame(telemetry, frame_num, frame=annotated_frame)

            # Update the video preview at ~15 FPS to keep the UI responsive
            now = time.time()
            if now - last_display > 1.0 / 15.0:
                last_display = now
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                metric_ear.metric("EAR", f"{telemetry['ear']:.2f}")
                metric_mor.metric("MOR", f"{telemetry['mor']:.2f}")
                metric_gaze.metric("Gaze", telemetry['gaze_label'])
                metric_yaw.metric("Yaw", f"{telemetry['yaw']:+.1f}°")
                metric_pitch.metric("Pitch", f"{telemetry['pitch']:+.1f}°")
                metric_blinks.metric("Blinks", str(telemetry['blinks']))
                metric_yawns.metric("Yawns", str(telemetry['yawns']))
                metric_distractions.metric("Distractions", str(telemetry['distractions']))
                metric_perclos.metric("PERCLOS", f"{telemetry['perclos_pct']:.1f}%")
                metric_risk.metric("Fatigue Risk", f"{telemetry['fatigue_risk']:.0f}/100")
                metric_microsleeps.metric("Microsleeps", str(telemetry['microsleeps']))
                risk_bar.progress(min(100, int(round(telemetry['fatigue_risk']))))
                risk_caption.caption(
                    f"Risk level: {telemetry['risk_level']} • "
                    f"blink rate {telemetry['blink_rate']:.1f}/min • "
                    f"yawn rate {telemetry['yawn_rate']:.1f}/min"
                )

                # Virtual embedded actuators (PC simulation mode)
                risk = telemetry['fatigue_risk']
                vibration_hz = 100 + 2 * int(round(risk))
                if telemetry['is_drowsy']:
                    act_state, buzzer, led = "DROWSY", "ON", "🔴 red"
                    motor = f"{vibration_hz} Hz"
                elif telemetry['is_distracted']:
                    act_state, buzzer, led = "DISTRACTION", "PULSE", "🔵 blue"
                    motor = f"{vibration_hz} Hz"
                elif telemetry['is_yawning']:
                    act_state, buzzer, led = "YAWNING", "CHIME", "🟡 yellow"
                    motor = "Idle"
                else:
                    act_state, buzzer, led = "NORMAL", "OFF", "🟢 green"
                    motor = "Idle"
                metric_actuator.metric("Actuator state", act_state)
                metric_motor.metric("Seatbelt motor", motor)
                metric_buzzer.metric("Buzzer", buzzer)
                led_line.caption(f"LEDs: {led} • simulated output")

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
                chart_holder.html(_sparkline_svg(history))

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
    if input_mode == "Live Webcam Stream":
        st.info("Live monitoring stopped after the selected duration. "
                "Press **Start Monitoring** again to continue watching the camera.")
    st.subheader("🧾 Session Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Frames", summary['frames'])
    s2.metric("Drowsy events", summary['drowsy_events'])
    s3.metric("Yawn events", summary['yawn_events'])
    s4.metric("Gaze alerts", summary['gaze_distractions'])

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Mean PERCLOS", f"{summary['avg_perclos']}%")
    q2.metric("Peak Fatigue Risk", f"{summary['peak_fatigue_risk']}/100")
    q3.metric("Microsleeps", summary['microsleeps'])
    q4.metric("Final Risk Level", summary['final_risk_level'])

    if summary['events']:
        st.markdown(_events_markdown(summary['events']))

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
