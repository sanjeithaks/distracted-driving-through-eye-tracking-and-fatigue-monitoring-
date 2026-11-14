"""
HTML Session Report Generator
------------------------------
Turns a SessionLogger summary dictionary into a standalone, self-contained
HTML report (no external assets) that can be opened in any browser or attached
to the project documentation.

Usage:
    from src.report import generate_html_report
    generate_html_report(summary, "logs/report.html")
"""
import html
import os
from datetime import datetime


def _pct(value, total):
    if not total:
        return 0.0
    return 100.0 * float(value) / float(total)


def _card(label, value, hint=""):
    return f"""
      <div class="card">
        <div class="value">{html.escape(str(value))}</div>
        <div class="label">{html.escape(label)}</div>
        <div class="hint">{html.escape(hint)}</div>
      </div>"""


def _bar(label, count, total, color):
    pct = _pct(count, total)
    return f"""
      <div class="bar-row">
        <div class="bar-label">{html.escape(label)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
        <div class="bar-value">{count} ({pct:.0f}%)</div>
      </div>"""


def generate_html_report(summary, output_path, title="Driver Monitoring Session Report"):
    """Write a standalone HTML report and return its path."""
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    frames = summary.get("frames", 0) or 1
    states = summary.get("state_frames", {})
    gaze = summary.get("gaze_direction_counts", {})
    events = summary.get("events", [])

    cards = "".join([
        _card("Frames Processed", summary.get("frames", 0)),
        _card("Session Duration", f'{summary.get("duration_sec", 0)} s'),
        _card("Face Detection", f'{summary.get("face_detection_rate_pct", 0)} %'),
        _card("Blinks", summary.get("blinks", 0)),
        _card("Yawns", summary.get("yawns", 0)),
        _card("Distractions", summary.get("distractions", 0)),
        _card("Gaze Alerts", summary.get("gaze_distractions", 0)),
        _card("Average EAR", summary.get("avg_ear", 0)),
        _card("Minimum EAR", summary.get("min_ear", 0)),
        _card("Maximum MOR", summary.get("max_mor", 0)),
        _card("Average FPS", summary.get("avg_fps", 0)),
    ])

    state_bars = "".join([
        _bar("Normal driving", states.get("NORMAL", 0), frames, "#22c55e"),
        _bar("Drowsiness", states.get("DROWSY", 0), frames, "#ef4444"),
        _bar("Distraction", states.get("DISTRACTED", 0), frames, "#f97316"),
        _bar("Yawning", states.get("YAWNING", 0), frames, "#a855f7"),
    ])

    gaze_total = sum(gaze.values()) or 1
    gaze_bars = "".join([
        _bar(label, gaze.get(label, 0), gaze_total, color)
        for label, color in [
            ("CENTER", "#22c55e"), ("LEFT", "#f97316"), ("RIGHT", "#f97316"),
            ("DOWN", "#eab308"), ("UP", "#eab308"), ("CLOSED", "#64748b"),
            ("N/A", "#475569"),
        ]
        if gaze.get(label, 0)
    ])

    if events:
        event_rows = "".join(
            f"<tr><td>{html.escape(str(e.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(e.get('frame', '')))}</td>"
            f"<td><span class='tag {html.escape(str(e.get('type', '')).lower())}'>"
            f"{html.escape(str(e.get('type', '')))}</span></td>"
            f"<td>{html.escape(str(e.get('message', '')))}</td>"
            f"<td>{html.escape(str(e.get('reason', '')))}</td></tr>"
            for e in events
        )
    else:
        event_rows = "<tr><td colspan='5'>No alert events recorded.</td></tr>"

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family: "Segoe UI", Arial, sans-serif;
         background:#0b1220; color:#e2e8f0; }}
  header {{ padding:32px 40px; background:linear-gradient(120deg,#0f172a,#1e3a8a);
            border-bottom:1px solid #1e293b; }}
  h1 {{ margin:0 0 6px; font-size:26px; }}
  .sub {{ color:#93c5fd; font-size:14px; }}
  main {{ padding:28px 40px 48px; max-width:1200px; margin:0 auto; }}
  h2 {{ font-size:17px; text-transform:uppercase; letter-spacing:.08em;
        color:#94a3b8; margin:32px 0 14px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
            gap:14px; }}
  .card {{ background:#111c30; border:1px solid #1e293b; border-radius:12px;
           padding:16px 18px; }}
  .card .value {{ font-size:24px; font-weight:700; color:#7dd3fc; }}
  .card .label {{ font-size:13px; margin-top:4px; color:#cbd5e1; }}
  .card .hint {{ font-size:11px; color:#64748b; }}
  .panel {{ background:#111c30; border:1px solid #1e293b; border-radius:12px;
            padding:18px 20px; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; margin:10px 0; }}
  .bar-label {{ width:160px; font-size:13px; color:#cbd5e1; }}
  .bar-track {{ flex:1; height:12px; background:#0b1220; border-radius:999px;
                overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:999px; }}
  .bar-value {{ width:110px; text-align:right; font-size:12px; color:#94a3b8; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #1e293b; }}
  th {{ color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:11px;
        letter-spacing:.06em; }}
  .tag {{ padding:2px 8px; border-radius:999px; font-size:11px; font-weight:700; }}
  .tag.drowsy {{ background:#450a0a; color:#fca5a5; }}
  .tag.distracted {{ background:#431407; color:#fdba74; }}
  .tag.yawning {{ background:#3b0764; color:#e9d5ff; }}
  footer {{ margin-top:36px; color:#64748b; font-size:12px; }}
</style>
</head>
<body>
<header>
  <h1>🚗 {html.escape(title)}</h1>
  <div class="sub">Distracted Driving Through Eye Tracking &amp; Fatigue Monitoring Using Embedded System</div>
  <div class="sub">Session: {html.escape(str(summary.get('session', '')))} &nbsp;•&nbsp; Generated: {generated}</div>
</header>
<main>
  <h2>Session overview</h2>
  <div class="cards">{cards}</div>

  <h2>Driver state distribution</h2>
  <div class="panel">{state_bars}</div>

  <h2>Iris gaze distribution</h2>
  <div class="panel">{gaze_bars if gaze_bars else '<div class="sub">No gaze data recorded.</div>'}</div>

  <h2>Alert events</h2>
  <div class="panel">
    <table>
      <thead><tr><th>Time (s)</th><th>Frame</th><th>Type</th><th>Message</th><th>Reason</th></tr></thead>
      <tbody>{event_rows}</tbody>
    </table>
  </div>

  <footer>
    Generated by the Driver Fatigue &amp; Distraction Monitoring System.
    Metrics: Eye Aspect Ratio (EAR), Mouth Opening Ratio (MOR), 3D head pose (solvePnP),
    and iris-based eye-gaze tracking.
  </footer>
</main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(document)

    return output_path
