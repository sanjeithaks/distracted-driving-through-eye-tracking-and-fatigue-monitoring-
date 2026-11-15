"""
HTML Session Report Generator
------------------------------
Turns a SessionLogger summary dictionary into a standalone, self-contained
HTML report (no external assets) that can be opened in any browser or attached
to the project documentation.

The report includes:
- Session overview cards (frames, face detection, EAR/MOR, PERCLOS, risk)
- Driver-state distribution bars
- Iris gaze distribution
- PERCLOS / Fatigue Risk Index timeline (inline SVG)
- Explainable fatigue risk factor breakdown
- Alert-event table with evidence snapshot filenames

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


def _risk_factor_bar(label, value, max_value, color):
    pct = min(100.0, _pct(value, max_value))
    return f"""
      <div class="bar-row">
        <div class="bar-label">{html.escape(label)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
        <div class="bar-value">{value:.1f} pts</div>
      </div>"""


def _risk_chart(series, width=1000, height=230):
    """Build an inline SVG line chart of the Fatigue Risk Index over time."""
    if not series:
        return '<div class="sub">No fatigue-risk timeline recorded.</div>'

    pad_left, pad_right, pad_top, pad_bottom = 46, 16, 16, 34
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    max_frame = max(1, int(series[-1].get("frame", 1)))

    def xy(frame, risk):
        x = pad_left + (float(frame) / max_frame) * plot_w
        y = pad_top + (1.0 - max(0.0, min(100.0, float(risk))) / 100.0) * plot_h
        return x, y

    points = []
    for item in series:
        x, y = xy(item.get("frame", 0), item.get("risk", 0.0))
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    first_x, _ = xy(series[0].get("frame", 0), 0)
    last_x, _ = xy(series[-1].get("frame", 0), 0)
    area = f"{first_x:.1f},{pad_top + plot_h:.1f} {polyline} {last_x:.1f},{pad_top + plot_h:.1f}"

    grid = []
    for level, color in [(0, "#334155"), (25, "#1e4d3a"), (50, "#5a4815"),
                         (75, "#5c2b16"), (100, "#4c1d1d")]:
        _, y = xy(0, level)
        grid.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="4 4" />'
            f'<text x="8" y="{y + 4:.1f}" fill="#64748b" font-size="11">{level}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img"
         aria-label="Fatigue risk timeline">
      <defs>
        <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#f97316" stop-opacity="0.45"/>
          <stop offset="100%" stop-color="#f97316" stop-opacity="0.02"/>
        </linearGradient>
      </defs>
      {''.join(grid)}
      <polygon points="{area}" fill="url(#riskFill)" />
      <polyline points="{polyline}" fill="none" stroke="#fb923c" stroke-width="2.5"
                stroke-linejoin="round" stroke-linecap="round" />
      <text x="{pad_left}" y="{height - 8}" fill="#64748b" font-size="11">frame 0</text>
      <text x="{width - pad_right - 80}" y="{height - 8}" fill="#64748b" font-size="11">frame {max_frame}</text>
      <text x="{pad_left + 6}" y="{pad_top + 14}" fill="#94a3b8" font-size="12">Fatigue Risk Index (0-100)</text>
    </svg>"""


def generate_html_report(summary, output_path, title="Driver Monitoring Session Report"):
    """Write a standalone HTML report and return its path."""
    directory = os.path.dirname(os.path.abspath(output_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    frames = summary.get("frames", 0) or 1
    states = summary.get("state_frames", {})
    gaze = summary.get("gaze_direction_counts", {})
    events = summary.get("events", [])
    risk_series = summary.get("risk_series", [])
    factors = summary.get("final_risk_factors", {}) or {}

    risk_level = summary.get("final_risk_level", "LOW")
    risk_color = {"LOW": "#22c55e", "MODERATE": "#eab308",
                  "HIGH": "#f97316", "CRITICAL": "#ef4444"}.get(risk_level, "#94a3b8")

    cards = "".join([
        _card("Frames Processed", summary.get("frames", 0)),
        _card("Session Duration", f'{summary.get("duration_sec", 0)} s'),
        _card("Face Detection", f'{summary.get("face_detection_rate_pct", 0)} %'),
        _card("Blinks", summary.get("blinks", 0)),
        _card("Yawns", summary.get("yawns", 0)),
        _card("Distractions", summary.get("distractions", 0)),
        _card("Gaze Alerts", summary.get("gaze_distractions", 0)),
        _card("Microsleeps", summary.get("microsleeps", 0), "closures ≥ 0.5 s"),
        _card("Average EAR", summary.get("avg_ear", 0)),
        _card("Minimum EAR", summary.get("min_ear", 0)),
        _card("Maximum MOR", summary.get("max_mor", 0)),
        _card("Average FPS", summary.get("avg_fps", 0)),
        _card("Mean PERCLOS", f'{summary.get("avg_perclos", 0)} %'),
        _card("Peak PERCLOS", f'{summary.get("max_perclos", 0)} %'),
        _card("Peak Fatigue Risk", f'{summary.get("peak_fatigue_risk", 0)}/100'),
        _card("Final Risk Level", risk_level, f'FRI {summary.get("final_fatigue_risk", 0)}/100'),
        _card("Blink Rate", f'{summary.get("blink_rate", 0)} /min'),
        _card("Yawn Rate", f'{summary.get("yawn_rate", 0)} /min'),
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

    factor_panels = "".join([
        _risk_factor_bar("PERCLOS", float(factors.get("perclos", 0.0)), 50.0, "#38bdf8"),
        _risk_factor_bar("Microsleeps", float(factors.get("microsleep", 0.0)), 20.0, "#ef4444"),
        _risk_factor_bar("Yawning", float(factors.get("yawning", 0.0)), 20.0, "#a855f7"),
        _risk_factor_bar("Longest closure", float(factors.get("closure", 0.0)), 10.0, "#f59e0b"),
    ])

    has_snapshot = any(e.get("snapshot") for e in events)
    if events:
        header = ("<tr><th>Time (s)</th><th>Frame</th><th>Type</th><th>Message</th>"
                  "<th>Reason</th><th>FRI</th>" + ("<th>Snapshot</th>" if has_snapshot else "") + "</tr>")
        rows = []
        for e in events:
            cells = [
                html.escape(str(e.get('timestamp', ''))),
                html.escape(str(e.get('frame', ''))),
                f"<span class='tag {html.escape(str(e.get('type', '')).lower())}'>"
                f"{html.escape(str(e.get('type', '')))}</span>",
                html.escape(str(e.get('message', ''))),
                html.escape(str(e.get('reason', ''))),
                html.escape(str(e.get('fatigue_risk', ''))),
            ]
            if has_snapshot:
                cells.append(html.escape(str(e.get('snapshot', ''))))
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        event_rows = "".join(rows)
    else:
        header = "<tr><th>Time (s)</th><th>Frame</th><th>Type</th><th>Message</th><th>Reason</th><th>FRI</th></tr>"
        event_rows = "<tr><td colspan='6'>No alert events recorded.</td></tr>"

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
  .risk-banner {{ display:flex; align-items:center; justify-content:space-between;
                  gap:16px; background:#111c30; border:1px solid #1e293b;
                  border-left:6px solid {risk_color}; border-radius:12px;
                  padding:18px 22px; }}
  .risk-banner .big {{ font-size:34px; font-weight:800; color:{risk_color}; }}
  .risk-banner .level {{ font-size:20px; font-weight:700; color:{risk_color}; }}
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

  <h2>Fatigue assessment</h2>
  <div class="risk-banner">
    <div>
      <div class="big">{summary.get('final_fatigue_risk', 0)}<span style="font-size:16px;color:#64748b">/100</span></div>
      <div class="sub">Final Fatigue Risk Index (FRI)</div>
    </div>
    <div class="level">{risk_level}</div>
    <div class="sub">PERCLOS {summary.get('max_perclos', 0)}% peak &nbsp;•&nbsp;
      {summary.get('microsleeps', 0)} microsleep(s) &nbsp;•&nbsp;
      longest closure {summary.get('longest_closure_sec', 0)} s</div>
  </div>

  <h2>Fatigue risk timeline</h2>
  <div class="panel">{_risk_chart(risk_series)}</div>

  <h2>Risk factor breakdown (explainable)</h2>
  <div class="panel">{factor_panels}</div>

  <h2>Driver state distribution</h2>
  <div class="panel">{state_bars}</div>

  <h2>Iris gaze distribution</h2>
  <div class="panel">{gaze_bars if gaze_bars else '<div class="sub">No gaze data recorded.</div>'}</div>

  <h2>Alert events</h2>
  <div class="panel">
    <table>
      <thead>{header}</thead>
      <tbody>{event_rows}</tbody>
    </table>
  </div>

  <footer>
    Generated by the Driver Fatigue &amp; Distraction Monitoring System.
    Metrics: Eye Aspect Ratio (EAR), Mouth Opening Ratio (MOR), 3D head pose (solvePnP),
    iris-based eye-gaze tracking, PERCLOS (Wierwille et al., 1994; FHWA/NHTSA),
    microsleep detection and a weighted Fatigue Risk Index.
  </footer>
</main>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(document)

    return output_path
