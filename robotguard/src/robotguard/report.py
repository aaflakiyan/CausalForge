"""Stand-alone HTML report; no browser dependencies or uploaded data."""
from __future__ import annotations

import html
from pathlib import Path

from .core import Result, RobotSpec


COLORS = ("#69d2e7", "#b8a1ff", "#ffbe75", "#78dfb1", "#f6a4ca", "#9aafef")


def render_html(spec: RobotSpec, result: Result) -> str:
    points = result.traces
    n = len(spec.joint_names)
    width, height = 920, 290
    x0, x1, y0, y1 = 66, 886, 30, 253
    times = [p["t"] for p in points]
    tx_min = min(times, default=0.0)
    tx_max = max(times, default=1.0)
    if tx_min == tx_max:
        tx_max += 1
    low = min((bounds[0] for bounds in spec.position_limits), default=-1)
    high = max((bounds[1] for bounds in spec.position_limits), default=1)
    if low == high:
        high += 1
    padding = 0.08 * (high - low)
    low -= padding
    high += padding

    def sx(t: float) -> float:
        return x0 + (t - tx_min) / (tx_max - tx_min) * (x1 - x0)

    def sy(v: float) -> float:
        return y1 - (v - low) / (high - low) * (y1 - y0)

    curves = []
    for j in range(n):
        colour = COLORS[j % len(COLORS)]
        points_text = " ".join(f"{sx(p['t']):.1f},{sy(p['position'][j]):.1f}" for p in points)
        curves.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round" points="{points_text}"/>')
        curves.append(f'<circle cx="{sx(points[-1]["t"]):.1f}" cy="{sy(points[-1]["position"][j]):.1f}" r="4" fill="{colour}"/>' if points else "")
    bad = {v.sample for v in result.violations}
    for p in points:
        if p["sample"] in bad:
            curves.append(f'<line x1="{sx(p["t"]):.1f}" x2="{sx(p["t"]):.1f}" y1="{y0}" y2="{y1}" stroke="#fb7185" stroke-width="1.4" stroke-dasharray="4 5" opacity=".75"/>')
    ticks = []
    for idx in range(5):
        t = tx_min + (tx_max - tx_min) * idx / 4
        x = sx(t)
        ticks.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}" stroke="#29364b"/><text x="{x:.1f}" y="275" text-anchor="middle" fill="#90a1ba" font-size="12">{t:.2f}s</text>')
    for idx in range(5):
        val = low + (high - low) * idx / 4
        y = sy(val)
        ticks.append(f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="#29364b"/><text x="56" y="{y + 4:.1f}" text-anchor="end" fill="#90a1ba" font-size="12">{val:.1f}</text>')
    legend = "".join(f'<span><i style="background:{COLORS[j % len(COLORS)]}"></i>{html.escape(name)}</span>' for j, name in enumerate(spec.joint_names))
    if result.violations:
        rows = "".join(f'<tr><td>{v.sample}</td><td><code>{html.escape(v.code)}</code></td><td>{html.escape(v.message)}</td></tr>' for v in result.violations)
    else:
        rows = '<tr><td colspan="3">All configured kinematic checks passed.</td></tr>'
    status = "PASS" if result.ok else "FAIL"
    tone = "good" if result.ok else "bad"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RobotGuard · {status}</title>
<style>
:root{{color-scheme:dark;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif}}*{{box-sizing:border-box}}
body{{background:#0b1120;color:#eef4ff;max-width:1050px;margin:auto;padding:44px 22px;line-height:1.5}}
header{{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap}}
.eyebrow{{color:#7dd3fc;letter-spacing:.16em;font-weight:750;font-size:12px}}h1{{font-size:40px;letter-spacing:-.05em;margin:4px 0}}
p{{color:#a6b6cf}}h2{{margin:0 0 12px;font-size:19px}}.pill{{padding:10px 18px;border-radius:999px;font-weight:850;letter-spacing:.08em}}
.good{{background:#143b32;color:#7df4c0}}.bad{{background:#492431;color:#ff9fb0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:24px 0}}.metric,.panel{{border:1px solid #26344c;border-radius:15px;background:#131d30;padding:20px}}
.metric strong{{display:block;font-size:27px;letter-spacing:-.05em}}.metric small{{color:#9faec7}}.panel{{margin:20px 0;overflow:auto}}
svg{{width:100%;height:auto;min-width:480px;display:block}}.legend{{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 0}}.legend span{{display:flex;align-items:center;gap:7px;color:#b6c7df;font-size:13px}}
.legend i{{width:10px;height:10px;border-radius:50%;display:inline-block}}table{{width:100%;border-collapse:collapse}}td,th{{padding:11px;text-align:left;vertical-align:top;border-top:1px solid #28354d;font-size:13px}}
th{{color:#9cacc7;font-weight:500}}code{{color:#8dd6ff}}footer{{color:#8497b3;font-size:12px;margin-top:28px}}@media(max-width:600px){{.grid{{grid-template-columns:1fr}}h1{{font-size:32px}}}}
</style></head><body>
<header><div><div class="eyebrow">ROBOTGUARD / TRACE AUDIT</div><h1>{html.escape(spec.name)}</h1><p>Deterministic joint-position pre-flight report</p></div><span class="pill {tone}">{status}</span></header>
<div class="grid"><div class="metric"><small>Actions inspected</small><strong>{result.samples}</strong></div><div class="metric"><small>Violations</small><strong>{len(result.violations)}</strong></div><div class="metric"><small>Peak joint velocity</small><strong>{result.peak_velocity:.2f} <small>{html.escape(spec.units)}/s</small></strong></div></div>
<section class="panel"><h2>Joint trajectories</h2><svg viewBox="0 0 {width} {height}" role="img" aria-label="Recorded joint position versus time, with red dashed lines at flagged samples">{''.join(ticks)}{''.join(curves)}</svg><div class="legend">{legend}</div></section>
<section class="panel"><h2>Findings</h2><table><thead><tr><th>Sample</th><th>Code</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table></section>
<footer>Offline kinematic checks only. Passing this report does not establish collision clearance, actuator safety, or real-world deployability. No remote telemetry.</footer>
</body></html>'''


def write_report(path: str | Path, spec: RobotSpec, result: Result) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(spec, result), encoding="utf-8")
