"""Offline browser replay of measured MuJoCo positions.

No models are called from the HTML page. The arm image is reconstructed from
world-frame joint anchors and tool-tip positions *recorded by MuJoCo*; the
viewer deliberately refuses to invent geometry for unsupported scenes.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Mapping


class PlaygroundError(ValueError):
    """No trustworthy replay geometry was recorded for this run."""


def render_playground_html(sim: Mapping[str, Any]) -> str:
    rows = sim.get("records", [])
    if not rows or not all(
        isinstance(r.get("link_points_xy_m"), list)
        and len(r["link_points_xy_m"]) == 3
        and all(len(point) == 2 for point in r["link_points_xy_m"])
        for r in rows
    ):
        raise PlaygroundError(
            "No recorded planar joint-anchor and tool-tip geometry: "
            "the browser replay only supports explicitly measured scenes"
        )
    # JSON is embedded as Base64 so model names cannot inject HTML or script.
    payload = {
        "records": rows,
        "joint_names": list(sim["joint_names"]),
        "status": sim["status"],
        "robot_spec": sim["robot_spec"],
        "mujoco_version": str(sim["mujoco_version"]),
        "metrics": dict(sim["metrics"]),
        "model_sha256": sim["model_sha256"],
        "trace_sha256": sim["trace_sha256"],
        "simulated_duration_s": sim["simulated_duration_s"],
        "integration_steps": sim["integration_steps"],
    }
    packed = base64.b64encode(json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("ascii")).decode("ascii")
    arm = html.escape(str(sim["robot_spec"]))
    steps = int(sim["integration_steps"])
    rmse = float(sim["metrics"]["rmse_tracking_error_rad"])
    max_err = float(sim["metrics"]["max_tracking_error_rad"])
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobotGuard Playground · measured MuJoCo replay</title>
<style>
:root{color-scheme:dark;--bg:#0a1020;--pane:#111b2d;--edge:#2b3a53;--muted:#a5b4ca;--txt:#f2f6ff;--lime:#7bdcb5;--blue:#6dbaff}
*{box-sizing:border-box}body{font:15px/1.5 system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--txt);margin:0}
main{max-width:1150px;margin:auto;padding:34px 22px 60px}header{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}
h1{font-size:clamp(28px,4vw,43px);letter-spacing:-.045em;margin:6px 0}p{color:var(--muted)}
.tag{font-size:12px;border:1px solid #316758;color:var(--lime);border-radius:99px;padding:7px 12px}
.panel{border:1px solid var(--edge);background:var(--pane);border-radius:18px;padding:21px;min-width:0}
.layout{display:grid;grid-template-columns:1.5fr 1fr;gap:18px;margin:20px 0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.stat{padding:17px;background:#0b1526;border-radius:12px}.stat strong{font-size:24px;display:block}.stat span{font-size:12px;color:var(--muted)}
svg{width:100%;height:auto;background:radial-gradient(circle at center,#14243d,#0b1527);border:1px solid #2b3a53;border-radius:14px}
.controls{display:flex;gap:13px;align-items:center;margin-top:15px;flex-wrap:wrap}button{background:#336dec;border:0;color:white;border-radius:9px;padding:10px 18px;cursor:pointer;font-weight:650}
button:focus-visible,input:focus-visible{outline:3px solid #e8d388;outline-offset:3px}input[type=range]{flex:1;min-width:125px;accent-color:var(--lime)}
.mono{font-family:ui-monospace,SFMono-Regular,monospace;color:#d9e4f5}.key{display:flex;justify-content:space-between;gap:14px;border-bottom:1px solid #2d3b52;padding:10px 0}
.key:last-child{border:none}.key span{color:var(--muted)}nav a{display:inline-block;color:#91c8ff;margin-right:18px;margin-top:6px}
small{color:var(--muted)}.provenance{overflow-wrap:anywhere;font-size:12px}.notes{margin-top:20px;border-left:3px solid #dca66b;padding-left:15px}
@media(max-width:790px){.layout{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main>
<header><div><small>ROBOTGUARD / OPEN PLAYGROUND</small><h1>Watch what the robot actually did.</h1>
<p>Replay captured MuJoCo observations, frame by frame. No generated motion or API calls.</p></div>
<span class="tag">MEASURED PHYSICS REPLAY</span></header>
<div class="stats"><div class="stat"><strong id="stat-status">Measured</strong><span>Simulation result</span></div>
<div class="stat"><strong>__RMSE__ rad</strong><span>Joint-position tracking RMSE</span></div>
<div class="stat"><strong>__STEPS__</strong><span>Actual physics integration steps</span></div></div>
<div class="layout"><section class="panel"><h2>Robot motion</h2><p>A top-down XY projection from MuJoCo joint anchors and tool-tip world coordinates. Not camera footage or a 3D viewer.</p>
<svg id="scene" viewBox="0 0 700 530" role="img" aria-label="Top-down playback of observed robot arm positions">
<defs><pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" stroke="#24334a" stroke-width="1" fill="none"/></pattern></defs>
<rect width="700" height="530" fill="url(#grid)"/>
<path d="M 0 265 H700 M350 0V530" stroke="#45546e" stroke-width="1" stroke-dasharray="5 8"/>
<path id="trail" fill="none" stroke="#3b82f6" opacity=".32" stroke-width="2"/>
<path id="segment-a" fill="none" stroke="#5eb8f3" stroke-width="21" stroke-linecap="round"/>
<path id="segment-b" fill="none" stroke="#c4a7ff" stroke-width="17" stroke-linecap="round"/>
<g id="joints" fill="#e5edfd" stroke="#162b40" stroke-width="4"><circle id="j0" r="10"/><circle id="j1" r="9"/><circle id="tip" r="8" fill="#efbf73"/></g>
<text x="18" y="31" fill="#a5b4ca" font-size="14">Measured world XY (metres)</text>
</svg>
<div class="controls"><button id="play" aria-label="Play recorded simulation">▶ Play</button><label for="timeline">Time</label><input id="timeline" type="range" min="0" max="0" step="1" value="0"/><strong class="mono" id="clock">0.000 s</strong></div>
<p><small>Playback is driven by recorded samples only. The robot is a synthetic 2-joint test model, not a manufacturer-validated industrial arm.</small></p>
</section><aside class="panel"><h2>Evidence inspector</h2><div class="key"><span>Model</span><b>__ARM__</b></div>
<div class="key"><span>Physics engine</span><b id="version"></b></div><div class="key"><span>Sample</span><b id="counter"></b></div>
<div class="key"><span>Requested q (rad)</span><b class="mono" id="target"></b></div>
<div class="key"><span>Observed q (rad)</span><b class="mono" id="actual"></b></div>
<div class="key"><span>Error at sample (rad)</span><b class="mono" id="error"></b></div>
<div class="key"><span>Run maximum error</span><b>__MAX_ERROR__ rad</b></div>
<h3>Audit files</h3><nav><a href="simulation.html">Target vs actual chart</a><a href="rejected.html">Rejected trace</a><a href="simulation.json">Measured JSON</a></nav>
<h3>Reproducibility</h3><div class="provenance"><small>MJCF SHA-256</small><div class="mono" id="model-hash"></div>
<small>Action trace SHA-256</small><div class="mono" id="trace-hash"></div></div></aside></div>
<div class="notes"><strong>Scope, not a safety certificate.</strong><p>Passing static checks does not prove correct units, legitimate robot limits, collision clearance, force safety or real-hardware suitability. The model starts at the first setpoint by directly setting its state. All measured replay points come from MuJoCo; this page cannot infer unrecorded scenes or simulate a new command.</p></div>
</main><script>
"use strict";
const result=JSON.parse(atob("__DATA__"));
const records=result.records;
const byId=id=>document.getElementById(id);
const allPoints=records.flatMap(r=>r.link_points_xy_m);
const xs=allPoints.map(p=>p[0]),ys=allPoints.map(p=>p[1]);
const xMin=Math.min(...xs),xMax=Math.max(...xs),yMin=Math.min(...ys),yMax=Math.max(...ys);
const half=Math.max(xMax-xMin,yMax-yMin,0.3)/2*1.18;
const cx=(xMin+xMax)/2,cy=(yMin+yMax)/2;
const toScreen=p=>[350+240*(p[0]-cx)/half,265-240*(p[1]-cy)/half];
function fmt(v){return Number(v).toFixed(3)}
let index=0,playing=false,lastFrame=0,accum=0;
function draw(next){index=Math.max(0,Math.min(records.length-1,next));const r=records[index];
 const pts=r.link_points_xy_m.map(toScreen);const d=(a,b)=>`M ${a[0].toFixed(2)} ${a[1].toFixed(2)} L ${b[0].toFixed(2)} ${b[1].toFixed(2)}`;
 byId("segment-a").setAttribute("d",d(pts[0],pts[1]));byId("segment-b").setAttribute("d",d(pts[1],pts[2]));
 ["j0","j1","tip"].forEach((id,j)=>{byId(id).setAttribute("cx",pts[j][0]);byId(id).setAttribute("cy",pts[j][1])});
 byId("trail").setAttribute("d",records.slice(0,index+1).map((row,i)=>{const p=toScreen(row.link_points_xy_m[2]);return `${i?'L':'M'} ${p[0].toFixed(2)} ${p[1].toFixed(2)}`}).join(" "));
 byId("timeline").value=index;byId("clock").textContent=fmt(r.t)+" s";byId("counter").textContent=`${index+1} / ${records.length}`;
 byId("target").textContent=r.target.map(fmt).join(", ");byId("actual").textContent=r.actual.map(fmt).join(", ");
 byId("error").textContent=r.target.map((a,j)=>fmt(Math.abs(a-r.actual[j]))).join(", ");
}
function setPlay(p){playing=p;byId("play").textContent=p?"❚❚ Pause":"▶ Play";lastFrame=0;accum=0}
function tick(now){if(playing){if(!lastFrame)lastFrame=now;accum+=(now-lastFrame)/1000;lastFrame=now;
 while(index<records.length-1&&accum>=records[index+1].t-records[index].t){accum-=records[index+1].t-records[index].t;draw(index+1)}
 if(index>=records.length-1)setPlay(false)}requestAnimationFrame(tick)}
byId("timeline").max=records.length-1;byId("timeline").addEventListener("input",e=>{setPlay(false);draw(Number(e.target.value))});
byId("play").addEventListener("click",()=>{if(index>=records.length-1)draw(0);setPlay(!playing)});
byId("version").textContent="MuJoCo "+result.mujoco_version;byId("model-hash").textContent=result.model_sha256;
byId("trace-hash").textContent=result.trace_sha256;byId("stat-status").textContent=result.status;
draw(0);requestAnimationFrame(tick);
</script></body></html>'''.replace("__DATA__", packed).replace("__ARM__", arm).replace("__STEPS__", str(steps)).replace("__RMSE__", f"{rmse:.4f}").replace("__MAX_ERROR__", f"{max_err:.4f}")


def write_playground(output: str | Path, sim: Mapping[str, Any]) -> bool:
    """Write viewer only when a real run has the required measurements."""
    try:
        content = render_playground_html(sim)
    except PlaygroundError:
        return False
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "playground.html").write_text(content, encoding="utf-8")
    return True
