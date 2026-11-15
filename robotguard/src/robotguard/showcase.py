"""Reproducible end-to-end integration and offline visual comparison.

The comparison is only constructed from actual MuJoCo result dictionaries returned
by simulate_trace. Baselines are hand-coded motion profiles, not pretrained AI.
"""
from __future__ import annotations

import base64
import html
from importlib import resources
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .core import RobotSpec, validate_trace
from .integration import baseline_trace, write_actions_jsonl
from .report import write_report
from .simulation import SimulationError, simulate_trace, write_simulation


class ComparisonError(ValueError):
    pass


def _verified_payload(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(results) != {"linear", "smoothstep"}:
        raise ComparisonError("Comparison requires both real linear and smoothstep simulations")
    a, b = results["linear"], results["smoothstep"]
    for name, sim in results.items():
        if sim.get("status") != "SIMULATION_COMPLETED" or sim.get("preflight_status") != "PASS":
            raise ComparisonError(f"{name}: physics replay was not completed")
        if sim.get("physical_safety_certified") is not False:
            raise ComparisonError("Simulation must not claim physical safety certification")
        if not isinstance(sim.get("integration_steps"), int) or sim["integration_steps"] < 1:
            raise ComparisonError(f"{name}: no recorded physics integration steps")
        records = sim.get("records")
        if not isinstance(records, list) or len(records) < 2:
            raise ComparisonError(f"{name}: missing actual simulator samples")
        for row in records:
            points = row.get("link_points_xy_m")
            if not isinstance(points, list) or len(points) != 3 or any(
                not isinstance(p, list) or len(p) != 2 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                for p in points
            ):
                raise ComparisonError(f"{name}: missing measured 2D joint positions")
            for key in ("actual", "target"):
                v = row.get(key)
                if not isinstance(v, list) or len(v) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
                    raise ComparisonError(f"{name}: missing measured {key} joint values")
        if not all(isinstance(sim.get(k), str) and len(sim[k]) == 64 and all(c in '0123456789abcdef' for c in sim[k]) for k in ("model_sha256", "trace_sha256")):
            raise ComparisonError(f"{name}: missing evidence hashes")
    if a["model_sha256"] != b["model_sha256"] or a["joint_names"] != b["joint_names"]:
        raise ComparisonError("Cannot compare different MJCF models or joint orders")
    if a["simulated_duration_s"] != b["simulated_duration_s"] or a["time_step_s"] != b["time_step_s"]:
        raise ComparisonError("Cannot compare different horizons or integration steps")
    for name, sim in results.items():
        if sim["records"][0]["target"] != sim["records"][0]["actual"]:
            raise ComparisonError(f"{name}: first commanded and observed position differ")
    return {
        "model_sha256": a["model_sha256"],
        "mujoco_version": str(a["mujoco_version"]),
        "labels": {"linear": "Linear baseline", "smoothstep": "Smoothstep baseline"},
        "policies": {
            name: {"metrics": sim["metrics"], "trace_sha256": sim["trace_sha256"],
                   "integration_steps": sim["integration_steps"],
                   "records": sim["records"]}
            for name, sim in results.items()
        },
        "joint_names": a["joint_names"],
        "duration_s": a["simulated_duration_s"],
        "physical_safety_certified": False,
        "policy_type": "deterministic test baselines; no machine-learning model executed",
    }


def render_comparison_html(results: Mapping[str, Mapping[str, Any]], rejected_count: int) -> str:
    payload = _verified_payload(results)
    if type(rejected_count) is not int or rejected_count <= 0:
        raise ComparisonError("Must include a rejected trace with actual audit findings")
    packed = base64.b64encode(json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()).decode()
    rmse_a = payload["policies"]["linear"]["metrics"]["rmse_tracking_error_rad"]
    rmse_b = payload["policies"]["smoothstep"]["metrics"]["rmse_tracking_error_rad"]
    template = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RobotGuard | Policy-to-Physics Playground</title><style>
:root{color-scheme:dark;--bg:#080e1b;--panel:#111d30;--line:#28384e;--ink:#eef5ff;--sub:#a8b9cf;--green:#76e1ad;--cyan:#66c7f4;--violet:#b9a0ff;--warn:#ffbd77}
*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;margin:0}
main{max-width:1220px;padding:24px 24px 70px;margin:auto}a{color:#9dcfff}nav{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;padding-bottom:22px;border-bottom:1px solid var(--line)}nav strong{letter-spacing:.12em;font-size:13px}nav small{color:var(--sub)}
h1{font-size:clamp(36px,5vw,62px);line-height:1.08;letter-spacing:-.055em;margin:24px 0 13px;max-width:900px}p{color:var(--sub);max-width:850px}h2{font-size:20px;margin:0 0 10px}h3{font-size:15px;margin:12px 0 6px}
.tag{display:inline-block;font-size:11px;letter-spacing:.08em;border-radius:20px;border:1px solid #347f68;color:var(--green);padding:6px 10px}
.panel{border:1px solid var(--line);background:var(--panel);border-radius:15px;padding:19px;min-width:0}.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:23px 0}.step{background:#121f33;border:1px solid #2e425a;border-radius:12px;padding:12px;text-align:center;min-width:0}.step b{display:block;font-size:13px}.step small{display:block;color:var(--sub);font-size:11px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:16px 0}.stat{padding:15px 16px;border:1px solid var(--line);border-radius:13px;background:#0d192b}.stat b{display:block;font-size:23px;font-variant-numeric:tabular-nums}.stat small{color:var(--sub)}
.bar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;margin:30px 0 13px}.bar p{margin:0}.controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:14px 16px;border:1px solid var(--line);border-radius:14px;background:#111e30;margin-bottom:14px}.controls button{background:#3b6df2;border:0;border-radius:9px;padding:9px 16px;color:white;cursor:pointer;font-weight:700}input[type=range]{flex:1;min-width:140px;accent-color:var(--green)}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #ffcf69;outline-offset:3px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:13px}.view{min-width:0}.view header{display:flex;align-items:center;justify-content:space-between;gap:10px}.view h2{margin:0}.view small,.meta{font-size:12px;color:var(--sub)}.visual{width:100%;display:block;height:auto;border:1px solid #2f4765;border-radius:12px;background:radial-gradient(circle at center,#15263d,#091423)}
.arm-a{stroke:var(--cyan);stroke-width:15;stroke-linecap:round}.arm-b{stroke:var(--violet);stroke-width:12;stroke-linecap:round}.joint{fill:#e6f1fe;stroke:#10233e;stroke-width:3}.tip{fill:#f8cb86}.telemetry{display:flex;justify-content:space-between;gap:5px;flex-wrap:wrap;padding-top:12px}.mono{font-family:ui-monospace,monospace;font-size:12px}.measure{color:var(--green)}
.chart{width:100%;height:auto;display:block;border-radius:9px;background:#0a1728}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--sub)}.legend em{font-style:normal;color:var(--cyan)}.legend .p{color:var(--violet)}.section{margin-top:26px}.sources{display:grid;grid-template-columns:1fr 1fr;gap:12px}.sources code{font-size:12px;overflow-wrap:anywhere}.status{border:1px solid #58613d;border-radius:10px;padding:12px;color:#fcd59c;background:#221d17}
.table{width:100%;border-collapse:collapse;font-size:13px}.table th,.table td{text-align:left;border-bottom:1px solid #2b3b50;padding:11px 8px;vertical-align:top}.table th{color:var(--sub);font-weight:600}.yes{color:var(--green)}.no{color:var(--warn)}footer{margin-top:32px;border-top:1px solid var(--line);padding-top:20px;color:var(--sub)}
@media(max-width:800px){.steps{grid-template-columns:repeat(3,1fr)}.stats{grid-template-columns:repeat(2,1fr)}.views,.sources{grid-template-columns:1fr}}
@media(max-width:460px){.steps{grid-template-columns:1fr 1fr}.stats{grid-template-columns:1fr 1fr}main{padding:16px}}
</style></head><body><main><nav><strong>◎ ROBOTGUARD <span style="color:var(--cyan)">/ PLAYGROUND</span></strong><small>Offline · no robot · no API key · measured MuJoCo physics</small></nav>
<section><span class="tag">EVIDENCE-FIRST ROBOT DEVELOPMENT</span><h1>Give your policy a robot.<br>See what really happens.</h1><p>Two deterministic reference policies, one identical physical model, separate MuJoCo executions and a shared timeline. This is a real replay, <b>not a pretrained AI demonstration</b>.</p></section>
<div class="steps" aria-label="Working integration path"><div class="step"><b>01 · Policy</b><small>Named joint positions</small></div><div class="step"><b>02 · Contract</b><small>Units + frame + time</small></div><div class="step"><b>03 · Gate</b><small>Reject invalid actions</small></div><div class="step"><b>04 · MuJoCo</b><small>Physics integration</small></div><div class="step"><b>05 · Evidence</b><small>Measured frames + hashes</small></div></div>
<div class="stats"><div class="stat"><b>2</b><small>Approved reference policies</small></div><div class="stat"><b class="measure">__STEPS__</b><small>Physics steps per policy</small></div><div class="stat"><b>__BAD__</b><small>Rejected unsafe-trace findings</small></div><div class="stat"><b>0</b><small>Hardware control commands</small></div></div>
<div class="bar"><div><h2>Side-by-side measured motion</h2><p>Both start from the same joint pose and seek the same target over the same time horizon.</p></div><span class="tag">MUJOCO __VERSION__</span></div>
<div class="controls"><button id="toggle" aria-label="Play the two recorded policies">▶ Play</button><label for="timeline">Replay position</label><input type="range" id="timeline" min="0" max="1000" step="1" value="0"/><b class="mono" id="clock">0.000 s</b><span class="meta" id="frame">frame 1</span></div>
<div class="views"><section class="panel view" id="linear-view"><header><h2>Linear</h2><small>Deterministic baseline</small></header><p>Constant-velocity reference between waypoints.</p><svg class="visual" id="linear-svg" viewBox="0 0 520 360" role="img" aria-label="Recorded MuJoCo two-link robot, linear reference"><defs><pattern id="lgrid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M30 0H0V30" stroke="#25384d" fill="none"/></pattern></defs><rect width="520" height="360" fill="url(#lgrid)"/><path id="linear-trail" stroke="#5bd3fc" opacity=".40" stroke-width="2" fill="none"/><path id="linear-a" class="arm-a"/><path id="linear-b" class="arm-b"/><circle id="linear-j0" r="9" class="joint"/><circle id="linear-j1" r="8" class="joint"/><circle id="linear-tip" r="8" class="tip"/></svg><div class="telemetry"><span class="mono">actual: <b id="linear-actual">—</b></span><span class="mono">target: <b id="linear-target">—</b></span></div><p class="meta">Tracking RMSE: <b>__A_RMSE__ rad</b> · <a href="linear/playground.html">Own replay</a> · <a href="linear/simulation.json">Raw evidence</a></p></section>
<section class="panel view" id="smoothstep-view"><header><h2>Smoothstep</h2><small>Deterministic baseline</small></header><p>Gradually varying reference between the same endpoints.</p><svg class="visual" id="smoothstep-svg" viewBox="0 0 520 360" role="img" aria-label="Recorded MuJoCo two-link robot, smoothstep reference"><defs><pattern id="sgrid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M30 0H0V30" stroke="#25384d" fill="none"/></pattern></defs><rect width="520" height="360" fill="url(#sgrid)"/><path id="smoothstep-trail" stroke="#5bd3fc" opacity=".40" stroke-width="2" fill="none"/><path id="smoothstep-a" class="arm-a"/><path id="smoothstep-b" class="arm-b"/><circle id="smoothstep-j0" r="9" class="joint"/><circle id="smoothstep-j1" r="8" class="joint"/><circle id="smoothstep-tip" r="8" class="tip"/></svg><div class="telemetry"><span class="mono">actual: <b id="smoothstep-actual">—</b></span><span class="mono">target: <b id="smoothstep-target">—</b></span></div><p class="meta">Tracking RMSE: <b>__B_RMSE__ rad</b> · <a href="smoothstep/playground.html">Own replay</a> · <a href="smoothstep/simulation.json">Raw evidence</a></p></section></div>
<section class="panel section"><h2>Commanded versus measured · joint 1</h2><p>Solid lines show observed MuJoCo qpos; dashed lines show commanded setpoints. Horizontal axis is simulation time.</p><svg class="chart" id="chart" viewBox="0 0 960 240" role="img" aria-label="Measured and requested shoulder joint traces for both simulated policies"><path d="M60 18V205H922" stroke="#657991" fill="none"/><path id="chart-lin-target" stroke="#66c7f4" stroke-width="2" stroke-dasharray="6 6" fill="none"/><path id="chart-lin-actual" stroke="#66c7f4" stroke-width="3" fill="none"/><path id="chart-smo-target" stroke="#b9a0ff" stroke-width="2" stroke-dasharray="6 6" fill="none"/><path id="chart-smo-actual" stroke="#b9a0ff" stroke-width="3" fill="none"/><line id="cursor" y1="18" y2="205" stroke="#f2c87c" stroke-width="1" stroke-dasharray="4 4"/></svg><div class="legend"><span><em>● Linear</em></span><span class="p">● Smoothstep</span><span>Solid = observed · dashed = requested</span></div></section>
<section class="sources section"><div class="panel"><h2>Safety gate · rejected input</h2><p class="status">__BAD__ findings recorded. The invalid trace was <strong>not</strong> sent to MuJoCo.</p><p class="meta">Inspect the actual failed input and diagnostic codes. An unsafe trace has no simulated result.</p><a href="rejected.html">Read failure report</a> · <a href="rejected.json">JSON findings</a></div><div class="panel"><h2>Provenance · same model</h2><p>MuJoCo version: __VERSION__</p><p class="meta">Model SHA-256</p><code id="model-hash">__MODEL_HASH__</code><p class="meta">Policy inputs and observations are archived separately in this output folder.</p><a href="comparison.json">Comparison JSON</a></div></section>
<section class="panel section"><h2>Where your own AI policy connects</h2><table class="table"><thead><tr><th>Input path</th><th>What works now</th><th>Next action</th></tr></thead><tbody><tr><td>Packaged example</td><td class="yes">Two actual MuJoCo replays</td><td><code>robotguard showcase</code></td></tr><tr><td>Python policy</td><td class="yes">Explicit named-joint adapter</td><td><code>python robotguard/examples/connect_my_policy.py</code></td></tr><tr><td>External policy JSONL</td><td class="yes">Preflight + supported MuJoCo model</td><td><code>robotguard simulate &lt;trace&gt; --spec ... --model ...</code></td></tr><tr><td>Dream-1 / LeRobot / OpenPI</td><td class="no">No native connector yet</td><td>Write &amp; test an embodiment-specific adapter first</td></tr><tr><td>ROS 2 / physical robot</td><td class="no">Not connected</td><td>Do not send validation output directly to hardware</td></tr></tbody></table></section>
<footer>Real simulator observations, not a generated robot video. This 2D visual shows recorded XY joint anchors of an intentionally synthetic arm. Model limits are illustrative, velocities are sampled estimates, and no collision, force or functional safety is certified. <b>Passing is not proof of hardware safety.</b></footer>
</main><script>"use strict";
const d=JSON.parse(atob("__PAYLOAD__"));const $=id=>document.getElementById(id);
const ids=["linear","smoothstep"];const all=ids.flatMap(k=>d.policies[k].records.flatMap(r=>r.link_points_xy_m));
const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);const xm=(Math.min(...xs)+Math.max(...xs))/2,ym=(Math.min(...ys)+Math.max(...ys))/2;
const radius=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys),.35)*.63;
const xy=p=>[260+(p[0]-xm)*150/radius,180-(p[1]-ym)*150/radius];
const at=(name,time)=>{const a=d.policies[name].records;let k=0;while(k<a.length-1&&a[k+1].t<=time+1e-7)k++;return [k,a[k]]};
const pt=p=>p.map(x=>x.toFixed(1)).join(',');
function draw(time){time=Math.max(0,Math.min(d.duration_s,time));for(const key of ids){const [index,row]=at(key,time),p=row.link_points_xy_m.map(xy);$(key+'-a').setAttribute('d','M'+pt(p[0])+' L'+pt(p[1]));$(key+'-b').setAttribute('d','M'+pt(p[1])+' L'+pt(p[2]));['j0','j1','tip'].forEach((id,i)=>{$(key+'-'+id).setAttribute('cx',p[i][0]);$(key+'-'+id).setAttribute('cy',p[i][1])});
$(key+'-trail').setAttribute('d',d.policies[key].records.slice(0,index+1).map((r,i)=>(i?'L':'M')+pt(xy(r.link_points_xy_m[2]))).join(' '));
$(key+'-actual').textContent=row.actual.map(v=>v.toFixed(3)).join(', ');$(key+'-target').textContent=row.target.map(v=>v.toFixed(3)).join(', ')}
$('clock').textContent=time.toFixed(3)+' s';$('frame').textContent='shared simulation time';$('timeline').value=Math.round(1000*time/d.duration_s);const xx=60+860*time/d.duration_s;$('cursor').setAttribute('x1',xx);$('cursor').setAttribute('x2',xx)}
const rows=ids.flatMap(k=>d.policies[k].records),vmin=Math.min(...rows.flatMap(r=>[r.target[0],r.actual[0]]))-.02,vmax=Math.max(...rows.flatMap(r=>[r.target[0],r.actual[0]]))+.02;
for(const [key,prefix] of [['linear','lin'],['smoothstep','smo']]){for(const [kind,kindName] of [['target','target'],['actual','actual']]){$('chart-'+prefix+'-'+kindName).setAttribute('d',d.policies[key].records.map((r,i)=>(i?'L':'M')+(60+860*r.t/d.duration_s).toFixed(1)+' '+(205-174*(r[kind][0]-vmin)/(vmax-vmin)).toFixed(1)).join(' '))}}
let playing=false,previous=0,elapsed=0;function animate(now){if(playing){if(previous){elapsed=Math.min(d.duration_s,elapsed+(now-previous)/1000);draw(elapsed);if(elapsed>=d.duration_s){playing=false;$('toggle').textContent='▶ Replay'}}previous=now}requestAnimationFrame(animate)}
$('timeline').addEventListener('input',e=>{playing=false;previous=0;$('toggle').textContent='▶ Play';elapsed=Number(e.target.value)/1000*d.duration_s;draw(elapsed)});
$('toggle').addEventListener('click',()=>{if(elapsed>=d.duration_s)elapsed=0;playing=!playing;previous=0;$('toggle').textContent=playing?'❚❚ Pause':'▶ Play';draw(elapsed)});
draw(0);requestAnimationFrame(animate);
</script></body></html>'''
    return (template.replace("__PAYLOAD__", packed)
            .replace("__BAD__", str(rejected_count))
            .replace("__STEPS__", str(payload["policies"]["linear"]["integration_steps"]))
            .replace("__VERSION__", html.escape(payload["mujoco_version"]))
            .replace("__MODEL_HASH__", payload["model_sha256"])
            .replace("__A_RMSE__", f"{rmse_a:.5f}")
            .replace("__B_RMSE__", f"{rmse_b:.5f}"))


def run_showcase(output: str | Path) -> dict[str, Any]:
    """Run two actual physics trials and create one portable comparison dashboard."""
    assets = resources.files("robotguard").joinpath("assets")
    spec = RobotSpec.from_dict(json.loads(assets.joinpath("demo_spec.json").read_text()))
    xml = assets.joinpath("demo_arm.xml").read_text()
    rejected_actions = [json.loads(x) for x in assets.joinpath("demo_unsafe.jsonl").read_text().splitlines() if x.strip()]
    rejected = validate_trace(spec, rejected_actions)
    if rejected.ok:
        raise ComparisonError("The deliberately invalid fixture unexpectedly passed")

    # Simulate in memory first. An error cannot leave a convincing HTML dashboard.
    data: dict[str, dict[str, Any]] = {}
    actions_by_policy: dict[str, list[dict[str, Any]]] = {}
    for name in ("linear", "smoothstep"):
        trace = baseline_trace(spec, name)
        audit = validate_trace(spec, trace)
        if not audit.ok:
            raise ComparisonError(f"{name} baseline unexpectedly violates preflight: {[v.code for v in audit.violations]}")
        sim = simulate_trace(spec, trace, xml)
        if sim["status"] != "SIMULATION_COMPLETED":
            raise SimulationError(f"{name} did not complete successfully: {sim['status']}")
        data[name], actions_by_policy[name] = sim, trace
    html_page = render_comparison_html(data, len(rejected.violations))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(exist_ok=True)
    write_actions_jsonl(output / "inputs" / "rejected.jsonl", rejected_actions)
    write_report(output / "rejected.html", spec, rejected)
    (output / "rejected.json").write_text(json.dumps(rejected.as_dict(), indent=2, allow_nan=False) + "\n")
    for name in ("linear", "smoothstep"):
        write_actions_jsonl(output / "inputs" / (name + ".jsonl"), actions_by_policy[name])
        write_simulation(output / name, data[name])
    comparison = _verified_payload(data)
    comparison.pop("policies")  # no duplicated traces in the summary JSON
    comparison["policies"] = {name: {"metrics": data[name]["metrics"],
                                   "integration_steps": data[name]["integration_steps"],
                                   "trace_sha256": data[name]["trace_sha256"],
                                   "input_path": f"inputs/{name}.jsonl",
                                   "measurement_path": f"{name}/simulation.json"}
                              for name in ("linear", "smoothstep")}
    comparison["rejected_violations"] = len(rejected.violations)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    (output / "index.html").write_text(html_page, encoding="utf-8")
    return comparison
