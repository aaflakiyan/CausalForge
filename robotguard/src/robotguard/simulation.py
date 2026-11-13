"""Optional MuJoCo replay of a validated synthetic joint-position trajectory.

The replay reports actual simulated motion; it is NOT a real-world safety proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import RobotSpec, validate_trace


class SimulationError(ValueError):
    """Simulation cannot be run without changing the requested validation contract."""


def _mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise SimulationError("MuJoCo is optional. Install: python -m pip install -e './robotguard[mujoco]'") from exc
    return mujoco


def _sha(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _configure(model: Any, spec: RobotSpec, mj: Any) -> tuple[list[int], list[int]]:
    """Fail closed unless each named hinge has exactly one unit-gear position servo."""
    if spec.units != "rad" or spec.frame != "joint":
        raise SimulationError("MuJoCo replay requires positions in radians and frame='joint'")
    if model.njnt != len(spec.joint_names) or model.nu != len(spec.joint_names):
        raise SimulationError("Demo adapter requires exactly one hinge joint and one servo per spec joint")
    qpos_addresses: list[int] = []
    actuator_ids: list[int] = []
    for name, (lo, hi) in zip(spec.joint_names, spec.position_limits):
        try:
            jid = int(model.joint(name).id)
        except (KeyError, ValueError) as exc:
            raise SimulationError(f"Model is missing the declared joint {name!r}") from exc
        if model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE:
            raise SimulationError(f"Joint {name!r} is not a one-DOF hinge")
        if not model.jnt_limited[jid]:
            raise SimulationError(f"Joint {name!r} has no enforced model joint range")
        mj_lo, mj_hi = model.jnt_range[jid]
        if lo < float(mj_lo) - 1e-9 or hi > float(mj_hi) + 1e-9:
            raise SimulationError(f"Spec limits for {name!r} exceed the MuJoCo joint range")
        matched = [aid for aid in range(model.nu) if
                   model.actuator_trntype[aid] == mj.mjtTrn.mjTRN_JOINT and
                   int(model.actuator_trnid[aid, 0]) == jid]
        if len(matched) != 1:
            raise SimulationError(f"Joint {name!r} must have exactly one joint actuator")
        aid = matched[0]
        if abs(float(model.actuator_gear[aid, 0]) - 1.0) > 1e-9:
            raise SimulationError(f"Joint {name!r}: only unit-gear position servos are supported")
        gain = float(model.actuator_gainprm[aid, 0])
        bias = float(model.actuator_biasprm[aid, 1])
        if not (gain > 0 and math.isclose(bias, -gain, rel_tol=1e-6, abs_tol=1e-6)):
            raise SimulationError(f"Joint {name!r}: actuator is not a recognised position servo")
        if not model.actuator_ctrllimited[aid]:
            raise SimulationError(f"Joint {name!r}: position servo has no control limits")
        ctrl_lo, ctrl_hi = model.actuator_ctrlrange[aid]
        if lo < float(ctrl_lo) - 1e-9 or hi > float(ctrl_hi) + 1e-9:
            raise SimulationError(f"Joint {name!r}: spec limits exceed actuator control range")
        qpos_addresses.append(int(model.jnt_qposadr[jid]))
        actuator_ids.append(aid)
    return qpos_addresses, actuator_ids


def simulate_trace(spec: RobotSpec, actions: Sequence[Mapping[str, Any]], model_xml: str,
                   *, hold_s: float = 0.2, max_steps: int = 200_000) -> dict[str, Any]:
    """Replay valid commands with piecewise-linear targets and log actual MuJoCo qpos.

    The initial simulated pose is SET to the first waypoint, which is a demo
    initialisation, not a feasible physical move. A pre-flight FAIL never steps
    physics. No collision/force/hardware suitability is asserted by this method.
    """
    result = validate_trace(spec, actions)
    if not result.ok:
        raise SimulationError(f"Pre-flight rejected {len(result.violations)} violation(s); no physics executed")
    if not math.isfinite(hold_s) or hold_s < 0:
        raise SimulationError("hold_s must be a finite non-negative number")
    mj = _mujoco()
    try:
        model = mj.MjModel.from_xml_string(model_xml)
    except Exception as exc:
        raise SimulationError(f"MuJoCo could not compile the provided MJCF: {exc}") from exc
    qadr, aids = _configure(model, spec, mj)
    dt = float(model.opt.timestep)
    if not math.isfinite(dt) or dt <= 0:
        raise SimulationError("Model timestep must be positive and finite")
    times = [float(a["t"]) for a in actions]
    positions = [tuple(float(x) for x in a["position"]) for a in actions]
    duration = times[-1] - times[0] + hold_s
    steps = max(1, math.ceil(duration / dt))
    if steps > max_steps:
        raise SimulationError(f"Replay would require {steps} steps (maximum {max_steps})")
    data = mj.MjData(model)
    for address, value in zip(qadr, positions[0]):
        data.qpos[address] = value
    mj.mj_forward(model, data)
    log_every = max(1, round(0.02 / dt))
    records: list[dict[str, Any]] = []
    max_error = 0.0
    error_square_sum = 0.0
    error_count = 0
    limit_violations = 0
    waypoint = 0
    for k in range(steps + 1):
        now = k * dt
        t = times[0] + now
        while waypoint < len(times) - 2 and t > times[waypoint + 1]:
            waypoint += 1
        if t >= times[-1]:
            target = positions[-1]
        else:
            left, right = times[waypoint:waypoint + 2]
            fraction = (t - left) / (right - left)
            target = tuple(a + fraction * (b - a) for a, b in zip(positions[waypoint], positions[waypoint + 1]))
        for aid, value in zip(aids, target):
            data.ctrl[aid] = value
        actual = [float(data.qpos[address]) for address in qadr]
        if not all(math.isfinite(x) for x in actual):
            raise SimulationError(f"Non-finite simulator state near t={now:g}s")
        for (lo, hi), observed, requested in zip(spec.position_limits, actual, target):
            max_error = max(max_error, abs(observed - requested))
            error_square_sum += (observed - requested) ** 2
            error_count += 1
            if observed < lo - 1e-7 or observed > hi + 1e-7:
                limit_violations += 1
        if k % log_every == 0 or k == steps:
            records.append({"t": round(now, 6), "target": [round(x, 8) for x in target],
                            "actual": [round(x, 8) for x in actual]})
        if k < steps:
            mj.mj_step(model, data)
    return {
        "status": "OBSERVED_LIMIT_VIOLATION" if limit_violations else "SIMULATION_COMPLETED",
        "preflight_status": "PASS", "physical_safety_certified": False,
        "model": "MJCF supplied to simulator",
        "mujoco_version": getattr(mj, "__version__", "unknown"),
        "model_sha256": _sha(model_xml),
        "trace_sha256": _sha(json.dumps(list(actions), sort_keys=True, separators=(",", ":"))),
        "robot_spec": spec.name, "joint_names": list(spec.joint_names), "time_step_s": dt,
        "simulated_duration_s": round(steps * dt, 8), "integration_steps": steps,
        "initialisation": "qpos set to first waypoint, qvel zero; not a physical start-up procedure",
        "interpolation": "piecewise-linear joint-position setpoints; discontinuous desired velocity at knots",
        "metrics": {"max_tracking_error_rad": round(max_error, 8),
                    "rmse_tracking_error_rad": round(math.sqrt(error_square_sum / error_count), 8),
                    "observed_joint_limit_violations": limit_violations},
        "records": records,
        "limitations": ["No collision or torque/force guarantee", "No real-robot execution",
                        "No assertion that the chosen spec limits are authoritative",
                        "Simulated controller dynamics are model-dependent"],
    }


def render_simulation_html(sim: Mapping[str, Any]) -> str:
    """Self-contained interactive-free comparison of commanded and observed motion."""
    rows = sim["records"]
    joints = sim["joint_names"]
    t_end = max(float(rows[-1]["t"]), 1e-9)
    values = [float(x) for row in rows for key in ("target", "actual") for x in row[key]]
    vmin, vmax = min(values), max(values)
    pad = max(0.08, (vmax - vmin) * 0.15)
    vmin -= pad
    vmax += pad
    colours = ["#38bdf8", "#c4b5fd", "#fbbf24", "#5eead4"]
    charts = []
    for j, name in enumerate(joints):
        def pts(which: str) -> str:
            return " ".join(f"{50 + 720 * float(r['t']) / t_end:.1f},{220 - 180*(float(r[which][j])-vmin)/(vmax-vmin):.1f}" for r in rows)
        colour = colours[j % len(colours)]
        charts.append(f'<g><polyline points="{pts("target")}" stroke="{colour}" stroke-width="2" stroke-dasharray="6 5" fill="none"/><polyline points="{pts("actual")}" stroke="{colour}" stroke-width="3" fill="none"/></g>')
    legend = "".join(f'<span style="color:{colours[j % len(colours)]}">● {html.escape(name)}</span>' for j, name in enumerate(joints))
    note = html.escape("; ".join(sim["limitations"]))
    metrics = sim["metrics"]
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RobotGuard / actual MuJoCo playback</title><style>body{{font:15px system-ui;background:#0b1220;color:#e4ecff;max-width:950px;margin:35px auto;padding:20px}}
article{{background:#132137;padding:22px;border:1px solid #324357;border-radius:14px;margin:18px 0}}h1{{font-size:32px}}p,small{{color:#afc0d7}}svg{{width:100%}}
.legend span{{margin-right:18px}}.kpi{{font-size:21px;font-weight:bold}}.warn{{color:#fbbf24}}</style><h1>RobotGuard / MuJoCo replay</h1>
<p>Actual physics steps, not invented motion. Model: {html.escape(sim['robot_spec'])} · MuJoCo {html.escape(sim['mujoco_version'])}</p>
<article><div class="kpi">{html.escape(sim['status'])}</div><p>Pre-flight PASS is not a hardware-safety certificate.</p>
<p>Tracking RMSE: {metrics['rmse_tracking_error_rad']:.5f} rad · Max error: {metrics['max_tracking_error_rad']:.5f} rad · Observed sampled limit violations: {metrics['observed_joint_limit_violations']}</p></article>
<article><h2>Commanded vs observed joint positions (radians)</h2>
<svg viewBox="0 0 820 265" role="img" aria-label="Dashed requested trajectories and solid simulated actual joint positions">
<path d="M50 25V220H770" fill="none" stroke="#64748b"/>{''.join(charts)}<text x="50" y="245" fill="#9ca3af">0s</text><text x="740" y="245" fill="#9ca3af">{t_end:.2f}s</text></svg>
<div class="legend">{legend}</div><p>Solid = actual MuJoCo qpos · dashed = requested servo target.</p></article>
<article><h2>Provenance</h2><small>Model SHA-256: {sim['model_sha256']}<br>Trace SHA-256: {sim['trace_sha256']}<br>Step: {sim['time_step_s']} s · {sim['integration_steps']} integration steps</small></article>
<p class="warn">Limitations: {note}. Initial pose is set directly. No collision, force, braking or hardware safety verification.</p></html>'''


def write_simulation(output: str | Path, sim: Mapping[str, Any]) -> None:
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "simulation.json").write_text(json.dumps(sim, indent=2) + "\n", encoding="utf-8")
    (path / "simulation.html").write_text(render_simulation_html(sim), encoding="utf-8")
