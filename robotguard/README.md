# RobotGuard

**A pre-flight check for AI-generated robot actions.** Catch joint-limit violations, unit and frame mismatches, excessive velocities, acceleration spikes and timing errors *before* sending a trajectory to a simulator.

> **Early developer preview (v0.1).** A deterministic joint-space trace validator, **not** certified safety software. This does **not** detect collisions, torque/force hazards, unreachable Cartesian poses or real-world control instability. Never use it as the sole safety mechanism for physical robots.

![RobotGuard example audit](assets/robotguard-preview.svg)

## Try it in under a minute

```bash
# From the CausalForge repository root:
python -m pip install -e ./robotguard
robotguard demo
# Open robotguard-demo.html in your browser.
```

To check your own trace:

```bash
robotguard check robotguard/examples/safe.jsonl --spec robotguard/examples/robot.json --html safe.html --json safe.json
# PASS — exit code 0

robotguard check robotguard/examples/unsafe.jsonl --spec robotguard/examples/robot.json --html unsafe.html --json unsafe.json
# FAIL — exit code 1 (expected for the unsafe example)
```

The generated HTML report runs completely offline, including the joint-position chart and all findings. The JSON output is suitable for CI, evaluations and agent workflows.

## Why this exists

A VLA or coding agent can output a plausible-looking action that is wrong for the robot: degrees instead of radians, a different reference frame, a large jump in 100 ms, an invalid joint index or a malformed timestamp. RobotGuard performs explicit checks with readable reasons and machine-readable results. It **rejects** invalid actions rather than silently clipping or converting them.

### Input format

`robot.json` defines `joint_names`, `position_limits`, `max_velocity`, `max_acceleration`, the declared `units` (`rad` or `deg`), expected `frame` and the largest accepted step interval `max_interval_s`. All position/velocity/acceleration values must use the same declared units, per second or per second squared. Angular commands only in v0.1; no Cartesian motion checking.

`actions.jsonl` contains one command per line:

```json
{"t": 0.1, "position": [0.005, 0.006, 0.004], "units": "rad", "frame": "joint"}
```

Timestamps must be finite, strictly increasing, non-negative, and expressed in **seconds**. The vector order must match `joint_names` exactly. Position limits are inclusive. Velocities use `Δposition/Δtime`; accelerations use successive segment velocities divided by the separation of their midpoints, which works for uneven sampling. There is no implicit coordinate conversion.

### Python API: guard a stream of proposed actions

```python
from robotguard import RobotSpec, StreamGuard

spec = RobotSpec.from_file("robotguard/examples/robot.json")
guard = StreamGuard(spec)

proposal = {"t": 0.0, "position": [0.0, 0.0, 0.0], "units": "rad", "frame": "joint"}
decision = guard.propose(proposal)
if decision.ok:
    # Only hand this to a separate, independently safeguarded simulator adapter.
    print("Accepted in offline pre-flight checks")
else:
    print([v.as_dict() for v in decision.violations])
```

`StreamGuard` keeps accepted actions and rejects a proposed sample if any check fails. It is suitable for low-volume demos and analysis, **not** real-time control: each proposal currently revalidates the accepted history. For long datasets use `validate_trace()` in batch mode.

## AI agent integration

The [`skills/robotguard/SKILL.md`](skills/robotguard/SKILL.md) file gives coding agents explicit instructions to produce a robot spec, generate an action trace and run validation. No credentials, cloud API or agent framework are required. Exit code **0** = pass, **1** = violations, **2** = bad inputs/configuration.

## Tests and roadmap

```bash
cd robotguard
PYTHONPATH=src python -m unittest discover -s tests -v
```

**What is implemented:** Python package, CLI, JSONL parser, specification validation, position/velocity/acceleration checks, strict units and reference-frame tags, timing checks, HTML and JSON reports, stream gate, fixtures and unit tests.

**What is not implemented:** collision checking, dynamic feasibility, ROS 2 bridge, MuJoCo policy execution, ISO-compliant safety controls, agent MCP server, certified hardware interfaces. Future work should integrate these as separate, clearly labelled adapters with their own tests.

Open issues and contributions are welcome. This project deliberately makes no claim that a PASS is sufficient for hardware deployment.

## Licence

MIT (same licence as CausalForge). See the repository-level `LICENSE`.
