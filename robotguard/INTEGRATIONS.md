# Integration guide: from policy output to measurable physics

This document describes **what is actually implemented**. The demonstration uses a synthetic 2-DOF position-servo arm and two **deterministic** reference motion profiles. No pretrained model is loaded. A PASS is **not** a real-robot safety certificate.

## Path A — first run (no model, robot or credentials)

At the CausalForge repository root:

```bash
python -m pip install -e './robotguard[mujoco]'
robotguard showcase --out robotguard-showcase
```

Open `robotguard-showcase/index.html` directly in any modern browser; it has no CDN, server or network dependency.

**Exactly what happens:**

1. The program loads the packaged MJCF synthetic arm and robot specification.
2. `baseline_trace()` produces two different **predefined, not AI-learned** joint-position trajectories with the same start, goal and duration.
3. Both samples pass the same static validator for joint limits, intervals and sampled velocity/acceleration.
4. `simulate_trace()` separately replays each set of setpoints through MuJoCo's real stepper, recording `MjData.qpos` and model world-frame joint anchors.
5. The dashboard shows both recorded sequences on a shared timeline. It also links the two separate input traces, per-run SHA-256 hashes and numerical tracking metrics.
6. A separate deliberately unsafe fixture triggers the pre-flight gate and **is not simulated**.

Output tree:

```text
robotguard-showcase/
├── index.html                     # Offline comparison / shared scrubber
├── comparison.json               # Comparable inputs + provenance
├── rejected.html                 # Why unsafe input was refused
├── rejected.json
├── inputs/
│   ├── linear.jsonl              # Exactly what was commanded
│   ├── smoothstep.jsonl
│   └── rejected.jsonl
├── linear/
│   ├── simulation.json           # Observed MuJoCo records and hashes
│   ├── simulation.html           # Joint-position chart
│   └── playground.html           # Single-policy replay
└── smoothstep/
    ├── simulation.json
    ├── simulation.html
    └── playground.html
```

The comparison's time horizon and MJCF hash must match. The visual is an actual measured top-down XY projection of the synthetic arm, **not** a 3D visualisation of an arbitrary robot.

## Path B — plug in your own Python policy

Run the working example:

```bash
python robotguard/examples/connect_my_policy.py
```

The example calls `actions_from_policy(spec, times, predict)` and writes `my-policy.jsonl`. You only replace the `predict(t)` callable. It **must return explicit labels**:

```python
{
    "position": {"shoulder": 0.02, "elbow": 0.01},
    "units": "rad",
    "frame": "joint"
}
```

RobotGuard refuses missing/extra joint names, NaN, infinity, duplicate/out-of-order timestamps, unsupported or mismatched units/frames. It orders the returned dictionary to the declared robot joint-name list, **without converting or clipping** the policy's values.

Then test and simulate the generated file (these commands match the packaged synthetic robot only):

```bash
robotguard check my-policy.jsonl \
  --spec robotguard/src/robotguard/assets/demo_spec.json \
  --json my-preflight.json --html my-preflight.html

robotguard simulate my-policy.jsonl \
  --spec robotguard/src/robotguard/assets/demo_spec.json \
  --model robotguard/src/robotguard/assets/demo_arm.xml \
  --out my-policy-simulation
```

A real policy is responsible for synchronising observation times, returning joint positions rather than increments, declaring units and frames correctly, and matching the chosen MJCF embodiment. **We cannot check whether a policy has lied about its units or whether the supplied robot model is physically accurate.** The callable is sampled offline by this example; there is no closed-loop feedback from MuJoCo into the callable.

## Path C — existing joint-position trace

Provide one JSON object per line:

```json
{"t":0.1,"position":[0.02,0.01],"units":"rad","frame":"joint"}
```

The `position` array order must match `spec.joint_names`. `spec.max_velocity` and `spec.max_acceleration` must use the same angular unit per second and per second squared. The simulation adapter currently requires one hinge and exactly one supported unit-gear, control-limited *position servo* per declared joint, plus a matching MJCF joint and actuator control range. Degrees are not supported by the MuJoCo adapter.

Simulation with a new MJCF is **not equivalent to validating its physical accuracy**. Do not reuse the packaged model's limits for a real robot, and do not interpret an observed pass as proving collision or force safety.

## Path D — research-model integrations (not yet available)

| Input source | What needs to be engineered and tested |
|---|---|
| LeRobot | Model-specific observation adapter, action dimension/order, units, timestep, embodiment and episode reset |
| OpenPI | Same plus supported checkpoint and inference environment, documented action-chunk interpretation |
| Dream-1 | Public API contract/permissions, embodiment-specific state/observation mapping and action interpretation |
| ROS 2 | Message timestamps, trajectory names, frame conventions and controller interface semantics |
| UR5e/Franka | Validated MJCF model, actual actuator mapping, joint/torque limits, collision/contact checks and simulation-to-hardware gap |

**Do not list an integration as supported until it has a reproducible example, negative contract tests and a real simulation CI result.**

## Validation protocol and honest claims

- **Offline unit tests** verify input validation, negative examples, explicit contracts and no-clipping behaviour. They cannot certify physics.
- **Real physics tests** install `mujoco` in GitHub Actions, compile MJCF, record `qpos`, check actual integration steps and insist both policies share model/hash/time horizon. `robotguard-showcase` artefacts are uploaded from CI.
- **Browser smoke tests** confirm offline dashboard JavaScript, playback and timeline change when a generated page is opened.
- The model begins at the first waypoint through a direct qpos assignment, and the setpoints are piecewise linear in MuJoCo. No collision, contact, torque, hardware or continuous-time guarantee is claimed.

Failing preflight: no MuJoCo call. Missing MuJoCo: command fails instead of writing invented physics. Missing measured geometry: no comparison dashboard. Each successful run archives the model/trace hashes, physics version and measured values.
