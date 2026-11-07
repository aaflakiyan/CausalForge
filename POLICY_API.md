# CausalForge policy API — verified dual-UR5e joint control

The public `evaluate(policy, robot="ur5e-dual", task="reach", menagerie=...)` API accepts a Python object with a `predict(observation)` method returning an explicit `ActionChunk`.

**Working scope:** 2 × official MuJoCo Menagerie UR5e, 12 absolute joint-position commands in radians, joint frame, 20 ms control samples. No Cartesian commands are silently interpreted as joint commands, and no unsupported Panda, UR10e, grasp or VLA architecture is claimed.

## Quickstart

```bash
git clone https://github.com/aaflakiyan/CausalForge.git
cd CausalForge
python -m pip install -e '.[mujoco]'
causalforge demo --trace --out runs/quickstart
```

The CLI fetches the licensed **pinned** official Menagerie UR5e assets into `~/.cache/causalforge/mujoco_menagerie` on the first run (internet and Git required). Subsequent runs verify the cached revision. Developers with an offline checkout can use `--menagerie /path/to/mujoco_menagerie` and avoid downloads.

```python
from vla_council.causalforge.dual_ur5e_policy import evaluate, BuiltinReachPolicy
from vla_council.causalforge.dual_ur5e_reach import load_reach_model

root = "/path/to/mujoco_menagerie"
model = load_reach_model(root)
result = evaluate(BuiltinReachPolicy(model), robot="ur5e-dual", task="reach", menagerie=root)
assert result["task_success"], result
```

The built-in policy uses **damped least-squares inverse kinematics** to generate 12-joint reach trajectories; it is not a learned VLA.

## What gets validated before physics steps?

- **Robot identity:** exact model `menagerie_dual_ur5e`.
- **Representation:** absolute `joint_position` commands only (not torque, velocity or pose).
- **Axis order:** exactly six official named joints per arm, left then right.
- **Units and frame:** `rad` for every joint and `joint` coordinate frame.
- **Timing:** 20 ms command period, exactly five 4 ms MuJoCo substeps; no interpolated or silent resampling.
- **Array integrity:** exact `[T,12]` shape and finite values with a nonempty bounded horizon.
- **Actuator and joint bounds:** command limits checked against the compiled MuJoCo model.
- **Continuity:** joint command changes, including the first step from home, cannot exceed 2.0 rad/s.
- **Execution:** no clipping, silent conversion or partial execution if *any* action is invalid.

The simulated controller then steps the physics, measures tool endpoints and contact occurrences, and declares pass/fail against explicit final-error and contact criteria. The controller does **not** make execution safe on physical hardware; this is a simulator acceptance boundary and does not include a complete collision-avoidance or emergency-stop system.

## Bring your own policy

```python
from vla_council.causalforge.dual_ur5e_policy import SPEC, evaluate
from vla_council.causalforge.vla_interface import ActionChunk

class MyPolicy:
    def predict(self, observation):
        # Construct [T,12] absolute UR5e joint commands in radians.
        actions = ...  # your model inference; must match SPEC exactly
        return ActionChunk(actions, SPEC, "my_policy")

report = evaluate(MyPolicy(), robot="ur5e-dual", task="reach", menagerie="my_menagerie")
```

For incompatible VLA outputs, developers must implement a **reviewed, robot-specific adapter** with documented axis, timing and unit transformations. The validator intentionally refuses ambiguous conversions.

## Validation

Run `python -m unittest discover -s tests -p 'test_dual_ur5e_policy.py' -v`.

The physics test exercises the built-in policy on the genuine Menagerie dual-UR5e environment and asserts success, zero inter-arm contacts, substantial motion, deterministic action-step mapping and correct trace length. Negative tests intentionally supply wrong robot name, action units, frame, joint order, representation, dt, discontinuous velocities, out-of-range values and NaN to prove they are rejected.

GitHub Actions records the report as an artifact. This is a validation of the **specific dual-UR joint-position interface**, not general VLA interoperability.
