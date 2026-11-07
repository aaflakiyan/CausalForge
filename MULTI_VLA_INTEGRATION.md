# Multi-VLA Integration — OpenPI, GR00T and LeRobot

CausalForge now provides a **tested adapter boundary** for three upstream VLA inference API families. This is not a bundled set of model checkpoints and does not claim pretrained foundation-model inference has been run.

The adapters operate on an already loaded native policy or its own remote client, with no vendor packages imported by CausalForge. Use the real libraries in a separate GPU environment when available.

## Supported upstream inference signatures

| Family | Documented upstream call | Output expected by this adapter |
| --- | --- | --- |
| [OpenPI](https://github.com/Physical-Intelligence/openpi) | `policy.infer(inputs)` | `{"actions": [T,D]}` |
| [NVIDIA GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `policy.get_action(inputs)` | `({"stream_name": [1,T,D]}, info)` |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) | `policy.predict_action_chunk(batch)` or `policy.select_action(batch)` | Tensor `[1,T,D]`, or `{"action": tensor}` |
| Custom Python callable | `policy(inputs)` | `[T,D]` |

The adapter **does not infer a model's robot embodiment or action semantics**. Many default checkpoints produce action vectors for robots other than UR5e, with 7-DoF manipulators, grippers, Cartesian deltas, normalised actions or different frequencies. Such outputs **cannot** run on the dual-UR5e simply because their arrays have a familiar shape.

A policy must produce an explicitly declared joint-space `ActionSpec`, with the same twelve independent UR5e joints and 20 ms action period, or be wrapped in a **separately validated** robot-to-robot conversion implementation. No unsupported conversion is performed.

## Example: OpenPI policy client

The following code runs after you provide an OpenPI client already configured for **dual-UR5e absolute joint-position output**. A stock Pi0 model trained on a different embodiment will not meet that contract without dedicated fine-tuning and action transformations.

```python
import causalforge
from vla_council.causalforge.vla_bridge import (
    ExplicitActionMapping, VLABackend,
)

native_client = ...  # OpenPI WebsocketClientPolicy, or a loaded OpenPI Policy

def preprocess(obs):
    # Use your checkpoint's required images, state, language and normalisation.
    # This must be built/tested for the specific trained checkpoint.
    return ...

backend = VLABackend(
    name="my_openpi_ur5e",
    family="openpi",
    client=native_client,
    mapping=ExplicitActionMapping(causalforge.SPEC),
    observation_adapter=preprocess,
)

report = causalforge.evaluate(
    backend,
    robot="ur5e-dual",
    task="reach",
    menagerie="/path/to/mujoco_menagerie",
)
```

**If your backend produces a permuted joint order in degrees**, specify its exact source contract and explicitly enable conversion. Axis names must match the destination set exactly.

```python
from dataclasses import replace
from vla_council.causalforge.vla_bridge import ExplicitActionMapping

source = replace(
    causalforge.SPEC,
    ordered_axes=tuple(reversed(causalforge.SPEC.ordered_axes)),
    units=("deg",) * 12,
)
mapping = ExplicitActionMapping(source, degrees_to_radians=True)
```

This performs a verified permutation and degree-to-radian conversion, *not* a learned or heuristic interpretation of VLA outputs.

## Explicit action-frequency conversion

Many VLA policies run at a different frequency from the 50 Hz UR5e command interface. There is now a **strict opt-in** linear interpolation for absolute joint targets at a slower period that is an exact integer multiple of the 20 ms controller tick.

```python
from dataclasses import replace
from vla_council.causalforge.vla_bridge import ExplicitActionMapping

source = replace(causalforge.SPEC, dt=0.10)  # source emits 10 Hz
mapping = ExplicitActionMapping(source, resample_from_current=True)
```

To generate 50 Hz commands, the adapter interpolates each original 10 Hz setpoint from the *measured current joint configuration* or previous setpoint. It never silently repeats, truncates or extrapolates a model action. A 10-step source chunk yields 50 controller ticks over exactly one second.

Unsupported timing ratios, missing current state and unauthorised conversion fail before execution. The resulting commands must still pass the MuJoCo policy validator's joint, actuator and **2 rad/s command-rate** limits.

**This is command interpolation, not proof that a lower-frequency VLA has appropriate closed-loop dynamics, latency or physical safety.** It does not support resampling Cartesian increments, torque or velocity controls. It should be evaluated on each robot/task separately.

## Combining two single-arm teachers

A GR00T policy for the left arm and a different policy for the right arm may be composed **only if** both produce [T,6] trajectories for the correct respective UR5e model in joint frame, at the same control rate and for the same number of steps.

Use `SingleArmMapping` and `DualArmComposer`. Missing arms cannot be silently padded. Different action horizons are rejected rather than secretly truncated, repeated or interpolated.

## Comparing teachers fairly

```python
from vla_council.causalforge.vla_bridge import evaluate_teachers

result = evaluate_teachers(
    {"openpi": openpi_backend, "groot": groot_backend},
    menagerie="/path/to/mujoco_menagerie",
)
```

Every policy is simulated independently from a fresh robot reset. Results include actual task error, collisions and a list of rejected teacher outputs.

**Important:** `offline_best_by_goal_error` is an oracle using measured results and must **not** be used as an online supervisor. A true council requires an online confidence/physics model that does not observe future outcomes, then matched-budget prospective evaluation.

## Validation and limitations

```bash
pip install -e '.[mujoco]'
python -m unittest discover -s tests -p 'test_vla_bridge.py' -v
```

The CI workflow runs mock OpenPI/GR00T/LeRobot inference signatures, checks unit/frame/robot/axis/timestep/horizon contracts, executes two mock teachers on genuine Menagerie dual-UR5e physics, and verifies both satisfy the reaching acceptance criteria. **The mock backends reuse a known IK trajectory. They are interface checks, not accuracy evaluations or demonstrations of pretrained models.**

No validated image-preprocessing pipelines, native GR00T/OpenPI/LeRobot dependencies or model weights are distributed with this project. Model-specific input transform, normalisation, calibration, inverse kinematics, control latency and gripper support must be provided by the integrator and separately tested.
