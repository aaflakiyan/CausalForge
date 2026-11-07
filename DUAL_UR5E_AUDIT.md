# Dual-UR5e independent audit and corrected reaching benchmark

**Status:** Simulation tests executed successfully on GitHub Actions using genuine MuJoCo Menagerie UR5e models. **No pretrained VLA, grasping, or real robot validation.**

## Why the previous demonstration was inadequate

The legacy `dual_ur5e.py` benchmark specified targets as small joint-angle offsets rather than independent physical goals. The observed initial tool sites were `(-0.864, +0.492, 1.268)` and `(+0.864, -0.492, 1.268)` metres, nearly **1.92 metres apart**. The robot geometry is real, but the arms barely moved toward a shared task; zero inter-arm contacts in such a scenario cannot establish collision avoidance. The previous GIF demonstrated very little.

Furthermore, the earlier comparison simulated a *right-arm setpoint scale*, not a torque or mechanical motor failure. These facts were not sufficiently clear in the original presentation.

## Replacement demonstration: independent Cartesian targets

We define two target positions in world coordinates and solve **position-only damped least-squares inverse kinematics**, rather than manufacturing goals from a joint-offset command.

| Property | Measured on GitHub MuJoCo runner |
|---|---:|
| Left target (m) | (-0.25, 0, 0.73) |
| Right target (m) | (+0.25, 0, 0.73) |
| Initial distance from each goal | 0.953 m |
| Actual end-effector travel per arm | **0.938 m** |
| Final error per arm | **0.0214 m** |
| Inverse-kinematics final residual | 0.00517 m |
| Arm–arm contact frames | **0** |
| Arm–table contact frames | **0** |
| Command duration | 3.0 s |

This is a successful **dual-arm Cartesian reach** to two separate points above a shared table. It is not a manipulation, handover, load-sharing, or collision-avoidance challenge. Zero contacts do not prove protection against deliberately colliding trajectories.

The GIF `docs/media/dual-ur5e-cartesian-reach.gif` is rendered from actual MuJoCo camera frames, not a schematic. The old `dual-ur5e-actual.gif` remains only as a historical recording of the inadequate motion.

## Fault-aware teacher selection on this corrected task

We compared two scripted controllers:
- **Nominal:** sends unmodified joint position setpoints.
- **Compensating:** multiplies the right-arm joint setpoint displacement by `1/0.55`.

The simulator independently attenuates delivered right-arm joint setpoints by either `1.0` (nominal), `0.55` (known fault), or `0.72` (unseen fault). This is **not a torque fault or realistic hardware health diagnosis**; it is a controlled command-calibration error. Each method sees the same task and sensor noise.

18 paired cases, one seed, probe cost 0.28 seconds:

| Probe method | Mean tool error | Full two-arm reach success |
|---|---:|---:|
| None | 144.09 mm | 33.3% |
| Fixed left (uninformative) | 144.09 mm | 33.3% |
| Random left/right | 128.29 mm | 38.9% |
| **Fixed right** | **59.10 mm** | **66.7%** |
| **Active** | **59.10 mm** | **66.7%** |

A critical negative result: **active probing equals a fixed right-arm probe** because the fault can only affect the right arm. Thus this benchmark does **not** establish any advantage for active experiment selection. Unseen-condition error remains around **135.8 mm** for both methods; the model has no explicit unknown-fault hypothesis.

The result supports a much narrower statement: detecting right-arm command attenuation can help select a compensating controller when that fault was anticipated.

## Reproduce

```bash
pip install -e '.[mujoco]'
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie sparse-checkout set universal_robots_ur5e
export CAUSALFORGE_MENAGERIE="$PWD/third_party/mujoco_menagerie"
python -m unittest discover -s tests -p 'test_dual_ur5e_reach.py' -v
python -m unittest discover -s tests -p 'test_dual_ur5e_recovery.py' -v
python -m vla_council.causalforge.dual_ur5e_reach --menagerie "$CAUSALFORGE_MENAGERIE" --out runs/reach.json
python -m vla_council.causalforge.dual_ur5e_recovery --menagerie "$CAUSALFORGE_MENAGERIE" --episodes 18 --out runs/recovery.json
```

To render the actual MuJoCo scene (Linux with OSMesa):

```bash
MUJOCO_GL=osmesa python -m vla_council.causalforge.dual_ur5e_reach --menagerie "$CAUSALFORGE_MENAGERIE" --render runs/dual-ur5e-cartesian-reach.gif
```

The independent GitHub Actions workflow is `.github/workflows/dual-ur5e-audit.yml`. The original Menagerie geometry and meshes are downloaded at runtime; third-party model licensing remains with MuJoCo Menagerie.

## Research conclusions and missing tests

**Validated engineering:** two real UR5e models, position-only IK, dynamic control, substantial motion, collision-contact recording, a repeatable physical fault surrogate, and measured tool accuracy.

**Not validated:** obstacle avoidance under deliberately unsafe plans; cooperative grasping or object handover; genuine faulty torque/actuator dynamics; pretrained VLA policies; student training from physical demonstrations; cross-simulator transfer or physical hardware safety. The active probe has not beaten the right-arm fixed baseline.

A stronger research benchmark should distribute faults across arms and multiple actuator modes, use held-out combinations and cost-matched diagnostic strategies, explicitly reject unknown physics, and compare to realistic learned policies on an object-interaction task.
