# Included MuJoCo simulation environments

## Dual UR5e Cartesian reach (corrected, physically validated)

**Two actual Universal Robots UR5e manipulators**, each with six revolute joints, are assembled from Google DeepMind's MuJoCo Menagerie at a pinned upstream revision. The fixed workbench, two coloured 3D goal markers, facing mount frames and camera are included as an inspectable scene description in [`assets/mujoco/dual_ur5e_reach_arena.xml`](assets/mujoco/dual_ur5e_reach_arena.xml).

The complete articulated UR5e XML and meshes come from the licensed upstream model; they are not silently replaced by simplified cylinders. [`dual_ur5e_reach.py`](src/vla_council/causalforge/dual_ur5e_reach.py) attaches those models to the included arena using MuJoCo MjSpec. A regression test checks that the visible XML exactly matches the scene builder.

[Watch the verified 3-second dual-arm Cartesian reach](docs/media/dual-ur5e-cartesian-reach.gif). Each tool moves approximately **0.94 m** and finishes approximately **0.021 m** from its specified 3D target, with no recorded arm–arm or arm–table contact. A second [recovery benchmark](DUAL_UR5E_AUDIT.md) evaluates script selection under an artificial right-arm command-setpoint error and includes the essential fixed-right diagnostic baseline, which **ties the active method**.

```bash
pip install -e '.[mujoco]'
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie fetch --depth 1 origin 98a70c569c4c1b6da467a20012f766ee1f8dc521
git -C third_party/mujoco_menagerie checkout FETCH_HEAD
git -C third_party/mujoco_menagerie sparse-checkout set universal_robots_ur5e
python -m vla_council.causalforge.dual_ur5e_reach --menagerie third_party/mujoco_menagerie --trace --out runs/dual-ur5e-reach.json
python -m vla_council.causalforge.dual_ur5e_recovery --menagerie third_party/mujoco_menagerie --episodes 18 --out runs/dual-ur5e-recovery.json
```

This is **independent Cartesian reaching**, not grasping or cooperative object manipulation, and neither controller has a pretrained VLA checkpoint. The GitHub Actions audit runs these checks and renders actual MuJoCo video frames.

---


## 1. CausalForge pusher (implemented, reproducible)

The repository includes [assets/mujoco/pusher.xml](assets/mujoco/pusher.xml), the **exact MJCF scene** used by `mujoco_push.py`, `demo.py`, the causal/OOD benchmarks and student-distillation experiments. It is a one-axis motorised pusher that contacts a one-axis sliding payload. The simulation deliberately uses zero gravity and slider damping: **this is not an ordinary table-contact/friction benchmark, a full robot arm, or a humanoid**.

No external asset downloads are required.

Install and execute:

```bash
python -m pip install -e '.[mujoco]'
python -m vla_council.causalforge.simulate_scene --out runs/example --force 7 --duration 0.8
```

Outputs `runs/example/trajectory.csv` (simulated joint positions, velocities and commands) and `runs/example/summary.json` (parameters, limits and measured displacements). Change `--motor-scale 0.42` or `--payload-mass 3` to simulate different model hypotheses. Use `--xml runs/example/pusher.xml` to export an identical copy of the scene.

For a sequence of rendered PNG frames, install `imageio` and set up a working OpenGL/EGL context before running with `--render`:

```bash
python -m pip install imageio
MUJOCO_GL=egl python -m vla_council.causalforge.simulate_scene --render --out runs/frames-demo
```

The rendering path is optional and may require graphics/EGL drivers. CI exercises the physics calculations and trajectory output, **not guaranteed headless GPU rendering**.

## 2. Synthetic teacher council (no MuJoCo required)

`vla_council.sim` and the council/student examples are numerical tasks, not robot scenes. Their synthetic outcomes should not be presented as physics validation.

## 3. Humanoid and dual-arm manipulation (not yet included)

This version does **not** contain a full humanoid MJCF, a Panda manipulation task, pretrained VLA weights, or converted GR00T/pi0 checkpoints. Adding a realistic robot environment requires compatible robot meshes, model licences, controllers, observational camera setup and action normalization. Do not describe the existing pusher as a humanoid test.

## Test the exact scene

```bash
python -m unittest discover -s tests -v
python -m vla_council.causalforge.physical_distillation --backend mujoco --train 36 --test 24 --out runs/student-mujoco.json
```

The `mujoco` GitHub Actions job executes these tests on CPU and retains JSON outputs as artifacts. Comparison between two simulated physical hypotheses does not constitute sim-to-real validation, and known noise/model assumptions can make active information-gain probing appear artificially effective.
## 4. Tabletop pushing with gravity and friction (experimental)

[assets/mujoco/tabletop_push.xml](assets/mujoco/tabletop_push.xml) is a **second** MJCF model with a gravity-loaded free-body payload on a plane and an actuated sliding pusher. It is closer to ordinary contact manipulation than the first zero-gravity model, but still has only one actuated degree of freedom. It is not a humanoid or Panda manipulation benchmark.

```bash
python -m vla_council.causalforge.tabletop_push --friction 0.2 --out runs/tabletop/low
python -m vla_council.causalforge.tabletop_push --friction 1.2 --out runs/tabletop/high
```

The commands export a `trajectory.csv` and `summary.json` per condition. GitHub's **MuJoCo tabletop physics** workflow runs the physics integration tests and uploads both conditions as artifacts. The high/low friction results are a *sanity comparison*, not trained-policy evaluation.

