# Included MuJoCo simulation environments

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