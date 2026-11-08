# CausalForge MuJoCo Lab — complete dual-UR5e simulation kit

This bundle contains **the actual CausalForge source snapshot, its inspected MJCF arena, the official pinned MuJoCo Menagerie UR5e XML and meshes, a recorded example GIF, launchers for Windows and Linux, and measured sample reports**.

Both robots have six articulated UR5e joints. The arena includes facing mounts, a workbench, coloured 3D targets, floor and an overview camera. The model is dynamically assembled by MuJoCo's `MjSpec.attach()`. Do **not** open the arena XML on its own and expect the robots to appear: the assembly code attaches the real UR5e robots from the bundled Menagerie model.

## Quick start: Windows 10/11

Install **Python 3.11 or 3.12 (64-bit)** and ensure the `py` launcher works. The first setup requires internet access for Python packages; the robot meshes and CausalForge source are **already included**, so no GitHub authentication is required.

Extract the ZIP to a normal directory (e.g. `C:\Robotics\CausalForge-MuJoCo-Lab`), open PowerShell in that directory and run:

```powershell
.\run.ps1 check
.\run.ps1 view
.\run.ps1 render
```

If PowerShell blocks scripts, use `powershell -ExecutionPolicy Bypass -File .\run.ps1 check` for that command, or run the Python commands directly as described below. Do not change system-wide script policies.

To run automated tests or more experiments:

```powershell
.\run.ps1 tests
.\run.ps1 stress
.\run.ps1 student
.\run.ps1 all
```

## Quick start: Linux

Use Python 3.11 or 3.12, Python venv, and a graphical desktop for the viewer. On Ubuntu, install required packages:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libgl1 libosmesa6 libosmesa6-dev
```

Extract, then:

```bash
bash run.sh check
bash run.sh view
bash run.sh render
bash run.sh tests
```

To run offscreen rendering on Linux/CI, use `MUJOCO_GL=osmesa bash run.sh render`. The GUI viewer needs an actual display; do not use the OSMesa setting for `view`.

On a remote/headless machine, use `check`, `render`, `stress`, `student`, and `tests` rather than `view`.

## Commands and output

| Command | What is genuinely executed | Output |
| --- | --- | --- |
| `check` | Assemble actual dual-UR5e model, validate 12-joint action chunk, simulate the Cartesian reach | `results/reach/metrics.json`, `report.html` |
| `view` | Show both moving robots in the interactive MuJoCo GUI at actual physics rate | Interactive 3D window |
| `view --repeat` | Loop the physical reaching motion for inspection | Interactive 3D window |
| `render` | Render real MuJoCo camera frames and encode GIF | `results/render/dual_ur5e_reach.gif`, `render_metrics.json` |
| `tests` | Run repository's exact physics and unsafe-action rejection tests | Terminal pass/fail |
| `stress` | Run fixed command delays (0/40/100/200 ms), with paired goals | `results/stress/latency_stress.json` |
| `student` | Train on actual simulated scripted-controller outcomes and execute student on unseen goals | `results/distillation/student/` |
| `all` | Run check, render, stress and student | All above reports and media |
| `stress --full` | Three seeds × four goals × four delays = 48 physics trials | Larger stress report |
| `student --full` | Ten train + four held-out UR5e goal pairs | Full student experiment |

For a different output folder, add `--out PATH`. For example, `python CausalForge/tools/mujoco_lab.py check --out my_results`.

## Without the wrapper scripts

Create and activate a Python virtual environment, then install the bundled project:

```bash
python -m pip install -e "./CausalForge[mujoco]"
python -m pip install "imageio>=2.34" pillow
python CausalForge/tools/mujoco_lab.py check
python CausalForge/tools/mujoco_lab.py view
```

The simulator executable reads `./mujoco_menagerie/universal_robots_ur5e/ur5e.xml` from the extracted bundle. It never requires the model weights of GR00T, π₀ or other pretrained VLAs.

## Examine the actual simulation scene

- Arena XML: `CausalForge/assets/mujoco/dual_ur5e_reach_arena.xml`
- MuJoCo assembly, IK and contact logic: `CausalForge/src/vla_council/causalforge/dual_ur5e_reach.py`
- Action validation and physics evaluator: `CausalForge/src/vla_council/causalforge/dual_ur5e_policy.py`
- Interactive viewer, renderer and command-line front-end: `CausalForge/tools/mujoco_lab.py`
- Official UR5e XML and meshes: `mujoco_menagerie/universal_robots_ur5e/`
- Example real MuJoCo recording: `examples/dual_ur5e_reach.gif`
- Example CI result: `examples/dual_ur5e_reach_metrics.json`

To adjust physical goals in code, use `CausalForge/src/vla_council/causalforge/dual_ur5e_reach.py`, but be aware that the coloured scene target markers are fixed in the current XML. For tests with different goals, `stress` and `student` change the actual tool goals; they do not dynamically relocate the static decorative markers. The controller and metrics always use the requested numeric targets.

## What the tests prove and what they don't

The validated experiment uses **actual MuJoCo rigid-body physics**, official robot geometry, a damped least-squares Cartesian inverse-kinematics controller and simulated 12-actuator joint-position control. It includes position error, collision-contact counters and measured state traces.

The code rejects wrong robot identity, joint ordering, degrees versus radians, wrong reference frame, incompatible timing, unsafe joint setpoints, and excessive command discontinuity **before** physics executes.

This is **not** a real pretrained VLA test, a robot-hand grasp, a proven collision-avoidance controller or a validated real-world UR hardware driver. The scene's two coloured targets represent independent reaching, not object handover.

## Version, licensing and provenance

The included `CausalForge` folder is a **snapshot of this GitHub Actions run**. `BUILD_MANIFEST.json` stores the source commit and pinned robot revision. The official robot assets are sourced from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at revision `98a70c569c4c1b6da467a20012f766ee1f8dc521`. Preserve the bundled source and model licences if redistributing.

Python dependencies must be installed for the user's OS and architecture. This ZIP does not ship pretrained VLA checkpoints, proprietary firmware or a Python interpreter. It is self-contained **with respect to CausalForge source and the physical UR5e model assets**.

Troubleshooting: if a GUI does not open, try `check` first and verify that a desktop display and working graphics drivers exist. If offscreen rendering fails on Linux, install `libosmesa6` and set `MUJOCO_GL=osmesa` before starting Python. If pip fails, check Python version and internet access. On Windows, use the native viewer rather than OSMesa.
