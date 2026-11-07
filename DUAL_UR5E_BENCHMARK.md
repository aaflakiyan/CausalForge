# Two facing Universal Robots UR5e arms — MuJoCo validation

> **Historical benchmark — do not use as evidence of validated active-learning superiority.** The original demonstration used small joint-offset motions with tool sites roughly 1.92 m apart and a known right-side setpoint fault. It did not test shared-workspace manipulation, contact avoidance or pretrained VLA learning. See the independently verified [corrected Cartesian-reaching audit](DUAL_UR5E_AUDIT.md), where active probing is shown to **tie** with a fixed right-arm probe.


## Actual robot source
Uses [Google DeepMind MuJoCo Menagerie UR5e](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e), BSD-3-Clause. Downloaded from upstream in CI rather than replacing genuine robot geometry with a hand-made approximated arm. The left base is (-0.73, 0, 0.78) metres, and right base (+0.73, 0, 0.78) metres, rotated 180 degrees around vertical. Their six joints and six actuators each form a 12-DoF system.

The CI job checks out the pinned Menagerie commit `98a70c569c4c1b6da467a20012f766ee1f8dc521`, uses MuJoCo `3.17.0`, and records the source revision in the downloadable artifact.

## Evaluating the proposed method

Five scripted teacher policies propose robot joint trajectories: synchronous, left-first, right-first, conservative, and right-arm compensation (1.9x right-joint motion, which can improve weak actuation but overshoot under normal conditions). These are *not* pretrained VLA teachers. Competing right-arm tracking hypotheses are 1.0 (nominal) and 0.45 (degraded); 0.70 is an unseen configuration.

Compare no probe, a fixed left-arm probe, a randomly chosen left/right probe, and a disagreement-driven probe. All active/fixed/random probes have equal duration and angular amplitude. The council uses its posterior belief to select the lowest expected end-effector error plus inter-arm collision penalty. It never sees the hidden true tracking scale; only oracle metrics use truth.

Reported metrics: left/right tool target error, total inter-arm contact steps, minimum tool separation, probe duration, posterior, chosen teacher, and comparison with an oracle. The same simulation engine underlies predictions and testing; physical model mismatch and trained multi-VLA performance are NOT established. The task does not currently include grasping or force control.

## Reproduce

First install MuJoCo, then retrieve the licensed source model:

    pip install -e '.[mujoco]'
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
    git -C third_party/mujoco_menagerie sparse-checkout set universal_robots_ur5e
    git -C third_party/mujoco_menagerie checkout 98a70c569c4c1b6da467a20012f766ee1f8dc521
    export CAUSALFORGE_MENAGERIE="$PWD/third_party/mujoco_menagerie"
    python -m unittest discover -s tests -p 'test_dual_ur5e.py' -v
    python -m vla_council.causalforge.dual_ur5e --menagerie "$CAUSALFORGE_MENAGERIE" --episodes 6 --duration 1.5 --out runs/dual-ur5e.json

Optional rendered video (requires a working OpenGL/EGL installation):

    pip install imageio pillow
    MUJOCO_GL=egl python -m vla_council.causalforge.dual_ur5e --menagerie "$CAUSALFORGE_MENAGERIE" --episodes 1 --out runs/preview.json --render runs/dual-ur5e.gif

For verified five-seed numbers, including the poor unseen-fault result, see [DUAL_UR5E_RESULTS.md](DUAL_UR5E_RESULTS.md). The robot recording is in [docs/media/dual-ur5e-actual.gif](docs/media/dual-ur5e-actual.gif). Follow the GitHub Actions benchmark artifact for the raw JSON. The model does not represent UR16e specifically, and this is not a real-hardware safety assessment.
