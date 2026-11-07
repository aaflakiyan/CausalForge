# Simulation gallery provenance

This folder contains vector figures that help explain the CausalForge MuJoCo experiments.

| File | Source | Interpretation |
| --- | --- | --- |
| `pusher-trajectories.svg` | Nominal, weak-motor and heavy-payload MuJoCo pusher trajectories | Measured payload position vs time |
| `tabletop-friction.svg` | MuJoCo tabletop push at friction 0.2 and 1.2 | Measured payload position vs time |
| `tabletop-replay.svg` | MuJoCo tabletop position samples | Animated **schematic** showing relative pusher/payload motion |

These are **not screenshots from an interactive MuJoCo viewer**. Curves and animated object positions come from actual simulator state trajectories; geometry is simplified for the README. The animation may appear static depending on GitHub SVG rendering.

Regenerate all three files with:

```bash
pip install -e '.[mujoco]'
python scripts/generate_media.py --out docs/media
```

The generator executes the two MuJoCo environments afresh. No downloaded model weights or GPU is required. This is documentation media for small-scale contact experiments, not evidence of humanoid or pretrained VLA performance.

See [Simulation environments](../../SIMULATION_ENVIRONMENTS.md) and [MuJoCo CI results](../../MUJOCO_CI_RESULTS.md).
