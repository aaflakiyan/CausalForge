# Verified two-UR5e MuJoCo results — 7 November 2026

> **Historical benchmark — do not use as evidence of validated active-learning superiority.** The original demonstration used small joint-offset motions with tool sites roughly 1.92 m apart and a known right-side setpoint fault. It did not test shared-workspace manipulation, contact avoidance or pretrained VLA learning. See the independently verified [corrected Cartesian-reaching audit](DUAL_UR5E_AUDIT.md), where active probing is shown to **tie** with a fixed right-arm probe.


## Experiment

- Robots: two official Google DeepMind MuJoCo Menagerie **Universal Robots UR5e** models, facing each other across a central table.
- Source revision: [`98a70c569c4c1b6da467a20012f766ee1f8dc521`](https://github.com/google-deepmind/mujoco_menagerie/commit/98a70c569c4c1b6da467a20012f766ee1f8dc521). Menagerie model licence: BSD-3-Clause.
- Engine: MuJoCo on GitHub Actions Linux runner, Python 3.11, 12 joints and 12 actuators.
- Five scripted joint-space teacher plans, including a fault-compensating right-arm teacher. **No foundation-model checkpoint or trained VLA was executed.**
- Known right-arm tracking factors: 1.0 and 0.45. Out-of-distribution factor: 0.70.
- Trials: **5 independent seeds × 30 paired scenarios = 150 per method**.
- Probe duration: 0.22 s in each probing method; no-probe duration 0.
- Goal error: mean Euclidean tool-site target error over both arms, in metres.
- Recorded inter-arm contact steps: 0 in the tested task family.

## Main comparison

| Selection strategy | Mean tool-target error, mm | Between-seed SD, mm | Probe time |
| --- | ---: | ---: | ---: |
| No probe | 22.00 | 0.82 | 0 s |
| Fixed left probe | 22.00 | 0.82 | 0.22 s |
| Random left/right probe | 17.70 | 1.65 | 0.22 s |
| **Active fault-discriminating probe** | **13.92** | **0.35** | **0.22 s** |

Active probing reduced mean error by **36.7% relative to no probe** and approximately **21.3% relative to random probing** in these scripted and simulated settings. All five seed-level paired differences favour active probing. These are *sample results*, not real-world performance guarantees or statistical proof of humanoid generalisation.

## Critical out-of-distribution finding

In the single-seed 30-case benchmark, the mean tool error was:

| Method | Known physical factors (mm) | Unseen factor 0.70 (mm) |
| --- | ---: | ---: |
| No probe | 21.94 | 23.03 |
| Fixed left probe | 21.94 | 23.03 |
| Random probe | 15.45 | 22.84 |
| Active probe | **10.04** | **22.57** |

The active strategy is substantially better on the **predeclared fault hypotheses**, but gains almost nothing on the **unseen intermediate fault**. This is a central open issue: a two-hypothesis council can diagnose the known cases while remaining overconfident or insufficiently adaptive elsewhere.

## What the validation does and does not establish

**Verified:** both genuine licensed UR5e models compile together; six joints and six actuators per arm; finite tool poses and simulation trajectories; predictable degradation when right-arm tracking is reduced; action critic and posterior selection run on the exact compiled dynamics; paired baseline reports execute; OSMesa renders real MuJoCo camera frames; all added robot-specific tests passed.

**Not verified:** pretrained VLA proposals, grasping or physical object handover, collision recovery (there were no inter-arm contacts), learned dual-arm control, model-to-real transfer, real robotic arm safety or a humanoid student policy. The tracking degradation is applied to **joint position commands**, not a validated emulation of a defective physical motor. The critic uses the same robot dynamics as evaluation except for the controlled right-arm command scaling. Therefore the study is not a strong model-mismatch or real-hardware benchmark.

## Reproduce

Follow [DUAL_UR5E_BENCHMARK.md](DUAL_UR5E_BENCHMARK.md) to fetch the genuine model assets and run the experiment. The automated workflow [`dual-ur5e.yml`](.github/workflows/dual-ur5e.yml) produces:
- `runs/dual-ur5e.json`: per-episode actions, errors, oracle comparisons and fault hypotheses;
- `runs/dual-ur5e-multiseed.json`: five-seed means, deviations and paired differences;
- `runs/dual-ur5e.gif`: **actual rendered MuJoCo camera recording**;
- `menagerie-revision.txt`: precise model source revision.

The repository embeds the same rendered GIF at [`docs/media/dual-ur5e-actual.gif`](docs/media/dual-ur5e-actual.gif). These results remain separate from earlier one-axis pushing results and are not interchangeable.
