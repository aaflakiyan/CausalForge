# Independent MuJoCo robustness audit — 2026-11-08

This audit tests genuine MuJoCo Menagerie **dual UR5e six-joint cobots**, with the upstream model pinned to `98a70c569c4c1b6da467a20012f766ee1f8dc521`. The code is in `src/vla_council/causalforge/` and checks are in `.github/workflows/mujoco-robustness.yml` and `mujoco-crossseed.yml`. Reports are uploaded from GitHub Actions as JSON artifacts.

## What was added and why

The previous examples mostly ran on ideal position-setpoint delivery and four held-out goals. That is insufficient evidence of controller reliability.

1. **Actual command-transport delay.** The evaluator validates the whole action plan before simulation, then may hold previously sent setpoints by an explicitly specified number of **20 ms control ticks**, while running the normal MuJoCo physics at five substeps per command. The recorded trace preserves intended and physically applied commands separately. Invalid delays are rejected *before any MuJoCo step*.
2. **Independent-seed physical student evaluation.** Each of three seeds trains its own student on eight separate reaching goals, tests it on four unseen goals, and compares it against the scripted IK teacher and a biased under-reaching teacher on exactly the same goals. Reports include success, errors, contacts and seed-block bootstrap uncertainty.
3. **Student versus teacher under matched delays.** The same trained student and IK teacher are evaluated with identical test goals and fixed command latencies. This makes latency sensitivity testable rather than a claim.
4. **Inter-tool distance.** The evaluator now records the minimum measured tool-site separation. Zero collision contacts during spatially separated reaches **does not prove** collision avoidance.
5. **Physical contact versus task success.** Earlier evidence mistakenly defined a teacher as "safe" only when it had also achieved the task goal. The revised evidence stores a conservative *contact-free* flag independent of the separate success measurement. Even contact-free is not a hardware safety certification.

## Verified command-delay stress test

48 genuine MuJoCo physics trials = 12 paired goals (three seeds × four goals) × four delays.

| Setpoint delay | Trials | Mean tool-position error | Reach success, existing 70 mm criterion | Arm–arm contacts | Arm–table contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 ms | 12 | 20.48 mm | 12/12 | 0 | 0 |
| 40 ms | 12 | 24.71 mm | 12/12 | 0 | 0 |
| 100 ms | 12 | 33.02 mm | 12/12 | 0 | 0 |
| 200 ms | 12 | 51.72 mm | 12/12 | 0 | 0 |

The fixed 200 ms command delay increased mean error by approximately **153%** versus zero delay, although all trials still met the pre-existing, fairly generous 70 mm end-point threshold. This is a **fixed transport latency** test, *not* a model of stochastic network jitter, torque lag or physical hardware failure.

## Verified multi-seed teacher/student test

Three independently fitted students, one per seed (41, 42, 43), each with eight training goals and four held-out goals: **12 held-out reaching trials in total per method**.

| Method | Reach success | Mean held-out error |
| --- | ---: | ---: |
| Scripted inverse-kinematics teacher | 12/12 | 20.98 mm |
| Learned ridge-regression student | 12/12 | 20.85 mm |
| Intentionally under-reaching scripted teacher | 0/12 | 198.55 mm |

The student–IK paired difference is **−0.13 mm** on average (student minus IK), with a seed-block resampling interval approximately **−1.15 to +1.03 mm**. **There is no defensible performance superiority claim**: the difference is tiny, and the interval includes zero. The student's held-out trials had **zero recorded contacts**.

**Only 4 of 12** held-out tasks fell within the student model's strict per-feature training range. That indicates its range-based abstention can reject many nearby valid tasks. This heuristic should not be equated with calibrated predictive uncertainty or real-world safety.

The two teachers here were **scripted controllers, not pretrained VLAs**. The trained student is a simple linear shared-encoder/action-head baseline, not a multimodal foundation model.

## Matched student-versus-IK communication latency (completed)

An additional audit trained **one** ridge-regression student on eight measured training goal pairs (seed 42) and tested it on four held-out goals. Both student and reference IK followed the same actual MuJoCo physics and exact command-delay conditions.

| Fixed setpoint delay | Student mean error | Reference IK mean error | Student successes | IK successes |
| --- | ---: | ---: | ---: | ---: |
| 0 ms | 20.09 mm | 21.25 mm | 4/4 | 4/4 |
| 100 ms | 32.58 mm | 33.68 mm | 4/4 | 4/4 |
| 200 ms | 51.36 mm | 52.33 mm | 4/4 | 4/4 |

No arm–arm or arm–table contacts were recorded in these 24 paired policy-condition trials. The differences between policies are small, with only four independent test goals; **this is not evidence that the student is more robust or accurate in general**. Both degrade under delay.

## Reproduce

```bash
pip install -e '.[mujoco]'
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie fetch --depth 1 origin 98a70c569c4c1b6da467a20012f766ee1f8dc521
git -C third_party/mujoco_menagerie checkout FETCH_HEAD
git -C third_party/mujoco_menagerie sparse-checkout set universal_robots_ur5e

python -m vla_council.causalforge.latency_audit \
  --menagerie third_party/mujoco_menagerie --seeds 41 42 43 --goals-per-seed 4 \
  --delay-ticks 0 2 5 10 --out runs/latency.json

python -m vla_council.causalforge.crossseed_audit \
  --menagerie third_party/mujoco_menagerie --seeds 41 42 43 \
  --train 8 --test 4 --out runs/crossseed

python -m vla_council.causalforge.student_latency_audit \
  --menagerie third_party/mujoco_menagerie --seed 42 \
  --train 8 --test 4 --delay-ticks 0 5 10 --out runs/student-latency
```

The reports are machine-readable and include per-trial physics outcomes. Use a Linux/Python environment with the official upstream assets for reproducibility.

## Unresolved scientific and engineering limitations

- Reaching goals are close to one another, with **no grasp, contact manipulation, force regulation, load-sharing or intentional near-collision tests**.
- Actual checkpoint inference from GR00T, π₀ or SmolVLA is **not** performed. Mock native-interface tests do not demonstrate learned VLA skill.
- No domain randomisation of inertial parameters, perception occlusion, realistic sensor jitter, distributed control failure, adversarial disturbances or hardware-in-the-loop verification in these audits.
- These examples use open-loop joint-target chunks, not closed-loop visual correction.
- Small sample sizes and three seeds cannot establish broad generalisation.
- The minimum measured tool separation and zero collision-contact events in these runs do not guarantee collision avoidance in other configurations.

**Evaluation rule:** New benchmark results must be compared with the same seeded goal distribution, equivalent action constraints, matching physics budget, independent train/test goals and both per-condition failure counts and success rates. Never interpret a green CI badge as proof of general-purpose Physical AI intelligence.
