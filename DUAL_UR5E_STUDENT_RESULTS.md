# Dual-UR5e distilled plan student — held-out MuJoCo result

> **Historical benchmark — do not use as evidence of validated active-learning superiority.** The original demonstration used small joint-offset motions with tool sites roughly 1.92 m apart and a known right-side setpoint fault. It did not test shared-workspace manipulation, contact avoidance or pretrained VLA learning. See the independently verified [corrected Cartesian-reaching audit](DUAL_UR5E_AUDIT.md), where active probing is shown to **tie** with a fixed right-arm probe.


The small student learns **which scripted joint-space plan to execute**, based only on goal magnitude and the council's posterior belief about right-arm tracking. It is not a VLA and does not generate continuous 12-DoF trajectories.

## Experimental protocol

- Real UR5e models from MuJoCo Menagerie, two opposed six-joint arms.
- Teacher choices: five scripted plans, including one compensating for weakened right-arm position tracking.
- Student: ridge multi-class classifier over [bias, posterior nominal, goal magnitude, interaction].
- Student training: 36 scenarios total; 24 known-condition examples after excluding unseen right tracking factor 0.70.
- Held-out test: 18 separate scenarios with a different seed; includes known factors 1.0 and 0.45 and unseen factor 0.70.
- The student never sees the hidden physical-condition label as an input.
- All three probing methods use the same 0.22-second probe duration and angular magnitude.
- Actual teacher and student outcomes are evaluated by **MuJoCo rollout**, not inferred from text or from an oracle.

## Verified held-out result (seed 71)

| Method | Student mean tool-position error | Council error | Council plan agreement |
| --- | ---: | ---: | ---: |
| No probe | 24.70 mm | 24.70 mm | 100% |
| Fixed left probe | 24.70 mm | 24.70 mm | 100% |
| Random probe | 20.20 mm | 20.20 mm | 100% |
| Active probe | **15.79 mm** | **15.79 mm** | 100% |

For the active-probe student, the measured errors are **14.24 mm** on known conditions and **18.89 mm** on unseen factor 0.70.

These results show that the *simple discrete council plan rule* is representable by the small classifier on this test split. The 100% agreement does **not** imply that the system learned robotic motor skills or discovered a new policy representation. A larger and more diverse task distribution is required.

## Caveats

The controller and teacher choices are **scripted**, not GR00T, pi0, OpenVLA or any pretrained VLA. No high-rate joint policy is trained. The simulated degradation rescales the commanded right-arm joint-position delta; it does not model a physical motor failure or sensor fault comprehensively. There is no demonstrated task involving coordinated grasping, handover, object contact or collision recovery. World-model mismatch is restricted to a chosen out-of-hypothesis tracking scale.

The student trains to reproduce the council, so it cannot automatically outperform the teacher on the same task distribution. Broad claims of scientific novelty or sim-to-real robustness are not warranted by these results.

## Reproduction

The dedicated [dual UR5e workflow](.github/workflows/dual-ur5e.yml) runs:
```bash
python -m vla_council.causalforge.dual_ur5e_student \
  --menagerie "$CAUSALFORGE_MENAGERIE" \
  --seed 71 --train 36 --test 18 --duration 1.0 \
  --out runs/dual-ur5e-student.json
```

The workflow uploads the raw student results together with the robot simulation and multiseed teacher comparisons. Model assets and MuJoCo runtime are pinned in the workflow.
