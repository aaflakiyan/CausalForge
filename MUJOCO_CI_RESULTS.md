# Verified MuJoCo CI results — 7 November 2026

GitHub Actions run: https://github.com/aaflakiyan/CausalForge/actions/runs/42358370776

Source commit tested: `f42f46f3eddf64781a7b6ac93e7e66860d76d610`.

All four jobs succeeded: Python 3.10, 3.11 and 3.12 core checks, plus a MuJoCo 3.x job. The MuJoCo job executes the physics-dependent tests and produces the `mujoco-results` workflow artifact containing `diagnostic.json`, `benchmark-mujoco.json` and `ood-mujoco-ci.json`.

## Four-trial controlled diagnosis comparison

| Probe method | Known-fault accuracy (unconditional) | Mean probe cost | Mean task goal error (m) |
| --- | ---: | ---: | ---: |
| Active | 4/4 | 0.2500 | 0.008812 |
| Random | 3/4 | 0.1825 | 0.008812 |
| Fixed | 0/4 (abstained) | 0.0800 | 0.012837 |
| No probe | 0/4 (abstained) | 0 | 0.012837 |

The four trials are too few to establish superiority. Active and random produce **identical mean task error** in this experiment. Costs are not controlled to be equal.

## Out-of-distribution physics comparison

This benchmark uses three episodes per condition across two declared and two unseen dynamics configurations (12 episodes per method). The simulated system is **a one-axis contact pusher**, not a humanoid.

| Method | Accuracy on known conditions | Rejection rate on unseen conditions | Mean task loss (m) | Mean probe force |
| --- | ---: | ---: | ---: | ---: |
| No probe | 50% | 0% | 0.01980 | 0 |
| Fixed | 100% | 0% | 0.01795 | 2.000 |
| Random | 100% | 0% | 0.01795 | 3.625 |
| Active | 100% | 0% | 0.01795 | 2.000 |

There is **no improvement over the fixed probe in task loss or unseen-physics rejection** in this run. This failure to recognise unfamiliar dynamics is a major open research problem, not a solved feature.

## Interpretation and limitations

- The workflow validates MuJoCo installation and execution, basic contact/dynamics simulation, tests, and reproducible outputs on a GitHub-hosted Linux runner.
- The simulator used to generate observations also underlies the model's predictions; true sim-to-real mismatch is untested.
- The toy teachers are candidate controls, **not** GR00T, pi0, or learned humanoid policies.
- No pretrained VLA model was executed; no pretrained teacher-to-student distillation or humanoid benchmark was validated.
- Training and evaluation must be extended to matched intervention budgets, unseen object geometries, model mismatch, independent seeds, and actual robot/teacher models before performance claims can be made.

The intended next milestone is a reproducible two-teacher manipulation benchmark on one compatible embodiment. Report both task success and the physical cost of diagnosis.
