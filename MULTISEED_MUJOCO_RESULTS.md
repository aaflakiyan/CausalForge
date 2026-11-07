# Five-seed MuJoCo student benchmark — 7 November 2026

GitHub Actions: [successful experimental run](https://github.com/aaflakiyan/CausalForge/actions/runs/42359928732). Artifact `mujoco-results/student-mujoco-aggregate.json` and individual seed JSON files.

Configuration: MuJoCo physics backend; 36 training and 24 test examples per seed; seeds 41, 42, 43, 44 and 45; noise SD 0.015; ridge penalty 0.025. Same scene distributions, seeds and outcomes across methods within a seed. Teacher actions are three **synthetic fixed-force controls**, not real VLA checkpoints.

| Probe method | Mean held-out student task loss | SD over 5 seeds |
| --- | ---: | ---: |
| No probe | 0.031610 | 0.002109 |
| Fixed | 0.030289 | 0.002651 |
| Random | 0.030809 | 0.003145 |
| Active information gain | 0.030289 | 0.002651 |

The active-minus-fixed **paired** difference is `0.000000` on all five seeds.

This is a **negative result for the purported advantage of active probing**, not a success claim. The fixed probe is already sufficient for the discrete council to select the same actions in this task, or the active criterion has insufficient discriminatory power; the current benchmark cannot distinguish these possibilities. Mean losses are weighted displacement errors and not generalised robot task success. Physics probes match intervention time but not force/energy. One-axis manipulation and a small sample cannot establish humanoid robustness or statistical significance.

Next work: redesign hypotheses/observations to make probes selectively informative, evaluate matched control effort, then integrate **actual model checkpoints** using the documented VLA recording and action-space protocol. Re-run paired multi-seed comparisons and report confidence intervals on real task success before making claims about Physical AI improvements.
