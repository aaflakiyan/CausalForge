# Physics-backed synthetic teacher-to-student benchmark

This benchmark connects the existing causal probe selection and teacher/student
training loop to the **actual MuJoCo dynamics backend**, rather than only the
algebraic surrogate. All teacher actions are fixed candidate forces: **there
is no pretrained VLA inference in this experiment**.

```bash
pip install -e '.[mujoco]'
python -m vla_council.causalforge.physical_distillation --backend mujoco --train 36 --test 24 --seed 42 --out runs/student-mujoco.json
```

It uses disjoint random-number streams for training and evaluation. Only the
two prespecified physical hypotheses appear in training; evaluation also
includes two unknown physical parameter combinations. Each method sees
paired test conditions and identical Gaussian observation noise draws.
The physics run function creates a fresh MuJoCo model and state, avoiding
contact/probe history leakage between candidate action evaluations.

Methods: no probe, fixed probe, random probe and expected information gain
(active probe). Probed methods share a nominal **0.32-second** intervention
budget, but probe force magnitudes differ. A separate supervised ridge
student is trained for each method using synthetic teacher action labels.

Measured outcome: task displacement loss (metres with a separate overshoot
penalty), council loss, action-oracle lower bound, train/test probing cost,
and error on familiar versus unfamiliar dynamics.

Interpretation and limitations:
- A low student loss is *not* proof that the approach works with GR00T,
  OpenPI or any other VLA, and not proof of humanoid performance.
- The model assumes two known hypotheses for selecting probes and actions.
  There is no learned digital twin or calibrated OOD detector in this module.
- The oracle selects among the same finite commands using the withheld true
  physics and is included *only* for measuring the lower bound.
- The students learn to imitate the model-based council, not human corrections.
- Do not compare probe methods without reporting intervention energy and
  computation cost as well as the nominal matched time budget.
- Headless MuJoCo CI produces machine-readable JSON results; review them
  before making any superiority claim.

For actual pretrained-VLA experiments see [RECORDED_VLA_BENCHMARK.md](RECORDED_VLA_BENCHMARK.md)
and [the checkpoint integration milestone](../../issues/2).
