# Verified MuJoCo student-distillation pilot — 7 November 2026

**Source:** GitHub Actions run [42359685941](https://github.com/aaflakiyan/CausalForge/actions/runs/42359685941), artifact `mujoco-results/student-mujoco-ci.json`.

Command:

```bash
python -m vla_council.causalforge.physical_distillation --backend mujoco --train 36 --test 24 --out runs/student-mujoco-ci.json
```

Independent simulation resets, 36 training examples and 24 evaluation examples, seed 42; two trained-on physics conditions and two unseen combinations. **This uses synthetic force teachers, not pretrained VLAs.**

| Student method | Mean task loss | Known-physics task loss | Unseen-physics task loss | Test probe cost |
| --- | ---: | ---: | ---: | ---: |
| No probe | 0.029922 | 0.024612 | 0.037355 | 0.00 |
| Fixed | 0.027530 | 0.021421 | 0.036083 | 0.32 |
| Random | 0.027530 | 0.021421 | 0.036083 | 0.32 |
| Active information-gain | 0.027530 | 0.021421 | 0.036083 | 0.32 |

The evaluation-only best candidate action, selected using withheld true physics, has mean loss **0.022514**. This is a lower bound, **not** a deployable controller.

**Interpretation:** The pilot demonstrates that the end-to-end teacher-to-student software can run against MuJoCo dynamics. It does not demonstrate that active probing is better than fixed or random probing. Equal losses may reflect high signal-to-noise, equivalent posterior decisions or a poor discriminative probe set; diagnose this before adding stronger claims.

**Limitations:** only one seed and 24 test episodes; one-axis contact benchmark with joint damping rather than real Coulomb sliding friction; three fixed force candidates; probe costs match time but not force/energy; synthetic training labels; no pretrained models; no trained humanoid, safe real-robot execution, or statistical significance claim.

**Next required experiment:** multiple seeds and environments with ambiguous hypotheses, delayed/noisy tactile observations, OOD dynamics, cost-matched force/time, and real checkpoint inference using [the recording protocol](RECORDED_VLA_BENCHMARK.md). Only move to full humanoid control after a manipulation study shows a clear advantage over matched baselines.
