# Reproducible synthetic experiment results

Results below are from `python -m vla_council.causalforge.suite --seeds 8 --train 400 --test 160 --out runs/suite.json`. Each method has its own ridge student; paired test scenarios are shared within each seed. **This is an algebraic model of two preselected physical failures, not a robotics/VLA benchmark.**

| Method | Mean held-out student loss (lower better) | Mean nominal diagnostic probe cost |
|---|---:|---:|
| No probe | 0.03960 | 0.000 |
| Fixed probe | 0.03964 | 0.080 |
| Random probe | 0.03864 | 0.210 |
| Active probe | 0.03795 | 0.350 |
| Oracle (sees true cause) | 0.03795 | 0.000 |

Active-minus-no-probe mean loss: **-0.00165**, active better in 8/8 seeds. Active-minus-random: **-0.00069**, active better in 8/8 seeds. These are *descriptive* comparisons; methods are not cost-matched, and the oracle is not deployable. The effect size is small and the synthetic physics assumptions are favourable to diagnosis.

This benchmark checks integration and supports software engineering decisions. It does **not** establish novelty, practical effectiveness, physical causality, humanoid generalisation, or VLA model superiority. Follow-up work must evaluate multiple realistic tasks, unmodelled failure causes, sim-to-real gaps, matched interaction and compute budgets, and real teacher checkpoints.

## v0.5 — independent unknown-parameter diagnostic benchmark (7 November 2026)

New artifact: [`examples/reports/ood-surrogate.json`](examples/reports/ood-surrogate.json),
seed 2026, 50 episodes per physical condition (200 total per strategy),
with two known and two unseen parameter configurations. One probe per episode,
equal 0.32-second probe duration and different force magnitudes.

| Selection | Known-label accuracy | Unseen-physics rejection | Mean task error (m) | Mean probe force (N) |
| --- | ---: | ---: | ---: | ---: |
| No probe | 50% | 0% | 0.01602 | 0.00 |
| Fixed | 97% | 0% | 0.01707 | 2.00 |
| Random | 99% | 1% | 0.01708 | 3.56 |
| Active | 100% | 2% | 0.01718 | 5.00 |

**Interpretation:** These numbers were produced with the algebraic surrogate,
not MuJoCo or pretrained VLAs. Active probing classifies the predeclared two
hypotheses well, but **does not improve held-out control task loss over the
baselines** in this comparison. The unseen-physics detector is ineffective at
the specified observation noise; this is a negative finding. Probe force differs
between the strategies, so even classification comparisons are not energy-matched.
The no-probe method has zero experimental cost, and is not an appropriate
claim of equivalent information. No statistically significant performance
claim is made; further task families, calibration, repeated seeds and
cost-matched interventions are required.
