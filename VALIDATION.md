# Validation record

## Scope and environment

Validation in this workspace was performed on the pure-Python / NumPy package. The editable install completed with `python -m pip install --no-build-isolation -e .` (the default isolated build attempted to fetch setuptools from an inaccessible package index). The environment has **no installed MuJoCo**. An attempted `pip install mujoco` failed because the package index could not be reached; it is not evidence of an upstream MuJoCo fault.

## Commands

```bash
python -m unittest discover -s tests -v
python -m vla_council.causalforge.closed_loop --train 800 --test 300 --out runs/closed-loop.json
python -m vla_council.causalforge.suite --seeds 8 --train 400 --test 160 --out runs/suite.json
python -m vla_council.causalforge.benchmark --backend surrogate --trials 100 --out runs/diagnostics.json
```

Tests cover: Bayesian posterior and information gain, missing-evidence abstention, probe-budget enforcement, invalid nonfinite data, agreement of action representations, collision vetoes, causal teacher-selection logic, student train/test separation, repeated-seed determinism and reproducible benchmark reports.

## Interpretation

- Synthetic tests assess *software correctness*, not effectiveness of pretrained VLAs.
- The oracle uses hidden physics and is not an eligible deployed baseline.
- Long contact probes naturally separate the two handcrafted hypotheses well; this does not establish open-world causal discovery.
- Repeated-seed differences are descriptive. Teacher compute, data quantity and probe cost are not fully matched.
- A correct test suite does not imply controller safety.

## Required external verification

On a local workstation or CI runner with the MuJoCo wheel installed, run:

```bash
python -m pip install -e '.[mujoco]'
python -m unittest discover -s tests -v
python -m vla_council.causalforge.demo --true both --out runs/diagnostic-physical.json
python -m vla_council.causalforge.benchmark --backend mujoco --trials 8 --out runs/benchmark-physical.json
```

Then inspect physical plausibility of contact trajectories. Validate mass/inertia updates, actuator force scaling, integration time, contact onset and whether different physical hypotheses are distinguishable over the permitted probe set. Do not infer that the MuJoCo part passed unless these commands actually execute.

## v0.5 validation (7 November 2026)

The current release was installed with
`python -m pip install --no-build-isolation -e .` and verified with:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m vla_council.causalforge.ood_benchmark --backend surrogate --episodes 50 --out examples/reports/ood-surrogate.json
python -m vla_council.causalforge.benchmark --backend surrogate --trials 100
python -m vla_council.causalforge.closed_loop --train 100 --test 50
```

Outcome: **50 test methods discovered; 43 passed, 7 skipped**, comprising
MuJoCo-dependent tests skipped because `mujoco` is not available. `compileall`,
the package installation and all listed surrogate commands completed.
An attempted installation of `mujoco>=3.2,<4` failed because the runtime
cannot resolve PyPI; therefore MuJoCo dynamics, contact onset and integration
**have not been executed locally**. The CI workflow provisions MuJoCo and
executes physical regression tests, but will only run after a remote push.

The unknown-dynamics benchmark is new and deliberately reports a negative
finding: active probing increases diagnosis accuracy but fails to improve task
loss or reject unseen parameter cases reliably. See `RESULTS.md`.


## Verified hosted MuJoCo run (7 November 2026)

The earlier local-wheel blocker was resolved through GitHub Actions: [run 42358370776](https://github.com/aaflakiyan/CausalForge/actions/runs/42358370776) **passed** all four jobs (Python 3.10, 3.11, 3.12 and MuJoCo). The MuJoCo CI artifact contains actual physics-derived diagnostics, comparison results and unknown-dynamics benchmarks. The test runner is not real humanoid hardware. See [MUJOCO_CI_RESULTS.md](MUJOCO_CI_RESULTS.md). Existing statements above about local wheel unavailability remain true of the local workspace, not the hosted CI environment.
