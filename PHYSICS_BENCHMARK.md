# Reproducible physics diagnosis benchmark

## What it establishes

`ood_benchmark.py` compares four probe-selection methods using a paired
scenario set: no probe, a fixed probe, a random probe, and expected information
gain. It evaluates two predeclared physical hypotheses and **two unseen**
configurations. Both NumPy algebraic and MuJoCo physics backends are available.
This is **not** a humanoid benchmark, and it does **not** execute GR00T or pi0.

```bash
python -m vla_council.causalforge.ood_benchmark --backend surrogate --episodes 50 --out runs/ood-surrogate.json
python -m vla_council.causalforge.ood_benchmark --backend mujoco --episodes 3 --out runs/ood-mujoco.json
```

- Every probed episode has the same **0.32-second intervention duration**.
- Probe magnitudes **are not matched**: all methods report mean commanded force.
- Every method receives paired physical conditions and the same observation-noise draws.
- The controller evaluates all candidate forces under the **inferred** posterior;
  the withheld true physics is only used to generate outcome metrics.
- The withheld oracle best action is measured purely as a performance ceiling.
- Predictions and observations use the same simulator implementation but
  different parameters on unseen conditions. This is **not** a test of transfer
  from a simulator to the real world.
- A residual above the configured threshold rejects the model family and
  blocks action selection in this demo. The residual threshold is heuristic,
  **not calibrated** for physical deployment.
- The random baseline's force distribution is not identical to active's.
  This is a remaining confound; reports include force and duration metrics.
- The known hypothesis support is narrow; performance cannot be generalised
  to real contact manipulation, humanoid balance or unknown dynamics families.

## Mandatory interpretation checks

Read both the means and the `per_condition` results. Particularly examine
`unknown_reject_rate`: a low rejection rate means the closed hypothesis model
is untrustworthy on unseen cases, even if known-case accuracy is excellent.
No gain should be claimed unless task loss also improves at a matched physical
intervention budget across multiple independently seeded runs.

## Actual MuJoCo validation requirements

GitHub Actions `mujoco` job installs the **real** `mujoco` Python wheel, executes
physics-specific regression tests, and uploads generated JSON files. A passed
NumPy-only test does not validate the MuJoCo model. Check CI logs for contact,
actuator and inertia regression tests before interpreting physics results.

To run headlessly:

```bash
python -m pip install -e '.[mujoco]'
python -c 'import mujoco; print(mujoco.__version__)'
python -m unittest discover -s tests -v
python -m vla_council.causalforge.ood_benchmark --backend mujoco --episodes 5
```

## Next external validation

1. Confirm the compiled MuJoCo model reaches stable contact and inspect trajectories.
2. Compare independent physics parameters (including friction, sensor delay, actuator gains), not only two predefined failures.
3. Add camera-driven perception and an accessible manipulation benchmark (e.g. RoboSuite Panda) with calibrated action adapters.
4. Run two genuinely pretrained VLA checkpoints and match observation/action frame, timestep, action horizon and robot embodiment.
5. Train one student and perform budget-matched, held-out evaluations across at least five seeds.
6. Treat all real-robot operation as requiring separate hardware risk assessment, hard safety bounds and supervision.