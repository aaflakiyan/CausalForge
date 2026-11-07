# CausalForge

> **Validation status (v0.5):** Core Python tests passed locally and on GitHub Actions.
> MuJoCo simulation tests and controlled physics benchmarks have now **passed in GitHub CI**
> ([results and limitations](MUJOCO_CI_RESULTS.md)). No real VLA checkpoints, humanoid
> policies, or physical-robot tests have been evaluated.


**Evidence-driven multi-teacher learning for Physical AI**

[![Tests](https://github.com/aaflakiyan/CausalForge/actions/workflows/tests.yml/badge.svg)](#verification) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml) [![Licence](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

> **Research prototype, not a humanoid foundation model.** The executable teacher/student examples are synthetic. No GR00T, OpenPI, SmolVLA, real robot deployment, or safety certification is included. The MuJoCo integration is separately installable and has passed hosted GitHub Actions tests; it remains unverified on real hardware.

## Reproduce and inspect the MuJoCo scenes

The **exact** one-axis MJCF pusher model is checked in at [assets/mujoco/pusher.xml](assets/mujoco/pusher.xml). See [SIMULATION_ENVIRONMENTS.md](SIMULATION_ENVIRONMENTS.md) for model limitations and reproduction commands. This is a constrained slider with zero gravity, **not** a Panda or humanoid environment.

```bash
python -m pip install -e '.[mujoco]'
python -m vla_council.causalforge.simulate_scene --out runs/scene --force 7 --duration 0.8
python -m vla_council.causalforge.rollout_actions examples/pusher_policy_chunk.json --out runs/policy-replay.json
```

The scene command produces trajectory CSV and summary JSON; `--render` additionally requires `imageio` and a working OpenGL context. The recorded-action replay validates actuator semantics (unit, frame, timestep, bounds) before applying commands to the MuJoCo model. The example is **synthetic**: no GR00T or pi0 inference has been run.

## Research question

Can *testable disagreements* among specialised robot policies be converted into cheap, targeted physical experiments, providing better supervision for a compact student policy than standard imitation or unverified model voting?

CausalForge is built around four boundaries:

1. **Teacher proposals:** heterogenous policies must output validated, embodiment-compatible action chunks.
2. **Independent criticism:** physically meaningful predicted/observed costs, collisions and joint constraints determine whether a proposal can be considered. Textual AI commentary does not certify safety.
3. **Active diagnosis:** an experiment selector chooses a bounded-cost probe to distinguish predeclared physical hypotheses, then updates Bayesian beliefs from simulated sensor data.
4. **Student distillation:** a small student learns to reproduce the critic-selected action, and is tested on held-out scenarios and competing baseline methods.


## New: recorded pretrained-policy evaluation and physics-backed students

Use the [recorded VLA protocol](RECORDED_VLA_BENCHMARK.md) to compare **externally executed** teacher checkpoint predictions in a common validated action space. The offline benchmark trains and evaluates simple supervised students; action imitation MSE is not physical task success.

The [physical student benchmark](PHYSICAL_STUDENT_BENCHMARK.md) runs the causal experiment selector, synthetic teachers and trained student on independent MuJoCo states:

```bash
python -m pip install -e '.[mujoco]'
python -m vla_council.causalforge.physical_distillation --backend mujoco --train 36 --test 24 --out runs/student-mujoco.json
python -m vla_council.causalforge.recorded_vla data/teacher_predictions.jsonl --out runs/teacher-imitation.json
```

The second command requires user-provided JSONL recorded predictions. The test suite does **not** download or run GR00T, pi0 or SmolVLA.

**Verified result:** [Five independent MuJoCo seeds](MULTISEED_MUJOCO_RESULTS.md) found equal student task loss for active and fixed probing. Read the negative results and limitations before citing any improvement. The next pretrained-model milestone is tracked in [GitHub issue #2](https://github.com/aaflakiyan/CausalForge/issues/2).

## Architecture

```text
     Task, observation, embodiment
                |
       Specialist policy proposals
                |
        Action-spec validation
                |
   Independent rollout/physics critic ----> veto infeasible actions
                |
      Disagreement / hypothesis set
                |
       Budgeted experiment selector
                |
   Measurement -> belief update -> action selection
                |
     Corrective dataset -> student training
                |
       Held-out paired evaluation
```

**What is genuinely connected today?** `closed_loop.py` connects the synthetic action-teacher selector, active probe, belief, student imitation and held-out outcome evaluation. `teacher_council.py` is a validated interface for future real-model proposals. The synthetic training loop now also runs against MuJoCo via `physical_distillation.py`. This is **not** a real-VLA training environment; see the recorded-policy interface and results below.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v

# Single-seed connected synthetic teacher -> probe -> student benchmark
python -m vla_council.causalforge.closed_loop --train 800 --test 300 --out runs/closed-loop.json

# Repeated seeds and paired differences; slower
python -m vla_council.causalforge.suite --seeds 8 --train 400 --test 160 --out runs/suite.json

# Original physical-hypothesis diagnostics benchmark (algebraic backend)
python -m vla_council.causalforge.benchmark --backend surrogate --trials 100 --out runs/diagnostics.json
```

To exercise the *actual MuJoCo integration*, install the optional dependency:

```bash
python -m pip install -e '.[mujoco]'
python -m vla_council.causalforge.demo --true both --out runs/mujoco-diagnosis.json
python -m vla_council.causalforge.benchmark --backend mujoco --trials 8 --out runs/mujoco-benchmark.json
```

Windows PowerShell: use `.venv\Scripts\Activate.ps1` instead of `source ...`.

## What the experiments measure

| Experiment | Implemented? | Outcome |
|---|---|---|
| Synthetic teacher proposal and independent critic | Yes | Candidate veto / selection |
| Active vs random/fixed/no-probe hypothesis diagnosis | Yes | Coverage, calibration, nominal probe cost, goal error |
| Teacher -> active experiments -> distilled student | Yes, synthetic only | Held-out task loss, imitation fidelity, paired comparisons |
| Independent MuJoCo contact benchmark | Tested in hosted CI | Encoder-based diagnosis and target displacement; see [measured results](MUJOCO_CI_RESULTS.md) |
| Real VLA teachers on shared robot embodiment | No | Required next step |
| Panda/Unitree humanoid learning in MuJoCo | No | Required next step |
| Student VLA checkpoint trained on real demonstrations | No | Required next step |

The synthetic environment has **two predefined hypotheses** (`weak_motor`, `heavy_payload`) and **three scripted control candidates**. An active decision system cannot discover a cause that is not in its hypothesis family. The surrogate's movement is algebraic, not a contact dynamics engine. Do not interpret successful toy benchmarks as humanoid robotics performance.

### Baselines and leakage controls

Every method uses the same train/test task sets, with independent train and test random seeds. Student features contain only the target and posterior probabilities; the hidden dynamics label is never directly encoded. The `oracle` receives the hidden label *deliberately* as a non-deployable performance ceiling. Each method trains a separate small ridge-classification student and reports the task-loss gap versus its corresponding teacher council and oracle.

**Caveat:** average experiment costs differ by method, and the two-hypothesis evaluation is not out-of-distribution. Do not interpret any raw win as a cost-matched scientific result. The repeated-seed report provides across-seed variability, not formal causal identification or a real-world confidence guarantee.

## Project layout

```text
src/vla_council/
  cli.py, core.py, council.py, sim.py, student.py, teachers.py
  causalforge/
    vla_interface.py     # strict robot/action/time/frame contract
    teacher_council.py   # independent critic and veto
    experiment.py        # hypothesis prediction and Bayesian update
    active.py            # information-gain acquisition under probe budget
    mujoco_push.py       # optional compiled MuJoCo contact model
    benchmark.py         # diagnosis baselines on surrogate/MuJoCo
    closed_loop.py       # connected toy teacher/probe/student pipeline
    suite.py             # repeated-seed paired evaluation
    demo.py              # MuJoCo diagnostic demo
tests/                   # deterministic and optional MuJoCo checks
.github/workflows/       # Python matrix and MuJoCo CI definition
```

## VLA integration contract

`ActionSpec` specifies the robot, action representation, ordered axes, units, coordinate frame and sampling period. `ActionChunk` holds an `[T, D]` prediction. `teacher_council.decide()` requires a common representation and independent evaluator; it abstains when all candidates violate the provided constraints. It is **not** a safety controller. To add a real model, write a model-specific adapter that explicitly converts its outputs into a matching `ActionSpec`; validate its observation schema, gripper mapping and time horizon. Do not average unaligned action tensors.

Recommended *future* research integration: begin with two policies that demonstrably support the exact same Panda embodiment and shared manipulation task, then benchmark individual teachers, simple ensemble, distillation, council gating and active teacher selection. Only after controlled comparisons progress to humanoid whole-body manipulation with a separate real-time controller.

## Verification

Run `python -m unittest discover -s tests -v`. GitHub Actions defines a Python 3.10/3.11/3.12 matrix and an independent MuJoCo job. All four jobs passed in [GitHub Actions run 42358370776](https://github.com/aaflakiyan/CausalForge/actions/runs/42358370776), including the MuJoCo test job. See [MUJOCO_CI_RESULTS.md](MUJOCO_CI_RESULTS.md) for measured data and limits.

## Scientific and safety limitations

No model weights, other projects' code, or datasets are redistributed. Real hardware requires independently tested collision, joint-limit, force and emergency-stop controls. The simulator cannot certify a real robot's safety. Metrics from simulated probes are only as reliable as the model, its noise assumptions and the scenarios tested. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

MIT licence for this repository's original code. Third-party software/checkpoints retain their own licences and usage terms.

## Unknown-physics comparison (v0.5)

```bash
python -m vla_council.causalforge.ood_benchmark --backend surrogate --episodes 50
python -m vla_council.causalforge.ood_benchmark --backend mujoco --episodes 3
```

Results identify the backend, distinguish known and unseen dynamics, report
unknown-condition rejection, physical intervention duration and magnitude, and
task loss against an oracle ceiling. See `PHYSICS_BENCHMARK.md`. The callable
`CallableVLATeacher` interface enables explicit checkpoint integration without
pretending that a placeholder is a pretrained model.
