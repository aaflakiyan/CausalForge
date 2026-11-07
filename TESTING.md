# Experimental protocol for CausalForge

## Scope of included benchmark

The MuJoCo add-on is a **restricted 1D contact task**: a motor-controlled slider pushes a sliding payload. The rail has no realistic ground/Coulomb friction; payload joint damping is a simple resistance model. The two explicit hypotheses are **weakened motor actuation** and **increased payload mass**. Two simple probes use identical action units; measured displacement updates a Bayesian hypothesis distribution. Three *synthetic* candidate controls are subsequently scored using the inferred physics model. No actual VLA checkpoints or learned VLA student are integrated.

The code tests *plumbing and physical diagnostic assumptions*; it does **not** test VLA policy distillation, real-world robotic safety, humanoid manipulation, or actual human feedback.

## How to run

From the repository root, on a machine with Internet access to install dependencies:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[mujoco]'
python -m unittest discover -s tests -v
python -m vla_council.causalforge.demo --true both --out runs/diagnostic.json
```

The demo prints the chosen experiment, measured pusher/payload displacement, posterior support for each cause, candidate control scores, and actual-vs-baseline displacement errors.

**Verification status:** Pure-Python causal inference tests and existing VLA Council toy tests pass. MuJoCo runtime could not be installed in the authoring container, so the MJCF model, contact outcomes, and benchmark claims must be verified locally. Do not report simulated success or physical validity until the MuJoCo smoke test passes.

## First meaningful physics experiments

1. **Calibration:** Inspect contact state and log `qpos`, `qvel`, actuator force and payload position throughout free motion, contact onset and recovery. Verify no penetration or solver instability. Check timestep convergence at 0.001, 0.002 and 0.004 s.
2. **Identifiability:** Test whether reduced motor strength, high payload mass and high slider damping can be distinguished. If multiple hypotheses produce statistically indistinguishable measurements, report *unidentifiable* instead of forcing a diagnosis.
3. **Noise robustness:** Add realistic encoder noise, control delay, force estimation error and friction mismatch; recalibrate posterior probability. Use multiple independent seeds.
4. **Active versus random:** Hold the **number and risk of diagnostic probes equal**; measure causal diagnosis accuracy, task success after correction, probe energy and induced collisions.
5. **Planning baselines:** Compare fixed nominal controller, oracle physics controller (upper bound), non-causal teacher selection and active-probe teacher selection.
6. **Beyond the toy:** Replace slider model with a Franka/Panda manipulation task from robosuite or MuJoCo Playground, where real contact friction and grasp pose matter.

## Hypotheses worth publishing (once real VLA teachers are integrated)

H1: With a fixed total number of teacher queries, critic-guided selection increases held-out success over training on an equally sized mixture of teacher demonstrations.

H2: With a fixed experiment budget and equivalent intervention risk, causal active sampling produces more recoverable student-policy failures than entropy-only sampling, random perturbation or naive disagreement sampling.

H3: Failure-conditioned student training improves recovery on unseen scene parameter combinations, *without* losing more than an agreed tolerance on previous skills.

## Data collection and splitting

Use a common robot, task, observation/action schema and scene randomisation for every teacher. Fix train/validation/test split **before** performing active sampling. Partition by object geometries, surfaces, perturbation combinations and scene assets where possible (not just different random initial positions). Hold out unseen combinations, not just random samples of seen combinations. Evaluate using identical seeds and matched action/compute/data budgets. Avoid revealing true simulator randomisation parameters to any teacher at evaluation time; use them only for evaluator metrics.

Recommended data record: `episode_id`, `seed`, `robot_model`, `task_id`, `observation`, `timestamp`, `requested_action_chunk`, `action_frame`, `action_units`, `teacher_id`, `teacher_candidate`, `critic_features`, `critic_prediction`, `diagnostic_probe`, `probe_cost`, `measured_effect`, `posterior`, `selected_action`, `actual_effect`, `termination_reason`, `success`, `constraint_violation`.

## Evaluation table

| Measurement | Reason |
|---|---|
| Task completion (%) | Real outcome, not model's self-reported confidence |
| Recovery success (%) after perturbations | Improvement on failures |
| Teacher queries / successful task | Data efficiency and fairness |
| Episodes / GPU-hours / wall time | Total cost (including digital experiments) |
| Causal diagnosis accuracy + posterior calibration | Whether explanations are grounded |
| Collision / joint-limit / balance violations | Safety trade-off; report separately |
| False rejection of feasible actions | Critic over-conservatism |
| Student-only inference latency | Whether unified deployment is realistic |
| Retained baseline skills | Catastrophic forgetting |

Use at least several independent training seeds and report uncertainty intervals; number of episodes depends on outcome variance. Separate evaluation of the teacher council from **student-only** policy evaluation. A causal claim should rely on controlled interventions or identifiable counterfactual predictions, not text explanations from LLMs.

## Real VLA integration order

- First, a **single Panda embodiment** and aligned checkpoints with identical camera setup and command semantics; start with the VLA Evaluation Harness/robosuite stack where available.
- Build and unit test each `VLATeacher` adapter: camera data, task instruction, joint ordering, units, reference frame, action time base, padding, normalisation and safety checks. Do not average joint actions and Cartesian motions.
- Record all teachers' proposed action **chunks**, independent critic outputs, and measured outcomes. Candidate branching in a MuJoCo clone is available **only in simulation**; in the real world the system must predict with calibrated approximate models.
- Train an identical-capacity student on (i) one teacher, (ii) mixed teachers, (iii) outcome-selected teacher demonstrations, (iv) failure-directed DAgger corrections, and (v) the full causal experiment proposal. Match total query budgets.
- Only after tabletop experiments are convincing: use the Unitree G1 Menagerie model for simulation, adding independent WBC, state estimation, support constraints, external disturbances and monitoring. Do not run the prototype on hardware without a separately engineered safety layer.
