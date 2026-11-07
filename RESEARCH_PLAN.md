# Research roadmap: from synthetic tests to humanoid Physical AI

## Question

Can structured, physically tested disagreement between specialised robot policies improve learning efficiency and recovery under out-of-distribution physical conditions, compared against single teachers, conventional ensembling and multi-teacher imitation?

## R0 — Repository and software correctness (implemented)

- Auditable action specification: robot, units, coordinate frame, time step, action ordering and horizon.
- Separately scored candidate proposals, feasibility veto and abstention.
- Bayesian belief over a specified physics-hypothesis family and budgeted probe selection.
- Held-out synthetic student distillation with identical train/test scenarios across baselines.
- Independent test suite, repeated random seeds and honest result reporting.

## R1 — Real MuJoCo dynamics (implementation exists; run externally)

- Install `mujoco` and verify MJCF compilation/trajectory logging.
- Inspect contact onset, object mass/inertia consistency and actuator response.
- Add sensor noise, randomisation, systematic calibration and automated rollout videos.
- Compare measured contact trajectories against the algebraic surrogate; quantify misspecification.
- Run cost-matched random/fixed/active probes and vary noise, budget and prior.

## R2 — Common-embodiment real VLA integration (not implemented)

- Pick one robot and task family supported by at least two actually available checkpoints (e.g. Panda manipulation if each checkpoint supplies a compatible embodiment head).
- Log dataset provenance, observation schemas, camera calibration, action frames/horizons, gripper semantics and normalisation.
- Implement adapters **per model**; reject incompatible actions rather than silently averaging dimensions.
- Use the exact same train/test tasks and maximum teacher queries for each baseline.
- Implement predictive critics backed by physical rollouts; evaluate critic calibration, false negatives and abstention.
- Distil into a shared student only after valid teacher demonstrations are obtained, with held-out policy evaluation.

## R3 — Scientific novelty tests (not implemented)

- Evaluate whether disagreement-driven probes outperform uncertainty-only and random selection under matched total simulator seconds, compute and environment interactions.
- Introduce *unknown* failure modes, counterfactual model-misspecification tests and simultaneous changes in friction, mass, occlusion and actuator latency.
- Explicitly quantify distribution shift and catastrophic forgetting.
- Ablate peer textual feedback, physical validation, experiment selection, memory and student architecture independently.
- Use a preregistered evaluation protocol with independent seeds and confidence intervals over *task episodes*, not just random seed averages.

## R4 — Humanoid embodiment (not implemented)

- Begin with manipulation while standing, then add balance-constrained reaching and whole-body control.
- Keep the student at policy level and a verified real-time controller in charge of joint limits, collision, force caps and emergency stops.
- Use MuJoCo Menagerie embodiments only after validating asset licences and intended use; do not infer hardware safety from simulation.

## Exit criteria

A research claim requires reproducible positive results on at least two physical task families, full baseline parity, a nontrivial held-out perturbation split, calibrated criticism, and independent reruns. The existing two-hypothesis surrogate is deliberately insufficient for this claim.
