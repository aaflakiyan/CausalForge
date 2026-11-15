# RobotGuard Playground: product scope and evidence contract

*Decision dated 15 November 2026. Design note; features described as "planned" are not implemented.*

## Context

Dream Labs' Dream-1 (https://dreamlabs.ai/research/dream-1/) is a **robot action-generation model** with a public inference API, embodiment adaptation and a real-robot playground. Its ability to generate a task-conditioned action is **not** functionality RobotGuard currently offers. It also states that Dream-1 is a research beta and not validated for safety-critical use.

RobotGuard's existing product is different: it checks typed, time-stamped joint-position traces against explicitly supplied kinematic limits, then performs an actual MuJoCo replay and records what the simulator observed. This is a useful component but not, by itself, a distinctive general-purpose robot agent platform.

Competitors/projects to track as of November 2026 include OpenRAL (policy orchestration), Aegis (policy safety and evaluation), SafeVLA (runtime VLA safety), Hugging Face LeRobot (data and policy training), OpenPI (model research) and RoboUse/Inspect Robots (evaluation and policy running). Names in this document are **context**, not claims of integration or endorsement.

## Product decision

**RobotGuard Playground = an accessible, open-source, evidence-first testing workbench for robot action policies.**

Public promise: **Connect a policy trace; validate its declared contract; replay it using real MuJoCo physics; inspect measured behaviour and export reproducible evidence.**

The public README must distinguish *what exists now* from *what is planned*. Do not market it as a universal WAM, safety-certified controller or model-agnostic execution engine before adapters are tested.

## The three layers

1. **Action contract:** explicit timestamp, joint order, action type, units, reference frame and rate. Require versioned limits. Fail closed on missing, non-finite or inconsistent fields; never reinterpret EEF poses as joint angles without a validated inverse-kinematics adapter.
2. **Physics/evidence:** pre-flight before stepping; model and actuator mapping validation; record actual physics engine and version, commands and positions, integration count, error metrics, model hash and action hash. Distinguish observed simulation outcomes from task success, safety and real-world transfer.
3. **Experience:** one command for a bundled synthetic arm, an offline HTML playback of *recorded MuJoCo world coordinates*, scrub/play, target-vs-observed chart and a rejected-trace explanation. No paid provider, GPU, web app or robot needed for the bundled demo.

## v0.3 actual scope

- Local Python CLI `robotguard playground --out output-dir`
- Synthetic two-joint MJCF with position-servos and a fixed safe trace
- Deterministic invalid trace rejected without physics execution
- Real MuJoCo stepping, recorded joint positions **and** joint-anchor/tool-tip coordinates
- Browser replay with those recorded planar XY positions; **not** camera footage, a live digital twin, or a 3D rendered simulation
- Separate offline numerical chart, JSON provenance and rejected trace report
- Tests and CI for no-fabrication, replay evidence, packaging and known boundaries

## Acceptance criteria for wider release

- [ ] Validate on fresh Linux, Windows and macOS environments; provide readable installation diagnostics
- [ ] Ship a real measured replay preview, using recorded simulation only; never a rendered fictional demo
- [ ] Implement collision and contact metrics with explicit scene assumptions; no certification claims
- [ ] Introduce a **single** small, versioned policy trace schema with strict action semantics and unit tests
- [ ] Add a reproducible, independently sourced UR5e or Franka MJCF model (licence checked), plus baseline controller and multiple seeded cases
- [ ] Add a documented, opt-in Python policy adapter and at least one verified third-party integration
- [ ] Benchmark against a comparable existing tool on setup time, correctness and reporting value
- [ ] Test browser controls with automated end-to-end tests and accessibility review
- [ ] Publish exact CI logs, artefacts, model versions, benchmark failures and limitations

## What we must NOT promise

- **Not a Dream-1 integration.** The public Dream client needs credentials, camera/state observations and a robot-specific control loop. We have not called the service or validated its returned action schema. Do not treat its product page as a ready-made conversion specification.
- **Not universal safety.** A numerical PASS is only about the supplied contract and discrete samples. Neither simulation nor tracking error alone proves collision clearance, force safety, braking feasibility, synchronisation or hardware validity.
- **Not automatic robot support.** A matching joint name and a synthetic MJCF do not make a robot compatible. FK, collision geometry, control and sensor models need independent verification.
- **Not task success.** Tracking a trajectory does not imply the policy accomplished a user instruction.
- **Not live robot control.** `playground.html` replays logged simulation positions. It cannot operate hardware, generate actions, or call a WAM.

## Decision on growth

The viral hook should be a **reproducible demonstration** that a developer can run without a robot and inspect in a browser. A visually pleasing interface is only valuable if its motion comes from actual logged physics. The public release should prioritise real use, external issue reports and reproducibility over stars alone.

Keep this as a feature branch until MuJoCo CI passes after each physics change; seek third-party testing before describing a broad release.
