# RobotGuard adversarial design review (v0.2)

**Problem statement.** An AI-generated trajectory can be numerically wrong, but checking a sequence of joint positions cannot prove that a robot can safely execute it. RobotGuard deliberately separates (1) static trace checking, (2) model-compatibility verification and (3) actual physics replay. None of the three certifies real-world functional safety.

## Challenge the premise

**Should we build a generic Physical AI workbench instead?** No, not yet. A giant integration framework can attract attention but is hard to test honestly. A narrow pre-flight validator and reproducible MuJoCo demonstration is usable immediately. It is also independently testable and useful as a CausalForge component.

**Why not call this a robot safety system?** Because it does not establish collision-free paths, emergency stopping, torque limits, trajectory smoothness, contact safety, uncertainty bounds, actuator dynamics outside the configured model, or compliance with any safety standard. The word *Guard* is an interface metaphor, not a certification claim.

**Why not execute invalid trajectories to show the collision?** Executing invalid commands would contradict the core guarantee. Instead, the rejected trace generates a static audit, and only an approved trajectory is stepped in MuJoCo. This prevents a misleading screenshot of unsafe motion being treated as an accepted run.

**Can an agent game the validator by labelling degrees as radians?** Yes. Labels can only be checked against the declared contract. A mislabelled source or invented specification can still produce a PASS. The simulation adapter checks that commands use radians and the spec does not exceed MJCF joint and servo ranges, but this **does not verify** a manufacturer's real-world limits.

**Does passing sampled acceleration constraints mean continuous-time feasibility?** No. The checker uses finite differences between positions and segment-average velocities, not jerk-bounded interpolation or inverse dynamics. The MuJoCo demonstration uses piecewise-linear servo targets with velocity discontinuities at knots. The simulation produces observations, not a continuous-time feasibility proof.

**Could a simulator 'confirm' a wrong model?** Yes. A matching joint name and servo is not enough to establish that MJCF masses, geometries, damping and limits represent physical hardware. The versioned synthetic scene must not be described as UR5e, Franka, or manufacturer-validated. The report records model and trace hashes and the exact MuJoCo version to support reproducibility.

**What if the simulation doesn't run?** Missing MuJoCo, malformed MJCF, unsupported actuators, model mismatch or a pre-flight FAIL returns a non-zero code. No physics run or measured tracking is fabricated. CI installs real MuJoCo and tests this behaviour; no mock physics output is passed off as a real result.

## Evidence ledger

| Statement | Evidence and verification approach | What it does *not* demonstrate |
|---|---|---|
| Invalid sample/units/frame/joint/time is rejected | Deterministic unit tests with negative cases and non-zero CLI status | Correctness of user-supplied physical limits |
| Extreme numeric values don't silently become valid JSON results | Adversarial tests for overflow and non-finite values | Exhaustive numeric proof |
| Safe synthetic trace can be stepped in MuJoCo | GitHub Actions job with a real MuJoCo install, actual `mj_step`, qpos recording, output artefacts | Real robot operation or collision clearance |
| Wrong joint/actuator mappings abort rather than guessing | Integration tests altering spec and actuator gear | Universal compatibility with arbitrary robot MJCF files |
| Report contains provenance | SHA-256 hashes and simulation version in `simulation.json` | Trustworthiness of the underlying MJCF or dataset |
| Unchecked or invalid trajectory is not replayed | `simulate_trace` checks pre-flight before importing MuJoCo, plus CLI rejection test | Malicious bypasses outside the provided entry point |

## Release acceptance checklist

- [x] Static checks and adversarial tests can run with Python only
- [x] Real MJCF model, trajectories and optional MuJoCo adapter included in source
- [x] No implicit clamping, unit conversion or fabrication of safety limits
- [x] Distinct pre-flight and actual simulation statuses and reports
- [x] Document what initialisation, discretisation and actuator assumptions are used
- [ ] Confirm real MuJoCo CI job passes and artefact contains measured trajectories
- [ ] Test clean installs and graphical viewer on Windows/macOS, beyond CI Linux
- [ ] Add collision/contact checking and more representative robot models, with independent tests
- [ ] Implement uncertainty-aware model validation before any hardware-facing interface

**Review gate:** do not merge or market this as tested MuJoCo simulation until the real-physics CI passes. Even after it passes, no hardware deployment claim is appropriate.
