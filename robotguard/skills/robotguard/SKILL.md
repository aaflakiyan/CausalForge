---
name: robotguard
summary: Audit AI-generated joint-space trajectories, then optionally replay approved traces in actual MuJoCo physics without inventing results.
---

# RobotGuard agent skill

Use this for static joint-position checks and, when MuJoCo is installed, for a narrowly supported synthetic hinge-arm replay. This skill does **not** certify hardware safety.

1. Confirm the robot's named joint order, joint ranges, units, reference frame, maximum velocities and accelerations from a traceable source. **Never guess manufacturer limits** or derive them from an AI-generated claim. Demo limits are synthetic, not deployable.
2. Produce JSON spec and JSONL actions with explicit `units`, `frame`, numeric `t` seconds and ordered `position` arrays. A wrong source value can still pass checks; explain the provenance and uncertainty.
3. Run `robotguard check PATH.jsonl --spec ROBOT.json --json preflight.json --html preflight.html`. Exit 0 only means the configured sampled checks passed. Exit 1 means findings, exit 2 means an input or setup problem.
4. For an actual MuJoCo physics simulation, install the optional `[mujoco]` extra and run `robotguard simulate PATH.jsonl --spec ROBOT.json --model SCENE.xml --out OUTPUT_DIR` only for a compatible joint-position-servo MJCF. Never claim simulation occurred unless `simulation.json` exists and records MuJoCo version and integration steps.
5. Use `robotguard mujoco-demo --out robotguard-output` for the packaged reproducible synthetic demo. The unsafe trace **must** be rejected without physics stepping. The approved trace is physically stepped and compared with measured `qpos`.
6. If model validation or replay fails, keep the failure; do not force a pass by silently changing limits, gear, units, actuator types or thresholds. Report the reasons.
7. Never describe a pre-flight PASS or successful simulation as collision-free, torque-safe, certified, hardware-ready or proof of a valid real-world model.

There are no LLM or web calls in RobotGuard's checker or simulator. **Do not fabricate measurements, citations, screenshots, simulation runs or benchmarks.** Link reproducible evidence, not hypothetical achievements.
