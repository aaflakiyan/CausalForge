---
name: robotguard
summary: Validate AI-generated robot joint-space actions for explicit units, frames, timing and kinematic limits before running a simulator.
---

# RobotGuard pre-flight skill

Use this skill when writing, testing or reviewing generated joint-position trajectories for robotics simulation.

1. Read the specific robot's joint names, ordering, joint position bounds, maximum velocities and accelerations. **Never fabricate safety limits.** If unknown, ask for or retrieve an authoritative spec. Use the provided DemoArm-3 specification only for its synthetic example.
2. Emit a JSON robot spec matching `robotguard/examples/robot.json`. Use seconds for timestamps, and declare units and frame explicitly on each action.
3. Write the action trajectory as one JSON object per line with `t`, `position`, `units` and `frame`. Do not mix degrees and radians or change joint order.
4. Run `robotguard check PATH.jsonl --spec ROBOT.json --json audit.json --html audit.html`. Interpret exit code 0 as passing *only the configured kinematic checks*, exit 1 as violations, exit 2 as an input/configuration error.
5. For any failure, inspect each finding. Correct the source policy or trajectory, then rerun. Never suppress findings, enlarge limits or silently clip values to force a pass.
6. Never send unchecked action commands to physical hardware. A PASS does not address collision clearance, contact forces, torque constraints, braking, functional safety or real-world deployment.

Do not invent result figures or claim a simulated run happened when only static trace checks were performed.
