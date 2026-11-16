# Lab 01 — Your first intelligent robot

**Prerequisites:** basic Python variables and vectors. No ROS 2, GPU or physical robot required.

**Outcome:** explain why a robot needs task goals, sensed state, a controller and a feedback loop. Inspect live MuJoCo state and change the controller gain, sensor noise and target position.

## Before you begin: the system in one minute

1. **Goal:** move the end-effector towards a stationary cup target.
2. **Sense:** read the two joint angles from MuJoCo. In this introductory lab, the target position is provided directly; no camera-based detection yet.
3. **Understand:** compute `e = target - tip_position` in the same world frame, in metres.
4. **Plan:** choose a joint velocity using a damped least-squares Jacobian inverse.
5. **Control:** turn that command into small joint-position setpoints for MuJoCo actuators.
6. **Act:** the physics engine integrates the robot's joint dynamics. Read the new state and repeat.

The robot model has two rotational joints in a horizontal plane, *not* a full Franka or UR robot. The cup is only a visual target marker, not an object the robot can physically grasp. Grasping, RGB-D perception, collision avoidance and robot-specific models are later labs.

## Try it

From the `academy` folder:

```bash
python -m pip install -e '.[dev]'
python -m uvicorn physical_ai_lab.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Click **Run physics**. Watch the end-effector move towards the cup. It has reached the goal when the error is less than 0.025 m (2.5 cm).

**Exercise A — gain:** reset, then run with gain 0.4 and with gain 4.0. Predict the difference before measuring it. Faster is not always more stable.

**Exercise B — noise:** reset, keep the target fixed and compare noise 0 m with 0.04 m. Why does a controller need trustworthy sensing?

**Exercise C — reachability:** try a difficult but reachable target, e.g. x=0.20 and y=-0.40. The interface rejects targets outside the workspace.

**Exercise D — code:** open [`kinematics.py`](../physical_ai_lab/kinematics.py), identify the two link lengths, and explain what `jacobian()` computes. Then locate the `mj_step()` call in [`engine.py`](../physical_ai_lab/engine.py).

## The four key lines

```python
error = target - forward(q)       # metres: desired minus observed tip
J = jacobian(q)                    # metres / radian
dq = J.T @ solve(J@J.T + damping**2*I, gain*error)
q_command = q + delta_t * dq       # radians: joint setpoint
```

This is an *explanatory sketch*, not copy-paste executable code. In the real code, `dq` is clipped, targets are checked for reachability, the robot's joint servo tracks the setpoint and physics advances in discrete steps.

## What to measure

The app measures position error over time. To run a repeatable headless benchmark:

```bash
physical-ai-lab --steps 1200 --out results/benchmark.json
```

Compare final error and success rate across the four preselected targets. This baseline is not a claim of general robot intelligence: it is one planar reaching task, with known target coordinates.

## Check your understanding

1. Why must target and measured end-effector position use the same coordinate frame?
2. What does changing the gain change? How could it create instability?
3. Why is a Jacobian needed? What do its rows and columns represent?
4. What role does MuJoCo play that is different from the controller's role?
5. Which modules are missing before this robot can identify and pick up a real cup?

**Expected answers, briefly:** (1) otherwise subtracting positions is meaningless; (2) error-to-command sensitivity, with a speed/stability trade-off; (3) conversion between joint velocity and Cartesian tip velocity; (4) simulated dynamics vs decision-making; (5) camera sensing/calibration, object state estimation, grasp planning, gripper and contact-aware control, safety.
