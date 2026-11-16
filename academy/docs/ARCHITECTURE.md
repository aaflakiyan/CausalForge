# Architecture and integration contract

```text
Browser (HTML/CSS/JS, top-down visualisation + controls)
  |  GET /api/state; POST /api/config, /api/reset, /api/step
  v
FastAPI (single-user API)
  |  state -> plan -> command -> mj_step -> new state
  v
Physical AI Lab / MuJoCo 4 core (CPU physics)
  |  MJCF model: 2 hinge joints + 2 position servos
  v
Report: joint state, target, tip position, error, success
```

**Coordinate convention:** world XY plane, origin at the shoulder; metres for positions, radians for joint angles, seconds for time. Both joints rotate around world Z. Positive rotation follows the right-hand rule. The two links are 0.4 m and 0.3 m. Initial joint state is `[-0.65, 1.35]` radians.

**Control frequency:** MuJoCo physics timestep = 0.005 s (200 Hz); a new Cartesian command every five physics steps (40 Hz). The browser typically advances ten physics steps per update; it is not a hard real-time client.

**Control mode:** DLS produces joint velocities [rad/s]. We integrate a short interval into joint-position setpoints [rad] for MuJoCo's built-in position actuators. We clamp the command to `[-3.12,3.12]` radians. The model does not guarantee collision avoidance or actuator saturation handling appropriate for real hardware.

**Sensing noise:** perturbs the measured Cartesian tip position for the controller (Gaussian standard deviation in metres). It does not change the actual physical target, nor does it simulate a camera. The app reports the true MuJoCo tip position so learners can compare actual tracking error.

**Safety:** pure simulation. There is no ROS 2 or hardware command bridge. Never directly feed these commands to industrial robots.

## API contract

| Method | Endpoint | Function |
|---|---|---|
| GET | `/api/health` | Server status (does not initialise MuJoCo) |
| GET | `/api/state` | Current physics state |
| POST | `/api/config` | Target, controller gain, observation noise |
| POST | `/api/step` | Advance 1–100 physics steps |
| POST | `/api/reset` | Reset robot and deterministic noise RNG |

Example: `curl -X POST http://127.0.0.1:8000/api/step -H 'Content-Type: application/json' -d '{"steps":10}'`

## Extension points

- **Lab 02:** camera observations and calibration; avoid hard-coded target coordinates.
- **Lab 03:** Cartesian trajectories, collision checking, joint limits and obstacle avoidance.
- **Lab 04:** contact forces, a simulated gripper and a physically graspable cup.
- **Later:** a real Franka/UR model from the MuJoCo Menagerie, ROS 2 bridge, imitation/RL policies and independent Isaac Lab tasks.

ROS 2, Isaac Sim, VLA models and physical robot deployment are **not implemented** in Lab 01. They are deliberately separated to keep the entry point lightweight.
