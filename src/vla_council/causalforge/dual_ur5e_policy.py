"""Fail-closed joint-policy integration for two actual MuJoCo Menagerie UR5e arms.

Policy output metadata is not inferred. It must match the simulator's 12 ordered
joint actuators, radians, joint frame, absolute position control and 20 ms tick.
Action vectors must be finite, inside model actuator limits, rate-limited and
start continuously from home. Validation finishes BEFORE the first physics step.
No VLA checkpoints are supplied; BuiltinReachPolicy is a baseline example.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json
import math
import numpy as np

from .dual_ur5e import HOME, JOINTS, ids
from .dual_ur5e_reach import GOALS, load_reach_model, reset, solve_ik, contacts
from .vla_interface import ActionSpec, ActionChunk

DT = .02
MAX_SPEED_RAD_S = 2.0
SPEC = ActionSpec(
    robot="menagerie_dual_ur5e",
    representation="joint_position",
    ordered_axes=tuple(f"{side}_{j}" for side in ("left", "right") for j in JOINTS),
    units=("rad",) * 12,
    frame="joint",
    dt=DT,
)


class Policy(Protocol):
    def predict(self, observation: dict[str, Any]) -> ActionChunk: ...


def _validate(chunk: ActionChunk, model, home_q: np.ndarray) -> tuple[np.ndarray, int]:
    """Validate complete action horizon without stepping the simulator."""
    if not isinstance(chunk, ActionChunk):
        raise TypeError("Policy.predict must return ActionChunk")
    if chunk.spec != SPEC:
        raise ValueError(f"Action specification mismatch; expected {SPEC}, received {chunk.spec}")
    actions = np.asarray(chunk.actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 12 or not len(actions):
        raise ValueError("Expected [T,12] absolute joint positions")
    if not np.isfinite(actions).all():
        raise ValueError("Joint commands must be finite")
    if len(actions) > 3000:
        raise ValueError("Action horizon exceeds 60 seconds")
    ratio = SPEC.dt / float(model.opt.timestep)
    if abs(ratio - round(ratio)) > 1e-8 or ratio < 1:
        raise ValueError("Control period must be an integer multiple of MuJoCo timestep")
    _, actuators, _ = ids(model)
    lower = model.actuator_ctrlrange[actuators.ravel(), 0]
    upper = model.actuator_ctrlrange[actuators.ravel(), 1]
    # Menagerie controllers have position setpoints, so the same units apply.
    if np.any(actions < lower - 1e-9) or np.any(actions > upper + 1e-9):
        raise ValueError("Commands exceed actuator setpoint limits")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Non-finite actuator bounds")
    # Check physical joint limits independently of actuator limits.
    q, _, _ = ids(model)
    for i, address in enumerate(q.ravel()):
        joint_id = int(model.dof_jntid[int(address)])
        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]
            if np.any(actions[:, i] < lo - 1e-9) or np.any(actions[:, i] > hi + 1e-9):
                raise ValueError(f"Joint {SPEC.ordered_axes[i]} exceeds joint limits")
    increments = np.diff(np.vstack([home_q.reshape(1, 12), actions]), axis=0)
    if np.max(np.abs(increments)) > MAX_SPEED_RAD_S * SPEC.dt + 1e-9:
        raise ValueError("Discontinuous or excessive joint command rate")
    return actions.copy(), int(round(ratio))


@dataclass
class BuiltinReachPolicy:
    """Reference motion generator using verified Cartesian IK, NOT a pretrained VLA."""
    model: Any
    duration: float = 3.0

    def predict(self, observation: dict[str, Any]) -> ActionChunk:
        targets = np.asarray(observation["goal_xyz_m"], dtype=float)
        if targets.shape != (2, 3) or not np.isfinite(targets).all():
            raise ValueError("Expected two finite XYZ targets")
        q_goal = np.stack([solve_ik(self.model, side, targets[side])[0] for side in (0, 1)])
        start = np.asarray(observation["joint_positions_rad"], dtype=float).reshape(2, 6)
        ticks = round(self.duration / DT)
        if ticks < 1 or abs(ticks * DT - self.duration) > 1e-8:
            raise ValueError("Duration must be positive multiple of policy dt")
        t = np.arange(1, ticks + 1) / ticks
        blend = t*t*(3-2*t)
        actions = start.reshape(1,12) + blend[:,None]*(q_goal-start).reshape(1,12)
        return ActionChunk(actions, SPEC, "builtin_cartesian_reach")


def evaluate(
    policy: Policy,
    *,
    robot: str = "ur5e-dual",
    task: str = "reach",
    menagerie: str | Path,
    goals: np.ndarray = GOALS,
    record_trace: bool = False,
    include_rgb: bool = False,
    image_width: int = 320,
    image_height: int = 240,
    command_delay_ticks: int = 0,
) -> dict:
    """Evaluate a joint-position policy on verified MuJoCo dual-arm reaching.

    Policy outputs are validated in full before stepping. No conversion or clipping
    is silently performed. Controls are held for exact integer physics steps.
    """
    if type(command_delay_ticks) is not int or not (0 <= command_delay_ticks <= 25):
        raise ValueError("Command delay must be an integer number of 20 ms ticks in [0,25]")
    if robot != "ur5e-dual" or task != "reach":
        raise ValueError("Only robot='ur5e-dual', task='reach' is implemented")
    model = load_reach_model(Path(menagerie))
    data = reset(model)
    q, ctrl, sites = ids(model)
    home_q = data.qpos[q].copy()
    goals = np.asarray(goals,dtype=float)
    if goals.shape != (2,3) or not np.isfinite(goals).all():
        raise ValueError("Two finite Cartesian goals are required")
    observation = {
        "robot":"menagerie_dual_ur5e",
        "joint_positions_rad":home_q.copy(),
        "goal_xyz_m":goals.copy(),
        "control_dt_s":DT,
        "action_spec":SPEC,
        "task_instruction":"Reach the two coloured target markers above the shared workbench.",
    }
    if include_rgb:
        if not (64 <= image_width <= 1024 and 64 <= image_height <= 1024):
            raise ValueError("Image dimensions must be within [64,1024]")
        import mujoco
        renderer = mujoco.Renderer(model, width=image_width, height=image_height)
        try:
            renderer.update_scene(data, camera="overview")
            image = renderer.render().copy()
        finally:
            renderer.close()
        if image.dtype != np.uint8 or image.shape != (image_height, image_width, 3):
            raise RuntimeError("Unexpected MuJoCo camera output")
        observation["rgb_overview"] = image
        observation["camera_frame"] = "mujoco/overview"
        observation["camera_image_encoding"] = "RGB8"
    proposal = policy.predict(observation)
    commands, substeps = _validate(proposal, model, home_q)
    initial_tool = data.site_xpos[sites].copy()
    trace = []
    arm_contacts = 0
    table_contacts = 0
    max_tracking = 0.
    min_tool_separation_m = float(np.linalg.norm(initial_tool[0]-initial_tool[1]))
    import mujoco
    for index, command in enumerate(commands):
        # Simulate held position setpoints during a measured communication delay.
        # This is transport-command latency, NOT a simulated actuator/torque fault.
        applied = (home_q.ravel() if index < command_delay_ticks
                   else commands[index - command_delay_ticks])
        data.ctrl[ctrl.ravel()] = applied
        for _ in range(substeps):
            mujoco.mj_step(model,data)
            inter, table = contacts(model,data)
            arm_contacts += bool(inter)
            table_contacts += bool(table)
            min_tool_separation_m = min(min_tool_separation_m,
                float(np.linalg.norm(data.site_xpos[sites[0]]-data.site_xpos[sites[1]])))
        max_tracking = max(max_tracking, float(np.max(np.abs(data.qpos[q].ravel() - applied))))
        if record_trace:
            trace.append({
                "time_s": float(data.time),
                "q_rad": data.qpos[q].tolist(),
                "command_rad": command.tolist(),
                "applied_command_rad": np.asarray(applied).tolist(),
                "eef_xyz_m": data.site_xpos[sites].tolist(),
            })
    measured = data.site_xpos[sites].copy()
    err = np.linalg.norm(measured-goals,axis=1)
    return {
        "robot":robot, "task":task, "policy_name":proposal.model_name,
        "observation_included_rgb":bool(include_rgb),
        "observation_rgb_shape":([image_height,image_width,3] if include_rgb else None),
        "validated_action_spec": {
            "robot":SPEC.robot, "representation":SPEC.representation,
            "ordered_axes":list(SPEC.ordered_axes), "units":list(SPEC.units),
            "frame":SPEC.frame, "dt":SPEC.dt,
        },
        "actions_checked_before_execution":int(len(commands)),
        "physics_substeps_per_action":substeps,
        "command_delay_ticks":command_delay_ticks,
        "command_delay_s":command_delay_ticks*DT,
        "duration_s":float(data.time),
        "initial_tool_xyz_m":initial_tool.tolist(),
        "final_tool_xyz_m":measured.tolist(),
        "goal_xyz_m":goals.tolist(),
        "tool_displacement_m":np.linalg.norm(measured-initial_tool,axis=1).tolist(),
        "tool_error_m":err.tolist(),
        "inter_arm_contact_steps":arm_contacts,
        "minimum_tool_site_separation_m":min_tool_separation_m,
        "table_contact_steps":table_contacts,
        "max_joint_tracking_error_rad":max_tracking,
        "task_success": bool(np.all(err < .07) and arm_contacts==0 and table_contacts==0),
        "trace": trace if record_trace else None,
    }


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("runs/validated-policy.json"))
    p.add_argument("--trace",action="store_true")
    args=p.parse_args()
    model=load_reach_model(args.menagerie)
    result=evaluate(BuiltinReachPolicy(model),menagerie=args.menagerie,record_trace=args.trace)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="trace"},indent=2))
    if not result["task_success"]:
        raise SystemExit("Validation completed but reaching task failed")


if __name__=="__main__":
    main()
