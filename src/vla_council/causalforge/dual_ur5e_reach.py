"""Validated Cartesian reaching demonstration with two *actual* MuJoCo Menagerie UR5e robots.

Two independently specified 3D tool targets lie over the shared table.
A damped least squares inverse-kinematics solver plans joint targets; a
smooth position-target controller executes them under MuJoCo dynamics.
This is a kinematic/physics test, NOT grasping, VLA inference, or safety certification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .dual_ur5e import HOME, JOINTS, ids, ARENA

LEFT_GOAL = np.array([-.25, 0., .73])
RIGHT_GOAL = np.array([.25, 0., .73])
GOALS = np.stack([LEFT_GOAL, RIGHT_GOAL])

def reach_arena():
    text = ARENA.replace(
        'pos="0 -2.5 2.2" xyaxes="1 0 0 0 0.45 0.89"',
        'pos="0 -2.35 1.80" xyaxes="1 0 0 0 0.46 0.887919"'
    )
    marker = '    <frame name="left_mount"'
    addition = (
        '    <geom name="left_target" type="sphere" pos="-0.25 0 0.73"'
        ' size="0.046" rgba="0.15 0.9 0.45 0.85" contype="0" conaffinity="0"/>\n'
        '    <geom name="right_target" type="sphere" pos="0.25 0 0.73"'
        ' size="0.046" rgba="0.9 0.55 0.12 0.85" contype="0" conaffinity="0"/>\n'
    )
    if marker not in text or "camera" not in text:
        raise RuntimeError("Arena XML changed unexpectedly")
    return text.replace(marker, addition + marker)

def load_reach_model(menagerie: Path):
    import mujoco
    filename = menagerie / "universal_robots_ur5e" / "ur5e.xml"
    if not filename.exists():
        raise FileNotFoundError(filename)
    spec = mujoco.MjSpec.from_string(reach_arena())
    for prefix, frame in (("left_", "left_mount"), ("right_", "right_mount")):
        spec.attach(mujoco.MjSpec.from_file(str(filename)), frame=frame, prefix=prefix)
    model = spec.compile()
    if model.nq != 12 or model.nu != 12:
        raise RuntimeError(f"Expected 12 DOF/actuators, got {model.nq}/{model.nu}")
    return model

def reset(model):
    import mujoco
    data = mujoco.MjData(model)
    q, actuator, _ = ids(model)
    for side in (0, 1):
        data.qpos[q[side]] = HOME
        data.ctrl[actuator[side]] = HOME
    mujoco.mj_forward(model, data)
    return data

def solve_ik(model, side, goal, tolerance=.015, max_iters=650):
    """Position-only DLS IK, with measured world-frame site Jacobian."""
    import mujoco
    q, _, site = ids(model)
    best = None
    # Repeated starts improve robustness; deterministic for reproducibility.
    rng = np.random.default_rng(11 + side)
    for start_index in range(5):
        data = reset(model)
        if start_index:
            data.qpos[q[side]] = HOME + rng.uniform(-.65, .65, size=6)
            mujoco.mj_forward(model, data)
        for _ in range(max_iters):
            e = np.asarray(goal) - data.site_xpos[site[side]]
            dist = float(np.linalg.norm(e))
            if best is None or dist < best[0]:
                best = (dist, data.qpos[q[side]].copy())
            if dist < tolerance:
                return data.qpos[q[side]].copy(), dist
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, int(site[side]))
            j = jacp[:, q[side]]
            # Damped least squares, prioritising position over orientation.
            step = j.T @ np.linalg.solve(j @ j.T + .025**2 * np.eye(3), e)
            norm = np.linalg.norm(step)
            if norm > .12:
                step *= .12 / norm
            data.qpos[q[side]] += step
            # Respect joint limits if they exist.
            for j_idx, addr in enumerate(q[side]):
                joint_id = int(model.dof_jntid[addr])
                if model.jnt_limited[joint_id]:
                    lo, hi = model.jnt_range[joint_id]
                    data.qpos[addr] = np.clip(data.qpos[addr], lo + .01, hi - .01)
            mujoco.mj_forward(model, data)
    raise RuntimeError(
        f"Arm {side} cannot reach requested site {list(goal)} within {tolerance} m: "
        f"closest IK residual {best[0]:.4f} m"
    )

def contacts(model, data):
    """Count collisions between different arms and between either arm and table."""
    import mujoco
    inter, table = 0, 0
    for k in range(data.ncon):
        c = data.contact[k]
        a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[c.geom1])) or ""
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[c.geom2])) or ""
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if (a.startswith("left_") and b.startswith("right_")) or (a.startswith("right_") and b.startswith("left_")):
            inter += 1
        if (a.startswith(("left_", "right_")) and g2 == "table") or (b.startswith(("left_", "right_")) and g1 == "table"):
            table += 1
    return inter, table

def execute(model, goals=GOALS, duration=3., render=False, right_command_scale=1.0, right_compensation=1.0):
    import mujoco
    q, actuator, site = ids(model)
    ik = [solve_ik(model, s, goals[s]) for s in (0, 1)]
    joint_goals = np.stack([a for a, _ in ik])
    data = reset(model)
    home_sites = data.site_xpos[site].copy()
    n = round(duration / model.opt.timestep)
    if n <= 0:
        raise ValueError("Invalid duration")
    frames = []
    renderer = mujoco.Renderer(model, height=540, width=960) if render else None
    inter_contact_frames, table_contact_frames = 0, 0
    max_joint_error = 0.
    for i in range(n):
        t = (i + 1) / n
        # smoothstep ensures zero nominal velocity at both ends
        alpha = t*t*(3 - 2*t)
        for s in (0, 1):
            cmd = HOME + alpha*(joint_goals[s]-HOME)
            if s == 1:
                # Fault model: attenuated right-arm setpoint delivery, not motor torque.
                cmd = HOME + right_command_scale * right_compensation * (cmd - HOME)
            data.ctrl[actuator[s]] = np.clip(cmd, model.actuator_ctrlrange[actuator[s],0], model.actuator_ctrlrange[actuator[s],1])
        mujoco.mj_step(model, data)
        inter, tabletop = contacts(model, data)
        inter_contact_frames += bool(inter)
        table_contact_frames += bool(tabletop)
        max_joint_error = max(max_joint_error, float(np.max(np.abs(data.qpos[q]-data.ctrl[actuator]))))
        if renderer is not None and (i % max(1, n//50) == 0 or i == n-1):
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())
    if renderer is not None:
        renderer.close()
    final_sites = data.site_xpos[site].copy()
    initial_errors = np.linalg.norm(home_sites-goals,axis=1)
    final_errors = np.linalg.norm(final_sites-goals,axis=1)
    return {
        "task":"two independent Cartesian reaches above a shared workbench",
        "real_pretrained_vla":False,
        "goal_xyz_m":goals.tolist(),
        "home_ee_xyz_m":home_sites.tolist(),
        "final_ee_xyz_m":final_sites.tolist(),
        "initial_goal_errors_m":initial_errors.tolist(),
        "final_goal_errors_m":final_errors.tolist(),
        "actual_ee_displacements_m":np.linalg.norm(final_sites-home_sites,axis=1).tolist(),
        "ik_residuals_m":[float(err) for _,err in ik],
        "joint_goal_rad":joint_goals.tolist(),
        "inter_arm_contact_frames":int(inter_contact_frames),
        "table_contact_frames":int(table_contact_frames),
        "max_abs_joint_tracking_error_rad":float(max_joint_error),
        "duration_s":duration,
        "right_command_scale":float(right_command_scale),
        "right_compensation":float(right_compensation),
    }, frames

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/dual-ur5e-reach.json"))
    p.add_argument("--render",type=Path)
    p.add_argument("--duration",type=float,default=3.)
    args=p.parse_args()
    model=load_reach_model(args.menagerie)
    report,frames=execute(model,duration=args.duration,render=args.render is not None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if args.render:
        import imageio.v2 as imageio
        args.render.parent.mkdir(parents=True,exist_ok=True)
        imageio.mimsave(args.render,frames,duration=args.duration/max(1,len(frames)))

if __name__=="__main__":
    main()
