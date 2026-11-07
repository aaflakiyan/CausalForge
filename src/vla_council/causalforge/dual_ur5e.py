"""Two facing Menagerie UR5e arms: physics-backed teacher-plan evaluation.

This is *not* pretrained VLA inference. Teacher candidates are explicit
joint-target schedules. Test cases share goals, random seeds and runtime.
The same simulation model is used for planning and evaluation, except for a
withheld right-arm tracking scale. This is not a sim-to-real benchmark.

Requires: MuJoCo >=3.5 and google-deepmind/mujoco_menagerie UR5e assets.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.])
ARENA = """<mujoco model="dual_ur5e">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="540"/></visual>
  <worldbody>
    <light pos="0 -1 3.0" diffuse="1 1 1" dir="0 0 -1"/>
    <camera name="overview" pos="0 -2.5 2.2" xyaxes="1 0 0 0 0.45 0.89"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.13 0.17 0.23 1"/>
    <geom name="table" type="box" pos="0 0 0.34" size="0.64 0.54 0.035"
          rgba="0.4 0.45 0.5 1" contype="1" conaffinity="1"/>
    <frame name="left_mount" pos="-0.73 0 0.78"/>
    <frame name="right_mount" pos="0.73 0 0.78" quat="0 0 0 1"/>
  </worldbody>
</mujoco>"""


@dataclass(frozen=True)
class Config:
    episodes: int = 6
    seed: int = 42
    duration_s: float = 1.5
    probe_s: float = 0.22
    probe_delta_rad: float = 0.20
    noise_rad: float = 0.003
    penalty: float = 0.20

    def __post_init__(self):
        if self.episodes < 1 or self.duration_s <= 0 or self.probe_s <= 0:
            raise ValueError("Invalid episodes or durations")
        if self.noise_rad <= 0 or self.probe_delta_rad <= 0:
            raise ValueError("Probe magnitude and noise must be positive")


def load_model(menagerie: Path):
    """Assemble two independently namespaced, real Menagerie UR5e models."""
    import mujoco
    filename = menagerie / "universal_robots_ur5e" / "ur5e.xml"
    if not filename.is_file():
        raise FileNotFoundError(
            f"Missing {filename}. Clone https://github.com/google-deepmind/"
            "mujoco_menagerie and pass --menagerie <clone directory>."
        )
    arena = mujoco.MjSpec.from_string(ARENA)
    for side, frame_name in (("left", "left_mount"), ("right", "right_mount")):
        child = mujoco.MjSpec.from_file(str(filename))
        arena.attach(child, frame=frame_name, prefix=f"{side}_")
    model = arena.compile()
    if model.nq != 12 or model.nu != 12:
        raise RuntimeError(f"Expected 12 robot joints and 12 controls; got nq={model.nq}, nu={model.nu}")
    for side in ("left", "right"):
        for joint in JOINTS:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}") < 0:
                raise RuntimeError(f"Missing {side}_{joint}")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_attachment_site") < 0:
            raise RuntimeError("Attachment-site prefix not found")
    return model


def ids(model):
    import mujoco
    qadr, ctrl, sites = [], [], []
    ctrl_names = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
    for side in ("left", "right"):
        qadr.append([int(model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")]) for j in JOINTS])
        ctrl.append([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{j}")
                     for j in ctrl_names])
        sites.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_attachment_site"))
    return np.asarray(qadr), np.asarray(ctrl), np.asarray(sites)


def initial(model):
    import mujoco
    data = mujoco.MjData(model)
    q, act, sites = ids(model)
    for side in (0, 1):
        data.qpos[q[side]] = HOME
        data.ctrl[act[side]] = HOME
    mujoco.mj_forward(model, data)
    return data


def targets(model, magnitude):
    """Controlled 12-DoF joint targets, not Cartesian IK or VLA predictions."""
    import mujoco
    q, act, sites = ids(model)
    result = np.stack([HOME.copy(), HOME.copy()])
    result[0] += np.array([magnitude, -.5 * magnitude, .35 * magnitude, 0, 0, 0])
    result[1] += np.array([-magnitude, -.5 * magnitude, .35 * magnitude, 0, 0, 0])
    data = initial(model)
    for side in (0, 1):
        data.qpos[q[side]] = result[side]
    mujoco.mj_forward(model, data)
    return result, np.array(data.site_xpos[sites], copy=True)


def _factor(teacher, side, t):
    if teacher in ("synchronous", "compensate_right"):
        return min(1., t / .7)
    if teacher == "left_first":
        return min(1., t / .7) if side == 0 else min(1., max(0., t - .42) / .7)
    if teacher == "right_first":
        return min(1., max(0., t - .42) / .7) if side == 0 else min(1., t / .7)
    if teacher == "conservative":
        return min(1., t / 1.10)
    raise ValueError(f"Unknown teacher: {teacher}")


TEACHERS = ("synchronous", "left_first", "right_first", "conservative", "compensate_right")
HYPOTHESES = (1.0, .45)


def rollout(model, teacher, target_q, target_sites, right_scale=1., duration=1.5, render=False):
    import mujoco
    data = initial(model)
    q, act, site_ids = ids(model)
    n = int(round(duration / model.opt.timestep))
    if n <= 0:
        raise ValueError("Duration too short")
    hit_count = 0
    separation_min = float("inf")
    joint_error = 0.
    frames = []
    renderer = None
    if render:
        renderer = mujoco.Renderer(model, height=540, width=960)
    for k in range(n):
        t = k * model.opt.timestep
        for side in (0, 1):
            frac = _factor(teacher, side, t)
            cmd = HOME + frac * (target_q[side] - HOME)
            if side == 1:
                cmd = HOME + right_scale * (1.9 if teacher == "compensate_right" else 1.0) * (cmd - HOME)
            data.ctrl[act[side]] = cmd
        mujoco.mj_step(model, data)
        separation = float(np.linalg.norm(data.site_xpos[site_ids[0]] - data.site_xpos[site_ids[1]]))
        separation_min = min(separation_min, separation)
        if renderer is not None and k % max(1, n // 24) == 0:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())
        for idx in range(data.ncon):
            contact = data.contact[idx]
            b1 = int(model.geom_bodyid[contact.geom1])
            b2 = int(model.geom_bodyid[contact.geom2])
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            if ((name1.startswith("left_") and name2.startswith("right_"))
                    or (name1.startswith("right_") and name2.startswith("left_"))):
                hit_count += 1
    if renderer is not None:
        renderer.close()
    err = np.linalg.norm(data.site_xpos[site_ids] - target_sites, axis=1)
    result = {
        "teacher": teacher, "right_scale": float(right_scale),
        "left_ee_error_m": float(err[0]), "right_ee_error_m": float(err[1]),
        "combined_ee_error_m": float(np.mean(err)),
        "inter_arm_contact_steps": int(hit_count),
        "min_eef_separation_m": float(separation_min),
        "final_joint_error_rad": float(np.mean([
            np.linalg.norm(data.qpos[q[s]] - target_q[s]) for s in (0, 1)
        ])),
    }
    return result, frames


def probe_result(model, arm, scale, delta, duration):
    import mujoco
    data = initial(model)
    q, act, _ = ids(model)
    for k in range(int(round(duration / model.opt.timestep))):
        command = HOME.copy()
        command[0] += delta
        if arm == 1:
            command = HOME + scale * (command - HOME)
        data.ctrl[act[arm]] = command
        mujoco.mj_step(model, data)
    return float(data.qpos[q[arm, 0]] - HOME[0])


def posterior(observed, predictions, std):
    diffs = np.asarray([(observed - p) / std for p in predictions])
    logp = -.5 * diffs * diffs
    weights = np.exp(logp - np.max(logp))
    return (weights / weights.sum()).tolist()


def run_benchmark(menagerie: Path, cfg: Config):
    model = load_model(menagerie)
    predictions = {
        arm: [probe_result(model, arm, scale, cfg.probe_delta_rad, cfg.probe_s)
              for scale in HYPOTHESES] for arm in (0, 1)
    }
    # Select the probe with greater expected separation, at equal duration and amplitude.
    discriminations = [abs(x[0] - x[1]) for x in predictions.values()]
    active_arm = int(np.argmax(discriminations))
    rng = np.random.default_rng(cfg.seed)
    rows = []
    for episode in range(cfg.episodes):
        truth = float(HYPOTHESES[episode % 2] if episode % 3 != 2 else .70)
        magnitude = float(rng.uniform(.16, .36))
        goal_q, goal_sites = targets(model, magnitude)
        paired_sensor_noise = float(rng.normal(0, cfg.noise_rad))
        # Compare all candidates under the true physical condition (oracle only).
        actual = {teacher: rollout(model, teacher, goal_q, goal_sites,
                                   right_scale=truth, duration=cfg.duration_s)[0]
                  for teacher in TEACHERS}
        def objective(outcome):
            return outcome["combined_ee_error_m"] + cfg.penalty * outcome["inter_arm_contact_steps"] / 100
        oracle = min(TEACHERS, key=lambda k: objective(actual[k]))
        for method in ("no_probe", "fixed_left_probe", "random_probe", "active_probe"):
            if method == "no_probe":
                belief = [0.5, 0.5]
                arm = None
            else:
                arm = (0 if method == "fixed_left_probe" else
                       int(rng.integers(0, 2)) if method == "random_probe" else active_arm)
                sensor = probe_result(model, arm, truth, cfg.probe_delta_rad, cfg.probe_s)
                sensor += paired_sensor_noise
                belief = posterior(sensor, predictions[arm], cfg.noise_rad)
            # Predict under HYPOTHESES ONLY, never give the planner the withheld truth.
            scores = {}
            for teacher in TEACHERS:
                expected = 0.
                for p, scale in zip(belief, HYPOTHESES):
                    est = rollout(model, teacher, goal_q, goal_sites,
                                  right_scale=scale, duration=cfg.duration_s)[0]
                    expected += p * objective(est)
                scores[teacher] = expected
            selected = min(TEACHERS, key=lambda k: scores[k])
            outcome = actual[selected]
            rows.append({
                "episode": episode, "method": method,
                "right_scale_ground_truth": truth,
                "out_of_distribution": truth not in HYPOTHESES,
                "goal_magnitude_rad": magnitude,
                "probe_arm": ("none" if arm is None else "left" if arm == 0 else "right"),
                "posterior_nominal": belief[0],
                "selected_teacher": selected,
                "oracle_teacher": oracle,
                "ee_error_m": outcome["combined_ee_error_m"],
                "inter_arm_contact_steps": outcome["inter_arm_contact_steps"],
                "minimum_eef_separation_m": outcome["min_eef_separation_m"],
                "oracle_error_m": actual[oracle]["combined_ee_error_m"],
                "probe_duration_s": 0. if arm is None else cfg.probe_s,
            })
    summary = {}
    for method in ("no_probe", "fixed_left_probe", "random_probe", "active_probe"):
        subset = [r for r in rows if r["method"] == method]
        summary[method] = {
            "episodes": len(subset),
            "mean_ee_error_m": float(np.mean([r["ee_error_m"] for r in subset])),
            "mean_probe_duration_s": float(np.mean([r["probe_duration_s"] for r in subset])),
            "total_inter_arm_contact_steps": sum(r["inter_arm_contact_steps"] for r in subset),
            "oracle_agreement_rate": float(np.mean([r["selected_teacher"] == r["oracle_teacher"] for r in subset])),
        }
    return {
        "experiment": "dual_real_ur5e_scripted_teacher_benchmark",
        "real_vla": False, "robot_model": "MuJoCo Menagerie Universal Robots UR5e x2",
        "source": "https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e",
        "config": asdict(cfg), "declared_right_actuation_hypotheses": list(HYPOTHESES),
        "active_probe_arm": "left" if active_arm == 0 else "right",
        "diagnostic_prediction_separation_rad": discriminations,
        "summary": summary, "episodes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--out", type=Path, default=Path("runs/dual-ur5e.json"))
    parser.add_argument("--render", type=Path, help="Optional GIF file (requires imageio and working EGL/OpenGL)")
    args = parser.parse_args()
    cfg = Config(episodes=args.episodes, seed=args.seed, duration_s=args.duration)
    report = run_benchmark(args.menagerie, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if args.render:
        import imageio.v2 as imageio
        model = load_model(args.menagerie)
        goal_q, goal_sites = targets(model, .26)
        _, frames = rollout(model, "synchronous", goal_q, goal_sites, duration=cfg.duration_s, render=True)
        args.render.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.render, frames, duration=cfg.duration_s / max(len(frames), 1))


if __name__ == "__main__":
    main()
