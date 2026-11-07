"""Fault-aware scripted-teacher selection on actual dual-UR5e Cartesian reaching.

No pretrained VLA; no learned student. The right actuator interface is
modelled as a scale error in position setpoint delivery, not faulty torque.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from .dual_ur5e_reach import GOALS, load_reach_model, execute
from .dual_ur5e import probe_result, posterior

HYPOTHESES = (1.0, .55)
TEACHERS = {"nominal_setpoints": 1., "compensated_right": 1. / .55}
STRATEGIES = ("no_probe", "fixed_left_probe", "random_probe", "active_probe")

@dataclass(frozen=True)
class Config:
    episodes: int = 18
    seed: int = 42
    noise_rad: float = .004
    duration_s: float = 3.
    probe_duration_s: float = .28
    probe_delta_rad: float = .3

def benchmark(menagerie: Path, cfg: Config = Config()):
    model = load_reach_model(menagerie)
    rng = np.random.default_rng(cfg.seed)
    probe_predictions = np.array([
        [probe_result(model, arm, scale, cfg.probe_delta_rad, cfg.probe_duration_s)
         for scale in HYPOTHESES] for arm in (0, 1)])
    discrimination = np.abs(probe_predictions[:, 0] - probe_predictions[:, 1])
    active_arm = int(np.argmax(discrimination))
    cases = []
    for ep in range(cfg.episodes):
        truth = (1.0, .55, .72)[ep % 3]
        goal = GOALS.copy()
        # Independent Cartesian test targets around two distinct workstations.
        goal[:, 1] += rng.uniform(-.035, .035, size=2)
        goal[:, 2] += rng.uniform(-.025, .025, size=2)
        noise = float(rng.normal(0., cfg.noise_rad))
        outcomes = {
            name: execute(model, goals=goal, duration=cfg.duration_s,
                          right_command_scale=truth, right_compensation=factor)[0]
            for name, factor in TEACHERS.items()
        }
        def loss(out):
            # Safety violations are penalised, and still reported separately.
            return (float(np.mean(out["final_goal_errors_m"])) +
                .2 * (out["inter_arm_contact_frames"] + out["table_contact_frames"]))
        oracle = min(outcomes, key=lambda x: loss(outcomes[x]))
        for strategy in STRATEGIES:
            if strategy == "no_probe":
                arm = None
                belief = [0.5, 0.5]
            else:
                arm = (0 if strategy == "fixed_left_probe" else
                       int(rng.integers(0, 2)) if strategy == "random_probe" else active_arm)
                observation = probe_result(model, arm, truth, cfg.probe_delta_rad,
                                           cfg.probe_duration_s) + noise
                belief = posterior(observation, probe_predictions[arm], cfg.noise_rad)
            predicted = {}
            for name, factor in TEACHERS.items():
                predicted[name] = sum(
                    prior * loss(execute(model, goals=goal, duration=cfg.duration_s,
                                         right_command_scale=hyp, right_compensation=factor)[0])
                    for prior, hyp in zip(belief, HYPOTHESES))
            selected = min(predicted, key=predicted.get)
            out = outcomes[selected]
            cases.append({
                "episode":ep,"strategy":strategy,"truth_setpoint_scale":truth,
                "out_of_family":truth not in HYPOTHESES,
                "teacher":selected,"oracle_teacher":oracle,
                "probe_arm":arm,"probe_cost_s":0. if arm is None else cfg.probe_duration_s,
                "posterior_nominal":belief[0],
                "mean_error_m":float(np.mean(out["final_goal_errors_m"])),
                "max_error_m":float(max(out["final_goal_errors_m"])),
                "task_success":bool(max(out["final_goal_errors_m"]) < .07 and
                    not out["inter_arm_contact_frames"] and not out["table_contact_frames"]),
                "inter_arm_contact_frames":out["inter_arm_contact_frames"],
                "table_contact_frames":out["table_contact_frames"]
            })
    methods = {}
    for strategy in STRATEGIES:
        selected = [x for x in cases if x["strategy"] == strategy]
        methods[strategy] = {
            "episodes":len(selected),
            "mean_error_m":float(np.mean([x["mean_error_m"] for x in selected])),
            "task_success_rate":float(np.mean([x["task_success"] for x in selected])),
            "mean_probe_cost_s":float(np.mean([x["probe_cost_s"] for x in selected])),
            "oracle_agreement":float(np.mean([x["teacher"] == x["oracle_teacher"] for x in selected])),
            "unseen_condition_mean_error_m":float(np.mean([
                x["mean_error_m"] for x in selected if x["out_of_family"]]))
        }
    return {
        "task":"dual-UR5e independent 3D reaching with setpoint-bias diagnosis",
        "limitations":"Teachers are scripted; faults attenuate joint setpoint delivery, not torque; prediction and evaluation share simulator; no grasp or real VLA trained.",
        "config":asdict(cfg), "declared_hypotheses":list(HYPOTHESES),
        "active_probe_arm":active_arm,"probe_separation_rad":discrimination.tolist(),
        "summary":methods,"episodes":cases,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=18)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out",type=Path,default=Path("runs/dual-ur5e-recovery.json"))
    args=p.parse_args()
    report=benchmark(args.menagerie,Config(episodes=args.episodes,seed=args.seed))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))
if __name__=="__main__":main()
