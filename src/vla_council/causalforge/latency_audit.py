"""Measured latency-robustness audit for dual-UR5e action execution.

Independently repeats the same verified IK action chunk with different
transport delays; compares true MuJoCo outcome and contact counts. The source
policy, goals, model revision and entire validated action chunk are identical
across delay conditions. All error/success numbers are measured in physics,
not inferred from pre-programmed trajectories.

Separate from model-training evaluation: shows where even a good controller
can degrade under execution latency.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .dual_ur5e_policy import BuiltinReachPolicy, evaluate
from .dual_ur5e_reach import load_reach_model, GOALS
from .ur5e_teacher_student import generate_goals


def audit(menagerie: Path, *, seeds=(41, 42, 43), goals_per_seed=4,
          delay_ticks=(0, 2, 5, 10), out=None):
    if not seeds or len(set(seeds))!=len(seeds) or not delay_ticks:
        raise ValueError("Nonempty unique seeds and delay conditions are required")
    if goals_per_seed<1 or any(type(x) is not int or not 0<=x<=25 for x in delay_ticks):
        raise ValueError("Invalid goal count or delay ticks")
    # This model is used for IK only; evaluate loads a clean physics model
    # from the exact same Menagerie assets for each and every trial.
    model=load_reach_model(menagerie)
    policy=BuiltinReachPolicy(model)
    trials=[]
    for seed in seeds:
        for episode,goal in enumerate(generate_goals(goals_per_seed,seed)):
            for delay in delay_ticks:
                report=evaluate(policy,menagerie=menagerie,goals=goal,
                                command_delay_ticks=delay)
                trials.append(dict(
                    seed=seed,episode=episode,delay_ticks=delay,
                    delay_ms=20*delay,
                    left_error_m=report["tool_error_m"][0],
                    right_error_m=report["tool_error_m"][1],
                    mean_tool_error_m=float(np.mean(report["tool_error_m"])),
                    success=report["task_success"],
                    arm_contact_steps=report["inter_arm_contact_steps"],
                    table_contact_steps=report["table_contact_steps"],
                    max_tracking_error_rad=report["max_joint_tracking_error_rad"],
                    minimum_tool_site_separation_m=report["minimum_tool_site_separation_m"],
                    action_samples=report["actions_checked_before_execution"],
                ))
    summary={}
    for delay in delay_ticks:
        v=[x for x in trials if x["delay_ticks"]==delay]
        summary[str(delay)] = dict(
            cases=len(v),delay_ms=delay*20,
            mean_tool_error_m=float(np.mean([x["mean_tool_error_m"] for x in v])),
            max_tool_error_m=float(max(max(x["left_error_m"],x["right_error_m"]) for x in v)),
            task_success_rate=float(np.mean([x["success"] for x in v])),
            arm_contact_steps=sum(x["arm_contact_steps"] for x in v),
            table_contact_steps=sum(x["table_contact_steps"] for x in v),
            minimum_tool_site_separation_m=float(min(x["minimum_tool_site_separation_m"] for x in v)),
        )
    result=dict(
        experiment="measured_UR5e_position_command_transport_delay",
        robot="2x official MuJoCo Menagerie UR5e",
        pretrained_vla=False,
        seeds=list(seeds),goals_per_seed=goals_per_seed,
        delay_ticks=list(delay_ticks),results=summary,trials=trials,
        warning="Transport latency holds preceding joint position targets; not a torque failure, network jitter distribution, real-hardware safety validation or pretrained VLA benchmark."
    )
    if out:
        p=Path(out)
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",required=True,type=Path)
    p.add_argument("--out",default="runs/ur5e-latency-audit.json",type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=[41,42,43])
    p.add_argument("--goals-per-seed",type=int,default=4)
    p.add_argument("--delay-ticks",nargs="+",type=int,default=[0,2,5,10])
    args=p.parse_args()
    result=audit(args.menagerie,seeds=tuple(args.seeds),
                 goals_per_seed=args.goals_per_seed,
                 delay_ticks=tuple(args.delay_ticks),out=args.out)
    print(json.dumps(result["results"],indent=2))

if __name__=="__main__":
    main()
