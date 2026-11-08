"""Matched-delay robustness evaluation of a trained student and its IK teacher.

The student is fitted ONCE per seed from actual MuJoCo demonstrations and
then evaluated on the same held-out goals and action-command delays as the
reference IK controller. The benchmark reports failures and contacts without
discarding poor runs. It is not a pretrained VLA comparison.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .ur5e_teacher_student import run,generate_goals,StudentPolicy
from .universal_teacher import UnifiedStudent
from .dual_ur5e_reach import load_reach_model
from .dual_ur5e_policy import evaluate,BuiltinReachPolicy


def run_stress(menagerie:Path, out:Path, *, seed=42, train=8, test=4,
               delays=(0,5,10)):
    if train<3 or test<2 or not delays or len(set(delays))!=len(delays):
        raise ValueError("Invalid train/test count or delay list")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    source=run(menagerie,out/"student-source",seed=seed,train=train,test=test)
    student=UnifiedStudent.from_dict(json.loads(
        (out/"student-source"/"student.json").read_text(encoding="utf-8")))
    reference=BuiltinReachPolicy(load_reach_model(menagerie))
    policy={"student":StudentPolicy(student),"ik_teacher":reference}
    goals=list(generate_goals(train+test,seed))[train:]
    trials=[]
    for index,goal in enumerate(goals):
        coverage=student.coverage(tuple(goal.ravel()),robot="menagerie_dual_ur5e")
        for delay in delays:
            for name,controller in policy.items():
                result=evaluate(controller,menagerie=menagerie,goals=goal,
                                command_delay_ticks=delay)
                trials.append(dict(test_episode=index,method=name,delay_ticks=delay,
                    delay_ms=20*delay,within_training_range=coverage["within_training_feature_ranges"],
                    goal_error_m=float(np.mean(result["tool_error_m"])),
                    max_goal_error_m=max(result["tool_error_m"]),
                    success=result["task_success"],
                    arm_contacts=result["inter_arm_contact_steps"],
                    table_contacts=result["table_contact_steps"]))
    summary={}
    for delay in delays:
        for method in ("student","ik_teacher"):
            trials_subset=[x for x in trials if x["delay_ticks"]==delay and x["method"]==method]
            summary[f"{method}:{delay}"]=dict(
                delay_ms=20*delay,method=method,cases=len(trials_subset),
                mean_goal_error_m=float(np.mean([x["goal_error_m"] for x in trials_subset])),
                success_rate=float(np.mean([x["success"] for x in trials_subset])),
                total_contacts=sum(x["arm_contacts"]+x["table_contacts"] for x in trials_subset))
    report=dict(
        experiment="dual_ur5e_trained_student_versus_IK_under_transport_latency",
        seed=seed,train=train,test=test,delays=list(delays),
        pretrained_vla=False,student_source=source["experiment"],
        summary=summary,trials=trials,
        warning="Both methods use same held-out goals. The student is ridge regression on scripted teacher data; transport delay is simulated, and no pre-trained VLA or real robot is evaluated.",
    )
    (out/"stress-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("runs/student-latency"))
    p.add_argument("--train",type=int,default=8)
    p.add_argument("--test",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--delay-ticks",nargs="+",type=int,default=[0,5,10])
    args=p.parse_args()
    report=run_stress(args.menagerie,args.out,seed=args.seed,
                      train=args.train,test=args.test,delays=tuple(args.delay_ticks))
    print(json.dumps(report["summary"],indent=2))

if __name__=="__main__":
    main()
