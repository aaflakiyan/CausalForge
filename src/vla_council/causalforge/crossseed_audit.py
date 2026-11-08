"""Independent-seed evaluation of a trained student against scripted teachers.

Every seed creates disjoint train/test Cartesian goals. For each seed:
  * physically execute the two scripted teachers for all goal cases;
  * train a student from TRAIN outcomes only;
  * physically execute the student on held-out goals;
  * compare paired test errors, success rates and action coverage.

The confidence interval is a seed-block bootstrap of mean student-minus-IK
position error. It is a small-sample DESCRIPTIVE interval, not a promise of
generalisation to more difficult manipulation tasks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .ur5e_teacher_student import run, generate_goals
from .universal_teacher import UnifiedStudent


def bootstrap_seed_means(deltas: list[list[float]], *, seed=20261108, draws=2000):
    if not deltas or any(not x for x in deltas):
        raise ValueError("Need nonempty paired error differences per seed")
    values=np.asarray([np.mean(x) for x in deltas],dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite paired error differences")
    rng=np.random.default_rng(seed)
    sample=rng.integers(0,len(values),size=(draws,len(values)))
    distribution=values[sample].mean(axis=1)
    return dict(mean_student_minus_ik_m=float(values.mean()),
                seed_means_m=values.tolist(),
                bootstrap_95pct_seed_block_interval_m=np.quantile(distribution,[.025,.975]).tolist(),
                bootstrap_draws=draws)


def evaluate_seeds(menagerie:Path,out:Path,*,seeds=(41,42,43),train=8,test=4):
    if len(set(seeds))!=len(seeds) or not seeds or train<3 or test<2:
        raise ValueError("Invalid seed list or sample count")
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    per_seed=[]
    paired=[]
    for seed in seeds:
        experiment=run(menagerie,out/f"seed-{seed}",seed=seed,train=train,test=test)
        student=experiment["student_held_out"]
        ik=experiment["teacher_results"]["ik_teacher"][train:]
        underreach=experiment["teacher_results"]["underreach_teacher"][train:]
        assert len(student)==len(ik)==len(underreach)==test
        differences=[]
        for i,(s,t) in enumerate(zip(student,ik)):
            assert s["episode"]==t["episode"]==train+i
            differences.append(s["mean_goal_error_m"]-t["mean_goal_error_m"])
        paired.append(differences)
        model=UnifiedStudent.from_dict(json.loads(
            (out/f"seed-{seed}"/"student.json").read_text(encoding="utf-8")))
        goals=list(generate_goals(train+test,seed))
        # This diagnostic uses only the student's training feature bounds;
        # the held-out outcomes do not influence coverage decisions.
        coverage=[model.coverage(tuple(goal.ravel()),robot="menagerie_dual_ur5e")
                  ["within_training_feature_ranges"] for goal in goals[train:]]
        per_seed.append(dict(
            seed=seed,train=train,test=test,
            student_success=int(sum(s["success"] for s in student)),
            ik_success=int(sum(x["success"] for x in ik)),
            underreach_success=int(sum(x["success"] for x in underreach)),
            student_mean_error_m=float(np.mean([x["mean_goal_error_m"] for x in student])),
            ik_mean_error_m=float(np.mean([x["mean_goal_error_m"] for x in ik])),
            underreach_mean_error_m=float(np.mean([x["mean_goal_error_m"] for x in underreach])),
            student_contacts=int(sum(s["arm_contacts"]+s["table_contacts"] for s in student)),
            coverage_passed=sum(coverage),coverage_total=test,
            paired_student_minus_ik_m=differences,
        ))
    total_test=len(seeds)*test
    summary=dict(
        experiment="multi_seed_actual_mujoco_scripted_teacher_student_validation",
        pretrained_vla=False,
        seeds=list(seeds),training_goals_per_seed=train,held_out_goals_per_seed=test,
        total_held_out_goals=total_test,
        student_success_rate=sum(x["student_success"] for x in per_seed)/total_test,
        ik_success_rate=sum(x["ik_success"] for x in per_seed)/total_test,
        underreach_success_rate=sum(x["underreach_success"] for x in per_seed)/total_test,
        mean_student_error_m=float(np.mean([x["student_mean_error_m"] for x in per_seed])),
        mean_ik_error_m=float(np.mean([x["ik_mean_error_m"] for x in per_seed])),
        mean_underreach_error_m=float(np.mean([x["underreach_mean_error_m"] for x in per_seed])),
        student_contact_frames=sum(x["student_contacts"] for x in per_seed),
        heldout_within_training_range=sum(x["coverage_passed"] for x in per_seed),
        paired_comparison=bootstrap_seed_means(paired),
        per_seed=per_seed,
        warnings=[
            "Scripted IK/underreach teachers are not pretrained VLAs.",
            "Each seed fits its own student; results do not evaluate one shared cross-seed policy.",
            "Test goals are near training goals; skill transfer, contact tasks and hardware generalisation are untested.",
            "The seed-block bootstrap interval is descriptive and not a formal guarantee.",
            "Feature-range coverage is not calibrated uncertainty and is not a safety certificate.",
        ]
    )
    (out/"aggregate-report.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",required=True,type=Path)
    p.add_argument("--out",type=Path,default=Path("runs/ur5e-crossseed"))
    p.add_argument("--seeds",type=int,nargs="+",default=[41,42,43])
    p.add_argument("--train",type=int,default=8)
    p.add_argument("--test",type=int,default=4)
    args=p.parse_args()
    result=evaluate_seeds(args.menagerie,args.out,seeds=tuple(args.seeds),
                          train=args.train,test=args.test)
    print(json.dumps({x:result[x] for x in (
        "total_held_out_goals","student_success_rate","ik_success_rate",
        "underreach_success_rate","mean_student_error_m","mean_ik_error_m",
        "mean_underreach_error_m","heldout_within_training_range",
        "paired_comparison")},indent=2))

if __name__=="__main__":
    main()
