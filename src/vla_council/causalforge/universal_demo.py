"""Zero-GPU multi-teacher learning smoke test.

All teachers and data in this demo are SYNTHETIC. They represent deliberately
different native action shapes and two robot embodiments, but do not use real
VLA checkpoints or claim physical robot success.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .universal_teacher import (
    TeacherEvidence,UnifiedStudent,save_evidence,load_evidence,evidence_digest)
from .vla_interface import ActionChunk,ActionSpec
from .curriculum import disagreement_curriculum,reasoning_curriculum_note
from .physical_reasoning import PhysicalFact,ReasoningGate


def synthetic_evidence(seed=42,n=80):
    if n<16: raise ValueError("Need at least sixteen synthetic episodes")
    rng=np.random.default_rng(seed)
    rows=[]
    for ep in range(n):
        features=np.array([
            ep/max(n-1,1), np.sin(ep*.18), np.cos(ep*.1), float(ep%2),
        ])
        split="train" if ep<n-16 else "test"
        digest=hashlib.sha256(f"synthetic_obs_{ep}".encode()).hexdigest()
        for robot,dim in (("dual_ur5e_mock",12),("humanoid_leg_mock",6)):
            spec=ActionSpec(robot,"joint_position",tuple(f"joint_{j}" for j in range(dim)),
                            ("rad",)*dim,"joint",.02)
            target=(features[0]*.06 + features[1]*.03
                    + np.arange(dim)*.004)[None,:]+np.arange(4)[:,None]*.015
            # Teacher A is better in one task regime, B in another.
            for name in ("synthetic_pi_style","synthetic_groot_style"):
                is_good=((name=="synthetic_pi_style")==bool(ep%2))
                bias=(.004 if is_good else .06)
                actions=target+bias
                native=np.column_stack([actions,np.ones(4)*ep])
                measured=float(np.mean((actions-target)**2))
                rows.append(TeacherEvidence(
                    episode_id=f"ep{ep:04d}",split=split,teacher=name,
                    architecture="synthetic_array_generator",checkpoint="MOCK/no_weights",
                    task_features=tuple(float(x) for x in features),
                    action=ActionChunk(actions,spec,name),
                    measured_loss=measured,safe=True,observation_digest=digest,
                    native_schema="synthetic_padded_native_stream",
                    native_action=tuple(tuple(float(v) for v in line) for line in native),
                    evidence_ref=f"synthetic_ground_truth_{ep}_{robot}_{name}",
                ))
    return rows


def run(out:Path,seed=42,n=80):
    rows=synthetic_evidence(seed,n)
    out.mkdir(parents=True,exist_ok=True)
    dataset=out/"synthetic_teacher_evidence.jsonl"
    save_evidence(dataset,rows)
    loaded=load_evidence(dataset)
    student=UnifiedStudent(latent_dim=4,ridge=.0001).fit(loaded)
    scores=student.evaluate(loaded)
    (out/"student.json").write_text(
        json.dumps(student.to_dict(),indent=2)+"\n",encoding="utf-8")
    selected=disagreement_curriculum(loaded,budget=8)
    notes=reasoning_curriculum_note(
        task="Investigate disagreement in simulated joint targets",
        facts=[PhysicalFact("observed_action_disagreement",selected["selected"][0]["disagreement_norm"],
                            "rad_l2","synthetic_action_comparison")],
        teachers=["synthetic_pi_style","synthetic_groot_style"],
        reasoner=ReasoningGate(lambda context:dict(
            hypothesis="Different policies may respond differently to task context",
            verdict="uncertain",evidence_refs=["observed_action_disagreement"],
            suggested_probe="Run both candidates in the same controlled simulator episode",
            rationale="Action disagreement alone cannot determine which action succeeds")),
    )
    report=dict(experiment="synthetic_heterogeneous_distillation_smoke_test",
                seed=seed,episodes=n,teachers=["synthetic_pi_style","synthetic_groot_style"],
                embodiments=list(student.heads),
                native_evidence_sha256=evidence_digest(loaded),
                training_data=dataset.name,student_file="student.json",
                scores=scores,curriculum=selected,reasoning=notes,
                checkpoint_inference=False,physics_experiment=False,
                warning="All labels and policies are SYNTHETIC; not pretrained VLA inference or MuJoCo robot success.")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("runs/universal-student"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--episodes",type=int,default=80)
    args=p.parse_args()
    report=run(args.out,seed=args.seed,n=args.episodes)
    print(json.dumps({
        "experiment":report["experiment"],
        "embodiments":report["embodiments"],
        "held_out_scores":report["scores"]["per_embodiment"],
        "evidence":str(args.out/"synthetic_teacher_evidence.jsonl"),
        "trained_student":str(args.out/"student.json"),
        "warning":report["warning"],
    },indent=2))

if __name__=="__main__":
    main()
