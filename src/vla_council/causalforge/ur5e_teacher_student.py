"""Real MuJoCo dual-UR5e imitation/distillation experiment, WITHOUT VLA weights.

Two scripted teachers are run on identical Cartesian goals in real Menagerie
UR5e physics. Measured outcome determines an offline training label. A shared
student head fits the measured-safe action chunks. Test episodes are held out.
The student is evaluated by executing its predicted commands in fresh MuJoCo.
No pretrained GR00T/OpenPI/Cosmos model is claimed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .dual_ur5e_reach import GOALS,load_reach_model
from .dual_ur5e_policy import BuiltinReachPolicy,SPEC,evaluate
from .vla_interface import ActionChunk
from .universal_teacher import (
    TeacherEvidence,UnifiedStudent,save_evidence,evidence_digest
)

class FixedChunkPolicy:
    def __init__(self,chunk):
        self.chunk=chunk
    def predict(self,observation):
        return self.chunk

class StudentPolicy:
    def __init__(self,student):
        self.student=student
    def predict(self,observation):
        goal=np.asarray(observation["goal_xyz_m"],dtype=float).ravel()
        return self.student.predict(goal,robot=SPEC.robot)

def generate_goals(n,seed):
    rng=np.random.default_rng(seed)
    for i in range(n):
        goal=GOALS.copy()
        goal[:,0]+=rng.uniform(-.020,.020,size=2)
        goal[:,1]+=rng.uniform(-.025,.025,size=2)
        goal[:,2]+=rng.uniform(-.015,.015,size=2)
        yield goal

def run(menagerie:Path,out:Path,*,seed=42,train=10,test=4):
    if train<3 or test<2: raise ValueError("Need at least 3 train and 2 test episodes")
    model=load_reach_model(menagerie)
    ref=BuiltinReachPolicy(model)
    from .dual_ur5e_reach import reset
    from .dual_ur5e import ids
    home=reset(model).qpos[ids(model)[0]].ravel().copy()
    records=[]
    goals=[]
    teacher_scores={"ik_teacher":[],"underreach_teacher":[]}
    for episode,goal in enumerate(generate_goals(train+test,seed)):
        goal=np.asarray(goal,dtype=float)
        goals.append(goal)
        observation=dict(goal_xyz_m=goal,joint_positions_rad=home.reshape(2,6))
        good=ref.predict(observation)
        # A deliberately biased second controller. Both physical outcomes are
        # measured; no selected teacher gets oracle test metrics online.
        bad_commands=home.reshape(1,-1)+.82*(good.actions-home.reshape(1,-1))
        bad=ActionChunk(bad_commands,SPEC,"scripted_underreach")
        digest=hashlib.sha256(goal.tobytes()).hexdigest()
        for name,chunk in (("ik_teacher",good),("underreach_teacher",bad)):
            outcome=evaluate(FixedChunkPolicy(chunk),menagerie=menagerie,goals=goal)
            teacher_scores[name].append(dict(
                episode=episode,mean_goal_error_m=float(np.mean(outcome["tool_error_m"])),
                success=outcome["task_success"],
                contact_free=(outcome["inter_arm_contact_steps"]==0 and outcome["table_contact_steps"]==0)))
            # A controller can fail its task without making physical contact.
            # Keep the provisional contact-based safety label separate from
            # success; selection uses measured loss among contact-free teachers.
            safe=(outcome["inter_arm_contact_steps"]==0 and outcome["table_contact_steps"]==0)
            loss=float(np.mean(outcome["tool_error_m"]))
            records.append(TeacherEvidence(
                episode_id=f"mujoco-{episode:04d}",
                split="train" if episode<train else "test",
                teacher=name,architecture="scripted_kinematic_controller",
                checkpoint="NONE/no_model_weights",
                task_features=tuple(float(v) for v in goal.ravel()),
                action=chunk,
                measured_loss=loss,safe=bool(safe),
                observation_digest=digest,
                native_schema="UR5e_absolute_joint_positions_T150_D12",
                native_action=tuple(tuple(float(v) for v in line)
                                    for line in chunk.actions),
                evidence_ref=f"measured_mujoco_episode_{episode}_teacher_{name}",
            ))
    out.mkdir(parents=True,exist_ok=True)
    save_evidence(out/"mujoco_teacher_evidence.jsonl",records)
    student=UnifiedStudent(latent_dim=6,ridge=.005).fit(records)
    (out/"student.json").write_text(json.dumps(student.to_dict(),indent=2)+"\n")
    offline=student.evaluate(records)
    student_scores=[]
    for ep in range(train,train+test):
        report=evaluate(StudentPolicy(student),menagerie=menagerie,goals=goals[ep])
        student_scores.append(dict(
            episode=ep,success=report["task_success"],
            mean_goal_error_m=float(np.mean(report["tool_error_m"])),
            arm_contacts=report["inter_arm_contact_steps"],
            table_contacts=report["table_contact_steps"]))
    result=dict(
        experiment="actual_MuJoCo_UR5e_scripted_teacher_distillation",
        pretrained_vla=False,reasoning_checkpoint=False,seed=seed,
        train_episodes=train,test_episodes=test,
        native_teacher_data_sha256=evidence_digest(records),
        teacher_results=teacher_scores,student_held_out=student_scores,
        held_out_student_success_rate=float(np.mean([r["success"] for r in student_scores])),
        held_out_student_mean_goal_error_m=float(np.mean([r["mean_goal_error_m"] for r in student_scores])),
        offline_imitation=offline,
        warning="Scripted joint-space teachers only; student is a linear multihead baseline. No pretrained VLA, humanoid generalisation or skill-retention claim.",
    )
    (out/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/ur5e-teacher-student"))
    p.add_argument("--train",type=int,default=10)
    p.add_argument("--test",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    report=run(a.menagerie,a.out,seed=a.seed,train=a.train,test=a.test)
    print(json.dumps({key:report[key] for key in (
        "experiment","train_episodes","test_episodes",
        "held_out_student_success_rate","held_out_student_mean_goal_error_m","warning")},indent=2))

if __name__=="__main__":
    main()
