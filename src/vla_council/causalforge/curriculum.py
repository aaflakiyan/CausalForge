"""Active data-collection planning for heterogeneous teacher disagreement.

This is a BEFORE-label-selection planner. It uses policy disagreement,
embodiment and task features, not future success or test outcomes.
A reasoner can annotate a proposed physical probe, but only a subsequent
verified experiment can supply the labels used to train the student.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .universal_teacher import TeacherEvidence, validate_groups
from .physical_reasoning import PhysicalFact, ReasoningGate


def disagreement_curriculum(rows:list[TeacherEvidence], budget:int) -> dict:
    if budget<1: raise ValueError("Budget must be positive")
    groups=validate_groups(rows)
    train=[(key,items) for key,items in groups.items() if items[0].split=="train"]
    if not train: raise ValueError("No training episodes")
    scored=[]
    for (episode,robot),items in train:
        # Candidates share the same physical space, but may be unverified.
        if len(items)<2: continue
        actions=np.stack([item.action.actions for item in items])
        distances=np.linalg.norm(actions[:,None]-actions[None,:],axis=(2,3))
        disagreement=float(np.sum(np.triu(distances,1)) / (len(items)*(len(items)-1)/2))
        scored.append(dict(episode_id=episode,robot=robot,
                           teacher_names=sorted(item.teacher for item in items),
                           disagreement_norm=disagreement))
    scored.sort(key=lambda item:(-item["disagreement_norm"],item["robot"],item["episode_id"]))
    # No hidden test access; no privileged future losses or oracle scores.
    return dict(method="unlabelled_pairwise_action_disagreement",
                selected=scored[:budget],candidates=len(scored),budget=budget,
                warning="Priority list for data collection ONLY. Train after actual measurement and safety checks.")


def reasoning_curriculum_note(*,task:str,facts:list[PhysicalFact],
                              teachers:list[str],reasoner:ReasoningGate) -> dict:
    """Add a grounded, non-executable suggestion to the collection plan."""
    return reasoner.assess(task=task,facts=facts,proposals=teachers)
