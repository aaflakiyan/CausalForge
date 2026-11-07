"""Offline teacher data collection, student failure correction and evaluation."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable
import numpy as np
from .core import Observation
from .council import Council
from .teachers import default_teachers
from .sim import collision_free, make_episode, step, done
from .student import LinearStudent


def collect(episodes: int, out: Path, start_seed: int = 0) -> dict:
    council = Council(default_teachers())
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, interventions = 0, 0
    with out.open("w") as f:
        for ep in range(start_seed, start_seed + episodes):
            obs = make_episode(ep)
            for _ in range(80):
                if done(obs):
                    break
                decision = council.decide(obs)
                if decision.needs_intervention:
                    interventions += 1
                    break  # A real system would request an operator correction.
                f.write(json.dumps({"observation": asdict(obs), "action": decision.action,
                                    "teacher": decision.teacher, "source": "toy-council",
                                    "peer_reviews": [asdict(r) for r in decision.reviews],
                                    "critic_assessments": [{"subject": c.proposal.name,
                                       "feasible": c.feasible, "score": c.score,
                                       "reason": c.reason} for c in decision.critiques]}) + "\n")
                rows += 1
                obs = step(obs, np.asarray(decision.action))
    return {"episodes": episodes, "examples": rows, "intervention_requests": interventions, "path": str(out)}


def load_dataset(path: Path) -> tuple[list[Observation], list[tuple[float, float]]]:
    obs, actions = [], []
    for line in path.read_text().splitlines():
        obj = json.loads(line)
        obs.append(Observation(**obj["observation"]))
        actions.append(tuple(obj["action"]))
    return obs, actions


def rollout(seed: int, controller: Callable[[Observation], np.ndarray], max_steps: int = 100) -> dict:
    obs = make_episode(seed)
    rejections = 0
    for i in range(max_steps):
        if done(obs):
            return {"success": True, "steps": i, "rejections": rejections}
        action = controller(obs)
        if not collision_free(obs, action):
            rejections += 1
            # Simulated fail-closed behaviour, no action executed.
            return {"success": False, "steps": i, "rejections": rejections}
        obs = step(obs, action)
    return {"success": done(obs), "steps": max_steps, "rejections": rejections}


def evaluate(student: LinearStudent, episodes: int, start_seed: int = 1000) -> dict:
    council = Council(default_teachers())
    strategies = {
        "single_direct_teacher": lambda o: np.array(default_teachers()[0].propose(o).action),
        "council": lambda o: np.array(council.decide(o).action),
        "student_with_gate": student.act,
    }
    result = {}
    for name, controller in strategies.items():
        outcomes = [rollout(seed, controller) for seed in range(start_seed, start_seed + episodes)]
        result[name] = {"success_rate": round(sum(x["success"] for x in outcomes) / episodes, 3),
                        "rejected_runs": sum(x["rejections"] > 0 for x in outcomes),
                        "mean_steps": round(float(np.mean([x["steps"] for x in outcomes])), 1)}
    return result


def refine(student: LinearStudent, dataset: Path, out: Path, episodes: int = 30, start_seed: int = 3000) -> dict:
    """DAgger-inspired: query teacher council at states induced by the student.

    Safety gate rejects unsafe student commands; teacher labels are NOT ground truth
    and should be checked by actual task outcome before using on hardware.
    """
    council = Council(default_teachers())
    observations, actions = load_dataset(dataset)
    acquired, rejected = 0, 0
    for seed in range(start_seed, start_seed+episodes):
        o = make_episode(seed)
        for _ in range(80):
            if done(o): break
            decision = council.decide(o)
            if decision.needs_intervention: break
            observations.append(o)
            actions.append(decision.action)
            acquired += 1
            proposed = student.act(o)
            if collision_free(o, proposed):
                o = step(o, proposed)
            else:
                rejected += 1
                o = step(o, np.asarray(decision.action))
    updated = LinearStudent.fit(observations, actions)
    updated.save(out)
    return {"teacher_corrections": acquired, "unsafe_student_proposals": rejected,
            "total_training_examples": len(actions), "model": str(out)}
