"""Paired MuJoCo/surrogate diagnosis -> synthetic council -> ridge student.

No pretrained VLA weights are used. Teacher controls are fixed-force
candidates, and no withheld true physical condition enters teacher selection.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from .active import select_active_probe, update_belief
from .closed_loop import RidgeStudent, features
from .experiment import Probe
from .mujoco_push import HYPOTHESES
from .ood_benchmark import PROBES, CONTROLS, CONTROL_DURATION, UNKNOWN, evaluator


@dataclass(frozen=True)
class Config:
    backend: str = "surrogate"
    train: int = 80
    test: int = 60
    seed: int = 42
    noise: float = 0.015
    ridge: float = 0.025

    def __post_init__(self):
        if self.backend not in ("surrogate", "mujoco") or self.train < 1 or self.test < 1:
            raise ValueError("Invalid backend or episode count")
        if not np.isfinite(self.noise) or self.noise <= 0 or self.ridge < 0:
            raise ValueError("Invalid noise or ridge strength")


def evaluate(cfg: Config = Config()) -> dict:
    simulate = evaluator(cfg.backend)
    physics = {**HYPOTHESES, **UNKNOWN}

    @lru_cache(maxsize=256)
    def outcome(name, force, duration):
        # Every simulation starts from an independently initialised state.
        return simulate(physics[name], Probe("execute", float(force), float(duration))).copy()

    known = tuple(HYPOTHESES)
    prior = {h: 1. / len(known) for h in known}
    predict = lambda h, p: outcome(h, p.force, p.duration)
    active_probe = select_active_probe(
        list(PROBES), prior, predict, cfg.noise,
        remaining_budget=.32, samples_per_hypothesis=48, seed=cfg.seed).probe
    if active_probe is None:
        raise RuntimeError("No active probe selected")

    train_rng = np.random.default_rng(cfg.seed)
    test_rng = np.random.default_rng(cfg.seed + 1000003)

    def scenarios(rng, count, labels):
        return [(str(rng.choice(labels)), float(rng.uniform(.01, .12)),
                 rng.standard_normal(2)) for _ in range(count)]

    training = scenarios(train_rng, cfg.train, list(known))
    testing = scenarios(test_rng, cfg.test, list(physics))

    def posterior_for(example, method, episode_seed):
        truth, _, noise_draw = example
        if method == "none":
            return dict(prior), 0.
        if method == "fixed":
            probe = PROBES[0]
        elif method == "random":
            probe = PROBES[int(np.random.default_rng(episode_seed).integers(len(PROBES)))]
        elif method == "active":
            probe = active_probe
        else:
            raise ValueError("Unknown method")
        measured = outcome(truth, probe.force, probe.duration) + noise_draw * cfg.noise
        posterior = update_belief(measured, probe, prior, predict, cfg.noise)
        return posterior, probe.cost

    def choose(belief, goal):
        scores = [
            sum(belief[h] * (
                abs(outcome(h, f, CONTROL_DURATION)[1] - goal) +
                .5 * max(0., outcome(h, f, CONTROL_DURATION)[1] - goal))
                for h in known)
            for f in CONTROLS]
        return int(np.argmin(scores))

    def loss(truth, force, goal):
        position = outcome(truth, force, CONTROL_DURATION)[1]
        return abs(position - goal) + .5 * max(0., position - goal)

    report = {
        "experiment": "physical_synthetic_teacher_distillation",
        "backend": cfg.backend, "real_vla": False, "config": asdict(cfg),
        "training_conditions": list(known), "test_conditions": list(physics),
        "teacher_forces": list(CONTROLS), "methods": {}}

    for method in ("none", "fixed", "random", "active"):
        xs, labels, train_cost = [], [], []
        for i, example in enumerate(training):
            posterior, cost = posterior_for(example, method, cfg.seed + 17*i)
            xs.append(features(example[1], posterior))
            labels.append(choose(posterior, example[1]))
            train_cost.append(cost)
        student = RidgeStudent.fit(np.stack(xs), np.asarray(labels), cfg.ridge)
        student_losses, council_losses, oracle_losses, test_cost = [], [], [], []
        known_losses, unseen_losses = [], []
        for i, example in enumerate(testing):
            truth, goal, _ = example
            posterior, cost = posterior_for(example, method, cfg.seed + 111119 + 17*i)
            student_idx = student.action_index(features(goal, posterior))
            council_idx = choose(posterior, goal)
            student_loss = loss(truth, CONTROLS[student_idx], goal)
            student_losses.append(student_loss)
            council_losses.append(loss(truth, CONTROLS[council_idx], goal))
            oracle_losses.append(min(loss(truth, force, goal) for force in CONTROLS))
            (known_losses if truth in HYPOTHESES else unseen_losses).append(student_loss)
            test_cost.append(cost)
        report["methods"][method] = {
            "student_task_loss": float(np.mean(student_losses)),
            "council_task_loss": float(np.mean(council_losses)),
            "action_oracle_task_loss": float(np.mean(oracle_losses)),
            "known_test_loss": float(np.mean(known_losses)) if known_losses else None,
            "unseen_physics_test_loss": float(np.mean(unseen_losses)) if unseen_losses else None,
            "mean_train_probe_cost": float(np.mean(train_cost)),
            "mean_test_probe_cost": float(np.mean(test_cost)),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["surrogate", "mujoco"], default="surrogate")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs/physical-distillation.json"))
    args = parser.parse_args()
    result = evaluate(Config(backend=args.backend, train=args.train, test=args.test, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
