"""Reproducible controlled diagnostics benchmark.

Two backends: `surrogate` (verified NumPy algebraic toy) and `mujoco`
(physical simulation requiring the separately installed MuJoCo package).
Not a VLA benchmark. Every method receives the same probe and cost budget.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .active import select_active_probe, update_belief, diagnose_or_abstain
from .experiment import Probe
from .mujoco_push import HYPOTHESES, predict as mujoco_predict, run as mujoco_run

TEACHER_FORCES = {'cautious': 2.5, 'nominal': 4.0, 'strong': 7.0}
GOAL_DISPLACEMENT = .05
CONTROL_DURATION = .42

PROBES = [
    Probe('free_short', 2.0, 0.10, cost=0.08),
    Probe('free_long', 3.0, 0.18, cost=0.15),
    Probe('contact_short', 3.0, 0.30, cost=0.25),
    Probe('contact_long', 4.0, 0.48, cost=0.35),
]


def surrogate_predict(hypothesis: str, probe: Probe) -> np.ndarray:
    """Transparent algebraic toy; not a physical dynamics solver.

    Early free motion primarily reveals motor response; contact motion also
    reveals payload response. Parameter dependence is intentionally simple.
    """
    physics = HYPOTHESES[hypothesis]
    t = probe.duration
    free = 0.5 * physics.motor_scale * probe.force * t * t
    contact = max(0., t - 0.22)
    payload = (physics.motor_scale / (physics.payload_mass + 0.3)) * probe.force * contact**2 / 2
    return np.array([free, payload], dtype=float)


def choose_teacher_force(belief: dict[str, float], predict, goal: float = GOAL_DISPLACEMENT) -> tuple[str, float]:
    """Minimise posterior expected terminal displacement error, without oracle access."""
    command_scores = {}
    for teacher, force in TEACHER_FORCES.items():
        probe = Probe('control', force, CONTROL_DURATION)
        command_scores[teacher] = sum(
            probability * abs(predict(hypothesis, probe)[1] - goal)
            for hypothesis, probability in belief.items()
        )
    teacher = min(command_scores, key=command_scores.get)
    return teacher, TEACHER_FORCES[teacher]


def run_benchmark(*, backend: str = 'surrogate', trials: int = 100,
                  noise_std: float = .015, confidence: float = .9,
                  budget: float = .35, seed: int = 42,
                  methods: tuple[str, ...] = ('active', 'random', 'fixed', 'none')) -> dict:
    if trials < 1 or noise_std <= 0 or budget < 0:
        raise ValueError('Invalid benchmark configuration')
    if backend not in {'surrogate', 'mujoco'}:
        raise ValueError('Unknown backend')
    if any(m not in {'active', 'random', 'fixed', 'none'} for m in methods):
        raise ValueError('Unknown method')
    predict = surrogate_predict if backend == 'surrogate' else mujoco_predict
    if backend == 'mujoco':
        # Explicitly fail instead of silently substituting toy results.
        from .mujoco_push import mujoco
        if mujoco is None:
            raise RuntimeError('MuJoCo backend requested, but mujoco is not installed')
    hypotheses = tuple(HYPOTHESES)
    rng = np.random.default_rng(seed)
    truths = [hypotheses[int(x)] for x in rng.integers(len(hypotheses), size=trials)]
    observations_noise = rng.normal(0, noise_std, size=(trials, 2))
    report = {'backend': backend, 'seed': seed, 'trials': trials,
              'noise_std': noise_std, 'confidence': confidence,
              'budget_per_trial': budget, 'methods': {}}
    for method in methods:
        correct = 0
        decided = 0
        spent = 0.
        brier = 0.
        goal_error = 0.
        oracle_goal_error = 0.
        selected_teacher_counts = {t: 0 for t in TEACHER_FORCES}
        selection_counts = {p.name: 0 for p in PROBES}
        selection_counts['none'] = 0
        for idx, truth in enumerate(truths):
            prior = {h: 1. / len(hypotheses) for h in hypotheses}
            affordable = [p for p in PROBES if p.cost <= budget]
            if method == 'none' or not affordable:
                probe = None
            elif method == 'fixed':
                probe = affordable[0]
            elif method == 'random':
                # Per-trial RNG, independent of method execution order.
                probe = affordable[int(np.random.default_rng(seed + idx * 8191).integers(len(affordable)))]
            else:
                decision = select_active_probe(affordable, prior, predict, noise_std,
                                               remaining_budget=budget, seed=seed + idx,
                                               samples_per_hypothesis=64)
                probe = decision.probe
            if probe is not None:
                measurement = predict(truth, probe) + observations_noise[idx]
                prior = update_belief(measurement, probe, prior, predict, noise_std)
                spent += probe.cost
                selection_counts[probe.name] += 1
            else:
                selection_counts['none'] += 1
            guess = diagnose_or_abstain(prior, confidence)
            if guess is not None:
                decided += 1
                correct += int(guess == truth)
            brier += sum((prior[h] - float(h == truth)) ** 2 for h in hypotheses)
            teacher, force = choose_teacher_force(prior, predict)
            selected_teacher_counts[teacher] += 1
            actual = predict(truth, Probe('control', force, CONTROL_DURATION))[1]
            goal_error += abs(actual - GOAL_DISPLACEMENT)
            oracle_goal_error += min(abs(predict(truth, Probe('control', candidate, CONTROL_DURATION))[1] - GOAL_DISPLACEMENT)
                                     for candidate in TEACHER_FORCES.values())
        report['methods'][method] = {
            'diagnostic_accuracy_all_trials': round(correct / trials, 4),
            'coverage': round(decided / trials, 4),
            'accuracy_when_decided': round(correct / decided, 4) if decided else None,
            'mean_probe_cost': round(spent / trials, 4),
            'mean_brier_score': round(brier / trials, 4),
            'mean_control_goal_error': round(goal_error / trials, 6),
            'oracle_goal_error': round(oracle_goal_error / trials, 6),
            'selected_teacher_counts': selected_teacher_counts,
            'probe_counts': selection_counts,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='CausalForge diagnostic benchmark (no real VLAs)')
    parser.add_argument('--backend', choices=['surrogate', 'mujoco'], default='surrogate')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('runs/benchmark.json'))
    args = parser.parse_args()
    report = run_benchmark(backend=args.backend, trials=args.trials, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
