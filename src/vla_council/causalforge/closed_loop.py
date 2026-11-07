"""End-to-end *synthetic* teacher -> experiment -> distillation evaluation.

No pretrained VLA, robot controller, or real-world safety guarantee is implied.
The student's decision features contain posterior beliefs and goals, never the
hidden dynamics label. Training and test scenarios use disjoint RNG streams.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from .active import select_active_probe, update_belief
from .benchmark import PROBES, TEACHER_FORCES, CONTROL_DURATION, surrogate_predict
from .experiment import Probe
from .mujoco_push import HYPOTHESES

HYPOTHESIS_NAMES = tuple(HYPOTHESES)
FORCES = np.asarray(list(TEACHER_FORCES.values()), dtype=float)


@dataclass(frozen=True)
class Config:
    train_episodes: int = 160
    test_episodes: int = 120
    noise_std: float = 0.015
    probe_budget: float = 0.35
    seed: int = 42
    ridge: float = 0.025
    # Deliberately use the same two-hypothesis model in training/testing.
    # This is an in-distribution proof of mechanics, not a generalisation claim.

    def __post_init__(self):
        if min(self.train_episodes, self.test_episodes) < 1 or self.noise_std <= 0 or self.probe_budget < 0 or self.ridge < 0:
            raise ValueError('Invalid configuration')


def features(goal: float, belief: dict[str, float]) -> np.ndarray:
    """Nonlinear goal basis with belief conditioning; no hidden physics label.

    Gaussian goal kernels help a small linear student imitate the discrete
    candidate selection boundary, avoiding an artificial polynomial bottleneck.
    """
    if not np.isfinite(goal) or goal <= 0 or not belief:
        raise ValueError('Invalid goal or belief')
    p = np.asarray([belief[h] for h in HYPOTHESIS_NAMES])
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1.0):
        raise ValueError('Belief must be finite probability distribution')
    centres = np.linspace(.005, .175, 18)
    kernels = np.exp(-.5 * ((goal - centres) / .011) ** 2)
    return np.r_[1., goal, p[0], kernels, kernels * p[0]]


def teacher_action(belief: dict[str, float], goal: float) -> tuple[int, np.ndarray]:
    """Posterior expected loss of each synthetic teacher, including overshoot."""
    errors = np.array([
        sum(belief[h] * (abs(surrogate_predict(h, Probe('control', float(f), CONTROL_DURATION))[1] - goal)
                        + 0.5 * max(0., surrogate_predict(h, Probe('control', float(f), CONTROL_DURATION))[1] - goal))
            for h in HYPOTHESIS_NAMES)
        for f in FORCES
    ])
    return int(np.argmin(errors)), errors


def sample_scenarios(count: int, seed: int) -> list[tuple[str, float, np.ndarray]]:
    """Paired observations shared across every comparison method."""
    rng = np.random.default_rng(seed)
    names = rng.choice(HYPOTHESIS_NAMES, size=count)
    goals = rng.uniform(.015, .16, size=count)
    noise = rng.standard_normal((count, 2))
    return [(str(h), float(g), n.copy()) for h, g, n in zip(names, goals, noise)]


@lru_cache(maxsize=64)
def _initial_active_probe(noise_std: float, budget: float):
    """One-step optimal probe at identical prior for all independent episodes.

    The prior, hypothesis family and candidate probes are fixed, so repeating
    the same Monte Carlo optimisation at every episode is wasteful. A constant
    acquisition seed also keeps paired experiments independent of episode ID.
    """
    candidates = [p for p in PROBES if p.cost <= budget]
    if not candidates:
        return None
    prior = {h: 1 / len(HYPOTHESIS_NAMES) for h in HYPOTHESIS_NAMES}
    return select_active_probe(candidates, prior, surrogate_predict, noise_std,
                               remaining_budget=budget, seed=113,
                               samples_per_hypothesis=96).probe

def observe(scenario, method: str, cfg: Config, episode_seed: int) -> tuple[dict[str, float], float, str]:
    truth, _, noise = scenario
    prior = {h: 1 / len(HYPOTHESIS_NAMES) for h in HYPOTHESIS_NAMES}
    candidates = [p for p in PROBES if p.cost <= cfg.probe_budget]
    if method == 'none' or not candidates:
        return prior, 0., 'none'
    if method == 'fixed':
        probe = candidates[0]
    elif method == 'random':
        probe = candidates[int(np.random.default_rng(episode_seed).integers(len(candidates)))]
    elif method == 'active':
        probe = _initial_active_probe(cfg.noise_std, cfg.probe_budget)
        if probe is None:
            return prior, 0., 'none'
    elif method == 'oracle':
        # Oracle is an evaluation ceiling only; not available during training.
        return {h: float(h == truth) for h in HYPOTHESIS_NAMES}, 0., 'oracle'
    else:
        raise ValueError(f'Unknown method: {method}')
    measured = surrogate_predict(truth, probe) + noise * cfg.noise_std
    return update_belief(measured, probe, prior, surrogate_predict, cfg.noise_std), probe.cost, probe.name


@dataclass
class RidgeStudent:
    """Multiclass ridge imitation of discrete teacher choice (not a VLA)."""
    weights: np.ndarray

    @classmethod
    def fit(cls, xs: np.ndarray, classes: np.ndarray, ridge: float) -> 'RidgeStudent':
        if xs.ndim != 2 or classes.ndim != 1 or len(xs) != len(classes) or not len(xs):
            raise ValueError('Invalid training shapes')
        y = np.eye(len(FORCES))[classes]
        regularizer = ridge * np.eye(xs.shape[1]); regularizer[0, 0] = 0
        # Least-squares handles a singular design without leaking held-out data.
        weights = np.linalg.lstsq(np.vstack((xs, np.sqrt(regularizer))),
                                  np.vstack((y, np.zeros((xs.shape[1], len(FORCES))))), rcond=None)[0]
        return cls(weights)

    def action_index(self, x: np.ndarray) -> int:
        if x.shape != (self.weights.shape[0],):
            raise ValueError('Unexpected feature dimension')
        return int(np.argmax(x @ self.weights))


def evaluate(cfg: Config = Config()) -> dict:
    methods = ('none', 'fixed', 'random', 'active')
    train = sample_scenarios(cfg.train_episodes, cfg.seed)
    test = sample_scenarios(cfg.test_episodes, cfg.seed + 1_000_003)
    report = {'experiment': 'synthetic_closed_loop', 'not_real_vla': True,
              'backend': 'algebraic_surrogate', 'config': asdict(cfg), 'methods': {}}
    for method in (*methods, 'oracle'):
        # Each method trains its own student, the identical train/test tasks are
        # used across methods. Oracle reveals ground truth and is an upper bound.
        xs = []; ys = []
        train_costs = []
        for i, scenario in enumerate(train):
            posterior, cost, _ = observe(scenario, method, cfg, cfg.seed + 43 * i)
            goal = scenario[1]
            label, _ = teacher_action(posterior, goal)
            xs.append(features(goal, posterior)); ys.append(label)
            train_costs.append(cost)
        student = RidgeStudent.fit(np.stack(xs), np.asarray(ys), cfg.ridge)
        losses = []; direct_losses = []; oracle_losses = []; class_agreement = []; test_costs = []
        probe_counts = {}
        for i, scenario in enumerate(test):
            truth, goal, _ = scenario
            posterior, cost, probe_name = observe(scenario, method, cfg, cfg.seed + 171_071 + 43 * i)
            probe_counts[probe_name] = probe_counts.get(probe_name, 0) + 1
            x = features(goal, posterior)
            chosen = student.action_index(x)
            direct, _ = teacher_action(posterior, goal)
            class_agreement.append(chosen == direct)
            values = np.array([surrogate_predict(truth, Probe('control', float(f), CONTROL_DURATION))[1]
                               for f in FORCES])
            def loss(j):
                return abs(values[j] - goal) + .5 * max(0., values[j] - goal)
            losses.append(loss(chosen))
            direct_losses.append(loss(direct))
            oracle_losses.append(min(loss(j) for j in range(len(FORCES))))
            test_costs.append(cost)
        report['methods'][method] = {
            'student_mean_task_loss': float(np.mean(losses)),
            'council_mean_task_loss': float(np.mean(direct_losses)),
            'oracle_mean_task_loss': float(np.mean(oracle_losses)),
            'student_teacher_agreement': float(np.mean(class_agreement)),
            'mean_probe_cost_train': float(np.mean(train_costs)),
            'mean_probe_cost_test': float(np.mean(test_costs)),
            'test_probe_counts': probe_counts,
            'student_weights': student.weights.tolist(),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=160)
    parser.add_argument('--test', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=Path, default=Path('runs/closed-loop.json'))
    args = parser.parse_args()
    result = evaluate(Config(train_episodes=args.train, test_episodes=args.test, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Synthetic benchmark (not real VLA; lower loss is better):')
    for method, metrics in result['methods'].items():
        print(f"{method:>8}: student_loss={metrics['student_mean_task_loss']:.5f}  "
              f"council_loss={metrics['council_mean_task_loss']:.5f}  "
              f"probe_cost={metrics['mean_probe_cost_test']:.3f}")
    print(f'Report: {args.out}')


if __name__ == '__main__':
    main()
