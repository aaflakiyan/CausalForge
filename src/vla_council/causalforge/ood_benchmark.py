"""Out-of-distribution *physics diagnosis* benchmark, not a real VLA benchmark.

A single probe per episode, identical duration and nominal budget across
active/random/fixed strategies. Evaluate known hypotheses and unseen intermediate
parameter values. A predictive-residual gate can explicitly reject the closed
hypothesis set instead of confidently mislabelling an unknown configuration.

The physical backend executes MuJoCo for each observation and outcome. In the
surrogate backend the observation generator is algebraic and remains synthetic.
Both backends are exposed; results must always identify which was used.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from functools import lru_cache
from typing import Callable
import numpy as np
from .active import select_active_probe, update_belief
from .experiment import Probe
from .mujoco_push import HYPOTHESES, Physics, run, mujoco

# Duration/cost matched; commanded force is a separately reported intervention.
PROBES = (
    Probe('gentle', 2.0, 0.32, cost=0.32),
    Probe('moderate', 3.5, 0.32, cost=0.32),
    Probe('firm', 5.0, 0.32, cost=0.32),
)
CONTROLS = (2.5, 4.0, 7.0)
CONTROL_DURATION = 0.42
TARGET = 0.05
UNKNOWN = {
    'intermediate_dynamics': Physics(motor_scale=.72, payload_mass=1.75),
    'weak_and_heavy': Physics(motor_scale=.42, payload_mass=3.0),
}


@dataclass(frozen=True)
class Config:
    backend: str = 'surrogate'
    episodes_per_condition: int = 15
    seed: int = 2026
    observation_noise: float = .015
    model_noise: float = .015
    novelty_threshold: float = 3.5
    mc_samples: int = 32

    def __post_init__(self):
        if self.backend not in {'surrogate', 'mujoco'}:
            raise ValueError('backend must be surrogate or mujoco')
        if self.episodes_per_condition <= 0 or self.mc_samples <= 0:
            raise ValueError('episode and sample counts must be positive')
        if not (np.isfinite(self.observation_noise) and self.observation_noise > 0 and
                np.isfinite(self.model_noise) and self.model_noise > 0 and
                np.isfinite(self.novelty_threshold) and self.novelty_threshold > 0):
            raise ValueError('noise and novelty threshold must be positive finite')


def _surrogate(physics: Physics, probe: Probe) -> np.ndarray:
    """Algebraic observation generator: explicitly NOT MuJoCo or measured reality."""
    t = probe.duration
    return np.array([
        .5 * physics.motor_scale * probe.force * t*t,
        physics.motor_scale / (physics.payload_mass + .3) * probe.force * max(0., t-.22)**2 / 2,
    ])


def evaluator(backend: str) -> Callable[[Physics, Probe], np.ndarray]:
    if backend == 'surrogate':
        return _surrogate
    if backend == 'mujoco':
        if mujoco is None:
            raise RuntimeError('MuJoCo not installed; physics benchmark cannot be run')
        @lru_cache(maxsize=128)
        def cached(physics: Physics, force: float, duration: float):
            return run(physics, force, duration).sensors()
        return lambda p, q: cached(p, q.force, q.duration).copy()
    raise ValueError('Invalid backend')


def novelty_score(observation: np.ndarray, predictions: dict[str, np.ndarray], std: float) -> float:
    """Minimum RMS standardised residual across known physical hypotheses.

    A heuristic model-mismatch indicator, not a calibrated OOD probability.
    """
    if not predictions or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid novelty model')
    observation = np.asarray(observation, dtype=float)
    if not observation.size or not np.all(np.isfinite(observation)):
        raise ValueError('Invalid observation')
    distances = []
    for prediction in predictions.values():
        p = np.asarray(prediction, dtype=float)
        if p.shape != observation.shape or not np.all(np.isfinite(p)):
            raise ValueError('Invalid prediction')
        distances.append(float(np.sqrt(np.mean(((observation-p)/std)**2))))
    return min(distances)


def evaluate(cfg: Config = Config()) -> dict:
    simulate = evaluator(cfg.backend)
    known = dict(HYPOTHESES)
    conditions = {**known, **UNKNOWN}
    if len(conditions) != len(known)+len(UNKNOWN):
        raise ValueError('Duplicate condition labels')
    prior = {h: 1/len(known) for h in known}
    # Cache teacher predictions, NOT truth/observations from test episodes.
    prediction_cache = {(h, p.name): simulate(physics, p)
                        for h, physics in known.items() for p in PROBES}
    def predict(h: str, probe: Probe) -> np.ndarray:
        return prediction_cache[h, probe.name]
    seed = np.random.SeedSequence(cfg.seed)
    children = seed.spawn(cfg.episodes_per_condition * len(conditions))
    episodes = []
    for condition_idx, (name, phys) in enumerate(conditions.items()):
        for i in range(cfg.episodes_per_condition):
            child = children[condition_idx * cfg.episodes_per_condition + i]
            episodes.append((name, phys, i, condition_idx,
                             np.random.default_rng(child).standard_normal(2)))
    report = {'experiment': 'closed_hypothesis_diagnosis_with_unseen_physics',
              'real_vla': False, 'backend': cfg.backend,
              'config': asdict(cfg), 'hypotheses': {k: asdict(v) for k,v in known.items()},
              'unseen_conditions': {k: asdict(v) for k,v in UNKNOWN.items()},
              'methods': {}}
    for method in ('none', 'fixed', 'random', 'active'):
        results = []
        for name, physics, i, condition_idx, noise in episodes:
            is_known = name in known
            if method == 'none':
                probe = None
            elif method == 'fixed':
                probe = PROBES[0]
            elif method == 'random':
                # Independent RNG, shared episode seed; method order cannot change choice.
                probe = PROBES[int(np.random.default_rng(cfg.seed + 7919*i +
                                 17*condition_idx).integers(len(PROBES)))]
            else:
                probe = select_active_probe(list(PROBES), prior, predict,
                         cfg.model_noise, remaining_budget=.32,
                         samples_per_hypothesis=cfg.mc_samples, seed=cfg.seed).probe
            posterior = dict(prior)
            reject = False
            residual = None
            if probe is not None:
                measurement = simulate(physics, probe) + noise * cfg.observation_noise
                predictions = {h: predict(h, probe) for h in known}
                residual = novelty_score(measurement, predictions, cfg.model_noise)
                reject = residual > cfg.novelty_threshold
                posterior = update_belief(measurement, probe, prior, predict, cfg.model_noise)
            # Select a *synthetic* discrete controller, not a joint-space command.
            # The novelty gate flags inadequate model support; a conservative
            # fallback and independent safety controller remain future work.
            losses = {}
            for command in CONTROLS:
                control = Probe('execute', command, CONTROL_DURATION)
                losses[command] = sum(prob * abs(simulate(hphysics, control)[1] - TARGET)
                                     for hphysics, prob in ((known[h], p) for h,p in posterior.items()))
            # A rejected physical hypothesis set must not dispatch a command.
            # No-op is modelled as zero displacement in this one-axis demo.
            command = None if reject else min(losses, key=losses.get)
            actual = (0.0 if command is None else
                      simulate(physics, Probe('execute', command, CONTROL_DURATION))[1])
            oracle = min(abs(simulate(physics, Probe('execute', c, CONTROL_DURATION))[1] - TARGET)
                         for c in CONTROLS)
            results.append({'condition': name, 'known': is_known,
                            'probe': probe.name if probe else None,
                            'probe_duration': probe.duration if probe else 0.,
                            'probe_force': probe.force if probe else 0.,
                            'rejected': reject, 'novelty_score': residual,
                            'posterior': posterior,
                            'correct_label': (not reject and max(posterior, key=posterior.get)==name) if is_known else None,
                            'command': command, 'task_loss': abs(actual-TARGET),
                            'oracle_task_loss': oracle})
        known_results = [r for r in results if r['known']]
        unknown_results = [r for r in results if not r['known']]
        probed = [r for r in results if r['probe'] is not None]
        report['methods'][method] = {
            'known_classification_accuracy_unconditional': float(np.mean([r['correct_label'] for r in known_results])),
            'known_reject_rate': float(np.mean([r['rejected'] for r in known_results])),
            'unknown_reject_rate': float(np.mean([r['rejected'] for r in unknown_results])),
            'mean_task_loss': float(np.mean([r['task_loss'] for r in results])),
            'mean_oracle_task_loss': float(np.mean([r['oracle_task_loss'] for r in results])),
            'mean_probe_seconds': float(np.mean([r['probe_duration'] for r in results])),
            'mean_probe_force': float(np.mean([r['probe_force'] for r in results])),
            'probe_coverage': len(probed) / len(results),
            'probes': {p.name: sum(r['probe']==p.name for r in results) for p in PROBES},
            'per_condition': {c: {
                 'reject_rate': float(np.mean([r['rejected'] for r in results if r['condition']==c])),
                 'mean_task_loss': float(np.mean([r['task_loss'] for r in results if r['condition']==c]))
                 } for c in conditions},
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Paired known/unknown physics benchmark, NOT VLA evaluation')
    parser.add_argument('--backend', choices=['surrogate','mujoco'], default='surrogate')
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', type=Path, default=Path('runs/ood_benchmark.json'))
    args = parser.parse_args()
    result = evaluate(Config(backend=args.backend, episodes_per_condition=args.episodes, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'hypotheses' and k != 'unseen_conditions'}, indent=2))


if __name__ == '__main__':
    main()
