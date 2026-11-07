"""Repeated-seed, paired synthetic benchmarking with uncertainty intervals.

All comparisons use the same held-out scenarios within each seed. Intervals
summarise variability across independent seeds, not real-robot reliability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .closed_loop import Config, evaluate


def run_suite(*, seeds: int = 8, train: int = 400, test: int = 160,
              base_seed: int = 2026) -> dict:
    if seeds < 2 or train < 1 or test < 1:
        raise ValueError('At least two seeds and positive episode counts required')
    experiments = [evaluate(Config(train_episodes=train, test_episodes=test, seed=base_seed + k * 1009))
                   for k in range(seeds)]
    methods = tuple(experiments[0]['methods'])
    summary = {}
    for method in methods:
        values = np.array([e['methods'][method]['student_mean_task_loss'] for e in experiments])
        costs = np.array([e['methods'][method]['mean_probe_cost_test'] for e in experiments])
        summary[method] = {
            'mean_task_loss': float(values.mean()),
            'std_task_loss_across_seeds': float(values.std(ddof=1)),
            'mean_probe_cost': float(costs.mean()),
        }
    # Paired differences, not separate confidence intervals for each method.
    comparisons = {}
    for baseline in ('none', 'fixed', 'random'):
        d = np.array([e['methods']['active']['student_mean_task_loss'] -
                      e['methods'][baseline]['student_mean_task_loss'] for e in experiments])
        # Use empirical percentiles across seeds, not an inferential 95% CI.
        comparisons[f'active_minus_{baseline}'] = {
            'mean_difference': float(d.mean()),
            'seed_percentiles_2_5_97_5': [float(x) for x in np.percentile(d, [2.5, 97.5])],
            'active_better_in_n_seeds': int(np.sum(d < 0)),
        }
    return {'benchmark': 'synthetic_surrogate', 'is_real_vla': False,
            'note': 'No paired compute-cost matching and no generalisation beyond two predefined hypotheses.',
            'seeds': seeds, 'train_per_seed': train, 'test_per_seed': test,
            'summary': summary, 'paired_comparisons': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=8)
    parser.add_argument('--train', type=int, default=400)
    parser.add_argument('--test', type=int, default=160)
    parser.add_argument('--out', type=Path, default=Path('runs/repeated-seeds.json'))
    args = parser.parse_args()
    result = run_suite(seeds=args.seeds, train=args.train, test=args.test)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
