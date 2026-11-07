"""Reproducible experiment: diagnosis and model-based teacher selection.

No reported numbers are intended to transfer to humanoids or real VLAs.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from .experiment import Probe, choose_probe, posterior
from .mujoco_push import HYPOTHESES, predict, run, select_command


PROBES = [
    Probe('free_motion', force=3.0, duration=0.12, cost=0.0),
    Probe('contact_motion', force=3.0, duration=0.38, cost=40.0),
]


def run_case(true_hypothesis: str, verbose: bool = False) -> dict:
    hypotheses = list(HYPOTHESES)
    chosen = choose_probe(PROBES, hypotheses, predict, noise_std=0.01, risk_weight=1.0)
    actual = run(HYPOTHESES[true_hypothesis], chosen.probe.force, chosen.probe.duration).sensors()
    belief = posterior(actual, hypotheses, chosen.probe, predict, noise_std=0.01)
    inferred = max(belief, key=belief.get)
    action, scores = select_command(HYPOTHESES[inferred])
    outcome = run(HYPOTHESES[true_hypothesis], action, duration=0.42)
    fixed = run(HYPOTHESES[true_hypothesis], 4.0, duration=0.42)
    goal = 0.14
    result = {
        'true_hypothesis': true_hypothesis,
        'selected_experiment': chosen.probe.name,
        'experiment_scores': chosen.scores,
        'measurement': actual.tolist(),
        'posterior': belief,
        'inferred_hypothesis': inferred,
        'selected_teacher_force': action,
        'candidate_scores': scores,
        'selected_payload_displacement': outcome.payload_displacement,
        'fixed_baseline_payload_displacement': fixed.payload_displacement,
        'selected_absolute_goal_error': abs(outcome.payload_displacement-goal),
        'fixed_absolute_goal_error': abs(fixed.payload_displacement-goal),
    }
    if verbose:
        print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description='MuJoCo diagnostic physics toy demo')
    parser.add_argument('--true', choices=['weak_motor','heavy_payload','both'], default='both')
    parser.add_argument('--out', default=None, help='optional JSON output path')
    args = parser.parse_args()
    cases = list(HYPOTHESES) if args.true == 'both' else [args.true]
    results = [run_case(h) for h in cases]
    print(json.dumps(results, indent=2))
    if args.out:
        from pathlib import Path
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
