"""Aggregate independent-seed dual UR5e teacher selection comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .dual_ur5e import Config, run_benchmark

METHODS = ("no_probe", "fixed_left_probe", "random_probe", "active_probe")


def aggregate(menagerie: Path, seeds=(42, 43, 44, 45, 46), episodes=30, duration=1.5):
    reports = [
        run_benchmark(menagerie, Config(seed=int(seed), episodes=episodes, duration_s=duration))
        for seed in seeds
    ]
    methods = {}
    for method in METHODS:
        vals = np.asarray([r["summary"][method]["mean_ee_error_m"] for r in reports])
        methods[method] = {
            "mean_ee_error_m": float(vals.mean()),
            "std_across_seeds_m": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "errors_by_seed_m": vals.tolist(),
            "mean_probe_duration_s": float(np.mean([
                r["summary"][method]["mean_probe_duration_s"] for r in reports])),
        }
    for method in METHODS[1:]:
        diff = np.asarray(methods["no_probe"]["errors_by_seed_m"]) - np.asarray(
            methods[method]["errors_by_seed_m"])
        methods[method]["paired_reduction_vs_no_probe_m"] = float(diff.mean())
        methods[method]["paired_reductions_by_seed_m"] = diff.tolist()
    return {
        "experiment": "dual_ur5e_multiseed_scripted_policy_selection",
        "real_pretrained_vla": False, "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "methods": methods,
        "warning": "The teacher policies are scripted; the council predictions and test environment share a model apart from controlled right-arm tracking changes. Results do not establish sim-to-real or pretrained VLA improvements."
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--out", type=Path, default=Path("runs/dual-ur5e-multiseed.json"))
    args = p.parse_args()
    report = aggregate(args.menagerie, tuple(args.seeds), args.episodes, args.duration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
