"""Aggregate paired student benchmark results across independent random seeds."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(paths):
    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("At least two independent seeds required")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    base = runs[0]
    seeds = [r["config"]["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate random seed")
    for run in runs:
        if run["experiment"] != "physical_synthetic_teacher_distillation" or run["real_vla"] is not False:
            raise ValueError("Incompatible experiment type")
        if run["backend"] != base["backend"] or set(run["methods"]) != set(base["methods"]):
            raise ValueError("Incompatible backend or baseline methods")
        for key in ("train", "test", "noise", "ridge"):
            if run["config"][key] != base["config"][key]:
                raise ValueError("Experiments use different configurations")
    metrics = {}
    for name in sorted(base["methods"]):
        losses = np.array([r["methods"][name]["student_task_loss"] for r in runs], dtype=float)
        if not np.isfinite(losses).all():
            raise ValueError("Invalid task loss")
        metrics[name] = {
            "mean_student_task_loss": float(np.mean(losses)),
            "sample_std": float(np.std(losses, ddof=1)),
            "per_seed_loss": losses.tolist()}
    diff = np.asarray(metrics["active"]["per_seed_loss"]) - np.asarray(metrics["fixed"]["per_seed_loss"])
    return {
        "experiment": "paired_seed_aggregate", "backend": base["backend"],
        "seeds": seeds, "runs": len(runs), "methods": metrics,
        "active_minus_fixed": {
            "mean": float(np.mean(diff)),
            "sample_std": float(np.std(diff, ddof=1)),
            "per_seed": diff.tolist()},
        "limitations": "Independent seeds, no power analysis, VLA inference or physical hardware."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/mujoco-seed-summary.json"))
    args = parser.parse_args()
    result = aggregate(args.files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
