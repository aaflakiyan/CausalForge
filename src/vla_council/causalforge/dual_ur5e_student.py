"""Distil simulated dual-UR5e council decisions into a compact ridge student.

The teacher policies are *scripted joint trajectories*, not pretrained VLAs.
Train/test episode seeds differ, and OOD tracking factors are excluded from
student training. Test ground truth is used only to measure final task loss.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .dual_ur5e import Config, TEACHERS, load_model, targets, rollout, run_benchmark

METHODS = ("no_probe", "fixed_left_probe", "random_probe", "active_probe")


def feature(row):
    p = float(row["posterior_nominal"])
    g = float(row["goal_magnitude_rad"])
    return np.array([1., p, g / .3, p * g / .3], dtype=float)


class RidgePlanStudent:
    """Fixed-size teacher-class logits, with no hidden physical truth in features."""
    def __init__(self, weight):
        self.weight = np.asarray(weight, dtype=float)

    @classmethod
    def fit(cls, rows, ridge=.03):
        if not rows:
            raise ValueError("No student training examples")
        x = np.stack([feature(r) for r in rows])
        labels = np.zeros((len(rows), len(TEACHERS)))
        for i, row in enumerate(rows):
            labels[i, TEACHERS.index(row["selected_teacher"])] = 1.
        reg = np.eye(x.shape[1]) * ridge
        reg[0, 0] = 0.
        return cls(np.linalg.solve(x.T @ x + reg, x.T @ labels))

    def predict(self, row):
        return TEACHERS[int(np.argmax(feature(row) @ self.weight))]


def evaluate(menagerie: Path, seed=71, train=36, test=18, duration=1.0):
    if train < 3 or test < 3:
        raise ValueError("Need at least 3 episodes per split")
    model = load_model(menagerie)
    train_rep = run_benchmark(menagerie, Config(seed=seed, episodes=train, duration_s=duration))
    test_rep = run_benchmark(menagerie, Config(seed=seed + 100_003, episodes=test, duration_s=duration))
    methods = {}
    for method in METHODS:
        train_rows = [row for row in train_rep["episodes"]
                      if row["method"] == method and not row["out_of_distribution"]]
        heldout = [row for row in test_rep["episodes"] if row["method"] == method]
        student = RidgePlanStudent.fit(train_rows)
        cached = {}
        errors, teacher_errors, agreements, known_err, ood_err = [], [], [], [], []
        for row in heldout:
            proposed = student.predict(row)
            key = (row["episode"], proposed)
            if key not in cached:
                goal_q, sites = targets(model, row["goal_magnitude_rad"])
                metrics, _ = rollout(
                    model, proposed, goal_q, sites, row["right_scale_ground_truth"],
                    duration=duration)
                cached[key] = metrics["combined_ee_error_m"]
            err = cached[key]
            errors.append(err)
            teacher_errors.append(float(row["ee_error_m"]))
            agreements.append(proposed == row["selected_teacher"])
            (ood_err if row["out_of_distribution"] else known_err).append(err)
        methods[method] = {
            "training_episodes_known_only": len(train_rows),
            "heldout_episodes": len(heldout),
            "student_mean_ee_error_m": float(np.mean(errors)),
            "council_mean_ee_error_m": float(np.mean(teacher_errors)),
            "student_council_agreement": float(np.mean(agreements)),
            "student_known_fault_error_m": float(np.mean(known_err)) if known_err else None,
            "student_unseen_fault_error_m": float(np.mean(ood_err)) if ood_err else None,
            "mean_probe_duration_s": float(test_rep["summary"][method]["mean_probe_duration_s"]),
        }
    return {
        "experiment": "dual_ur5e_scripted_teacher_student_distillation",
        "real_vla_checkpoint": False, "method": "ridge_multiclass_plan_student",
        "training_seed": seed, "test_seed": seed + 100_003,
        "train_episodes_total": train, "test_episodes_total": test,
        "known_conditions_for_training": [1., .45],
        "unseen_test_condition": .70,
        "teacher_plans": list(TEACHERS),
        "student_features": ["bias", "posterior_nominal", "goal_magnitude", "interaction"],
        "methods": methods,
        "note": "Plan selection is distilled, not 12-DOF continuous motor control. The teacher council and rollouts use the same robot model apart from a withheld tracking scale.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--train", type=int, default=36)
    parser.add_argument("--test", type=int, default=18)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("runs/dual-ur5e-student.json"))
    args = parser.parse_args()
    result = evaluate(args.menagerie, args.seed, args.train, args.test, args.duration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    main()
