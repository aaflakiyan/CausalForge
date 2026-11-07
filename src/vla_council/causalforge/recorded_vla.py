"""Offline evaluation of recorded VLA actions, not physical task success.

Recordings must be produced externally using compatible robot-specific adapters.
Training labels may select training teachers; test targets are used only for scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .vla_interface import ActionSpec


def load_records(path):
    records, seen, canonical, names, observation_dim = [], set(), None, None, None
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"episode_id", "split", "observation", "target_action", "teacher_actions", "action_spec"}
        if not required.issubset(item):
            raise ValueError(f"Line {number}: missing {sorted(required-set(item))}")
        identifier = str(item["episode_id"])
        if not identifier or identifier in seen:
            raise ValueError("Blank or duplicate episode_id")
        seen.add(identifier)
        if item["split"] not in ("train", "test"):
            raise ValueError("Split must be train or test")
        spec = item["action_spec"]
        parsed = ActionSpec(
            robot=spec["robot"], representation=spec["representation"],
            ordered_axes=tuple(spec["ordered_axes"]), units=tuple(spec["units"]),
            frame=spec["frame"], dt=float(spec["dt"]))
        if canonical is None:
            canonical = parsed
        elif parsed != canonical:
            raise ValueError("Mixed action frames/units/timesteps/robots: explicit conversion required")
        observation = np.asarray(item["observation"], dtype=float)
        target = np.asarray(item["target_action"], dtype=float)
        teachers = {str(k): np.asarray(v, dtype=float) for k, v in item["teacher_actions"].items()}
        if observation.ndim != 1 or not observation.size or target.shape != (len(parsed.ordered_axes),):
            raise ValueError("Observation or target dimensions invalid")
        if len(teachers) < 2 or any(not key for key in teachers):
            raise ValueError("At least two named teachers required")
        if any(value.shape != target.shape for value in teachers.values()):
            raise ValueError("Teacher action dimensions invalid")
        if not (np.isfinite(observation).all() and np.isfinite(target).all()
                and all(np.isfinite(value).all() for value in teachers.values())):
            raise ValueError("Non-finite action or observation")
        if names is None:
            names, observation_dim = set(teachers), observation.size
        elif set(teachers) != names or observation.size != observation_dim:
            raise ValueError("Teacher names or observation dimensions changed")
        records.append({
            "episode_id": identifier, "split": item["split"],
            "observation": observation, "target_action": target,
            "teacher_actions": teachers,
        })
    if {r["split"] for r in records} != {"train", "test"}:
        raise ValueError("Nonempty train and test splits required")
    return records


def ridge_fit(x, targets, ridge):
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Invalid ridge strength")
    design = np.column_stack((np.ones(len(x)), x))
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(
        np.vstack((design, np.sqrt(penalty))),
        np.vstack((targets, np.zeros((design.shape[1], targets.shape[1])))),
        rcond=None)[0]


def evaluate(path, ridge=1e-3):
    records = load_records(path)
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    names = sorted(train[0]["teacher_actions"])
    def unpack(rows):
        x = np.stack([r["observation"] for r in rows])
        target = np.stack([r["target_action"] for r in rows])
        teachers = np.stack([
            np.stack([r["teacher_actions"][name] for name in names])
            for r in rows])
        return x, target, teachers
    x_train, y_train, t_train = unpack(train)
    x_test, y_test, t_test = unpack(test)
    # Supervised router: intentionally uses TRAIN targets, never test targets.
    routing = np.argmin(np.mean((t_train - y_train[:, None, :])**2, axis=2), axis=1)
    chosen = t_train[np.arange(len(train)), routing]
    models = {
        "supervised_teacher_selection_student": ridge_fit(x_train, chosen, ridge),
        "average_teacher_student": ridge_fit(x_train, t_train.mean(axis=1), ridge),
        "ground_truth_supervised_student": ridge_fit(x_train, y_train, ridge),
    }
    def mse(y):
        return float(np.mean((y - y_test)**2))
    scores = {f"teacher/{name}": mse(t_test[:, index]) for index, name in enumerate(names)}
    scores["teacher_average"] = mse(t_test.mean(axis=1))
    design = np.column_stack((np.ones(len(test)), x_test))
    for name, weights in models.items():
        scores[name] = mse(design @ weights)
    return {
        "experiment": "recorded_action_imitation",
        "real_robot_success_claim": False,
        "provenance": "External recordings; no checkpoint verification in this command",
        "train_examples": len(train), "test_examples": len(test),
        "teacher_names": names, "action_mse": scores,
        "test_split_sha256": hashlib.sha256(",".join(sorted(r["episode_id"] for r in test)).encode()).hexdigest(),
        "limitations": "Action MSE is not task success or safety; routing uses supervised training labels."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("runs/recorded-vla.json"))
    args = parser.parse_args()
    report = evaluate(args.recordings, args.ridge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
