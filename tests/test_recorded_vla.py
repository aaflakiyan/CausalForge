"""Tests for action-spec validation and held-out recorded teacher evaluation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from vla_council.causalforge.recorded_vla import load_records, evaluate

SPEC = {
    "robot": "panda", "representation": "joint_velocity",
    "ordered_axes": ["joint1", "joint2"],
    "units": ["rad/s", "rad/s"], "frame": "robot_base", "dt": 0.05,
}

def make_record(i, split="train"):
    x = [float(i % 7) / 7, float(i % 5) / 5]
    y = [x[0] + x[1], x[0] - x[1]]
    return {
        "episode_id": str(i), "split": split, "observation": x,
        "target_action": y, "action_spec": SPEC,
        "teacher_actions": {"model_a": y, "model_b": [y[0] + 0.2, y[1] - 0.2]}
    }


class RecordedTests(unittest.TestCase):
    def write(self, path, items):
        path.write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")

    def test_evaluate(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "examples.jsonl"
            self.write(p, [make_record(i, "train" if i < 30 else "test") for i in range(40)])
            r = evaluate(p)
            self.assertEqual((r["train_examples"], r["test_examples"]), (30, 10))
            self.assertAlmostEqual(r["action_mse"]["teacher/model_a"], 0)
            self.assertGreater(r["action_mse"]["teacher/model_b"], 0)

    def test_reject_incompatible_units(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "examples.jsonl"
            items = [make_record(0), make_record(1, "test")]
            items[1]["action_spec"] = dict(SPEC, units=["degree/s", "degree/s"])
            self.write(p, items)
            with self.assertRaises(ValueError):
                load_records(p)

    def test_reject_duplicates(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "examples.jsonl"
            self.write(p, [make_record(1), make_record(1, "test")])
            with self.assertRaises(ValueError):
                load_records(p)

    def test_reject_non_finite(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "examples.jsonl"
            items = [make_record(0), make_record(1, "test")]
            items[1]["target_action"] = [float("nan"), 0]
            self.write(p, items)
            with self.assertRaises(ValueError):
                load_records(p)

    def test_test_target_not_used_for_training(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "examples.jsonl"
            rows = [make_record(i, "train" if i < 30 else "test") for i in range(40)]
            self.write(p, rows)
            original = evaluate(p)
            for row in rows[30:]:
                row["target_action"] = [v + 1 for v in row["target_action"]]
            self.write(p, rows)
            changed = evaluate(p)
            self.assertEqual(original["test_split_sha256"], changed["test_split_sha256"])
            self.assertNotEqual(
                original["action_mse"]["ground_truth_supervised_student"],
                changed["action_mse"]["ground_truth_supervised_student"]
            )


if __name__ == "__main__":
    unittest.main()
