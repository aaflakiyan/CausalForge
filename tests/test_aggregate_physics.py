import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from vla_council.causalforge.aggregate_physics import aggregate


class AggregateTests(unittest.TestCase):
    def test_paired_differences_and_duplicates(self):
        with TemporaryDirectory() as directory:
            paths = []
            for seed in (2, 3):
                path = Path(directory) / f"{seed}.json"
                path.write_text(json.dumps({
                    "experiment": "physical_synthetic_teacher_distillation",
                    "backend": "mujoco", "real_vla": False,
                    "config": {"seed": seed, "train": 30, "test": 20, "noise": .015, "ridge": .025},
                    "methods": {
                        "fixed": {"student_task_loss": float(seed)},
                        "active": {"student_task_loss": float(seed) + .01}
                    }}), encoding="utf-8")
                paths.append(path)
            result = aggregate(paths)
            self.assertEqual(result["runs"], 2)
            self.assertAlmostEqual(result["active_minus_fixed"]["mean"], .01)
            with self.assertRaises(ValueError):
                aggregate([paths[0], paths[0]])


if __name__ == "__main__":
    unittest.main()
