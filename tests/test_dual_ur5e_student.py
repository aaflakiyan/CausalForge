"""Unit and optional real-UR5e MuJoCo integration tests for plan distillation."""
import os
from pathlib import Path
import unittest
import numpy as np
from vla_council.causalforge.dual_ur5e_student import feature, RidgePlanStudent


class StudentUnitTests(unittest.TestCase):
    def test_features_use_only_goal_and_posterior(self):
        row = {"posterior_nominal": .7, "goal_magnitude_rad": .3}
        x = feature(row)
        self.assertTrue(np.allclose(x, [1., .7, 1., .7]))
        self.assertEqual(x.shape, (4,))

    def test_train_and_predict_scripted_teacher_class(self):
        rows = [
            {"posterior_nominal": p, "goal_magnitude_rad": .2,
             "selected_teacher": "synchronous" if p > .5 else "compensate_right"}
            for p in np.linspace(.05, .95, 40)
        ]
        student = RidgePlanStudent.fit(rows)
        self.assertEqual(student.predict(rows[0]), "compensate_right")
        self.assertEqual(student.predict(rows[-1]), "synchronous")

    def test_empty_train_rejected(self):
        with self.assertRaises(ValueError):
            RidgePlanStudent.fit([])


@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"), "Menagerie unavailable")
class StudentIntegrationTests(unittest.TestCase):
    def test_student_heldout_physics_benchmark(self):
        from vla_council.causalforge.dual_ur5e_student import evaluate
        rep = evaluate(Path(os.environ["CAUSALFORGE_MENAGERIE"]),
                       seed=71, train=12, test=6, duration=.8)
        self.assertNotEqual(rep["training_seed"], rep["test_seed"])
        for method, item in rep["methods"].items():
            self.assertTrue(np.isfinite(item["student_mean_ee_error_m"]))
            self.assertGreaterEqual(item["student_council_agreement"], 0.)
            self.assertLessEqual(item["student_council_agreement"], 1.)
            self.assertGreater(item["training_episodes_known_only"], 0)
            self.assertLess(item["training_episodes_known_only"], rep["train_episodes_total"])


if __name__ == "__main__":
    unittest.main()
