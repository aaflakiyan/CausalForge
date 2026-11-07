import unittest
from vla_council.causalforge.physical_distillation import Config, evaluate


class PhysicalDistillationTests(unittest.TestCase):
    def test_surrogate_pipeline(self):
        result = evaluate(Config(backend="surrogate", train=24, test=20, seed=99))
        self.assertFalse(result["real_vla"])
        self.assertEqual(set(result["methods"]), {"none", "fixed", "random", "active"})
        self.assertEqual(result["methods"]["none"]["mean_test_probe_cost"], 0.)
        for name, metrics in result["methods"].items():
            self.assertGreaterEqual(metrics["student_task_loss"], 0)
            self.assertGreaterEqual(metrics["unseen_physics_test_loss"], 0)
            if name != "none":
                self.assertAlmostEqual(metrics["mean_test_probe_cost"], .32)

    def test_repeatable(self):
        cfg = Config(backend="surrogate", train=12, test=12, seed=14)
        self.assertEqual(evaluate(cfg), evaluate(cfg))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Config(backend="fake")
        with self.assertRaises(ValueError):
            Config(noise=0)


if __name__ == "__main__":
    unittest.main()
