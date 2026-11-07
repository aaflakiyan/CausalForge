import unittest
import numpy as np
from vla_council.causalforge.ood_benchmark import Config, evaluate, novelty_score, evaluator


class TestOODBenchmark(unittest.TestCase):
    def test_deterministic_surrogate(self):
        cfg = Config(episodes_per_condition=2, mc_samples=8)
        self.assertEqual(evaluate(cfg), evaluate(cfg))

    def test_known_and_unknown_recorded(self):
        result = evaluate(Config(episodes_per_condition=2, mc_samples=8))
        self.assertEqual(set(result['methods']), {'none','random','active','fixed'})
        for method in result['methods'].values():
            self.assertEqual(set(method['per_condition']), {
                'weak_motor','heavy_payload','intermediate_dynamics','weak_and_heavy'})
            self.assertGreaterEqual(method['mean_task_loss'], 0.)
            self.assertLessEqual(method['probe_coverage'], 1.)
            if method['probe_coverage']:
                self.assertAlmostEqual(method['mean_probe_seconds'], .32)

    def test_novelty_rejects_foreign_observation(self):
        expected = {'a': np.zeros(2), 'b': np.ones(2)}
        self.assertLess(novelty_score(np.array([0., .02]),expected,.02), 3.5)
        self.assertGreater(novelty_score(np.array([10., 10.]),expected,.02), 3.5)
        with self.assertRaises(ValueError):
            novelty_score(np.array([np.nan]),expected,.1)

    def test_invalid_configs(self):
        with self.assertRaises(ValueError): Config(episodes_per_condition=0)
        with self.assertRaises(ValueError): Config(observation_noise=float('nan'))
        with self.assertRaises(ValueError): evaluator('bad')

    def test_mujoco_physics_integration(self):
        from vla_council.causalforge.mujoco_push import mujoco
        if mujoco is None:
            self.skipTest('MuJoCo not installed')
        result = evaluate(Config(backend='mujoco', episodes_per_condition=1, mc_samples=6))
        self.assertEqual(result['backend'], 'mujoco')
        self.assertTrue(all(np.isfinite(m['mean_task_loss']) for m in result['methods'].values()))
