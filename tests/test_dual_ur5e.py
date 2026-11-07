"""Dual real UR5e MuJoCo benchmark validation (requires Menagerie asset checkout)."""
import os
from pathlib import Path
import unittest
import numpy as np
from vla_council.causalforge.dual_ur5e import Config, posterior, _factor


class UniversalRobotUnitTests(unittest.TestCase):
    def test_schedule_bounded_and_monotone(self):
        for name in ("synchronous", "left_first", "right_first", "conservative"):
            for side in (0, 1):
                values = [_factor(name, side, x) for x in np.linspace(0, 1.5, 30)]
                self.assertTrue(all(0 <= x <= 1. for x in values))
                self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
    def test_posterior_normalises(self):
        p = posterior(.07, [.02, .09], .01)
        self.assertAlmostEqual(sum(p), 1.)
        self.assertGreater(p[1], p[0])
    def test_invalid_episode_count(self):
        with self.assertRaises(ValueError):
            Config(episodes=0)


@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"), "Menagerie path not provided")
class UniversalRobotIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e import load_model
        cls.root = Path(os.environ["CAUSALFORGE_MENAGERIE"])
        cls.model = load_model(cls.root)

    def test_two_ur5e_six_dof(self):
        self.assertEqual(self.model.nq, 12)
        self.assertEqual(self.model.nu, 12)

    def test_opposing_bases_and_finite_sites(self):
        from vla_council.causalforge.dual_ur5e import initial, ids
        data = initial(self.model)
        _, _, sites = ids(self.model)
        self.assertEqual(len(sites), 2)
        self.assertTrue(np.isfinite(data.site_xpos[sites]).all())
        self.assertGreater(np.linalg.norm(data.site_xpos[sites[0]] - data.site_xpos[sites[1]]), .05)

    def test_reproducible_motion_and_fault_effect(self):
        from vla_council.causalforge.dual_ur5e import targets, rollout
        goal, xyz = targets(self.model, .25)
        nominal, _ = rollout(self.model, "synchronous", goal, xyz, right_scale=1., duration=.7)
        degraded, _ = rollout(self.model, "synchronous", goal, xyz, right_scale=.45, duration=.7)
        self.assertTrue(np.isfinite(nominal["combined_ee_error_m"]))
        self.assertTrue(np.isfinite(degraded["combined_ee_error_m"]))
        self.assertGreater(degraded["right_ee_error_m"], nominal["right_ee_error_m"])

    def test_paired_benchmark(self):
        from vla_council.causalforge.dual_ur5e import run_benchmark
        report = run_benchmark(self.root, Config(episodes=3, duration_s=.7))
        self.assertEqual(set(report["summary"]), {
            "no_probe", "fixed_left_probe", "random_probe", "active_probe"})
        self.assertEqual(len(report["episodes"]), 12)
        for record in report["episodes"]:
            self.assertTrue(np.isfinite(record["ee_error_m"]))
            self.assertGreaterEqual(record["probe_duration_s"], 0)


if __name__ == "__main__":
    unittest.main()
