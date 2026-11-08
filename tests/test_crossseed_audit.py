"""Seed-block uncertainty and per-case train/test diagnostics."""
import unittest
from vla_council.causalforge.crossseed_audit import bootstrap_seed_means

class CrossSeedMathTests(unittest.TestCase):
    def test_fixed_seed_is_deterministic(self):
        diffs=[[.01,.02],[.03,.04],[-.01,.0]]
        a=bootstrap_seed_means(diffs,seed=2026,draws=1000)
        b=bootstrap_seed_means(diffs,seed=2026,draws=1000)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["mean_student_minus_ik_m"],.015,places=12)
        lo,hi=a["bootstrap_95pct_seed_block_interval_m"]
        self.assertLessEqual(lo,.015)
        self.assertGreaterEqual(hi,.015)
    def test_empty_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_seed_means([])
        with self.assertRaises(ValueError):
            bootstrap_seed_means([[float("nan")]])

if __name__=="__main__":
    unittest.main()
