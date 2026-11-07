import unittest
import numpy as np
from vla_council.causalforge.suite import run_suite

class SuiteTests(unittest.TestCase):
    def test_paired_reproducibility_and_validity(self):
        x = run_suite(seeds=2, train=30, test=20, base_seed=73)
        self.assertEqual(x, run_suite(seeds=2, train=30, test=20, base_seed=73))
        self.assertEqual(set(x['summary']), {'none', 'fixed', 'random', 'active', 'oracle'})
        self.assertTrue(np.isfinite(x['paired_comparisons']['active_minus_none']['mean_difference']))

    def test_invalid(self):
        with self.assertRaises(ValueError): run_suite(seeds=1)
