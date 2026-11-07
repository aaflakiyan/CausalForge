import math
import unittest
import numpy as np
from vla_council.causalforge.active import (
    normalize, update_belief, entropy, expected_information_gain,
    select_active_probe, diagnose_or_abstain,
)
from vla_council.causalforge.experiment import Probe
from vla_council.causalforge.benchmark import run_benchmark, surrogate_predict


class ActiveLearningTests(unittest.TestCase):
    def setUp(self):
        self.probes = [Probe('no_signal', 1., .1, .1), Probe('signal', 1., .3, .2)]
        self.predict = lambda h, p: np.array([0. if h == 'a' or p.name == 'no_signal' else 1.])
        self.belief = {'a': .5, 'b': .5}

    def test_prior_validation(self):
        for prior in ({}, {'a': -1}, {'a': float('nan')}, {'a': 0.}):
            with self.assertRaises(ValueError): normalize(prior)

    def test_posterior(self):
        out = update_belief(np.array([1.]), self.probes[1], self.belief, self.predict, .1)
        self.assertGreater(out['b'], .99)
        self.assertAlmostEqual(sum(out.values()), 1.)

    def test_uninformative_probe_has_zero_gain(self):
        gain = expected_information_gain(self.probes[0], self.belief, self.predict, .1, seed=0)
        self.assertLess(gain, 1e-9)

    def test_informative_probe_high_gain_and_selected(self):
        gain = expected_information_gain(self.probes[1], self.belief, self.predict, .1, seed=0)
        self.assertGreater(gain, .5)
        selected = select_active_probe(self.probes, self.belief, self.predict, .1,
                                       remaining_budget=.25, seed=0)
        self.assertEqual(selected.probe.name, 'signal')

    def test_no_budget_abstains(self):
        decision = select_active_probe(self.probes, self.belief, self.predict, .1,
                                       remaining_budget=.05)
        self.assertIsNone(decision.probe)
        self.assertEqual(decision.reason, 'no_affordable_probe')

    def test_no_information_abstains(self):
        decision = select_active_probe(self.probes[:1], self.belief, self.predict, .1,
                                       remaining_budget=1.)
        self.assertIsNone(decision.probe)

    def test_diagnosis_abstains_when_uncertain(self):
        self.assertIsNone(diagnose_or_abstain(self.belief))
        self.assertEqual(diagnose_or_abstain({'a': .95, 'b': .05}), 'a')

    def test_impossible_data_fails_fast(self):
        with self.assertRaises(ValueError):
            update_belief(np.array([math.nan]), self.probes[1], self.belief, self.predict, .1)
        with self.assertRaises(ValueError):
            expected_information_gain(self.probes[1], self.belief, self.predict, -1.)


class BenchmarkTests(unittest.TestCase):
    def test_reproducible_and_all_methods(self):
        a = run_benchmark(trials=8, seed=5)
        b = run_benchmark(trials=8, seed=5)
        self.assertEqual(a, b)
        self.assertEqual(set(a['methods']), {'active', 'random', 'fixed', 'none'})
        for entry in a['methods'].values():
            self.assertTrue(0 <= entry['coverage'] <= 1)
            self.assertTrue(0 <= entry['diagnostic_accuracy_all_trials'] <= 1)
            self.assertTrue(entry['mean_probe_cost'] <= a['budget_per_trial'])

    def test_no_hidden_truth_access_in_probe_choice(self):
        self.assertEqual(surrogate_predict('weak_motor', Probe('probe', 2, .2)).shape, (2,))

    def test_rejects_missing_mujoco(self):
        from vla_council.causalforge.mujoco_push import mujoco
        if mujoco is not None:
            self.skipTest('MuJoCo is installed')
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            run_benchmark(backend='mujoco', trials=1)


if __name__ == '__main__':
    unittest.main()

class InterfaceValidationTests(unittest.TestCase):
    def test_nan_sample_period_rejected(self):
        from vla_council.causalforge.vla_interface import ActionSpec
        with self.assertRaises(ValueError):
            ActionSpec('Panda', 'joint_velocity', ('j1',), ('rad/s',), 'base', float('nan'))
