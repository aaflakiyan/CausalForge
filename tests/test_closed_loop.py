import unittest
import numpy as np
from vla_council.causalforge.closed_loop import (
    Config, evaluate, features, sample_scenarios, teacher_action, observe,
    RidgeStudent, HYPOTHESIS_NAMES,
)
from vla_council.causalforge.teacher_council import Critique, decide
from vla_council.causalforge.vla_interface import ActionChunk, ActionSpec


class ClosedLoopTests(unittest.TestCase):
    def test_deterministic_full_pipeline(self):
        config = Config(train_episodes=35, test_episodes=25, seed=16)
        self.assertEqual(evaluate(config), evaluate(config))

    def test_all_baselines_and_finite_metrics(self):
        r = evaluate(Config(train_episodes=25, test_episodes=20))
        self.assertEqual(set(r['methods']), {'none', 'fixed', 'random', 'active', 'oracle'})
        for m in r['methods'].values():
            for key in ('student_mean_task_loss', 'council_mean_task_loss',
                        'oracle_mean_task_loss', 'mean_probe_cost_test'):
                self.assertTrue(np.isfinite(m[key]))
                self.assertGreaterEqual(m[key], 0)
            self.assertLessEqual(m['mean_probe_cost_test'], r['config']['probe_budget'])
        self.assertEqual(r['methods']['none']['mean_probe_cost_test'], 0.)

    def test_train_test_scenarios_separate(self):
        self.assertNotEqual(sample_scenarios(1, 42)[0][1],
                            sample_scenarios(1, 42 + 1_000_003)[0][1])

    def test_observable_student_features_do_not_depend_on_hidden_truth(self):
        b = {h: .5 for h in HYPOTHESIS_NAMES}
        scenario_a = ('weak_motor', .04, np.zeros(2))
        scenario_b = ('heavy_payload', .04, np.zeros(2))
        cfg = Config()
        pa, _, _ = observe(scenario_a, 'none', cfg, 10)
        pb, _, _ = observe(scenario_b, 'none', cfg, 10)
        np.testing.assert_array_equal(features(.04, pa), features(.04, pb))
        self.assertEqual(teacher_action(pa, .04)[0], teacher_action(pb, .04)[0])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): Config(train_episodes=0)
        with self.assertRaises(ValueError): features(-1, {'weak_motor': .5, 'heavy_payload': .5})
        with self.assertRaises(ValueError): RidgeStudent.fit(np.empty((0, 6)), np.empty((0,), dtype=int), .1)
        with self.assertRaises(ValueError): observe(('weak_motor', .02, np.zeros(2)), 'nonexistent', Config(), 12)


class CouncilTests(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec('panda', 'joint_velocity', ('joint1',), ('rad/s',), 'base', .05)
        self.proposals = [ActionChunk(np.array([[x], [x]]), self.spec, name)
                          for name, x in [('alpha', .1), ('beta', .2), ('gamma', .3)]]

    def test_picks_lowest_valid_physical_score(self):
        score = {'alpha': Critique(.01, 1, 0., .1),
                 'beta': Critique(.03, 0, 0., .1),
                 'gamma': Critique(.08, 0, 0., .1)}
        result = decide(self.proposals, lambda p: score[p.model_name])
        self.assertEqual(result.selected_model, 'beta')
        self.assertGreater(result.disagreement, 0)

    def test_abstains_when_no_safe_candidate(self):
        result = decide(self.proposals, lambda _: Critique(0., 2, 0., .1))
        self.assertEqual(result.abstained_reason, 'all_proposals_vetoed')
        self.assertIsNone(result.selected)

    def test_rejects_bad_scoring_and_horizons(self):
        with self.assertRaises(ValueError): Critique(float('nan'), 0, 0, 0)
        different_horizon = ActionChunk(np.array([[.1]]), self.spec, 'delta')
        with self.assertRaisesRegex(ValueError, 'horizon'):
            decide([self.proposals[0], different_horizon], lambda _: Critique(0, 0, 0, 0))


if __name__ == '__main__': unittest.main()
