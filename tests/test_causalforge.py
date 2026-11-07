import unittest
import numpy as np
from vla_council.causalforge.experiment import Probe, choose_probe, posterior, information_proxy


class TestCausalExperiments(unittest.TestCase):
    def test_disagreement(self):
        self.assertAlmostEqual(information_proxy({'a': np.array([0.]), 'b': np.array([.1])}, .1), .5)

    def test_chooses_informative_safe_probe(self):
        probes = [Probe('ambiguous',1.,.1), Probe('diagnostic',1.,.2), Probe('hazardous',1.,.3,cost=100)]
        def predict(h, p):
            if p.name == 'ambiguous': return np.array([0.])
            return np.array([0.]) if h == 'a' else np.array([.2])
        result = choose_probe(probes,['a','b'],predict, noise_std=.1)
        self.assertEqual(result.probe.name, 'diagnostic')

    def test_identifies_supported_hypothesis(self):
        probe = Probe('test',1.,.1)
        def predict(h, p): return np.array([0.]) if h == 'a' else np.array([1.])
        belief = posterior(np.array([1.]), ['a','b'], probe, predict, noise_std=.1)
        self.assertGreater(belief['b'], .999)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            information_proxy({'a':np.array([1.]),'b':np.array([2.,3.])})


class TestMuJoCoOptional(unittest.TestCase):
    def test_simulation_smoke(self):
        from vla_council.causalforge.mujoco_push import mujoco, Physics, run
        if mujoco is None:
            self.skipTest('MuJoCo not installed in this environment')
        result = run(Physics(), force=3, duration=.1)
        self.assertTrue(np.isfinite(result.pusher_displacement))
        self.assertGreater(result.pusher_displacement,0)


if __name__ == '__main__':
    unittest.main()

class TestVLAAdapterBoundary(unittest.TestCase):
    def test_rejects_incompatible_robot_frames(self):
        from vla_council.causalforge.vla_interface import ActionSpec, ActionChunk, assert_same_action_space
        a = ActionSpec('Panda','ee_delta_pose',('x',),('m',),'base',.05)
        b = ActionSpec('Panda','ee_delta_pose',('x',),('m',),'world',.05)
        with self.assertRaises(ValueError):
            assert_same_action_space([ActionChunk([[.01]],a,'teacher_a'), ActionChunk([[.02]],b,'teacher_b')])

    def test_rejects_nonfinite_actions(self):
        from vla_council.causalforge.vla_interface import ActionSpec, ActionChunk
        a = ActionSpec('Panda','ee_delta_pose',('x',),('m',),'base',.05)
        with self.assertRaises(ValueError):
            ActionChunk([[float('nan')]],a,'teacher_a')
