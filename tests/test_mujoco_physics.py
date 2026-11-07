"""Physics integration checks. CI must install mujoco to execute these."""
import unittest
import numpy as np
from vla_council.causalforge.mujoco_push import Physics, mujoco, run


@unittest.skipIf(mujoco is None, 'MuJoCo unavailable; CI mujoco job must run this')
class TestPhysicalPusher(unittest.TestCase):
    def test_deterministic(self):
        a = run(Physics(), 5., .6)
        b = run(Physics(), 5., .6)
        self.assertEqual(a, b)

    def test_zero_force_no_uncommanded_motion(self):
        result = run(Physics(), 0., .3)
        self.assertLess(abs(result.pusher_displacement), 1e-8)
        self.assertLess(abs(result.payload_displacement), 1e-8)

    def test_strong_push_contacts_payload(self):
        result = run(Physics(), 7., .8)
        self.assertGreater(result.pusher_displacement, .1)
        self.assertGreater(result.payload_displacement, 0.001,
                           'Payload not reached: check geometry, actuator and contact settings')

    def test_weak_motor_less_motion(self):
        nominal = run(Physics(), 5., .5)
        weak = run(Physics(motor_scale=.42), 5., .5)
        self.assertGreater(nominal.pusher_displacement, weak.pusher_displacement)

    def test_limits_and_finite(self):
        result = run(Physics(motor_scale=.65, payload_mass=2.), 4., .4)
        self.assertTrue(np.all(np.isfinite(result.sensors())))
        with self.assertRaises(ValueError): run(Physics(), 15., .4)
        with self.assertRaises(ValueError): run(Physics(payload_mass=0.), 4., .4)
