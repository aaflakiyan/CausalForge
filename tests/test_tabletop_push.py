import unittest
import numpy as np
from vla_council.causalforge.mujoco_push import mujoco
from vla_council.causalforge.tabletop_push import SCENE_PATH, simulate


class TabletopScenePresence(unittest.TestCase):
    def test_bundled_model_exists(self):
        text = SCENE_PATH.read_text()
        self.assertIn('gravity="0 0 -9.81"', text)
        self.assertIn('freejoint', text)


@unittest.skipIf(mujoco is None, 'MuJoCo required; run in CI')
class TabletopDynamicsTests(unittest.TestCase):
    def test_compiles_and_runs(self):
        model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.assertEqual(model.nu, 1)
        rows, report = simulate(force=7., duration=.9)
        self.assertGreater(len(rows), 10)
        self.assertTrue(all(np.isfinite(r['payload_x_m']) for r in rows))
        self.assertGreater(report['payload_displacement_x_m'], .001)

    def test_deterministic_and_no_uncommanded_x_motion(self):
        a = simulate(force=0., duration=.4)
        self.assertEqual(a, simulate(force=0., duration=.4))
        self.assertLess(abs(a[1]['payload_displacement_x_m']), .001)

    def test_invalid_physics_rejected(self):
        with self.assertRaises(ValueError):
            simulate(friction=0.)
        with self.assertRaises(ValueError):
            simulate(payload_mass=float('nan'))


if __name__ == '__main__':
    unittest.main()
