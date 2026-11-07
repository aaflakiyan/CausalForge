"""Scene export, input validation and CPU MuJoCo dynamics regression checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from vla_council.causalforge.mujoco_push import XML, Physics, mujoco, run
from vla_council.causalforge.simulate_scene import export_scene, simulate


class SceneExportTests(unittest.TestCase):
    def test_checked_in_model_matches_canonical_xml(self):
        asset = Path(__file__).resolve().parents[1] / 'assets' / 'mujoco' / 'pusher.xml'
        self.assertEqual(asset.read_text(encoding='utf-8'), XML.rstrip() + '\n')

    def test_scene_export_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = export_scene(Path(tmp) / 'mujoco' / 'pusher.xml')
            self.assertEqual(out.read_text(), XML.rstrip() + '\n')

    def test_invalid_inputs_do_not_require_mujoco(self):
        if mujoco is None:
            self.skipTest('MuJoCo only')
        with self.assertRaises(ValueError):
            simulate(Physics(), float('nan'), 0.5)
        with self.assertRaises(ValueError):
            simulate(Physics(), 5, 0.5, stride=0)


@unittest.skipIf(mujoco is None, 'MuJoCo required; runs in physics CI')
class ScenePhysicsTests(unittest.TestCase):
    def test_model_compiles_from_bundled_xml(self):
        model = mujoco.MjModel.from_xml_string(XML)
        path = Path(__file__).resolve().parents[1] / 'assets' / 'mujoco' / 'pusher.xml'
        bundled = mujoco.MjModel.from_xml_path(str(path))
        self.assertEqual(model.nq, bundled.nq)
        self.assertEqual(model.nu, bundled.nu)

    def test_trajectory_matches_existing_benchmark(self):
        physics = Physics(motor_scale=.42, payload_mass=2.0)
        rows, summary = simulate(physics, 7.0, 0.8)
        expected = run(physics, 7.0, 0.8)
        self.assertAlmostEqual(summary['pusher_displacement_m'], expected.pusher_displacement, places=9)
        self.assertAlmostEqual(summary['payload_displacement_m'], expected.payload_displacement, places=9)
        self.assertGreater(len(rows), 2)
        self.assertTrue(all(np.isfinite(row['payload_x_m']) for row in rows))


if __name__ == '__main__':
    unittest.main()
