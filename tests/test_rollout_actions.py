"""Validate external policy metadata before replay in MuJoCo."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from vla_council.causalforge.rollout_actions import PUSHER_SPEC, execute, parse_record
from vla_council.causalforge.mujoco_push import mujoco
from vla_council.causalforge.vla_interface import ActionChunk


def fixture():
    return {'model_name': 'synthetic_test_policy',
            'action_spec': {'robot': 'causalforge_pusher', 'representation': 'motor_control',
                            'ordered_axes': ['pusher_motor'], 'units': ['actuator_ctrl'],
                            'frame': 'actuator', 'dt': 0.02},
            'actions': [[7.0]] * 40}


class RecordTests(unittest.TestCase):
    def read_record(self, record):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / 'actions.json'
            file.write_text(json.dumps(record))
            return parse_record(file)

    def test_valid_record(self):
        chunk = self.read_record(fixture())
        self.assertEqual(chunk.actions.shape, (40, 1))
        self.assertEqual(chunk.spec, PUSHER_SPEC)

    def test_incompatible_units_rejected(self):
        record = fixture()
        record['action_spec']['units'] = ['N']
        with self.assertRaises(ValueError):
            self.read_record(record)

    def test_bounds_rejected(self):
        record = fixture()
        record['actions'][0] = [15]
        with self.assertRaises(ValueError):
            self.read_record(record)


@unittest.skipIf(mujoco is None, 'MuJoCo required; validated in CI')
class PolicyPhysicsTests(unittest.TestCase):
    def test_chunk_runs_and_is_deterministic(self):
        chunk = ActionChunk(np.full((40,1), 7.0), PUSHER_SPEC, 'mock_policy')
        report = execute(chunk)
        self.assertGreater(report['payload_displacement_m'], 0.001)
        self.assertEqual(report, execute(chunk))
        self.assertEqual(len(report['samples']), 40)


if __name__ == '__main__':
    unittest.main()
