import unittest
from vla_council.causalforge.vla_interface import ActionChunk, ActionSpec, CallableVLATeacher


class TestTeacherAdapter(unittest.TestCase):
    def setUp(self):
        self.spec = ActionSpec('panda', 'ee_delta_pose', ('x',), ('m',), 'base', .05)

    def test_callable_adapter(self):
        teacher = CallableVLATeacher('unit', self.spec,
                  lambda obs, task: ActionChunk([[.01]], self.spec, 'unit'))
        self.assertEqual(teacher.propose({}, 'reach').actions.shape, (1,1))

    def test_rejects_bad_metadata(self):
        teacher = CallableVLATeacher('unit', self.spec,
                  lambda obs, task: ActionChunk([[.01]], self.spec, 'wrong'))
        with self.assertRaises(ValueError): teacher.propose({}, 'reach')
        with self.assertRaises(ValueError): teacher.propose({}, '')

    def test_rejects_unvalidated_raw_array(self):
        teacher = CallableVLATeacher('unit', self.spec, lambda obs, task: [[.01]])
        with self.assertRaises(TypeError): teacher.propose({}, 'reach')
