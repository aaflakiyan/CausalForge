"""Opt-in RGB camera observation to external VLA policies, run on MuJoCo."""
import os
import unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"MuJoCo Menagerie required")
class VisionObservationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model
        cls.root=Path(os.environ["CAUSALFORGE_MENAGERIE"])
        cls.model=load_reach_model(cls.root)

    def test_rgb_is_actual_camera_frame_and_policy_can_use_it(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        class VisionAwarePolicy:
            def __init__(self,model):
                self.ref=BuiltinReachPolicy(model)
                self.observation=None
            def predict(self,observation):
                self.observation=observation
                return self.ref.predict(observation)
        policy=VisionAwarePolicy(self.model)
        result=evaluate(policy,menagerie=self.root,include_rgb=True)
        image=policy.observation["rgb_overview"]
        self.assertEqual(image.shape,(240,320,3))
        self.assertEqual(image.dtype,np.uint8)
        self.assertGreater(float(image.std()),2)
        self.assertEqual(policy.observation["camera_frame"],"mujoco/overview")
        self.assertEqual(policy.observation["camera_image_encoding"],"RGB8")
        self.assertTrue(policy.observation["task_instruction"].strip())
        self.assertTrue(result["task_success"],result)
        self.assertEqual(result["observation_rgb_shape"],[240,320,3])

    def test_invalid_camera_size_rejected(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        with self.assertRaises(ValueError):
            evaluate(BuiltinReachPolicy(self.model),menagerie=self.root,
                     include_rgb=True,image_width=32)

if __name__=="__main__":
    unittest.main()
