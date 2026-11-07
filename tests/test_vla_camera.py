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

    def test_visual_openpi_interface_receives_real_mujoco_rgb(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate,SPEC
        from vla_council.causalforge.vla_bridge import ExplicitActionMapping,VLABackend
        class VisualMockClient:
            def __init__(self,model):
                self.ref=BuiltinReachPolicy(model)
            def infer(self,observation):
                return {"actions": self.ref.predict(observation).actions}
        observed=[]
        def preprocess(obs):
            assert obs["rgb_overview"].shape == (240,320,3)
            assert obs["camera_frame"] == "mujoco/overview"
            observed.append(float(obs["rgb_overview"].std()))
            return obs
        backend=VLABackend(
            "mock_visual_openpi","openpi",VisualMockClient(self.model),
            ExplicitActionMapping(SPEC),preprocess)
        report=evaluate(backend,menagerie=self.root,include_rgb=True)
        self.assertTrue(report["task_success"],report)
        self.assertEqual(len(observed),1)
        self.assertGreater(observed[0],2.0)

    def test_invalid_camera_size_rejected(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        with self.assertRaises(ValueError):
            evaluate(BuiltinReachPolicy(self.model),menagerie=self.root,
                     include_rgb=True,image_width=32)

if __name__=="__main__":
    unittest.main()
