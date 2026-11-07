"""Multi-VLA bridge adapter tests using documented interfaces and test doubles.

No mock is presented as a pretrained checkpoint. MuJoCo integration runs in CI.
"""
import os
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from vla_council.causalforge.vla_bridge import (
    ExplicitActionMapping, VLABackend, SingleArmMapping,
    DualArmComposer, evaluate_teachers,
)
from vla_council.causalforge.dual_ur5e_policy import SPEC, BuiltinReachPolicy
from vla_council.causalforge.vla_interface import ActionSpec, ActionChunk

class OpenPIMock:
    def __init__(self, output): self.output=output
    def infer(self,batch): return {"actions":self.output}

class GR00TMock:
    def __init__(self, output): self.output=output
    def get_action(self,batch): return {"arm_action":self.output[None]}, {}

class LeRobotMock:
    def __init__(self, output): self.output=output
    def predict_action_chunk(self,batch): return {"action":self.output[None]}

class BridgeUnitTests(unittest.TestCase):
    def test_openpi_explicit_degrees_and_axis_reordering(self):
        source=replace(SPEC, ordered_axes=tuple(reversed(SPEC.ordered_axes)), units=("deg",)*12)
        map=ExplicitActionMapping(source,degrees_to_radians=True)
        joint=np.tile(np.linspace(-.4,.5,12),(3,1))
        reversed_degrees=np.rad2deg(joint[:,::-1])
        adapter=VLABackend("pi_mock","openpi",OpenPIMock(reversed_degrees),map,lambda obs:obs)
        result=adapter.predict({"state":123})
        np.testing.assert_allclose(result.actions,joint,atol=1e-12)
        self.assertEqual(result.spec,SPEC)
    def test_degrees_without_opt_in_rejected(self):
        with self.assertRaises(ValueError):
            ExplicitActionMapping(replace(SPEC,units=("deg",)*12))
    def test_incompatible_frame_mode_period_robot_rejected(self):
        for diff in [dict(frame="world"),dict(representation="ee_delta"),
                     dict(dt=.04),dict(robot="another_robot")]:
            with self.subTest(diff=diff),self.assertRaises(ValueError):
                ExplicitActionMapping(replace(SPEC,**diff))
    def test_missing_axis_rejected(self):
        with self.assertRaises(ValueError):
            ExplicitActionMapping(replace(SPEC,ordered_axes=SPEC.ordered_axes[:-1],
                                          units=SPEC.units[:-1]))
    def test_gr00t_one_batch_return(self):
        data=np.zeros((3,12))
        adapter=VLABackend("groot_mock","groot",GR00TMock(data),
                           ExplicitActionMapping(SPEC),lambda obs:obs, action_key="arm_action")
        self.assertEqual(adapter.predict({}).actions.shape,(3,12))
    def test_lerobot_action_chunk(self):
        data=np.zeros((4,12))
        adapter=VLABackend("smolvla_mock","lerobot",LeRobotMock(data),
                           ExplicitActionMapping(SPEC),lambda obs:obs)
        self.assertEqual(adapter.predict({}).actions.shape,(4,12))
    def test_invalid_batch_or_wrong_action_stream_rejected(self):
        adapter=VLABackend("groot","groot",GR00TMock(np.zeros((2,12))),
                           ExplicitActionMapping(SPEC),lambda obs:obs,action_key="missing")
        with self.assertRaises(ValueError): adapter.predict({})
    def test_invalid_dimensions_rejected(self):
        adapter=VLABackend("openpi","openpi",OpenPIMock(np.zeros((3,7))),
                           ExplicitActionMapping(SPEC),lambda obs:obs)
        with self.assertRaises(ValueError): adapter.predict({})
    def test_no_silent_arm_padding(self):
        original=SPEC.ordered_axes[:6]
        a=SingleArmMapping(ActionSpec("menagerie_left_ur5e","joint_position",
                                     original,("rad",)*6,"joint",SPEC.dt),"left")
        right=SPEC.ordered_axes[6:]
        b=SingleArmMapping(ActionSpec("menagerie_right_ur5e","joint_position",
                                     right,("rad",)*6,"joint",SPEC.dt),"right")
        composed=DualArmComposer(a,b)
        self.assertEqual(composed.compose(np.ones((5,6)),np.zeros((5,6))).actions.shape,(5,12))
        with self.assertRaises(ValueError):
            composed.compose(np.ones((5,6)),np.zeros((4,6)))

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"MuJoCo Menagerie required")
class BridgePhysicsTests(unittest.TestCase):
    def test_two_vendor_interfaces_run_on_same_robot(self):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model, reset,GOALS
        from vla_council.causalforge.dual_ur5e import ids
        root=Path(os.environ["CAUSALFORGE_MENAGERIE"])
        model=load_reach_model(root)
        ref=BuiltinReachPolicy(model)
        q=reset(model).qpos[ids(model)[0]].copy()
        original=ref.predict({"joint_positions_rad":q,"goal_xyz_m":GOALS})
        # The *same* recorded IK plan is exposed through two mock vendor APIs.
        # This checks interface and physics only; NOT independent policy quality.
        teacher_a=VLABackend("pi_test","openpi",OpenPIMock(original.actions),
                             ExplicitActionMapping(SPEC),lambda obs:obs)
        teacher_b=VLABackend("groot_test","groot",GR00TMock(original.actions),
                             ExplicitActionMapping(SPEC),lambda obs:obs,
                             action_key="arm_action")
        report=evaluate_teachers({"pi_test":teacher_a,"groot_test":teacher_b},menagerie=root)
        self.assertEqual(len(report["teachers"]),2)
        self.assertTrue(all(t["task_success"] for t in report["teachers"].values()))
        self.assertTrue(all(len(t["tool_error_m"])==2 for t in report["teachers"].values()))
        self.assertTrue(all(not t["inter_arm_contact_steps"] for t in report["teachers"].values()))

if __name__=="__main__":
    unittest.main()
