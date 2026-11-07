"""Fail-closed UR5e action validation tests; intentional unsafe inputs never step physics."""
import os
import unittest
from pathlib import Path
import numpy as np
from vla_council.causalforge.vla_interface import ActionChunk, ActionSpec
from vla_council.causalforge.dual_ur5e_policy import SPEC, _validate

class PolicyMetadataTests(unittest.TestCase):
    def test_fixed_axis_contract_has_12_unique_names(self):
        self.assertEqual(len(SPEC.ordered_axes),12)
        self.assertEqual(len(set(SPEC.ordered_axes)),12)
        self.assertEqual(SPEC.units,("rad",)*12)
        self.assertEqual(SPEC.frame,"joint")
        self.assertEqual(SPEC.dt,.02)
    def test_chunk_rejects_nonfinite_actions(self):
        with self.assertRaises(ValueError):
            ActionChunk(np.full((3,12),np.nan), SPEC, "bad")

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"Menagerie checkout required")
class PolicyMuJoCoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model, reset
        from vla_council.causalforge.dual_ur5e import ids
        cls.menagerie=Path(os.environ["CAUSALFORGE_MENAGERIE"])
        cls.model=load_reach_model(cls.menagerie)
        cls.home=reset(cls.model).qpos[ids(cls.model)[0]].ravel().copy()
    def chunk(self, spec=SPEC,actions=None):
        return ActionChunk(
            np.tile(self.home,(4,1)) if actions is None else actions,
            spec,"test")
    def test_valid_home_commands(self):
        result,n=_validate(self.chunk(),self.model,self.home)
        self.assertEqual(result.shape,(4,12))
        self.assertEqual(n,5)
    def test_wrong_robot_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,robot="ur10e")) ,self.model,self.home)
    def test_wrong_frame_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,frame="world")),self.model,self.home)
    def test_wrong_units_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,units=("deg",)*12)),self.model,self.home)
    def test_wrong_axis_order_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,ordered_axes=tuple(reversed(SPEC.ordered_axes)))),self.model,self.home)
    def test_wrong_representation_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,representation="joint_velocity")),self.model,self.home)
    def test_wrong_dt_rejected(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            _validate(self.chunk(replace(SPEC,dt=.025)),self.model,self.home)
    def test_excessive_speed_rejected(self):
        commands=np.tile(self.home,(4,1))
        commands[1,0]+=1.0
        with self.assertRaises(ValueError):
            _validate(self.chunk(actions=commands),self.model,self.home)
    def test_out_of_bounds_rejected(self):
        commands=np.tile(self.home,(4,1))
        commands[0,0]=999.0
        with self.assertRaises(ValueError):
            _validate(self.chunk(actions=commands),self.model,self.home)
    def test_reaching_runs_and_success(self):
        from vla_council.causalforge.dual_ur5e_policy import evaluate,BuiltinReachPolicy
        result=evaluate(BuiltinReachPolicy(self.model),menagerie=self.menagerie,record_trace=True)
        self.assertTrue(result["task_success"],result)
        self.assertEqual(result["actions_checked_before_execution"],150)
        self.assertEqual(result["physics_substeps_per_action"],5)
        self.assertEqual(len(result["trace"]),150)
        self.assertTrue(all(v>.7 for v in result["tool_displacement_m"]))
        self.assertTrue(all(v<.07 for v in result["tool_error_m"]))
        self.assertEqual(result["inter_arm_contact_steps"],0)
    def test_wrong_robot_task_fail_before_loading(self):
        from vla_council.causalforge.dual_ur5e_policy import evaluate,BuiltinReachPolicy
        with self.assertRaises(ValueError):
            evaluate(BuiltinReachPolicy(self.model),robot="panda",menagerie=self.menagerie)
        with self.assertRaises(ValueError):
            evaluate(BuiltinReachPolicy(self.model),task="grasp",menagerie=self.menagerie)
if __name__=="__main__":
    unittest.main()
