"""MuJoCo regression: verify action delay is real, bounded and fail-closed."""
import os
from pathlib import Path
import unittest
import numpy as np

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"Official Menagerie assets required")
class LatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model
        cls.menagerie=Path(os.environ["CAUSALFORGE_MENAGERIE"])
        cls.model=load_reach_model(cls.menagerie)

    def test_zero_delay_default_bitwise_same(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        policy=BuiltinReachPolicy(self.model)
        ref=evaluate(policy,menagerie=self.menagerie)
        zero=evaluate(policy,menagerie=self.menagerie,command_delay_ticks=0)
        self.assertEqual(ref["tool_error_m"],zero["tool_error_m"])
        self.assertEqual(ref["inter_arm_contact_steps"],zero["inter_arm_contact_steps"])

    def test_delay_applies_to_real_controls(self):
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        policy=BuiltinReachPolicy(self.model)
        baseline=evaluate(policy,menagerie=self.menagerie,record_trace=True)
        lagged=evaluate(policy,menagerie=self.menagerie,
                        command_delay_ticks=10,record_trace=True)
        self.assertTrue(baseline["task_success"])
        self.assertEqual(lagged["command_delay_s"],.2)
        self.assertEqual(lagged["actions_checked_before_execution"],150)
        for index in range(10):
            self.assertEqual(lagged["trace"][index]["applied_command_rad"],
                             lagged["trace"][0]["applied_command_rad"])
        self.assertGreater(float(np.max(np.abs(
            np.asarray(lagged["final_tool_xyz_m"])-
            np.asarray(baseline["final_tool_xyz_m"])))),.000001)

    def test_negative_delay_rejected_before_any_step(self):
        from unittest.mock import patch
        import mujoco
        from vla_council.causalforge.dual_ur5e_policy import BuiltinReachPolicy,evaluate
        for val in (-1,26,1.5,True):
            with self.subTest(val=val):
                with patch.object(mujoco,"mj_step",side_effect=AssertionError("physics stepped")):
                    with self.assertRaises(ValueError):
                        evaluate(BuiltinReachPolicy(self.model),
                                 menagerie=self.menagerie,command_delay_ticks=val)

    def test_paired_trials_and_negative_controls(self):
        from vla_council.causalforge.latency_audit import audit
        result=audit(self.menagerie,seeds=(41,42),goals_per_seed=2,
                     delay_ticks=(0,10))
        self.assertEqual(len(result["trials"]),8)
        for delay in ("0","10"):
            self.assertEqual(result["results"][delay]["cases"],4)
            self.assertEqual(result["results"][delay]["arm_contact_steps"],0)
            self.assertTrue(np.isfinite(result["results"][delay]["mean_tool_error_m"]))

if __name__=="__main__":
    unittest.main()
