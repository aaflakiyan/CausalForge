"""Task-level, physically grounded setpoint-fault tests with UR5e models."""
import os
from pathlib import Path
import unittest

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"MuJoCo Menagerie checkout required")
class FaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model
        cls.model=load_reach_model(Path(os.environ["CAUSALFORGE_MENAGERIE"]))
    def test_compensation_helpful_only_for_bias(self):
        from vla_council.causalforge.dual_ur5e_reach import execute
        normal,_=execute(self.model, duration=3., right_command_scale=1., right_compensation=1.)
        degraded,_=execute(self.model, duration=3., right_command_scale=.55, right_compensation=1.)
        compensated,_=execute(self.model, duration=3., right_command_scale=.55, right_compensation=1/.55)
        self.assertGreater(degraded["final_goal_errors_m"][1],
                           compensated["final_goal_errors_m"][1] + .06)
        self.assertLess(compensated["final_goal_errors_m"][1],.07)
        self.assertEqual(compensated["table_contact_frames"],0)
        self.assertEqual(compensated["inter_arm_contact_frames"],0)
    def test_small_paired_comparison(self):
        from vla_council.causalforge.dual_ur5e_recovery import benchmark, Config
        result=benchmark(Path(os.environ["CAUSALFORGE_MENAGERIE"]),Config(episodes=6))
        self.assertEqual(len(result["episodes"]),24)
        self.assertEqual(set(result["summary"]),{
            "no_probe","fixed_left_probe","random_probe","active_probe"})
        self.assertEqual(result["active_probe_arm"],1)
        self.assertLess(result["summary"]["active_probe"]["mean_error_m"],
                        result["summary"]["no_probe"]["mean_error_m"])

if __name__=="__main__":
    unittest.main()
