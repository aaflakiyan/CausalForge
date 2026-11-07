"""Three mocked upstream provider interfaces on official MuJoCo UR5e models."""
import os
import unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"MuJoCo Menagerie required")
class MultiProviderSmokeTests(unittest.TestCase):
    def test_three_mock_interfaces_execute_and_reach(self):
        from vla_council.causalforge.vla_bridge_demo import run
        report=run(Path(os.environ["CAUSALFORGE_MENAGERIE"]))
        self.assertEqual(len(report["teachers"]),3)
        self.assertEqual(report["rejected"],{})
        self.assertIn("MOCKED",report["checkpoint_status"])
        for result in report["teachers"].values():
            self.assertTrue(result["task_success"],result)
            self.assertEqual(result["actions_checked_before_execution"],150)
            self.assertTrue(all(x<.07 for x in result["tool_error_m"]))
            self.assertEqual(result["inter_arm_contact_steps"],0)

if __name__=="__main__":
    unittest.main()
