"""Command-delay student comparison smoke test on pinned UR5e MuJoCo."""
import json
import os
from pathlib import Path
import tempfile
import unittest

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"Menagerie required")
class StudentStressTests(unittest.TestCase):
    def test_small_paired_heldout_student_vs_ik(self):
        from vla_council.causalforge.student_latency_audit import run_stress
        with tempfile.TemporaryDirectory() as temp:
            report=run_stress(Path(os.environ["CAUSALFORGE_MENAGERIE"]),
                              Path(temp),seed=9,train=4,test=2,delays=(0,5))
            self.assertEqual(len(report["trials"]),8)
            self.assertEqual(set(report["summary"]),
                             {"student:0","student:5","ik_teacher:0","ik_teacher:5"})
            self.assertTrue((Path(temp)/"stress-report.json").exists())
            self.assertTrue(all(entry["total_contacts"]==0 for entry in report["summary"].values()))
            self.assertFalse(report["pretrained_vla"])

if __name__=="__main__":
    unittest.main()
