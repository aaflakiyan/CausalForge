"""Actual MuJoCo teacher-to-student rollout integration, not pretrained VLA."""
import os
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"Pinned Menagerie asset required")
class UR5eStudentTests(unittest.TestCase):
    def test_train_and_heldout_simulated_student(self):
        from vla_council.causalforge.ur5e_teacher_student import run
        from vla_council.causalforge.universal_teacher import load_evidence,UnifiedStudent
        import json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            result=run(Path(os.environ["CAUSALFORGE_MENAGERIE"]),path,
                       seed=42,train=6,test=2)
            self.assertFalse(result["pretrained_vla"])
            self.assertEqual(result["test_episodes"],2)
            self.assertTrue((path/"student.json").is_file())
            records=load_evidence(path/"mujoco_teacher_evidence.jsonl")
            self.assertEqual(len(records),16)
            # Physically contact-free but task-incomplete is not the same
            # as a dangerous controller. Labels must remain independent.
            failed_but_contact_free=[
                row for row in records
                if row.teacher=="underreach_teacher"
                and row.safe
            ]
            self.assertTrue(failed_but_contact_free)
            self.assertTrue(any(
                not episode["success"] and episode["contact_free"]
                for episode in result["teacher_results"]["underreach_teacher"]
            ))
            model=UnifiedStudent.from_dict(json.loads((path/"student.json").read_text()))
            self.assertIn("menagerie_dual_ur5e",model.heads)
            self.assertTrue(all(s["arm_contacts"]==0 and s["table_contacts"]==0
                                for s in result["student_held_out"]))
            self.assertLess(result["held_out_student_mean_goal_error_m"],.12)

if __name__=="__main__":
    unittest.main()
