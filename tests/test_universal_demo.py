"""Executable CPU-only unified-teacher distillation quickstart smoke test."""
import json
import tempfile
from pathlib import Path
import unittest
from vla_council.causalforge.universal_demo import run
from vla_council.causalforge.universal_teacher import UnifiedStudent,load_evidence

class DemoTests(unittest.TestCase):
    def test_complete_artifact_and_heldout_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            result=run(p,n=36)
            self.assertFalse(result["checkpoint_inference"])
            self.assertEqual(len(result["embodiments"]),2)
            self.assertEqual(result["scores"]["per_embodiment"]["dual_ur5e_mock"]["examples"],16)
            self.assertTrue((p/"student.json").exists())
            self.assertTrue((p/"synthetic_teacher_evidence.jsonl").exists())
            self.assertTrue((p/"report.json").exists())
            student=UnifiedStudent.from_dict(json.loads((p/"student.json").read_text()))
            rows=load_evidence(p/"synthetic_teacher_evidence.jsonl")
            self.assertEqual(set(student.heads),{"dual_ur5e_mock","humanoid_leg_mock"})
            self.assertEqual(len(rows),144)
            self.assertFalse(result["reasoning"]["control_authorised"])

if __name__=="__main__":
    unittest.main()
