"""Active training-data selection must not inspect held-out outcomes."""
import hashlib
import unittest
from dataclasses import replace
import numpy as np
from vla_council.causalforge.vla_interface import ActionSpec,ActionChunk
from vla_council.causalforge.universal_teacher import TeacherEvidence
from vla_council.causalforge.curriculum import disagreement_curriculum,reasoning_curriculum_note
from vla_council.causalforge.physical_reasoning import ReasoningGate,PhysicalFact

def row(ep,split,teacher,offset):
    spec=ActionSpec("toy_robot","joint_position",("j1","j2"),("rad","rad"),"joint",.02)
    return TeacherEvidence(
        episode_id=ep,split=split,teacher=teacher,architecture="mock",
        checkpoint="mock",task_features=(.1,.2),
        action=ActionChunk(np.ones((2,2))*offset,spec,teacher),
        measured_loss=.01,safe=True,
        observation_digest=hashlib.sha256(ep.encode()).hexdigest(),
        native_schema="mock",native_action=((1.,2.),),
        evidence_ref=ep+"-"+teacher)

class CurriculumTests(unittest.TestCase):
    def test_disagreement_prioritises_uncertain_case(self):
        rows=[row("ep1","train","a",0),row("ep1","train","b",.01),
              row("ep2","train","a",0),row("ep2","train","b",1),
              row("ep3","test","a",0),row("ep3","test","b",100)]
        report=disagreement_curriculum(rows,budget=2)
        self.assertEqual(report["selected"][0]["episode_id"],"ep2")
        self.assertEqual(report["candidates"],2)
        self.assertEqual(report["selected"][1]["episode_id"],"ep1")

    def test_test_outcome_cannot_change_selection(self):
        rows=[row("a","train","x",.2),row("a","train","y",.5),
              row("b","test","x",0),row("b","test","y",10)]
        base=disagreement_curriculum(rows,1)
        modified=[replace(x,measured_loss=1e12,safe=False)
                  if x.split=="test" else x for x in rows]
        self.assertEqual(disagreement_curriculum(modified,1),base)

    def test_grounded_reasoning_only_suggests_probe(self):
        facts=[PhysicalFact("slip",.02,"m","mujoco_sensor")]
        reasoning=ReasoningGate(lambda context:{
            "hypothesis":"insufficient friction","verdict":"uncertain",
            "evidence_refs":["slip"],"suggested_probe":"change friction in simulator",
            "rationale":"slip was measured, cause not yet established",
        })
        out=reasoning_curriculum_note(
            task="grasp",facts=facts,teachers=["pi","groot"],reasoner=reasoning)
        self.assertFalse(out["control_authorised"])
        self.assertEqual(out["claim"]["evidence_refs"],["slip"])

if __name__=="__main__":
    unittest.main()
