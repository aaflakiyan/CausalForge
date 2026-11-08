"""CPU tests for versioned heterogeneous evidence and one multihead student."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

from vla_council.causalforge.vla_interface import ActionChunk, ActionSpec
from vla_council.causalforge.universal_teacher import (
    TeacherEvidence, UnifiedStudent, load_evidence, save_evidence,
    validate_groups, verified_teacher, evidence_digest,
)


def records(n=32):
    result=[]
    for i in range(n):
        split="train" if i < n-8 else "test"
        task=np.array([i/30, (i%5)/5, 1.0])
        digest=hashlib.sha256(f"observation-{i}".encode()).hexdigest()
        for robot,dof in (("dual_ur5e",12),("toy_walker",5)):
            spec=ActionSpec(robot,"joint_position",tuple(f"joint_{j}" for j in range(dof)),
                            ("rad",)*dof,"joint",.02)
            baseline=np.outer(np.linspace(.01,.04,4),np.arange(1,dof+1)/dof)+task[0]*.02
            for teacher,offset,loss,ok in (
                ("pi0_mock",0.0,.02,True),
                ("groot_mock",.08,.08,True),
                ("unsafe_mock",-.1,.001,False),
            ):
                native=np.ones((3,dof+2))*(i+.123)
                result.append(TeacherEvidence(
                    episode_id=f"ep{i:03d}",split=split,teacher=teacher,
                    architecture=f"synthetic_{teacher}",checkpoint="NONE/mock",
                    task_features=tuple(task),
                    action=ActionChunk(baseline+offset,spec,teacher),
                    measured_loss=loss,safe=ok,observation_digest=digest,
                    native_schema=f"native_{teacher}_T3_D{dof+2}",
                    native_action=tuple(map(tuple,native)),evidence_ref=f"simulation-{i}-{teacher}",
                ))
    return result


class HeterogeneousEvidenceTests(unittest.TestCase):
    def test_roundtrip_preserves_native_action_and_architecture(self):
        items=records()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"data.jsonl"
            save_evidence(p,items)
            loaded=load_evidence(p)
        self.assertEqual(len(items),len(loaded))
        self.assertEqual(items[0].to_dict(),loaded[0].to_dict())
        self.assertEqual(evidence_digest(items),evidence_digest(loaded))

    def test_unsafe_teacher_is_not_chosen_despite_lower_loss(self):
        g=validate_groups(records())[("ep000","dual_ur5e")]
        self.assertEqual(verified_teacher(g).teacher,"pi0_mock")

    def test_student_multiple_embodiments_and_heldout(self):
        items=records()
        student=UnifiedStudent(latent_dim=3,ridge=.00001).fit(items)
        self.assertEqual(set(student.heads),{"dual_ur5e","toy_walker"})
        scores=student.evaluate(items)
        self.assertEqual(scores["per_embodiment"]["dual_ur5e"]["examples"],8)
        self.assertEqual(scores["per_embodiment"]["toy_walker"]["examples"],8)
        for score in scores["per_embodiment"].values():
            self.assertLess(score["mean_mse"],.001)
        restored=UnifiedStudent.from_dict(json.loads(json.dumps(student.to_dict())))
        for robot in student.heads:
            a=student.predict((.1,.2,1.),robot=robot)
            b=restored.predict((.1,.2,1.),robot=robot)
            np.testing.assert_allclose(a.actions,b.actions)
            self.assertEqual(a.spec,b.spec)
        with self.assertRaises(ValueError):
            student.predict((.1,.2,1.),robot="unknown")

    def test_reject_train_test_leakage(self):
        items=records()
        x=items[0]
        items.append(replace(x,split="test",teacher="different"))
        with self.assertRaisesRegex(ValueError,"leakage"):
            validate_groups(items)

    def test_reject_action_space_mismatch_within_embodiment(self):
        items=records()
        x=items[1]
        altered_spec=replace(x.action.spec,units=("deg",)*12)
        items[1]=replace(x,action=ActionChunk(x.action.actions,altered_spec,x.teacher))
        with self.assertRaisesRegex(ValueError,"Physical action space"):
            validate_groups(items)

    def test_reject_ambiguous_native_data_and_observation(self):
        x=records()[0]
        with self.assertRaises(ValueError):
            replace(x,native_action=((float("nan"),),))
        with self.assertRaises(ValueError):
            replace(x,observation_digest="not-a-sha256")

    def test_reject_horizon_padding_and_inconsistent_features(self):
        items=records()
        x=items[1]
        items[1]=replace(x,action=ActionChunk(x.action.actions[:2],x.action.spec,x.teacher))
        with self.assertRaisesRegex(ValueError,"horizon"):
            validate_groups(items)
        items=records()
        x=items[1]
        items[1]=replace(x,task_features=(999.,0.,1.))
        with self.assertRaisesRegex(ValueError,"Inconsistent observation"):
            validate_groups(items)

    def test_corrupt_serialised_student_rejected(self):
        student=UnifiedStudent().fit(records())
        state=student.to_dict()
        state["heads"]["toy_walker"]["horizon"]=999
        with self.assertRaisesRegex(ValueError,"Corrupt"):
            UnifiedStudent.from_dict(state)


if __name__ == "__main__":
    unittest.main()
