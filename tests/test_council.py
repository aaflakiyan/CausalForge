import unittest
import numpy as np
from vla_council.core import Observation, Proposal, bounded
from vla_council.sim import collision_free, step
from vla_council.council import Council
from vla_council.teachers import default_teachers
from vla_council.student import LinearStudent

class BadTeacher:
    name = "unsafe"
    def propose(self, obs):
        return Proposal("unsafe", (.5, .5), "violates per-tick displacement")

class Tests(unittest.TestCase):
    def test_bounded(self):
        self.assertAlmostEqual(np.linalg.norm(bounded([12, 5])), 0.08)
    def test_gate_rejects_swept_collision(self):
        o = Observation(.4, .5, .9, .5, .5, .5, .04)
        self.assertFalse(collision_free(o, np.array([.08, 0.])))
    def test_gate_rejects_excessive_displacement(self):
        self.assertFalse(collision_free(Observation(.1,.1,.9,.9), np.array([.5, 0.])))
    def test_fails_closed_if_all_unfeasible(self):
        d = Council([BadTeacher()]).decide(Observation(.1,.1,.9,.9))
        self.assertTrue(d.needs_intervention)
        self.assertEqual(d.teacher, "STOP")
    def test_safe_selection(self):
        o = Observation(.1,.1,.9,.9)
        d = Council(default_teachers()).decide(o)
        self.assertFalse(d.needs_intervention)
        self.assertTrue(collision_free(o, np.array(d.action)))
    def test_student_shapes(self):
        samples = [Observation(.1,.2,.9,.8), Observation(.2,.3,.9,.8)]
        model = LinearStudent.fit(samples, [(0.03, 0.04), (0.04, .03)])
        self.assertEqual(model.act(samples[0]).shape, (2,))
        self.assertTrue(np.all(np.isfinite(model.act(samples[0]))))
    def test_step_throws_on_bad_action(self):
        with self.assertRaises(ValueError):
            step(Observation(.1,.1,.9,.9), np.array([3., 0.]))

if __name__ == "__main__": unittest.main()

# Test module also includes a small integration smoke-check for peer review.
class PeerReviewTests(unittest.TestCase):
    def test_reviews_present_and_external_critic_independent(self):
        result = Council(default_teachers()).decide(Observation(.1,.5,.9,.5))
        self.assertEqual(len(result.reviews), 6)
        self.assertEqual(len(result.critiques), 3)
        self.assertTrue(all(r.reviewer != r.subject for r in result.reviews))
