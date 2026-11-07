"""Geometry-grounded acceptance tests for meaningful dual-UR5e reaching."""
import os
from pathlib import Path
import unittest
import numpy as np

@unittest.skipUnless(os.environ.get("CAUSALFORGE_MENAGERIE"),"Menagerie required")
class DualUR5eReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from vla_council.causalforge.dual_ur5e_reach import load_reach_model
        cls.model=load_reach_model(Path(os.environ["CAUSALFORGE_MENAGERIE"]))
    def test_actual_cartesian_task_not_joint_offset(self):
        from vla_council.causalforge.dual_ur5e_reach import GOALS, reset, ids
        data=reset(self.model)
        _,_,sites=ids(self.model)
        start=data.site_xpos[sites]
        self.assertTrue(np.all(np.linalg.norm(start-GOALS,axis=1) > .4))
        self.assertTrue(np.all(GOALS[:,2] > .4))
    def test_both_arms_physically_reach_and_really_move(self):
        from vla_council.causalforge.dual_ur5e_reach import execute
        out,_=execute(self.model,duration=3.)
        self.assertTrue(all(x < .07 for x in out["final_goal_errors_m"]),out)
        self.assertTrue(all(x > .35 for x in out["actual_ee_displacements_m"]),out)
        self.assertEqual(out["inter_arm_contact_frames"],0,out)
        self.assertEqual(out["table_contact_frames"],0,out)
    def test_reaching_across_reproducible_nearby_targets(self):
        from vla_council.causalforge.dual_ur5e_reach import execute, GOALS
        rng=np.random.default_rng(20261107)
        for case in range(8):
            goals=GOALS.copy()
            goals[:,0] += rng.uniform(-.035,.035,size=2)
            goals[:,1] += rng.uniform(-.045,.045,size=2)
            goals[:,2] += rng.uniform(-.03,.03,size=2)
            result,_=execute(self.model,goals=goals,duration=3.)
            self.assertLess(max(result["final_goal_errors_m"]),.07,(case,result))
            self.assertGreater(min(result["actual_ee_displacements_m"]),.7,(case,result))
            self.assertEqual(result["inter_arm_contact_frames"],0,(case,result))
            self.assertEqual(result["table_contact_frames"],0,(case,result))
    def test_deterministic_replay(self):
        from vla_council.causalforge.dual_ur5e_reach import execute
        a,_=execute(self.model,duration=3.,record_trace=True)
        self.assertGreater(len(a['trajectory']), 50)
        self.assertTrue(all(v['time_s'] > 0 for v in a['trajectory']))
        self.assertTrue(all(x < .07 for x in a['trajectory'][-1]['goal_errors_m']))
        b,_=execute(self.model,duration=3.)
        self.assertTrue(np.allclose(a["final_ee_xyz_m"],b["final_ee_xyz_m"],atol=1e-8))

if __name__=="__main__":
    unittest.main()
