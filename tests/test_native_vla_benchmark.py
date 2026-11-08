"""Tests for strict authentic-checkpoint parsing, never fake a result."""
import json
import tempfile
from pathlib import Path
import unittest
from vla_council.causalforge.native_vla_benchmark import (
    evaluate_native,parse_native_info
)
class NativeCheckpointTests(unittest.TestCase):
    def test_require_genuine_evaluation_records(self):
        data={
            "aggregated":{"avg_max_reward":0.7,"pc_success":50.},
            "per_episode":[
                {"seed":42,"max_reward":.9,"success":True},
                {"seed":43,"max_reward":.5,"success":False},
            ],
        }
        parsed=parse_native_info(data,2)
        self.assertEqual(parsed["avg_max_reward"],.7)
        self.assertEqual(len(parsed["episodes"]),2)
        with self.assertRaises(ValueError):parse_native_info(data,3)
        with self.assertRaises(ValueError):parse_native_info({},2)

    def test_bad_learned_result_rejected(self):
        with self.assertRaises(ValueError):
            parse_native_info({"aggregated":{"avg_max_reward":float("nan"),
                "pc_success":55}, "per_episode":[{"seed":42,"max_reward":1,"success":True}]},1)

    def test_unregistered_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"Unsupported checkpoint"):
                evaluate_native(models=("fake/foundation","aadarshram/act_pusht"),
                                out=Path(d),episodes=1)

    def test_missing_executable_creates_failure_not_fabricated_score(self):
        with tempfile.TemporaryDirectory() as d:
            result=evaluate_native(models=("aadarshram/act_pusht",),out=Path(d),
                   episodes=1,executable="no-such-lerobot-executable",timeout_s=10)
            self.assertFalse(result["all_models_ran"])
            self.assertFalse(result["policies"]["aadarshram/act_pusht"]["checkpoint_inference"])
            self.assertIsNone(result.get("best_mean_overlap_model"))
            self.assertTrue((Path(d)/"comparison.json").exists())

if __name__=="__main__":
    unittest.main()
