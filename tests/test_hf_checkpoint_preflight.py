"""Conformance checks reject two-dimensional policy as UR5e joint actuation."""
import unittest
from vla_council.causalforge.hf_checkpoint_preflight import inspect_config
class PreflightTests(unittest.TestCase):
    def test_two_dimensional_action_is_rejected(self):
        report=inspect_config("real/checkpoint",{
            "type":"smolvla","input_features":{"observation.image":{"shape":[3,96,96]}},
            "output_features":{"action":{"shape":[2]}}})
        self.assertEqual(report["action_dimension"],2)
        self.assertFalse(report["direct_ur5e_compatible"])
        self.assertIn("Action dimensions 2",report["refused_reasons"][0])
    def test_twleve_unknown_dimensions_still_rejected(self):
        report=inspect_config("unknown",{"output_features":{"action":{"shape":[12]}}})
        self.assertFalse(report["direct_ur5e_compatible"])
        self.assertTrue(any("named UR5e joint" in x for x in report["refused_reasons"]))
if __name__=="__main__":unittest.main()
