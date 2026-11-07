"""Unit checks for paired seed aggregation."""
import unittest
from vla_council.causalforge.dual_ur5e_multiseed import METHODS

class DualUR5eMultiseedTests(unittest.TestCase):
    def test_declared_methods(self):
        self.assertEqual(METHODS, ("no_probe", "fixed_left_probe", "random_probe", "active_probe"))

if __name__ == "__main__":
    unittest.main()
