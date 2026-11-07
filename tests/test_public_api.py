"""Test stable public import without implicitly installing MuJoCo."""
import unittest
import causalforge

class PublicAPITests(unittest.TestCase):
    def test_public_exports(self):
        for name in ("evaluate","BuiltinReachPolicy","ActionSpec","ActionChunk","SPEC","locate_menagerie"):
            self.assertTrue(hasattr(causalforge,name),name)

if __name__=="__main__":
    unittest.main()
