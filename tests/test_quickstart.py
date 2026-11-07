"""Quickstart CLI tests; no external network needed for these cases."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from vla_council.causalforge.quickstart import locate_menagerie,main

class QuickstartTests(unittest.TestCase):
    def test_missing_explicit_model_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                locate_menagerie(Path(tmp))
    def test_never_overwrites_populated_non_git_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"cache"
            root.mkdir()
            (root/"contents.txt").write_text("do not remove")
            with self.assertRaises(RuntimeError):
                locate_menagerie(cache=root)
    def test_unknown_command_rejected(self):
        with self.assertRaises(SystemExit):
            main(["not-a-command"])
    def test_explicit_valid_source_is_accepted_without_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            f=root/"universal_robots_ur5e"/"ur5e.xml"
            f.parent.mkdir()
            f.write_text("<mujoco/>")
            with patch("subprocess.run",side_effect=AssertionError("Should never use network")):
                self.assertEqual(locate_menagerie(root),root.resolve())

if __name__=="__main__":
    unittest.main()
