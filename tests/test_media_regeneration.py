"""Validate the MuJoCo-to-SVG regeneration path where MuJoCo is installed."""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


class MediaRegenerationTests(unittest.TestCase):
    @unittest.skipIf(importlib.util.find_spec("mujoco") is None, "MuJoCo not installed")
    def test_regenerate_svg_media(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            subprocess.run(
                [sys.executable, str(root / "scripts" / "generate_media.py"), "--out", str(output)],
                check=True, cwd=root, timeout=120,
            )
            for name in ("pusher-trajectories.svg", "tabletop-friction.svg", "tabletop-replay.svg"):
                tree = ET.parse(output / name)
                self.assertTrue(tree.getroot().tag.endswith("svg"))
                self.assertGreater((output / name).stat().st_size, 200)


if __name__ == "__main__":
    unittest.main()
