"""Browser replay is evidence-driven, never a fabricated simulation."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from robotguard.playground import PlaygroundError, render_playground_html, write_playground


class OfflinePlaygroundTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "records": [{"t": 0.0, "target": [0.0, 0.0], "actual": [0.0, 0.0],
                         "link_points_xy_m": [[0, 0], [0.4, 0], [0.7, 0]]},
                        {"t": 0.02, "target": [0.1, 0.1], "actual": [0.09, 0.09],
                         "link_points_xy_m": [[0, 0], [0.398, 0.036], [0.692, 0.092]]}],
            "joint_names": ["shoulder", "elbow"],
            "status": "SIMULATION_COMPLETED", "robot_spec": "Synthetic arm",
            "mujoco_version": "test-fixture-only", "metrics": {
                "rmse_tracking_error_rad": 0.01, "max_tracking_error_rad": 0.01},
            "model_sha256": "a" * 64, "trace_sha256": "b" * 64,
            "simulated_duration_s": 0.02, "integration_steps": 10,
        }

    def test_playground_contains_measured_frames_without_external_assets(self):
        page = render_playground_html(self.result)
        self.assertIn("playground", page.lower())
        self.assertIn("MEASURED PHYSICS REPLAY", page)
        self.assertIn("requestAnimationFrame", page)
        self.assertIn("simulation.json", page)
        self.assertNotIn("https://", page)
        self.assertIn("<svg", page)

    def test_cannot_generate_playground_without_actual_geometry(self):
        self.result["records"][0].pop("link_points_xy_m")
        with self.assertRaises(PlaygroundError):
            render_playground_html(self.result)
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(write_playground(directory, self.result))
            self.assertFalse((Path(directory) / "playground.html").exists())

    def test_escapes_untrusted_names_and_embeds_safe_payload(self):
        self.result["robot_spec"] = "</script><script>alert('oops')</script>"
        page = render_playground_html(self.result)
        self.assertNotIn("</script><script>alert", page)
        self.assertIn("&lt;/script&gt;", page)
        self.assertEqual(len(re.findall(r"<script>", page)), 1)
        self.assertIn("atob(", page)

    def test_writes_local_html(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(write_playground(directory, self.result))
            self.assertTrue((Path(directory) / "playground.html").is_file())


if __name__ == "__main__":
    unittest.main()
