"""Tests for optional CPU/GPU environment inspection."""
import unittest
from vla_council.causalforge.gpu_preflight import inspect_compute


class ComputePreflightTests(unittest.TestCase):
    def test_schema_and_no_assumed_gpu(self):
        report = inspect_compute()
        self.assertIn(report["status"], ("gpu_ready", "cpu_only"))
        self.assertIsInstance(report["gpu_devices"], list)
        self.assertEqual(report["status"] == "gpu_ready", report["cuda_available"])
        for device in report["gpu_devices"]:
            self.assertGreater(device["total_memory_gib"], 0)
            self.assertEqual(len(device["compute_capability"]), 2)


if __name__ == "__main__":
    unittest.main()
