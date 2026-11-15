"""The policy bridge and comparison must not make unearned simulation claims."""
from __future__ import annotations

from importlib import resources, util
import json
from pathlib import Path
import tempfile
import unittest

from robotguard.cli import main
from robotguard.core import RobotSpec, validate_trace
from robotguard.integration import PolicyContractError, actions_from_policy, baseline_trace, write_actions_jsonl
from robotguard.showcase import ComparisonError, render_comparison_html, run_showcase

SPEC = RobotSpec.from_dict(json.loads(resources.files("robotguard").joinpath("assets/demo_spec.json").read_text()))
HAS_MUJOCO = util.find_spec("mujoco") is not None


class IntegrationOfflineTests(unittest.TestCase):
    def test_baselines_both_pass_preflight_and_share_start_goal_horizon(self):
        a, b = baseline_trace(SPEC, "linear"), baseline_trace(SPEC, "smoothstep")
        self.assertTrue(validate_trace(SPEC, a).ok)
        self.assertTrue(validate_trace(SPEC, b).ok)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[-1], b[-1])
        self.assertNotEqual(a[1], b[1])

    def test_named_joint_order_independent_of_policy_mapping_insertion(self):
        result = actions_from_policy(SPEC, [0.0, 0.1], lambda t: {
            "units": "rad", "frame": "joint", "position": {"elbow": 0.3*t, "shoulder": 0.4*t}
        })
        self.assertAlmostEqual(result[1]["position"][0], 0.04)
        self.assertAlmostEqual(result[1]["position"][1], 0.03)
        self.assertTrue(validate_trace(SPEC, result).ok)

    def test_unit_frame_and_missing_joint_rejected(self):
        def policy(t):
            return {"units": "deg", "frame": "joint", "position": {"shoulder": 0.0, "elbow": 0.0}}
        with self.assertRaisesRegex(PolicyContractError, "units/frame"):
            actions_from_policy(SPEC, [0.0], policy)
        with self.assertRaisesRegex(PolicyContractError, "joints"):
            actions_from_policy(SPEC, [0.0], lambda t: {"units": "rad", "frame": "joint", "position": {"shoulder": 0.1}})

    def test_invalid_timestamps_nonfinite_and_extra_joint_rejected(self):
        valid = lambda t: {"units": "rad", "frame": "joint", "position": {"shoulder": 0.0, "elbow": 0.0}}
        for times in ([0.1, 0.1], [0.2, 0.1], [float('nan')], [-0.1]):
            with self.subTest(times=times), self.assertRaises(PolicyContractError):
                actions_from_policy(SPEC, times, valid)
        with self.assertRaisesRegex(PolicyContractError, "finite"):
            actions_from_policy(SPEC, [0.0], lambda t: {"units": "rad", "frame": "joint", "position": {"shoulder": float('inf'), "elbow": 0}})
        with self.assertRaisesRegex(PolicyContractError, "joints"):
            actions_from_policy(SPEC, [0.0], lambda t: {"units": "rad", "frame": "joint", "position": {"shoulder": 0, "elbow": 0, "wrist": 0}})

    def test_no_clipping_invalid_positions_are_detected_in_gate(self):
        trajectory = actions_from_policy(SPEC, [0, 0.1], lambda t: {
            "units": "rad", "frame": "joint", "position": {"shoulder": t*100, "elbow": 0}
        })
        self.assertFalse(validate_trace(SPEC, trajectory).ok)
        self.assertEqual(trajectory[1]["position"][0], 10.0)

    def test_jsonl_roundtrip_and_nonfinite_serialization_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            write_actions_jsonl(path, baseline_trace(SPEC, "linear"))
            self.assertEqual(len(path.read_text().splitlines()), 6)
            with self.assertRaises(ValueError):
                write_actions_jsonl(path, [{"t": float('nan')}])
            self.assertEqual(len(path.read_text().splitlines()), 6)

    def test_comparison_refuses_missing_or_unverified_physics_data(self):
        with self.assertRaises(ComparisonError):
            render_comparison_html({}, 4)
        with self.assertRaises(ComparisonError):
            render_comparison_html({"linear": {"status": "PASS"}, "smoothstep": {}}, 4)


@unittest.skipUnless(HAS_MUJOCO, "Install MuJoCo extra to verify actual comparison physics")
class RealIntegrationTests(unittest.TestCase):
    def test_end_to_end_two_policies_one_model_and_real_report(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / "reports"
            result = run_showcase(output)
            self.assertEqual(set(result["policies"]), {"linear", "smoothstep"})
            self.assertGreater(result["rejected_violations"], 0)
            self.assertEqual(result["physical_safety_certified"], False)
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "rejected.json").exists())
            self.assertFalse((output / "rejected" / "simulation.json").exists())
            page = (output / "index.html").read_text()
            self.assertIn("Side-by-side measured motion", page)
            self.assertNotIn('src="https://', page)
            for name in ("linear", "smoothstep"):
                self.assertTrue((output / name / "playground.html").exists())
                sim = json.loads((output / name / "simulation.json").read_text())
                self.assertEqual(sim["status"], "SIMULATION_COMPLETED")
                self.assertGreater(sim["integration_steps"], 100)
                self.assertEqual(sim["metrics"]["observed_joint_limit_violations"], 0)
            a = json.loads((output / "linear" / "simulation.json").read_text())
            b = json.loads((output / "smoothstep" / "simulation.json").read_text())
            self.assertEqual(a["model_sha256"], b["model_sha256"])
            self.assertNotEqual(a["trace_sha256"], b["trace_sha256"])
            self.assertEqual(a["simulated_duration_s"], b["simulated_duration_s"])

    def test_cli_showcase(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(["showcase", "--out", str(Path(d) / "output")]), 0)

if __name__ == "__main__":
    unittest.main()
