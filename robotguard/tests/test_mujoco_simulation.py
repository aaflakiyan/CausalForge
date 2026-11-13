"""Negative-contract tests plus real MuJoCo tests whenever it is installed."""
from __future__ import annotations

from importlib import resources, util
import json
from pathlib import Path
import tempfile
import unittest

from robotguard.cli import main
from robotguard.core import RobotSpec, validate_trace
from robotguard.simulation import SimulationError, render_simulation_html, simulate_trace


ASSETS = resources.files("robotguard").joinpath("assets")
SPEC = json.loads(ASSETS.joinpath("demo_spec.json").read_text())
XML = ASSETS.joinpath("demo_arm.xml").read_text()
SAFE = [json.loads(x) for x in ASSETS.joinpath("demo_safe.jsonl").read_text().splitlines() if x]
UNSAFE = [json.loads(x) for x in ASSETS.joinpath("demo_unsafe.jsonl").read_text().splitlines() if x]
HAS_MUJOCO = util.find_spec("mujoco") is not None


class OfflineSimulationContractTests(unittest.TestCase):
    def test_demonstration_is_a_real_preflight_failure(self):
        report = validate_trace(RobotSpec.from_dict(SPEC), UNSAFE)
        self.assertFalse(report.ok)
        self.assertIn("joint_limit", {v.code for v in report.violations})
        self.assertIn("unit_mismatch", {v.code for v in report.violations})

    def test_preflight_rejects_before_importing_physics(self):
        with self.assertRaisesRegex(SimulationError, "no physics executed"):
            simulate_trace(RobotSpec.from_dict(SPEC), UNSAFE, XML)

    def test_no_fabricated_physics_results_without_mujoco(self):
        if not HAS_MUJOCO:
            with self.assertRaisesRegex(SimulationError, "Install"):
                simulate_trace(RobotSpec.from_dict(SPEC), SAFE, XML)

    def test_static_demo_spec_is_not_ur5(self):
        self.assertIn("Synthetic", SPEC["name"])
        self.assertEqual(SPEC["joint_names"], ["shoulder", "elbow"])


@unittest.skipUnless(HAS_MUJOCO, "Install the MuJoCo extra to execute physical engine tests")
class RealMujocoTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec.from_dict(SPEC)

    def test_physics_steps_and_logged_observations(self):
        sim = simulate_trace(self.spec, SAFE, XML)
        self.assertEqual(sim["preflight_status"], "PASS")
        self.assertEqual(sim["status"], "SIMULATION_COMPLETED")
        self.assertGreater(sim["integration_steps"], 100)
        self.assertGreater(len(sim["records"]), 6)
        self.assertEqual(sim["physical_safety_certified"], False)
        self.assertEqual(sim["metrics"]["observed_joint_limit_violations"], 0)
        self.assertGreaterEqual(sim["metrics"]["max_tracking_error_rad"], 0)
        self.assertEqual(len(sim["model_sha256"]), 64)
        self.assertIn("Solid = actual", render_simulation_html(sim))

    def test_rejects_inconsistent_specification(self):
        tampered = dict(SPEC, position_limits=[[-2, 2], [-1.2, 1.2]])
        with self.assertRaisesRegex(SimulationError, "exceed the MuJoCo joint range"):
            simulate_trace(RobotSpec.from_dict(tampered), SAFE, XML)

    def test_rejects_mismatched_joint_order_or_names(self):
        altered = dict(SPEC, joint_names=["shoulder", "missing"])
        with self.assertRaisesRegex(SimulationError, "missing the declared joint"):
            simulate_trace(RobotSpec.from_dict(altered), SAFE, XML)

    def test_rejects_unsupported_degrees(self):
        altered = dict(SPEC, units="deg")
        actions = [dict(a, units="deg") for a in SAFE]
        with self.assertRaisesRegex(SimulationError, "requires positions in radians"):
            simulate_trace(RobotSpec.from_dict(altered), actions, XML)

    def test_rejects_wrong_servo_transmission(self):
        motor_xml = XML.replace('name="shoulder_servo" joint="shoulder" kp="35" kv="6"',
                                'name="shoulder_servo" joint="shoulder" kp="35" kv="6" gear="2"')
        with self.assertRaisesRegex(SimulationError, "unit-gear"):
            simulate_trace(self.spec, SAFE, motor_xml)

    def test_rejects_too_many_physics_steps(self):
        with self.assertRaisesRegex(SimulationError, "maximum"):
            simulate_trace(self.spec, SAFE, XML, max_steps=10)

    def test_mujoco_demo_cli_produces_real_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(main(["mujoco-demo", "--out", str(out)]), 0)
            data = json.loads((out / "simulation.json").read_text())
            rejected = json.loads((out / "rejected.json").read_text())
            self.assertEqual(data["status"], "SIMULATION_COMPLETED")
            self.assertEqual(rejected["status"], "FAIL")
            self.assertTrue((out / "simulation.html").is_file())
            self.assertTrue((out / "rejected.html").is_file())

    def test_cli_does_not_run_invalid_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            spec = tmp / "spec.json"
            trace = tmp / "unsafe.jsonl"
            model = tmp / "model.xml"
            spec.write_text(json.dumps(SPEC))
            trace.write_text("\n".join(json.dumps(a) for a in UNSAFE) + "\n")
            model.write_text(XML)
            out = tmp / "out"
            self.assertEqual(main(["simulate", str(trace), "--spec", str(spec), "--model", str(model), "--out", str(out)]), 1)
            self.assertFalse((out / "simulation.json").exists())
            self.assertEqual(json.loads((out / "preflight.json").read_text())["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
