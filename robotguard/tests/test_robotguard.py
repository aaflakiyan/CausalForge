import json
import math
import tempfile
import unittest
from pathlib import Path

from robotguard import ConfigurationError, RobotSpec, StreamGuard, validate_trace
from robotguard.cli import main
from robotguard.report import render_html


SPEC = dict(name="Test Arm", joint_names=["a", "b"], position_limits=[[-2, 2], [-2, 2]],
            max_velocity=[1, 1], max_acceleration=[2, 2], units="rad", frame="joint", max_interval_s=0.2)


def action(t, a=0, b=0, units="rad", frame="joint"):
    return dict(t=t, position=[a, b], units=units, frame=frame)


def codes(result):
    return {v.code for v in result.violations}


class RobotGuardTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec.from_dict(SPEC)

    def test_safe_trajectory(self):
        trace = [action(0), action(.1, .005), action(.2, .015), action(.3, .03)]
        self.assertTrue(validate_trace(self.spec, trace).ok)

    def test_excessive_speed_and_acceleration(self):
        r = validate_trace(self.spec, [action(0), action(.1, .01), action(.2, 1.7)])
        self.assertIn("velocity_limit", codes(r))
        self.assertIn("acceleration_limit", codes(r))

    def test_position_outside_limits(self):
        self.assertIn("joint_limit", codes(validate_trace(self.spec, [action(0, 2.1)])))

    def test_fails_closed_on_unit_or_frame_mismatch(self):
        r = validate_trace(self.spec, [action(0, units="deg"), action(.1, frame="base")])
        self.assertEqual(codes(r), {"unit_mismatch", "frame_mismatch"})

    def test_missing_units_not_silently_inferred(self):
        a = action(0)
        del a["units"]
        self.assertIn("unit_mismatch", codes(validate_trace(self.spec, [a])))

    def test_timestamp_not_increasing(self):
        self.assertIn("non_monotonic_time", codes(validate_trace(self.spec, [action(0), action(0)])))

    def test_large_gap(self):
        self.assertIn("timing_gap", codes(validate_trace(self.spec, [action(0), action(.3)])))

    def test_reject_nonfinite_or_boolean(self):
        for bad in (float("nan"), float("inf"), True):
            with self.subTest(bad=bad):
                self.assertIn("invalid_position", codes(validate_trace(self.spec, [action(0, bad)])))

    def test_reject_missing_joint_or_bad_sample(self):
        r = validate_trace(self.spec, [{"t": 0, "position": [0], "units": "rad", "frame": "joint"}, None])
        self.assertIn("invalid_position", codes(r))
        self.assertIn("invalid_sample", codes(r))

    def test_empty_trace_is_failure(self):
        self.assertIn("empty_trace", codes(validate_trace(self.spec, [])))

    def test_bad_config(self):
        for change in ({"joint_names": ["a", "a"]}, {"max_velocity": [1, -1]},
                       {"position_limits": [[1, -1], [-2, 2]]}, {"units": "meters"}):
            with self.subTest(change=change), self.assertRaises(ConfigurationError):
                RobotSpec.from_dict({**SPEC, **change})

    def test_stream_guard_rejects_without_accepting(self):
        guard = StreamGuard(self.spec)
        self.assertTrue(guard.propose(action(0)).ok)
        self.assertFalse(guard.propose(action(.1, 4)).ok)
        self.assertEqual(guard.accepted_count, 1)
        self.assertTrue(guard.propose(action(.1, .01)).ok)
        self.assertEqual(guard.accepted_count, 2)

    def test_html_escapes_untrusted_robot_name(self):
        spec = RobotSpec.from_dict({**SPEC, "name": "<script>alert(1)</script>"})
        markup = render_html(spec, validate_trace(spec, [action(0)]))
        self.assertIn("&lt;script&gt;", markup)
        self.assertNotIn("<script>alert", markup)

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            spec_path = p / "robot.json"
            spec_path.write_text(json.dumps(SPEC))
            trace_path = p / "actions.jsonl"
            trace_path.write_text(json.dumps(action(0)) + "\n" + json.dumps(action(.1, .01)) + "\n")
            self.assertEqual(main(["check", str(trace_path), "--spec", str(spec_path), "--html", str(p / "report.html"), "--json", str(p / "report.json")]), 0)
            self.assertIn("PASS", (p / "report.json").read_text())
            trace_path.write_text(json.dumps(action(0, 10)) + "\n")
            self.assertEqual(main(["check", str(trace_path), "--spec", str(spec_path)]), 1)
            trace_path.write_text('{"t":NaN}\n')
            self.assertEqual(main(["check", str(trace_path), "--spec", str(spec_path)]), 2)


if __name__ == "__main__":
    unittest.main()
