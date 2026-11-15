"""Minimal real integration point: replace predict() with your own callable.

From repo root (after pip install -e './robotguard[mujoco]'):
    python robotguard/examples/connect_my_policy.py
    robotguard simulate my-policy.jsonl \
      --spec robotguard/src/robotguard/assets/demo_spec.json \
      --model robotguard/src/robotguard/assets/demo_arm.xml --out my-policy-report

This is a deterministic example, NOT a pretrained AI/VLA model.
"""
import json
from importlib import resources

from robotguard.core import RobotSpec, validate_trace
from robotguard.integration import actions_from_policy, write_actions_jsonl

spec = RobotSpec.from_dict(json.loads(
    resources.files("robotguard").joinpath("assets/demo_spec.json").read_text()
))


def predict(t):
    """Replace this function with an inference call returning an explicit contract."""
    return {"position": {"shoulder": 0.4 * t, "elbow": 0.3 * t},
            "units": "rad", "frame": "joint"}


actions = actions_from_policy(spec, [i / 10 for i in range(6)], predict)
report = validate_trace(spec, actions)
if not report.ok:
    raise SystemExit(f"Policy output rejected: {[v.code for v in report.violations]}")
write_actions_jsonl("my-policy.jsonl", actions)
print("Wrote 6 checked samples to my-policy.jsonl; see command in this script to simulate")
