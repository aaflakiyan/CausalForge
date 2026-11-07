"""Reproducible three-provider mock VLA integration on real MuJoCo UR5e physics.

The provider shapes mirror documented GR00T, OpenPI and LeRobot interfaces,
but all three receive a single KNOWN, scripted IK trajectory. This is a
protocol/integration smoke test only, NOT trained-model evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from dataclasses import replace

from .dual_ur5e_policy import SPEC, BuiltinReachPolicy
from .dual_ur5e_reach import load_reach_model, reset, GOALS
from .dual_ur5e import ids
from .vla_bridge import ExplicitActionMapping, VLABackend, evaluate_teachers


class MockOpenPI:
    def __init__(self, actions): self.actions = actions
    def infer(self, batch): return {"actions":self.actions}


class MockGR00T:
    def __init__(self, actions): self.actions = actions
    def get_action(self, batch): return {"ur5e_joints":self.actions[None]}, {}


class MockLeRobot:
    def __init__(self, actions): self.actions = actions
    def predict_action_chunk(self, batch): return {"action":self.actions[None]}


def run(menagerie: Path, *, record_trace: bool = False):
    model = load_reach_model(menagerie)
    home = reset(model).qpos[ids(model)[0]].copy()
    known_plan = BuiltinReachPolicy(model).predict({
        "joint_positions_rad":home,
        "goal_xyz_m":GOALS,
    }).actions
    reversed_degrees = np.rad2deg(known_plan[:, ::-1])
    degrees_source = replace(
        SPEC,
        ordered_axes=tuple(reversed(SPEC.ordered_axes)),
        units=("deg",) * 12,
    )
    backend = {
        "OpenPI-compatible mock":VLABackend(
            "mock_openpi", "openpi", MockOpenPI(known_plan),
            ExplicitActionMapping(SPEC), lambda obs: obs),
        "GR00T-compatible mock":VLABackend(
            "mock_groot", "groot", MockGR00T(reversed_degrees),
            ExplicitActionMapping(degrees_source, degrees_to_radians=True),
            lambda obs: obs, action_key="ur5e_joints"),
        "LeRobot-compatible mock":VLABackend(
            "mock_lerobot", "lerobot", MockLeRobot(known_plan),
            ExplicitActionMapping(SPEC), lambda obs: obs),
    }
    result = evaluate_teachers(backend, menagerie=menagerie, record_trace=record_trace)
    result["checkpoint_status"] = "ALL PROVIDERS MOCKED: no pretrained models used"
    result["source_plan"] = "Single MuJoCo IK baseline shared across mocked inference APIs"
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--menagerie",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs/vla-protocol-demo.json"))
    a=p.parse_args()
    result=run(a.menagerie)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "checkpoint_status": result["checkpoint_status"],
        "teacher_results":{
            name:{"success":value["task_success"],"error_m":value["tool_error_m"]}
            for name,value in result["teachers"].items()
        },
    }, indent=2))
    if result["rejected"] or not all(v["task_success"] for v in result["teachers"].values()):
        raise SystemExit("VLA protocol simulation smoke test failed")


if __name__ == "__main__":
    main()
