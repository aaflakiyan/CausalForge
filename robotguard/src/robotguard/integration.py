"""A small, explicit bridge from a user-supplied policy to RobotGuard.

This bridge does not execute models, guess units, perform IK or certify actions.
Every policy callback must declare a joint-space position vector by joint name,
units and frame. Call validate_trace separately before any simulation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core import RobotSpec


class PolicyContractError(ValueError):
    """A policy did not meet the declared joint-position action contract."""


def actions_from_policy(
    spec: RobotSpec,
    timestamps: Sequence[float],
    predict: Callable[[float], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Sample a user callable with an *explicit*, inspectable action contract.

    Callable output is {"position": {joint_name: number, ...},
                        "units": "rad", "frame": "joint"}.
    Position values are ordered according to ``spec.joint_names``. There is no
    unit conversion, name guessing, clipping or modification of outputs.
    The caller must separately run ``validate_trace``; physics always enforces it.
    """
    if not timestamps:
        raise PolicyContractError("Provide one or more timestamps in seconds")
    actions: list[dict[str, Any]] = []
    prev = -1.0
    for index, t in enumerate(timestamps):
        if type(t) not in (int, float) or not math.isfinite(float(t)) or float(t) < 0 or float(t) <= prev:
            raise PolicyContractError(f"timestamp {index} must be finite, non-negative and strictly increasing")
        prev = float(t)
        result = predict(float(t))
        if not isinstance(result, Mapping):
            raise PolicyContractError(f"policy sample {index} must return a mapping")
        if set(result) != {"position", "units", "frame"}:
            raise PolicyContractError(f"policy sample {index} must contain exactly position, units and frame")
        if result["units"] != spec.units or result["frame"] != spec.frame:
            raise PolicyContractError(f"policy sample {index}: units/frame must explicitly match {spec.units}/{spec.frame}")
        pose = result["position"]
        if not isinstance(pose, Mapping) or set(pose) != set(spec.joint_names):
            raise PolicyContractError(f"policy sample {index}: expected exactly joints {list(spec.joint_names)}")
        values = [pose[joint] for joint in spec.joint_names]
        if any(type(v) not in (int, float) or not math.isfinite(float(v)) for v in values):
            raise PolicyContractError(f"policy sample {index}: joint positions must be finite numbers")
        actions.append({"t": float(t), "position": [float(v) for v in values],
                        "units": spec.units, "frame": spec.frame})
    return actions


def write_actions_jsonl(path: str | Path, actions: Sequence[Mapping[str, Any]]) -> Path:
    """Serialize only JSON-valid samples, no NaN or Infinity, atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Validate before writing; preserve original if any sample cannot serialize.
    content = "".join(json.dumps(dict(a), sort_keys=True, allow_nan=False) + "\n" for a in actions)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(target)
    return target


def baseline_trace(spec: RobotSpec, name: str) -> list[dict[str, Any]]:
    """Two *deterministic examples*, never presented as learned VLA models.

    Both follow the same start/goal/horizon in the synthetic 2-DOF scene.
    """
    if tuple(spec.joint_names) != ("shoulder", "elbow") or spec.units != "rad" or spec.frame != "joint":
        raise PolicyContractError("Reference baselines only support the packaged synthetic 2-joint arm")
    if name not in ("linear", "smoothstep"):
        raise PolicyContractError("Baseline must be 'linear' or 'smoothstep'")

    def predict(t: float) -> dict[str, Any]:
        u = t / 0.5
        weight = u if name == "linear" else 3 * u * u - 2 * u * u * u
        return {"position": {"shoulder": 0.2 * weight, "elbow": 0.15 * weight},
                "units": "rad", "frame": "joint"}

    return actions_from_policy(spec, [i * 0.1 for i in range(6)], predict)
