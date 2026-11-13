"""Deterministic, dependency-free pre-flight checks for joint-position traces.

This is a kinematic validator, not a controller, collision checker or safety PLC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigurationError(ValueError):
    """A robot specification is incomplete or inconsistent."""


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _positive_vector(data: object, n: int, name: str) -> tuple[float, ...]:
    if not isinstance(data, list) or len(data) != n or not all(_finite(x) and float(x) > 0 for x in data):
        raise ConfigurationError(f"{name} must contain {n} positive finite numbers")
    return tuple(float(x) for x in data)


@dataclass(frozen=True)
class RobotSpec:
    name: str
    joint_names: tuple[str, ...]
    position_limits: tuple[tuple[float, float], ...]
    max_velocity: tuple[float, ...]
    max_acceleration: tuple[float, ...]
    units: str = "rad"
    frame: str = "joint"
    max_interval_s: float = 0.2

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RobotSpec":
        if not isinstance(raw, dict):
            raise ConfigurationError("robot spec must be a JSON object")
        names = raw.get("joint_names")
        if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
            raise ConfigurationError("joint_names must be a non-empty list of names")
        if len(set(names)) != len(names):
            raise ConfigurationError("joint_names must be unique")
        n = len(names)
        limits = raw.get("position_limits")
        if (not isinstance(limits, list) or len(limits) != n or
                any(not isinstance(pair, list) or len(pair) != 2 or
                    not all(_finite(x) for x in pair) or float(pair[0]) >= float(pair[1])
                    for pair in limits)):
            raise ConfigurationError(f"position_limits must contain {n} finite [lower, upper] pairs")
        units = raw.get("units", "rad")
        if units not in ("rad", "deg"):
            raise ConfigurationError("units must be explicitly 'rad' or 'deg'")
        frame = raw.get("frame", "joint")
        if not isinstance(frame, str) or not frame:
            raise ConfigurationError("frame must be a non-empty string")
        gap = raw.get("max_interval_s", 0.2)
        if not _finite(gap) or gap <= 0:
            raise ConfigurationError("max_interval_s must be positive and finite")
        return cls(
            name=str(raw.get("name", "Unnamed robot")),
            joint_names=tuple(names),
            position_limits=tuple((float(lo), float(hi)) for lo, hi in limits),
            max_velocity=_positive_vector(raw.get("max_velocity"), n, "max_velocity"),
            max_acceleration=_positive_vector(raw.get("max_acceleration"), n, "max_acceleration"),
            units=units,
            frame=frame,
            max_interval_s=float(gap),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "RobotSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Violation:
    sample: int
    code: str
    message: str
    joint: str | None = None
    value: float | None = None
    limit: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"sample": self.sample, "code": self.code, "message": self.message,
                "joint": self.joint, "value": self.value, "limit": self.limit}


@dataclass
class Result:
    robot: str
    samples: int
    violations: list[Violation] = field(default_factory=list)
    peak_velocity: float = 0.0
    peak_acceleration: float = 0.0
    traces: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {"status": "PASS" if self.ok else "FAIL", "robot": self.robot,
                "samples": self.samples, "violations": [v.as_dict() for v in self.violations],
                "metrics": {"peak_joint_velocity": self.peak_velocity,
                            "peak_joint_acceleration": self.peak_acceleration}}


def validate_trace(spec: RobotSpec, actions: Sequence[Mapping[str, Any]]) -> Result:
    """Validate timestamped joint-position commands. All limits are per joint.

    Position values, velocity bounds and acceleration bounds must share ``spec.units``.
    Times are seconds; the ``frame`` identifies the command reference frame.
    Malformed samples fail closed and break finite-difference continuity.
    """
    report = Result(robot=spec.name, samples=len(actions))
    if not actions:
        report.violations.append(Violation(0, "empty_trace", "Trace contains no actions"))
        return report

    prev: tuple[float, tuple[float, ...]] | None = None
    prev_velocity: tuple[float, ...] | None = None
    prev_dt: float | None = None
    n = len(spec.joint_names)

    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            report.violations.append(Violation(i, "invalid_sample", "Action must be a JSON object"))
            prev = prev_velocity = prev_dt = None
            continue
        t = action.get("t")
        pos = action.get("position")
        malformed = False
        if not _finite(t) or float(t) < 0:
            report.violations.append(Violation(i, "invalid_time", "t must be a finite, non-negative number (seconds)"))
            malformed = True
        if not isinstance(pos, list) or len(pos) != n or not all(_finite(x) for x in pos):
            report.violations.append(Violation(i, "invalid_position", f"position must contain {n} finite joint values"))
            malformed = True
        if action.get("units") != spec.units:
            report.violations.append(Violation(i, "unit_mismatch", f"Expected explicit units={spec.units!r}; got {action.get('units')!r}"))
            malformed = True
        if action.get("frame") != spec.frame:
            report.violations.append(Violation(i, "frame_mismatch", f"Expected frame={spec.frame!r}; got {action.get('frame')!r}"))
            malformed = True
        if malformed:
            prev = prev_velocity = prev_dt = None
            continue
        time = float(t)
        values = tuple(float(x) for x in pos)
        report.traces.append({"sample": i, "t": time, "position": values})
        for j, (x, bounds) in enumerate(zip(values, spec.position_limits)):
            lo, hi = bounds
            if not lo <= x <= hi:
                report.violations.append(Violation(i, "joint_limit", f"{spec.joint_names[j]}: {x:.4g} outside [{lo:g}, {hi:g}] {spec.units}",
                                                   spec.joint_names[j], x, lo if x < lo else hi))
        if prev is not None:
            dt = time - prev[0]
            if dt <= 0:
                report.violations.append(Violation(i, "non_monotonic_time", f"Timestamp {time:g} is not after {prev[0]:g}"))
                prev = prev_velocity = prev_dt = None
                continue
            if dt > spec.max_interval_s + 1e-12:
                report.violations.append(Violation(i, "timing_gap", f"Step interval {dt:.4g}s exceeds {spec.max_interval_s:g}s", value=dt, limit=spec.max_interval_s))
                prev_velocity = prev_dt = None
            velocities = tuple((x - old) / dt for x, old in zip(values, prev[1]))
            for j, velocity in enumerate(velocities):
                report.peak_velocity = max(report.peak_velocity, abs(velocity))
                if abs(velocity) > spec.max_velocity[j] + 1e-9:
                    report.violations.append(Violation(i, "velocity_limit", f"{spec.joint_names[j]}: |velocity|={abs(velocity):.4g} exceeds {spec.max_velocity[j]:g} {spec.units}/s",
                                                       spec.joint_names[j], abs(velocity), spec.max_velocity[j]))
            if prev_velocity is not None and prev_dt is not None:
                # Unequal sample intervals: compare segment-average velocities over midpoint separation.
                step_between_midpoints = (dt + prev_dt) / 2.0
                for j, (current, previous) in enumerate(zip(velocities, prev_velocity)):
                    acc = abs(current - previous) / step_between_midpoints
                    report.peak_acceleration = max(report.peak_acceleration, acc)
                    if acc > spec.max_acceleration[j] + 1e-9:
                        report.violations.append(Violation(i, "acceleration_limit", f"{spec.joint_names[j]}: |acceleration|={acc:.4g} exceeds {spec.max_acceleration[j]:g} {spec.units}/s²",
                                                           spec.joint_names[j], acc, spec.max_acceleration[j]))
            prev_velocity, prev_dt = velocities, dt
        prev = (time, values)
    return report


class StreamGuard:
    """Fail-closed, stateful simulator-side gate for joint-position proposals.

    Rejected actions are not recorded. Never connect directly to actuators: this does
    not verify physical collision clearance, dynamics or control-loop stability.
    """
    def __init__(self, spec: RobotSpec):
        self.spec = spec
        self._accepted: list[dict[str, Any]] = []

    @property
    def accepted_count(self) -> int:
        return len(self._accepted)

    def propose(self, action: Mapping[str, Any]) -> Result:
        candidate = self._accepted + [dict(action) if isinstance(action, dict) else action]
        result = validate_trace(self.spec, candidate)
        if result.ok:
            self._accepted.append(dict(action))
        return result
