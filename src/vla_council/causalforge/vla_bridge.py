"""Bridge real VLA *interfaces* into validated CausalForge action chunks.

Supports the documented output method shapes for GR00T, OpenPI and LeRobot,
without importing the huge model-specific dependencies. A loaded pretrained
policy (or a remote client) must be supplied explicitly by the caller.

A model predicting actions for a different embodiment, frame, mode, sample
period or set of joints cannot be used by guessing conversions. Conversion
requires an explicit fully checked mapping and deliberate degree opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
import numpy as np

from .dual_ur5e_policy import SPEC
from .vla_interface import ActionChunk, ActionSpec


def _numpy(output: Any) -> np.ndarray:
    """Read torch/JAX/NumPy outputs without requiring any vendor library."""
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    return np.asarray(output, dtype=float)


@dataclass(frozen=True)
class ExplicitActionMapping:
    """Declare the full source action contract and every conversion policy.

    Only known joint-position axes are supported in this first working bridge.
    To move a gripper, control torque or apply Cartesian pose deltas, implement
    and validate a new robot/controller adapter rather than reusing this one.
    """
    source: ActionSpec
    target: ActionSpec = SPEC
    degrees_to_radians: bool = False

    def __post_init__(self):
        if self.source.robot != self.target.robot:
            raise ValueError("Robot embodiment differs; no implicit cross-robot conversion")
        if self.source.representation != "joint_position" or self.target.representation != "joint_position":
            raise ValueError("Only absolute joint_position actions supported")
        if self.source.frame != "joint" or self.target.frame != "joint":
            raise ValueError("Only joint-frame commands supported")
        if not np.isclose(self.source.dt, self.target.dt, atol=1e-12, rtol=0):
            raise ValueError("Action periods differ; resampling must be implemented and tested separately")
        if len(self.source.ordered_axes) != len(self.target.ordered_axes):
            raise ValueError("Every target joint must be supplied; no silent padding")
        if set(self.source.ordered_axes) != set(self.target.ordered_axes):
            raise ValueError("Missing, additional or unknown source joint names")
        if any(u not in ("rad", "deg") for u in self.source.units):
            raise ValueError("Unrecognised source units")
        if any(u != "rad" for u in self.target.units):
            raise ValueError("Destination must use radians")
        if "deg" in self.source.units and not self.degrees_to_radians:
            raise ValueError("Degree-to-radian conversion requires explicit opt-in")

    def convert(self, action: Any, model_name: str) -> ActionChunk:
        data = _numpy(action)
        if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != len(self.source.ordered_axes):
            raise ValueError("Expected source action chunk [T, source_joint_count]")
        if not np.isfinite(data).all():
            raise ValueError("Non-finite VLA output")
        indices = [self.source.ordered_axes.index(name) for name in self.target.ordered_axes]
        transformed = data[:, indices].copy()
        for col, idx in enumerate(indices):
            if self.source.units[idx] == "deg":
                transformed[:, col] = np.deg2rad(transformed[:, col])
        return ActionChunk(transformed, self.target, model_name)


@dataclass
class VLABackend:
    """An already loaded GR00T/OpenPI/LeRobot instance or client.

    observation_adapter is REQUIRED because models expect different camera,
    language, proprioception and normalisation conventions. The user-supplied
    adapter must produce correct model input including any batch dimensions.
    This class does not fabricate camera images or infer action semantics.
    """
    name: str
    family: str
    client: Any
    mapping: ExplicitActionMapping
    observation_adapter: Callable[[dict[str, Any]], Any]
    action_key: str = "actions"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A non-empty backend name is required")
        if self.family not in ("groot", "openpi", "lerobot", "callable"):
            raise ValueError("Unsupported VLA backend family")
        if not callable(self.observation_adapter):
            raise TypeError("Provide an explicit observation preprocessing function")
        method = {
            "groot": "get_action",
            "openpi": "infer",
            "lerobot": ("predict_action_chunk", "select_action"),
            "callable": "__call__",
        }[self.family]
        if isinstance(method, tuple):
            if not any(callable(getattr(self.client, key, None)) for key in method):
                raise TypeError(f"LeRobot backend requires one of {method}")
        elif not callable(getattr(self.client, method, None)):
            raise TypeError(f"Backend needs method {method}")

    def predict(self, observation: dict[str, Any]) -> ActionChunk:
        batch = self.observation_adapter(observation)
        if self.family == "groot":
            # GR00T: (dict[action_name, ndarray[B,T,D]], info).
            result = self.client.get_action(batch)
            if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], dict):
                raise ValueError("GR00T must return (action_dict, info)")
            if self.action_key not in result[0]:
                raise ValueError(f"Unknown GR00T action stream: {self.action_key}")
            raw = _numpy(result[0][self.action_key])
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError("Expected GR00T [B=1,T,D] stream")
            chunk = raw[0]
        elif self.family == "openpi":
            # OpenPI: policy.infer(...) returns dict with 'actions':[T,D].
            result = self.client.infer(batch)
            if not isinstance(result, dict) or self.action_key not in result:
                raise ValueError(f"OpenPI inference lacks '{self.action_key}'")
            chunk = _numpy(result[self.action_key])
        elif self.family == "lerobot":
            method = (getattr(self.client, "predict_action_chunk", None)
                      or getattr(self.client, "select_action", None))
            result = method(batch)
            if isinstance(result, dict):
                if "action" not in result:
                    raise ValueError("LeRobot policy result does not contain action")
                result = result["action"]
            chunk = _numpy(result)
            if chunk.ndim == 3:
                if chunk.shape[0] != 1:
                    raise ValueError("Only LeRobot batch size 1 is supported")
                chunk = chunk[0]
            elif chunk.ndim == 2 and chunk.shape[0] == 1 and hasattr(self.client, "select_action"):
                # select_action generally yields [B,D] for one control tick.
                pass
        else:
            chunk = _numpy(self.client(batch))
        return self.mapping.convert(chunk, self.name)


@dataclass(frozen=True)
class SingleArmMapping:
    """Explicit [T,6] arm contract for composing two single-arm policies."""
    source: ActionSpec
    side: str
    degrees_to_radians: bool = False

    def __post_init__(self):
        if self.side not in ("left", "right"):
            raise ValueError("Unknown arm")
        expected = tuple(f"{self.side}_{name}" for name in SPEC.ordered_axes[:6])
        # SPEC.ordered_axes already has prefixes; use the exact six target names.
        expected = SPEC.ordered_axes[:6] if self.side == "left" else SPEC.ordered_axes[6:]
        if self.source.robot != f"menagerie_{self.side}_ur5e":
            raise ValueError("Incorrect single-arm embodiment identity")
        if self.source.representation != "joint_position" or self.source.frame != "joint":
            raise ValueError("Single-arm source must produce absolute joint positions in joint frame")
        if set(self.source.ordered_axes) != set(expected) or len(self.source.ordered_axes) != 6:
            raise ValueError("Single-arm joint names do not exactly match target")
        if abs(self.source.dt-SPEC.dt)>1e-12:
            raise ValueError("Incompatible control period")
        if any(u not in ("rad", "deg") for u in self.source.units):
            raise ValueError("Unsupported unit")
        if "deg" in self.source.units and not self.degrees_to_radians:
            raise ValueError("Degrees require explicit opt-in")

    def convert(self, data) -> np.ndarray:
        a=_numpy(data)
        if a.ndim != 2 or a.shape[1] != 6 or not len(a) or not np.isfinite(a).all():
            raise ValueError("Expected finite [T,6]")
        names = SPEC.ordered_axes[:6] if self.side=="left" else SPEC.ordered_axes[6:]
        order=[self.source.ordered_axes.index(name) for name in names]
        out=a[:,order].copy()
        for j,i in enumerate(order):
            if self.source.units[i] == "deg":
                out[:,j]=np.deg2rad(out[:,j])
        return out


@dataclass
class DualArmComposer:
    """Join two independently predicted single-arm chunks, never pad/guess."""
    left_mapping: SingleArmMapping
    right_mapping: SingleArmMapping

    def compose(self, left: Any, right: Any, name="dual_teacher") -> ActionChunk:
        a = self.left_mapping.convert(left)
        b = self.right_mapping.convert(right)
        if a.shape[0] != b.shape[0]:
            raise ValueError("Different arm horizons: explicit synchronisation required")
        return ActionChunk(np.hstack([a,b]), SPEC, name)


def evaluate_teachers(teachers: dict[str, Any], *, menagerie, record_trace=False) -> dict:
    """Compare independent VLA proposals offline, with fresh MuJoCo resets.

    This is a benchmark, NOT a deployed online selector. Observed task outcomes
    cannot be used to select the next real action without information leakage.
    """
    from .dual_ur5e_policy import evaluate
    if len(teachers) < 2 or len(teachers) != len(set(teachers)) or any(not n for n in teachers):
        raise ValueError("Require at least two named teacher policies")
    scores={}
    errors={}
    for name,policy in teachers.items():
        try:
            report=evaluate(policy, menagerie=menagerie, record_trace=record_trace)
            scores[name]=report
        except (ValueError, TypeError, RuntimeError) as exc:
            errors[name]={"type":type(exc).__name__,"message":str(exc)}
    if not scores:
        raise ValueError(f"No valid policy could run: {errors}")
    return {
        "experiment":"independent_policy_comparison",
        "model_weights_verified":False,
        "provenance":"User-supplied model clients; check training checkpoints separately",
        "teachers":scores,
        "rejected":errors,
        "offline_best_by_goal_error":min(
            scores, key=lambda n:np.mean(scores[n]["tool_error_m"])),
        "warning":"Offline best uses measured outcomes and is an oracle comparison only. This is NOT online teacher selection."
    }
