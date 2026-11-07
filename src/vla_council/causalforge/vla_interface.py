"""Validation boundary for future real VLA teacher adapters.

A tensor of the same shape is NOT proof that two policies command the same
physical actions. Require an explicit embedding/action specification.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Protocol, Any
import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    robot: str
    representation: str  # e.g. 'joint_position', 'ee_delta_pose'
    ordered_axes: tuple[str, ...]
    units: tuple[str, ...]
    frame: str
    dt: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt) or self.dt <= 0 or not self.robot or not self.frame or not self.representation:
            raise ValueError('ActionSpec requires robot, frame, representation and dt > 0')
        if not self.ordered_axes or len(self.ordered_axes) != len(self.units):
            raise ValueError('Each action axis needs a unit')
        if len(set(self.ordered_axes)) != len(self.ordered_axes):
            raise ValueError('Action axes must be unique')


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray  # [T,D]
    spec: ActionSpec
    model_name: str

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=float)
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != len(self.spec.ordered_axes):
            raise ValueError('Expected actions of shape [T, number of action axes]')
        if not np.all(np.isfinite(actions)):
            raise ValueError('Action contains NaN or infinity')
        object.__setattr__(self, 'actions', actions.copy())


def assert_same_action_space(chunks: list[ActionChunk]) -> None:
    if not chunks:
        raise ValueError('No proposals')
    expected = chunks[0].spec
    for chunk in chunks[1:]:
        if chunk.spec != expected:
            raise ValueError(f'Incompatible action specifications: {expected} != {chunk.spec}. '
                             'Convert explicitly in a validated robot adapter.')


class VLATeacher(Protocol):
    name: str
    spec: ActionSpec

    def propose(self, observation: Any, task_instruction: str) -> ActionChunk: ...


class CallableVLATeacher:
    """Minimal adapter boundary for an externally loaded, real policy.

    No weights, vendor libraries or pretrained models are included. The caller
    must implement image preprocessing, robot-specific action conversion and
    checkpoint loading. This class *rejects* mismatched returned action specs.
    """
    def __init__(self, name: str, spec: ActionSpec, predictor):
        if not name or not callable(predictor):
            raise ValueError('Teacher name and callable predictor required')
        self.name, self.spec, self._predictor = name, spec, predictor

    def propose(self, observation: Any, task_instruction: str) -> ActionChunk:
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError('Nonempty task instruction required')
        result = self._predictor(observation, task_instruction)
        if not isinstance(result, ActionChunk):
            raise TypeError('Predictor must return an ActionChunk with explicit ActionSpec')
        if result.spec != self.spec or result.model_name != self.name:
            raise ValueError('Predictor produced incompatible frame/units/timestep/model')
        return result
