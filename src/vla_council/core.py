"""Canonical action and observation types for a toy 2D robot.

Real VLA adapters must convert embodiment-specific observations/actions, units,
coordinate frames, time horizons and timestamps explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass(frozen=True)
class Observation:
    x: float
    y: float
    gx: float
    gy: float
    ox: float = 0.5
    oy: float = 0.5
    obstacle_radius: float = 0.14

    def features(self) -> np.ndarray:
        dx, dy = self.gx - self.x, self.gy - self.y
        rx, ry = self.x - self.ox, self.y - self.oy
        od = np.hypot(rx, ry)
        return np.array([1., self.x, self.y, dx, dy, rx, ry,
                         1. / max(od, .07), dx * ry - dy * rx,
                         dx * rx + dy * ry], dtype=float)

@dataclass(frozen=True)
class Proposal:
    name: str
    action: tuple[float, float]  # 2D displacement per tick, not motor commands
    rationale: str

class Teacher(Protocol):
    name: str
    def propose(self, obs: Observation) -> Proposal: ...

MAX_STEP = 0.08

def bounded(action: np.ndarray | tuple[float, float], max_step: float = MAX_STEP) -> np.ndarray:
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError("Action must be a finite 2-vector")
    norm = float(np.linalg.norm(a))
    return a if norm <= max_step else a * max_step / norm
