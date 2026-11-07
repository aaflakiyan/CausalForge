"""Minimal student imitation baseline; not a VLA or flow-matching architecture."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .core import Observation, bounded

@dataclass
class LinearStudent:
    weights: np.ndarray

    @classmethod
    def fit(cls, observations: list[Observation], actions: list[tuple[float, float]], ridge: float = 1e-2):
        if len(observations) != len(actions) or not observations:
            raise ValueError("Provide matching non-empty examples")
        x = np.stack([o.features() for o in observations])
        y = np.asarray(actions)
        regularizer = ridge * np.eye(x.shape[1])
        regularizer[0, 0] = 0.
        weights = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        return cls(weights)

    def act(self, obs: Observation) -> np.ndarray:
        return bounded(obs.features() @ self.weights)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weights": self.weights.tolist(), "model_type": "linear-toy-student"}, indent=2))

    @classmethod
    def load(cls, path: Path):
        obj = json.loads(path.read_text())
        if obj.get("model_type") != "linear-toy-student":
            raise ValueError("Unsupported model file")
        weights = np.array(obj["weights"], dtype=float)
        if weights.shape != (10, 2):
            raise ValueError("Invalid weights shape")
        return cls(weights)
