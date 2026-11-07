"""Toy geometric world for testing arbitration. NOT a safety-certified controller."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .core import Observation, MAX_STEP

COLLISION_MARGIN = 0.015

def collision_free(obs: Observation, a: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        return False
    if np.linalg.norm(a) > MAX_STEP + 1e-9:
        return False
    start = np.array([obs.x, obs.y])
    end = start + a
    if np.any(end < 0.) or np.any(end > 1.):
        return False
    c = np.array([obs.ox, obs.oy])
    t = float(np.clip(np.dot(c - start, a) / max(float(a @ a), 1e-12), 0., 1.))
    nearest = start + t * a
    return bool(np.linalg.norm(nearest - c) > obs.obstacle_radius + COLLISION_MARGIN)

def step(obs: Observation, a: np.ndarray) -> Observation:
    if not collision_free(obs, a):
        raise ValueError("Unsafe action rejected by toy geometry gate")
    return replace(obs, x=obs.x + float(a[0]), y=obs.y + float(a[1]))

def done(obs: Observation) -> bool:
    return bool(np.hypot(obs.gx - obs.x, obs.gy - obs.y) <= 0.06)

def make_episode(seed: int) -> Observation:
    rng = np.random.default_rng(seed)
    # starts and goals deliberately cross the obstacle's neighbourhood
    yy = float(rng.uniform(0.28, 0.72))
    gy = float(np.clip(yy + rng.uniform(-0.18, 0.18), 0.13, .87))
    return Observation(.10, yy, .90, gy)
