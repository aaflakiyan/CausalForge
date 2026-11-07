"""Evidence-based arbitration with independent geometric checks.

Never blindly blend incompatible teacher actions or treat teacher self-reported
confidence as measured success probability.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .core import Observation, Proposal, Teacher
from .sim import collision_free

@dataclass(frozen=True)
class Critique:
    proposal: Proposal
    feasible: bool
    score: float
    reason: str

@dataclass(frozen=True)
class PeerReview:
    reviewer: str
    subject: str
    agrees: bool
    comment: str

@dataclass(frozen=True)
class Decision:
    action: tuple[float, float]
    teacher: str
    critiques: tuple[Critique, ...]
    reviews: tuple[PeerReview, ...]
    needs_intervention: bool

class GeometricCritic:
    def assess(self, obs: Observation, proposal: Proposal) -> Critique:
        a = np.asarray(proposal.action)
        if not collision_free(obs, a):
            return Critique(proposal, False, -1e9, "Collision, boundary or speed constraint")
        start_dist = float(np.hypot(obs.gx-obs.x, obs.gy-obs.y))
        next_xy = np.array([obs.x, obs.y]) + a
        end_dist = float(np.linalg.norm(np.array([obs.gx, obs.gy])-next_xy))
        # Score is a transparent heuristic, not learned/calibrated confidence.
        clearance = float(np.linalg.norm(next_xy - [obs.ox, obs.oy]) - obs.obstacle_radius)
        score = 4. * (start_dist - end_dist) + .12 * min(clearance, .3)
        return Critique(proposal, True, score, "Geometrically feasible (toy world)")

class Council:
    def __init__(self, teachers: Sequence[Teacher], critic: GeometricCritic | None = None):
        self.teachers = list(teachers)
        if not self.teachers:
            raise ValueError("At least one teacher required")
        self.critic = critic or GeometricCritic()

    def decide(self, obs: Observation) -> Decision:
        proposals = [t.propose(obs) for t in self.teachers]
        critiques = tuple(self.critic.assess(obs, p) for p in proposals)
        reviews = []
        for reviewer in proposals:
            for subject in proposals:
                if reviewer.name == subject.name:
                    continue
                a = np.asarray(reviewer.action)
                b = np.asarray(subject.action)
                cosine = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
                agrees = cosine >= 0.85
                reviews.append(PeerReview(reviewer.name, subject.name, agrees,
                                          "Directionally consistent" if agrees else
                                          "Disagrees on motion direction; requires external evaluation"))
        safe = [c for c in critiques if c.feasible]
        if not safe:
            return Decision((0., 0.), "STOP", critiques, tuple(reviews), True)
        best = max(safe, key=lambda c: c.score)
        return Decision(best.proposal.action, best.proposal.name, critiques,
                        tuple(reviews), False)
