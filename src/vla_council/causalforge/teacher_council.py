"""Model-independent VLA action council with independently scored proposals.

The critic accepts externally measured or simulated metrics. A model's own
confidence or peer comments are NOT evidence of task success or safety.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import math
import numpy as np
from .vla_interface import ActionChunk, assert_same_action_space


@dataclass(frozen=True)
class Critique:
    task_loss: float
    collision_count: int
    constraint_violation: float
    uncertainty: float

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in (
                self.task_loss, self.constraint_violation, self.uncertainty)):
            raise ValueError('Critique metrics must be finite and nonnegative')
        if not isinstance(self.collision_count, int) or self.collision_count < 0:
            raise ValueError('collision_count must be nonnegative integer')


@dataclass(frozen=True)
class CouncilResult:
    selected_model: str | None
    selected: ActionChunk | None
    critiques: dict[str, Critique]
    scores: dict[str, float]
    disagreement: float
    abstained_reason: str | None


def decide(proposals: Sequence[ActionChunk],
           evaluator: Callable[[ActionChunk], Critique],
           *, max_violation: float = 0., max_uncertainty: float = 1.,
           uncertainty_weight: float = .1) -> CouncilResult:
    """Evaluate candidates, veto collisions/violations, otherwise lowest loss.

    Evaluator should be an independent predictive/measurement process. In
    particular, it must not trust unverified textual teacher claims.
    """
    assert_same_action_space(list(proposals))
    if len({x.model_name for x in proposals}) != len(proposals):
        raise ValueError('Teacher names must be unique')
    if max_violation < 0 or max_uncertainty < 0 or uncertainty_weight < 0:
        raise ValueError('Thresholds/weights must be nonnegative')
    if len({p.actions.shape for p in proposals}) != 1:
        raise ValueError('All action chunks must have equal horizon and shape')
    vectors = np.stack([p.actions.flatten() for p in proposals])
    disagreement = (float(np.mean([np.linalg.norm(vectors[i] - vectors[j])
                                    for i in range(len(vectors)) for j in range(i+1, len(vectors))]))
                    if len(vectors) > 1 else 0.)
    critiques: dict[str, Critique] = {}
    scores: dict[str, float] = {}
    for proposal in proposals:
        critique = evaluator(proposal)
        if not isinstance(critique, Critique):
            raise TypeError('Evaluator must return a Critique')
        critiques[proposal.model_name] = critique
        if (critique.collision_count or critique.constraint_violation > max_violation or
                critique.uncertainty > max_uncertainty):
            continue
        scores[proposal.model_name] = critique.task_loss + uncertainty_weight * critique.uncertainty
    if not scores:
        return CouncilResult(None, None, critiques, scores, disagreement, 'all_proposals_vetoed')
    winner = min(scores, key=scores.get)
    selected = next(p for p in proposals if p.model_name == winner)
    return CouncilResult(winner, selected, critiques, scores, disagreement, None)
