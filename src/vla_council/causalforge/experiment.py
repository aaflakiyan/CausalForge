"""Active experiment selection over a small, explicit set of physics hypotheses.

Does not use a language model to grade its own claims. Hypotheses are only as
useful as the simulator and measurement model used to define them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping
import numpy as np


@dataclass(frozen=True)
class Probe:
    name: str
    force: float
    duration: float
    cost: float = 0.0  # experiment cost / nominal risk penalty, NOT a safety certificate


@dataclass(frozen=True)
class Selection:
    probe: Probe
    utility: float
    scores: dict[str, float]


def information_proxy(predictions: Mapping[str, np.ndarray], noise_std: float = 0.01) -> float:
    """Pairwise prediction disagreement, an information proxy (not true EIG).

    Predictions should be commensurate measured quantities. Dimensionless
    scaling or whitening is required when units / uncertainties differ.
    """
    if noise_std <= 0:
        raise ValueError('noise_std must be positive')
    vectors = [np.asarray(x, dtype=float).reshape(-1) for x in predictions.values()]
    if len(vectors) < 2:
        raise ValueError('at least two hypotheses are required')
    if len({v.shape for v in vectors}) != 1 or not all(np.all(np.isfinite(v)) for v in vectors):
        raise ValueError('predictions must be finite vectors of equal shape')
    distances = [float(np.sum((vectors[i]-vectors[j])**2)) / (2*noise_std**2)
                 for i in range(len(vectors)) for j in range(i+1, len(vectors))]
    return float(np.mean(distances))


def choose_probe(
    probes: list[Probe], hypotheses: list[str],
    predict: Callable[[str, Probe], np.ndarray], noise_std: float = 0.01,
    risk_weight: float = 1.0,
) -> Selection:
    """Select a probe on expected physical separation minus experiment cost.

    This is an experimental-design heuristic. Later versions should replace it
    with posterior expected information gain and calibrated safety bounds.
    """
    if not probes or len(hypotheses) < 2:
        raise ValueError('Need probes and at least two hypotheses')
    scores = {}
    for probe in probes:
        predictions = {h: predict(h, probe) for h in hypotheses}
        scores[probe.name] = information_proxy(predictions, noise_std) - risk_weight*probe.cost
    selected = max(probes, key=lambda x: scores[x.name])
    return Selection(probe=selected, utility=scores[selected.name], scores=scores)


def posterior(
    actual: np.ndarray, hypotheses: list[str], probe: Probe,
    predict: Callable[[str, Probe], np.ndarray], noise_std: float = 0.01,
    prior: dict[str, float] | None = None,
) -> dict[str, float]:
    """Gaussian likelihood update. Assumes independent equal-variance noise."""
    if noise_std <= 0 or not hypotheses:
        raise ValueError('invalid noise or hypotheses')
    obs = np.asarray(actual, dtype=float).reshape(-1)
    probs = prior or {h: 1.0/len(hypotheses) for h in hypotheses}
    weights = []
    for h in hypotheses:
        p = np.asarray(predict(h, probe), dtype=float).reshape(-1)
        if p.shape != obs.shape or not np.all(np.isfinite(p)):
            raise ValueError('prediction has incompatible shape or invalid numbers')
        weights.append(math.log(max(probs[h], 1e-300)) - 0.5 * float(np.sum(((p - obs) / noise_std)**2)))
    offset = max(weights)
    terms = np.exp(np.asarray(weights)-offset)
    terms = terms / terms.sum()
    return {h: float(v) for h, v in zip(hypotheses, terms)}
