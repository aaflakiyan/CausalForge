"""Calibrated, budgeted diagnostic probes with explicit abstention.

Synthetic scalar/vector Gaussian sensor model. This is NOT a physical safety
checker or a causal discovery algorithm; hypotheses must be supplied in advance.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
import math
import numpy as np
from .experiment import Probe


@dataclass(frozen=True)
class Decision:
    probe: Probe | None
    information_gain: float
    utilities: dict[str, float]
    reason: str


def normalize(prior: Mapping[str, float]) -> dict[str, float]:
    if not prior or any(not math.isfinite(v) or v < 0 for v in prior.values()):
        raise ValueError('Prior must have non-negative finite probabilities')
    total = sum(prior.values())
    if total <= 0:
        raise ValueError('Prior cannot have zero mass')
    return {k: float(v / total) for k, v in prior.items()}


def update_belief(
    observation: np.ndarray, probe: Probe, belief: Mapping[str, float],
    predict: Callable[[str, Probe], np.ndarray], noise_std: float,
) -> dict[str, float]:
    if noise_std <= 0 or not math.isfinite(noise_std):
        raise ValueError('noise_std must be positive and finite')
    belief = normalize(belief)
    observation = np.asarray(observation, dtype=float).reshape(-1)
    if not observation.size or not np.all(np.isfinite(observation)):
        raise ValueError('Observation must be nonempty and finite')
    log_weights = {}
    for hypothesis, probability in belief.items():
        expected = np.asarray(predict(hypothesis, probe), dtype=float).reshape(-1)
        if expected.shape != observation.shape or not np.all(np.isfinite(expected)):
            raise ValueError('Invalid predicted observation')
        log_weights[hypothesis] = (math.log(probability) if probability else -math.inf) - 0.5 * float(
            np.sum(((observation - expected) / noise_std) ** 2)
        )
    largest = max(log_weights.values())
    weights = {h: math.exp(value - largest) for h, value in log_weights.items()}
    return normalize(weights)


def entropy(belief: Mapping[str, float]) -> float:
    return -sum(p * math.log(p) for p in normalize(belief).values() if p > 0)


def expected_information_gain(
    probe: Probe, belief: Mapping[str, float],
    predict: Callable[[str, Probe], np.ndarray], noise_std: float,
    *, samples_per_hypothesis: int = 80, seed: int = 0,
) -> float:
    """Monte Carlo expected entropy reduction; local RNG gives reproducibility.

    Measurements are sampled from each hypothesis according to the current
    belief. No hidden simulator truth is passed to the selection algorithm.
    """
    if samples_per_hypothesis < 1:
        raise ValueError('Require at least one sample')
    prior = normalize(belief)
    rng = np.random.default_rng(seed)
    means = {h: np.asarray(predict(h, probe), dtype=float).reshape(-1) for h in prior}
    shapes = {x.shape for x in means.values()}
    if len(shapes) != 1 or not all(x.size and np.all(np.isfinite(x)) for x in means.values()):
        raise ValueError('Predicted measurements must be finite and have matching shapes')
    expected_entropy = 0.0
    # Common-random-number noise reduces variance when comparing probes.
    noise = rng.normal(0, noise_std, size=(samples_per_hypothesis, len(next(iter(means.values())))))
    for hypothesis, probability in prior.items():
        if probability == 0:
            continue
        for eps in noise:
            measurement = means[hypothesis] + eps
            expected_entropy += probability * entropy(
                update_belief(measurement, probe, prior, predict, noise_std)
            ) / samples_per_hypothesis
    return max(0., entropy(prior) - expected_entropy)


def select_active_probe(
    probes: list[Probe], belief: Mapping[str, float],
    predict: Callable[[str, Probe], np.ndarray], noise_std: float,
    *, remaining_budget: float, risk_weight: float = 0., min_gain: float = 1e-4,
    samples_per_hypothesis: int = 80, seed: int = 0,
) -> Decision:
    """Select informative affordable probe or abstain.

    ``cost`` is an externally specified *nominal* experiment cost, not a
    measured risk or substitute for independent safety checks.
    """
    if remaining_budget < 0 or risk_weight < 0 or not math.isfinite(remaining_budget):
        raise ValueError('Invalid budget or risk weight')
    if not 0 <= min_gain or not math.isfinite(min_gain):
        raise ValueError('Invalid minimum information gain')
    candidates = [p for p in probes if 0 <= p.cost <= remaining_budget]
    if not candidates:
        return Decision(None, 0., {}, 'no_affordable_probe')
    utilities = {}
    gains = {}
    for i, probe in enumerate(candidates):
        if probe.name in utilities:
            raise ValueError('Probe names must be unique')
        gain = expected_information_gain(probe, belief, predict, noise_std,
                                        samples_per_hypothesis=samples_per_hypothesis, seed=seed)
        gains[probe.name] = gain
        utilities[probe.name] = gain - risk_weight * probe.cost
    best = max(candidates, key=lambda p: utilities[p.name])
    if gains[best.name] < min_gain or utilities[best.name] <= 0:
        return Decision(None, gains[best.name], utilities, 'not_informative_enough')
    return Decision(best, gains[best.name], utilities, 'selected')


def diagnose_or_abstain(belief: Mapping[str, float], confidence: float = 0.9) -> str | None:
    if not 0.5 < confidence <= 1.:
        raise ValueError('Confidence threshold must be between 0.5 and 1')
    prior = normalize(belief)
    winner = max(prior, key=prior.get)
    return winner if prior[winner] >= confidence else None
