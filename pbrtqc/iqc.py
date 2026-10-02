"""Internal quality control: Westgard multirule evaluation, power functions, analytical statistics.

QC results are handled as z-scores relative to the stable (in-control) mean and SD of each level:
    z = (observed - target) / SD_stable
A "QC event" is one round of QC across all levels, so a series is an
(n_events, n_levels) matrix of z-scores in chronological order.
"""
from __future__ import annotations
import numpy as np

RULE_SET_DEFAULT = ("1-3s", "2-2s", "R-4s", "4-1s", "10x")
ALL_RULES = ("1-2s", "1-3s", "2-2s", "R-4s", "4-1s", "8x", "10x")


def _violates(z: np.ndarray, rules) -> bool:
    """Does the LAST QC event in z (n_events, n_levels) trigger a rejection?"""
    last = z[-1]
    for r in rules:
        if r == "1-2s" and np.any(np.abs(last) > 2):
            return True
        if r == "1-3s" and np.any(np.abs(last) > 3):
            return True
        if r == "2-2s":
            # within event: all levels beyond 2 on the same side
            if last.size >= 2 and (np.all(last > 2) or np.all(last < -2)):
                return True
            # across events: same level, two consecutive beyond 2 same side
            if z.shape[0] >= 2:
                pair = z[-2:]
                if np.any(np.all(pair > 2, axis=0) | np.all(pair < -2, axis=0)):
                    return True
        if r == "R-4s" and last.size >= 2:
            if last.max() - last.min() > 4:
                return True
        if r == "4-1s" and z.shape[0] >= 4:
            q = z[-4:]
            if np.any(np.all(q > 1, axis=0) | np.all(q < -1, axis=0)):
                return True
        if r in ("8x", "10x"):
            n = 8 if r == "8x" else 10
            if z.shape[0] >= n:
                q = z[-n:]
                if np.any(np.all(q > 0, axis=0) | np.all(q < 0, axis=0)):
                    return True
    return False


def evaluate_series(z: np.ndarray, rules=RULE_SET_DEFAULT) -> np.ndarray:
    """Rejection flag for every QC event in a series (n_events, n_levels)."""
    out = np.zeros(z.shape[0], bool)
    for i in range(z.shape[0]):
        out[i] = _violates(z[: i + 1], rules)
    return out


def power(se_sd: float, rules=RULE_SET_DEFAULT, n_levels=2, re_mult=1.0, n_sim=4000,
          hist=12, rng=None) -> float:
    """Probability of rejection at the first QC event after a shift of `se_sd` (in stable SD units).

    History before the shift is in control, which is the usual power-function convention.
    `re_mult` multiplies the SD (random-error increase).
    """
    rng = rng or np.random.default_rng(0)
    z = rng.normal(0, 1, (n_sim, hist + 1, n_levels))
    z[:, -1, :] = rng.normal(se_sd, re_mult, (n_sim, n_levels))
    return float(np.mean([_violates(z[k], rules) for k in range(n_sim)]))


def power_curve(se_grid, rules=RULE_SET_DEFAULT, n_levels=2, **kw) -> np.ndarray:
    return np.array([power(s, rules, n_levels, **kw) for s in se_grid])


def analytical_stats(values: np.ndarray, target: float, tea_frac: float) -> dict:
    """CVa, bias and sigma metric from an in-control QC series of one level.

    sigma = (TEa% - |bias%|) / CVa%   (bias measured against the assigned target)
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    cva = sd / mean
    bias = (mean - target) / target if target else 0.0
    sigma = (tea_frac - abs(bias)) / cva if cva > 0 else np.inf
    return dict(n=int(v.size), mean=mean, sd=sd, cva=cva, bias=bias, sigma=float(sigma),
                target=float(target))
