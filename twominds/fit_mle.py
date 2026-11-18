"""Maximum-likelihood fitting of the hybrid agent (and its reduced models) with SciPy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from twominds.agents import loglik

# theta = (alpha1, alpha2, lam, beta1, beta2, w, p)
BOUNDS = ((0, 1), (0, 1), (0, 1), (0, 20), (0, 20), (0, 1), (-2, 2))

# model -> {theta index: fixed value}
MODELS = {
    "hybrid": {},
    "hybrid_nop": {6: 0.0},
    "mf": {5: 0.0},
    "mb": {5: 1.0},
    "mf_nop": {5: 0.0, 6: 0.0},
    "mb_nop": {5: 1.0, 6: 0.0},
}


def _random_start(rng: np.random.Generator) -> np.ndarray:
    x = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
    x[3:5] = rng.uniform(1.0, 10.0, size=2)  # betas: avoid near-zero / extreme starts
    x[6] = rng.uniform(-0.5, 0.5)
    return x


def _n_obs(data) -> int:
    a1 = np.asarray(data["a1"])
    a2 = np.asarray(data["a2"])
    return int(np.sum(a1 >= 0) + np.sum((a1 >= 0) & (a2 >= 0)))


def fit_mle(data, n_starts: int = 20, rng: np.random.Generator | None = None, model="hybrid"):
    """L-BFGS-B on the negative log-likelihood from random starts; keep the best.

    `model` is one of MODELS: hybrid / mf (w = 0) / mb (w = 1), each optionally without
    perseveration (`_nop`, p = 0). Returns theta (full 7-vector, fixed values filled in), nll,
    aic, bic (n = number of answered choices, stage 1 + stage 2), converged (best run's flag),
    n_free, n_obs and model.
    """
    if rng is None:
        rng = np.random.default_rng()
    fixed = MODELS[model]
    free = [i for i in range(7) if i not in fixed]

    def assemble(x_free):
        theta = np.zeros(7)
        for i, v in fixed.items():
            theta[i] = v
        theta[free] = x_free
        return theta

    def nll(x_free):
        return -loglik(assemble(x_free), data)

    bounds = [BOUNDS[i] for i in free]
    best = None
    for _ in range(n_starts):
        x0 = _random_start(rng)[free]
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    k, n = len(free), _n_obs(data)
    return {
        "theta": assemble(best.x),
        "nll": float(best.fun),
        "aic": float(2 * best.fun + 2 * k),
        "bic": float(2 * best.fun + k * np.log(n)),
        "converged": bool(best.success),
        "n_free": k,
        "n_obs": n,
        "model": model,
    }


def fit_models(data, n_starts: int = 20, rng: np.random.Generator | None = None) -> dict:
    """Fit all six models; returns {model name: fit_mle result}."""
    if rng is None:
        rng = np.random.default_rng()
    return {m: fit_mle(data, n_starts, rng, model=m) for m in MODELS}
