"""Model-free behavioural signatures: stay probabilities and logistic regression."""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm


def _sessions(data) -> list[dict]:
    return [data] if isinstance(data, dict) else list(data)


def _stay_trials(data):
    """Return arrays (stay, prev_reward, prev_rare) over all valid trials t >= 1.

    A trial is valid if a1[t] and a1[t-1] were answered and trial t-1's stage 2 was answered
    (so its reward is known). Pairs are never formed across sessions.
    """
    stay, rew, rare = [], [], []
    for sess in _sessions(data):
        a1 = np.asarray(sess["a1"])
        a2 = np.asarray(sess["a2"])
        r = np.asarray(sess["r"])
        tr = np.asarray(sess["transition"])
        ok = (a1[1:] >= 0) & (a1[:-1] >= 0) & (a2[:-1] >= 0)
        stay.append((a1[1:] == a1[:-1])[ok])
        rew.append((r[:-1] > 0)[ok])
        rare.append((tr[:-1] == "rare")[ok])
    return (
        np.concatenate(stay).astype(float),
        np.concatenate(rew).astype(int),
        np.concatenate(rare).astype(int),
    )


def stay_table(data) -> dict:
    """P(repeat a1) by previous reward (0/1) x previous transition (0 common, 1 rare)."""
    stay, rew, rare = _stay_trials(data)
    n = np.zeros((2, 2), dtype=int)
    p = np.full((2, 2), np.nan)
    for i in (0, 1):
        for j in (0, 1):
            m = (rew == i) & (rare == j)
            n[i, j] = m.sum()
            if n[i, j]:
                p[i, j] = stay[m].mean()
    return {"stay": p, "n": n}


def stay_effects(data) -> dict:
    """Reward effect and reward x transition interaction, in stay-probability units."""
    stay, rew, _ = _stay_trials(data)
    tab = stay_table(data)["stay"]
    return {
        "reward": float(stay[rew == 1].mean() - stay[rew == 0].mean()),
        "interaction": float((tab[1, 0] - tab[1, 1]) - (tab[0, 0] - tab[0, 1])),
    }


def stay_logit(data) -> dict:
    """Logistic regression stay ~ reward * transition (both coded +/-1), pooled (fixed effects)."""
    stay, rew, rare = _stay_trials(data)
    x_r = 2.0 * rew - 1.0  # +1 rewarded
    x_t = 1.0 - 2.0 * rare  # +1 common
    X = np.column_stack([np.ones_like(x_r), x_r, x_t, x_r * x_t])
    fit = sm.Logit(stay, X).fit(disp=0)
    names = ["intercept", "reward", "transition", "interaction"]
    return {
        "coef": dict(zip(names, map(float, fit.params), strict=True)),
        "se": dict(zip(names, map(float, fit.bse), strict=True)),
    }
