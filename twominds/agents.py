"""Hybrid model-based / model-free agent for the two-step task (Daw et al., 2011).

theta = (alpha1, alpha2, lam, beta1, beta2, w, p). Model-free is w = 0, model-based is w = 1.

Data encoding (same as env.py): a1, a2 in {0, 1}; s2 in {1, 2}; r in {0, 1}.
Missed trials: a1 == -1 means stage 1 was missed (no likelihood, no update, and the
perseveration memory is reset). a1 >= 0 with a2 == -1 means stage 2 was missed (the stage-1
log-prob counts, no update; the stage-1 choice still counts as the previous choice).
"""

from __future__ import annotations

import numpy as np

from twominds.env import make_reward_schedule

P_COMMON = 0.7
# T[a1, s2 - 1] = P(s2 | a1). The agent knows the true transition structure.
T = np.array([[P_COMMON, 1 - P_COMMON], [1 - P_COMMON, P_COMMON]])


def _log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    return z - np.log(np.sum(np.exp(z)))


def _unpack(theta):
    alpha1, alpha2, lam, beta1, beta2, w, p = (float(v) for v in theta)
    return alpha1, alpha2, lam, beta1, beta2, w, p


class _Agent:
    """Value state shared by simulate, loglik and trace."""

    def __init__(self, theta):
        self.alpha1, self.alpha2, self.lam, self.beta1, self.beta2, self.w, self.p = _unpack(
            theta
        )
        self.q_mf = np.zeros(2)
        self.q2 = np.zeros((2, 2))
        self.prev_a1 = -1  # -1: no perseveration bonus on the next trial

    def values(self):
        q_mb = T @ self.q2.max(axis=1)
        q_net = self.w * q_mb + (1 - self.w) * self.q_mf
        return q_mb, q_net

    def logp1(self, q_net):
        rep = np.zeros(2)
        if self.prev_a1 >= 0:
            rep[self.prev_a1] = 1.0
        return _log_softmax(self.beta1 * (q_net + self.p * rep))

    def logp2(self, s2):
        return _log_softmax(self.beta2 * self.q2[s2 - 1])

    def update(self, a1, s2, a2, r):
        d1 = self.q2[s2 - 1, a2] - self.q_mf[a1]
        self.q_mf[a1] += self.alpha1 * d1
        d2 = r - self.q2[s2 - 1, a2]
        self.q2[s2 - 1, a2] += self.alpha2 * d2
        self.q_mf[a1] += self.alpha1 * self.lam * d2


def _replay(theta, data, record=False):
    """Run the agent over observed data. Returns (loglik, trace dict or None)."""
    agent = _Agent(theta)
    a1s, s2s, a2s, rs = (np.asarray(data[k]) for k in ("a1", "s2", "a2", "r"))
    n = len(a1s)
    tr = None
    if record:
        tr = {
            "q_mf": np.zeros((n, 2)),
            "q_mb": np.zeros((n, 2)),
            "q_net": np.zeros((n, 2)),
            "q2": np.zeros((n, 2, 2)),
        }
    ll = 0.0
    for t in range(n):
        q_mb, q_net = agent.values()
        if record:
            tr["q_mf"][t] = agent.q_mf
            tr["q_mb"][t] = q_mb
            tr["q_net"][t] = q_net
            tr["q2"][t] = agent.q2
        a1 = int(a1s[t])
        if a1 < 0:  # stage 1 missed: nothing counts, rep resets
            agent.prev_a1 = -1
            continue
        ll += agent.logp1(q_net)[a1]
        agent.prev_a1 = a1
        a2 = int(a2s[t])
        if a2 < 0:  # stage 2 missed: stage-1 term counted, no update
            continue
        s2 = int(s2s[t])
        ll += agent.logp2(s2)[a2]
        agent.update(a1, s2, a2, float(rs[t]))
    return float(ll), tr


def loglik(theta, data) -> float:
    """Sum of log P(a1) + log P(a2) over answered choices."""
    return _replay(theta, data)[0]


def trace(theta, data) -> dict:
    """Per-trial q_mf, q_mb, q_net (n, 2) and q2 (n, 2, 2) at the start of each trial."""
    return _replay(theta, data, record=True)[1]


def simulate(theta, n_trials, rng, schedule=None) -> dict:
    """Simulate a session. If `schedule` is None it is drawn from `rng` (env.py algorithm)."""
    if schedule is None:
        schedule = make_reward_schedule(n_trials, rng)
    agent = _Agent(theta)
    out = {k: np.zeros(n_trials, dtype=int) for k in ("a1", "s2", "a2", "r")}
    out["transition"] = np.empty(n_trials, dtype="<U6")
    out["p_a1"] = np.zeros(n_trials)
    out["p_a2"] = np.zeros(n_trials)
    for t in range(n_trials):
        _, q_net = agent.values()
        lp1 = agent.logp1(q_net)
        a1 = int(rng.random() < np.exp(lp1[1]))
        common = rng.random() < P_COMMON
        s2 = 1 + a1 if common else 2 - a1
        lp2 = agent.logp2(s2)
        a2 = int(rng.random() < np.exp(lp2[1]))
        r = int(rng.random() < schedule[t, s2 - 1, a2])
        out["a1"][t], out["s2"][t], out["a2"][t], out["r"][t] = a1, s2, a2, r
        out["transition"][t] = "common" if common else "rare"
        out["p_a1"][t] = np.exp(lp1[a1])
        out["p_a2"][t] = np.exp(lp2[a2])
        agent.prev_a1 = a1
        agent.update(a1, s2, a2, r)
    return out
