"""Hierarchical Bayesian hybrid model for the two-step task (TM-08), in PyMC.

Data
----
`datasets` is a list over participants. Each entry is either a dict keyed by condition
(``"no_pressure"`` / ``"time_pressure"`` as written by `io.load_session`, or 0 / 1) or a
``(no_pressure, time_pressure)`` pair. Each condition is a session dict with the keys of
`agents.simulate` / `io.load_session` (``a1, s2, a2, r``; missed stage 1: ``a1 = a2 = -1``;
missed stage 2: ``a2 = -1``). Condition code c = 0 is no pressure, c = 1 is time pressure.
Sessions of different lengths are padded at the end with missed trials (a1 = a2 = -1), which
contribute nothing to the likelihood and come after every real trial, so they change nothing.
Exclusion flags (``excluded``) are ignored here: the caller decides who goes in.

Likelihood
----------
Participants x conditions are flattened into one batch dimension (b = 2 * i + c). Two
implementations, both tested against `agents.loglik` to 1e-6 (including the missed-trial rules:
a missed stage 1 adds nothing, does no update and resets perseveration; a missed stage 2 counts
the stage-1 log-prob, does no update, and the stage-1 choice still sets the next trial's
perseveration):

* ``likelihood="scan"``: a `pytensor.scan` over trials (the spec's reference graph;
  `batch_loglik` uses it). Its gradient is slow on CPU (~56 ms at 40 participants x 2 x 200
  trials, plus ~30 s of graph rewriting per model), so it is not the sampling default.
* ``likelihood="numba"`` (default): a pytensor Op whose numba kernel returns the log-likelihood
  and its exact gradient (forward-mode sensitivities through the value recursion) in one pass,
  ~50x faster. Its value and gradient are tested against the scan graph / finite differences.

Model (per participant i, condition c)
--------------------------------------
Every per-participant parameter is non-centered: x_i = link^-1(mu_x + sd_x * z_i), z_i ~ N(0, 1).

* alpha1, alpha2, lam = invlogit(.)   mu ~ Normal(0, 1.5), sd ~ HalfNormal(1)
* beta1, beta2        = exp(.)        mu ~ Normal(1.5, 1), sd ~ HalfNormal(0.5)
  (log scale; exp(1.5) ~ 4.5, the centre of the realistic recovery prior)
* p                   = identity      mu ~ Normal(0, 1),   sd ~ HalfNormal(0.5)
* w_ic = invlogit(mu_w + delta_w * c + u_i + v_i * c), u_i = sigma_u * z_u_i,
  v_i = sigma_v * z_v_i, with mu_w ~ Normal(0, 1.5), delta_w ~ Normal(0, 1),
  sigma_u ~ HalfNormal(1), sigma_v ~ HalfNormal(0.5).
* ``beta1_by_condition=True``: beta1_ic = exp(mu_beta1 + delta_beta1 * c + sd_beta1 * z_i),
  delta_beta1 ~ Normal(0, 0.5) (a fixed condition effect on the log scale; separates "time
  pressure adds noise" from "time pressure lowers planning").
* ``model="mf"`` fixes w = 0 and ``model="mb"`` fixes w = 1 (no w parameters, so no delta_w);
  the rest of the hierarchy is the same.

All other parameters are shared across conditions within a participant.

LOO
---
`fit_hier` stores the total log-likelihood per (participant, condition) block in the
``log_likelihood`` group as ``"y"``, so `az.loo` / `az.compare` treat one participant-condition
session as one observation (leave-one-session-out). Per-trial log-likelihoods are added as
``"y_trial"`` with ``trial_loglik=True`` (then pass ``var_name="y"`` or ``"y_trial"`` to az.loo).
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from numba import njit
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.special import log_softmax

from twominds.agents import P_COMMON, simulate
from twominds.env import make_reward_schedule

CONDITIONS = ("no_pressure", "time_pressure")
MODELS = ("hybrid", "mf", "mb")
SUBJECT_PARAMS = ("alpha1", "alpha2", "lam", "beta1", "beta2", "p")
QUICK = {"draws": 300, "tune": 300, "chains": 2}


# --------------------------------------------------------------------------- data


def _condition_pair(entry):
    if isinstance(entry, dict):
        pair = []
        for c, name in enumerate(CONDITIONS):
            if name in entry:
                pair.append(entry[name])
            elif c in entry:
                pair.append(entry[c])
            else:
                raise ValueError(f"participant entry is missing condition {name!r} (or {c})")
        return pair
    if len(entry) != 2:
        raise ValueError("a participant entry must hold exactly two conditions")
    return list(entry)


def stack_datasets(datasets) -> dict:
    """Stack datasets into (T, N, 2) int arrays a1, s2, a2, r (end-padded with missed trials)."""
    pairs = [_condition_pair(e) for e in datasets]
    n_part = len(pairs)
    n_t = max(len(d["a1"]) for pair in pairs for d in pair)
    out = {
        "a1": np.full((n_t, n_part, 2), -1, dtype=int),
        "s2": np.ones((n_t, n_part, 2), dtype=int),
        "a2": np.full((n_t, n_part, 2), -1, dtype=int),
        "r": np.zeros((n_t, n_part, 2), dtype=int),
    }
    for i, pair in enumerate(pairs):
        for c, d in enumerate(pair):
            n = len(d["a1"])
            for k in out:
                out[k][:n, i, c] = np.asarray(d[k], dtype=int)
    return out


def _scan_inputs(stacked) -> dict:
    """One-hot / mask arrays (T, B, ...) with b = 2 * i + c, as float64."""
    n_t = stacked["a1"].shape[0]
    a1, s2, a2, r = (stacked[k].reshape(n_t, -1) for k in ("a1", "s2", "a2", "r"))
    v1 = a1 >= 0
    v2 = v1 & (a2 >= 0)
    o1 = (a1[..., None] == np.arange(2)) & v1[..., None]
    os_ = ((s2 - 1)[..., None] == np.arange(2)) & v2[..., None]
    oa = (a2[..., None] == np.arange(2)) & v2[..., None]
    rep = np.zeros_like(o1)
    rep[1:] = o1[:-1]  # previous answered stage-1 choice; a stage-1 miss leaves zeros (reset)
    f = np.float64
    return {
        "o1": o1.astype(f),
        "os": os_.astype(f),
        "oa": oa.astype(f),
        "r": (r * v2).astype(f),
        "v2": v2.astype(f),
        "rep": rep.astype(f),
    }


# --------------------------------------------------------------------------- likelihood


def _step(o1, os_, oa, r, v2, rep, qmf, q2, alpha1, alpha2, lam, beta1, beta2, w, p):
    # max over stage-2 actions; this form splits the gradient evenly at ties
    m = 0.5 * (q2[:, :, 0] + q2[:, :, 1] + pt.abs(q2[:, :, 0] - q2[:, :, 1]))
    pr = 1.0 - P_COMMON
    q_mb = pt.stack([P_COMMON * m[:, 0] + pr * m[:, 1], pr * m[:, 0] + P_COMMON * m[:, 1]], 1)
    q_net = w[:, None] * q_mb + (1.0 - w[:, None]) * qmf
    lp1 = (o1 * log_softmax(beta1[:, None] * (q_net + p[:, None] * rep), axis=1)).sum(1)
    q2s = (q2 * os_[:, :, None]).sum(1)  # values of the visited stage-2 state
    lp2 = (oa * log_softmax(beta2[:, None] * q2s, axis=1)).sum(1)
    qa = (q2s * oa).sum(1)
    d1 = qa - (qmf * o1).sum(1)
    d2 = r - qa
    answered = v2 > 0.5  # both stages answered: the only case that updates
    qmf_new = pt.switch(answered[:, None], qmf + (alpha1 * (d1 + lam * d2))[:, None] * o1, qmf)
    q2_new = pt.switch(
        answered[:, None, None],
        q2 + (alpha2 * d2)[:, None, None] * os_[:, :, None] * oa[:, None, :],
        q2,
    )
    return qmf_new, q2_new, lp1 + lp2


def _loglik_graph(inp: dict, params: dict):
    """Per-trial log-likelihood (T, B). `inp` from _scan_inputs (arrays or tensors)."""
    seqs = [pt.as_tensor_variable(inp[k]) for k in ("o1", "os", "oa", "r", "v2", "rep")]
    n_b = seqs[0].shape[1]
    qmf0 = pt.zeros((n_b, 2), dtype="float64")
    q20 = pt.zeros((n_b, 2, 2), dtype="float64")
    names = ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p")
    (_, _, ll), _ = pytensor.scan(
        _step,
        sequences=seqs,
        outputs_info=[qmf0, q20, None],
        non_sequences=[pt.as_tensor_variable(params[k]) for k in names],
        strict=True,
    )
    return ll


def _symbolic_inputs():
    inp = {
        "o1": pt.tensor3("o1"),
        "os": pt.tensor3("os"),
        "oa": pt.tensor3("oa"),
        "r": pt.matrix("r"),
        "v2": pt.matrix("v2"),
        "rep": pt.tensor3("rep"),
    }
    names = ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p")
    params = {k: pt.vector(k) for k in names}
    return inp, params


@lru_cache(maxsize=1)
def _compiled_loglik():
    inp, params = _symbolic_inputs()
    ll = _loglik_graph(inp, params)
    return pytensor.function(list(inp.values()) + list(params.values()), ll)


@njit(cache=True)
def _log_sigmoid(u):
    return -(max(-u, 0.0) + np.log1p(np.exp(-abs(u))))


@njit(cache=True)
def _ll_grad_kernel(theta, a1s, s2s, a2s, rs):
    """Log-likelihood per batch element and its gradient w.r.t. theta (B, 7).

    Same recursion as agents.loglik; the d* arrays carry d(state)/d(theta) forward in time
    (forward-mode sensitivities). Index order: alpha1, alpha2, lam, beta1, beta2, w, p.
    Data arrays are (T, B) ints. At a tie in max_a Q2 the gradient is split evenly (as in the
    scan graph); exact ties only occur between never-updated values with zero sensitivity.
    """
    n_t, n_b = a1s.shape
    pc = P_COMMON
    pr = 1.0 - P_COMMON
    ll = np.zeros(n_b)
    grad = np.zeros((n_b, 7))
    qmf = np.zeros(2)
    q2 = np.zeros((2, 2))
    m = np.zeros(2)
    dqmf = np.zeros((2, 7))
    dq2 = np.zeros((2, 2, 7))
    dm = np.zeros((2, 7))
    dz = np.zeros((2, 7))
    dqa = np.zeros(7)
    for b in range(n_b):
        al1, al2, lam = theta[b, 0], theta[b, 1], theta[b, 2]
        b1, b2, w, p = theta[b, 3], theta[b, 4], theta[b, 5], theta[b, 6]
        qmf[:] = 0.0
        q2[:] = 0.0
        dqmf[:] = 0.0
        dq2[:] = 0.0
        prev = -1
        for t in range(n_t):
            a1 = a1s[t, b]
            if a1 < 0:
                prev = -1
                continue
            for s in range(2):
                if q2[s, 0] > q2[s, 1]:
                    m[s] = q2[s, 0]
                    for k in range(7):
                        dm[s, k] = dq2[s, 0, k]
                elif q2[s, 1] > q2[s, 0]:
                    m[s] = q2[s, 1]
                    for k in range(7):
                        dm[s, k] = dq2[s, 1, k]
                else:
                    m[s] = q2[s, 0]
                    for k in range(7):
                        dm[s, k] = 0.5 * (dq2[s, 0, k] + dq2[s, 1, k])
            qmb0 = pc * m[0] + pr * m[1]
            qmb1 = pr * m[0] + pc * m[1]
            z0 = w * qmb0 + (1.0 - w) * qmf[0]
            z1 = w * qmb1 + (1.0 - w) * qmf[1]
            for k in range(7):
                dz[0, k] = w * (pc * dm[0, k] + pr * dm[1, k]) + (1.0 - w) * dqmf[0, k]
                dz[1, k] = w * (pr * dm[0, k] + pc * dm[1, k]) + (1.0 - w) * dqmf[1, k]
            dz[0, 5] += qmb0 - qmf[0]
            dz[1, 5] += qmb1 - qmf[1]
            if prev == 0:
                z0 += p
                dz[0, 6] += 1.0
            elif prev == 1:
                z1 += p
                dz[1, 6] += 1.0
            o = 1 - a1
            diff = (z1 - z0) if a1 == 1 else (z0 - z1)
            u = b1 * diff
            ll[b] += _log_sigmoid(u)
            g = 1.0 / (1.0 + np.exp(u))  # d log sigmoid(u) / du
            for k in range(7):
                grad[b, k] += g * b1 * (dz[a1, k] - dz[o, k])
            grad[b, 3] += g * diff
            prev = a1
            a2 = a2s[t, b]
            if a2 < 0:
                continue
            s = s2s[t, b] - 1
            qa = q2[s, a2]
            diff = qa - q2[s, 1 - a2]
            u = b2 * diff
            ll[b] += _log_sigmoid(u)
            g = 1.0 / (1.0 + np.exp(u))
            for k in range(7):
                grad[b, k] += g * b2 * (dq2[s, a2, k] - dq2[s, 1 - a2, k])
            grad[b, 4] += g * diff
            r = float(rs[t, b])
            d1 = qa - qmf[a1]
            d2 = r - qa
            # Q_MF(a1) += alpha1 * (d1 + lam * d2); Q2(s, a2) += alpha2 * d2
            for k in range(7):
                dqa[k] = dq2[s, a2, k]
                dqmf[a1, k] += al1 * (dqa[k] - dqmf[a1, k] - lam * dqa[k])
                dq2[s, a2, k] = dqa[k] - al2 * dqa[k]
            dqmf[a1, 0] += d1 + lam * d2
            dqmf[a1, 2] += al1 * d2
            dq2[s, a2, 1] += d2
            qmf[a1] += al1 * (d1 + lam * d2)
            q2[s, a2] = qa + al2 * d2
    return ll, grad


class _LoglikOp(Op):
    """theta (B, 7) -> (log-lik (B,), d log-lik / d theta (B, 7)); data fixed at construction."""

    def __init__(self, stacked):
        n_t = stacked["a1"].shape[0]
        self.data = tuple(
            np.ascontiguousarray(stacked[k].reshape(n_t, -1), dtype=np.int64)
            for k in ("a1", "s2", "a2", "r")
        )

    def make_node(self, theta):
        theta = pt.as_tensor_variable(theta)
        return Apply(self, [theta], [pt.dvector(), pt.dmatrix()])

    def perform(self, node, inputs, outputs):
        ll, grad = _ll_grad_kernel(np.asarray(inputs[0], dtype=np.float64), *self.data)
        outputs[0][0] = ll
        outputs[1][0] = grad

    def L_op(self, inputs, outputs, output_grads):
        from pytensor.gradient import DisconnectedType, grad_not_implemented

        g_ll, g_grad = output_grads
        if not isinstance(g_grad.type, DisconnectedType):
            return [grad_not_implemented(self, 0, inputs[0], "second derivatives")]
        return [g_ll[:, None] * outputs[1]]

    def connection_pattern(self, node):
        return [[True, True]]


def _eval_loglik(theta_b, inp):
    """theta_b (B, 7) -> per-trial log-lik (T, B) via the compiled scan graph."""
    f = _compiled_loglik()
    args = [inp[k] for k in ("o1", "os", "oa", "r", "v2", "rep")]
    return f(*args, *(np.ascontiguousarray(theta_b[:, k], dtype=float) for k in range(7)))


def batch_loglik(theta, datasets, per_trial=False) -> np.ndarray:
    """Log-likelihood from the scan graph; theta (N, 2, 7) in agents order.

    Returns (N, 2) totals, or (N, 2, T) per trial with ``per_trial=True``.
    """
    stacked = stack_datasets(datasets)
    n_t, n_part, _ = stacked["a1"].shape
    ll = _eval_loglik(np.asarray(theta, dtype=float).reshape(-1, 7), _scan_inputs(stacked))
    ll = ll.reshape(n_t, n_part, 2).transpose(1, 2, 0)
    return ll if per_trial else ll.sum(-1)


# --------------------------------------------------------------------------- model


def build_model(
    datasets,
    model: str = "hybrid",
    *,
    beta1_by_condition: bool = False,
    likelihood: str = "numba",
):
    """Build the hierarchical PyMC model (priors and likelihood options in the module
    docstring). Deterministic ``ll_pc`` (participant, condition) is the per-session log-lik."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; use one of {MODELS}")
    if likelihood not in ("numba", "scan"):
        raise ValueError("likelihood must be 'numba' or 'scan'")
    stacked = stack_datasets(datasets)
    n_t, n_part, _ = stacked["a1"].shape
    # Shared (not constant) data keeps the compiled graph independent of the data values, so
    # pytensor's C-module cache is reused across datasets.
    inp = None
    if likelihood == "scan":
        inp = {k: pytensor.shared(v, name=k) for k, v in _scan_inputs(stacked).items()}
    c = np.array([0.0, 1.0])
    coords = {"participant": np.arange(n_part), "condition": list(CONDITIONS)}
    with pm.Model(coords=coords) as m:

        def hier(name, mu_loc, mu_sd, sd_sd, link):
            mu_x = pm.Normal(f"mu_{name}", mu_loc, mu_sd)
            sd_x = pm.HalfNormal(f"sd_{name}", sd_sd)
            z = pm.Normal(f"z_{name}", 0.0, 1.0, dims="participant")
            return mu_x, sd_x, z, link(mu_x + sd_x * z)

        params = {}
        for name in ("alpha1", "alpha2", "lam"):
            *_, x = hier(name, 0.0, 1.5, 1.0, pm.math.invlogit)
            params[name] = pm.Deterministic(name, x, dims="participant")
        mu_b1, sd_b1, z_b1, b1 = hier("beta1", 1.5, 1.0, 0.5, pt.exp)
        if beta1_by_condition:
            delta_b1 = pm.Normal("delta_beta1", 0.0, 0.5)
            eta_b1 = mu_b1 + delta_b1 * c[None, :] + (sd_b1 * z_b1)[:, None]
            params["beta1"] = pm.Deterministic(
                "beta1", pt.exp(eta_b1), dims=("participant", "condition")
            )
        else:
            params["beta1"] = pm.Deterministic("beta1", b1, dims="participant")
        *_, b2 = hier("beta2", 1.5, 1.0, 0.5, pt.exp)
        params["beta2"] = pm.Deterministic("beta2", b2, dims="participant")
        *_, p = hier("p", 0.0, 1.0, 0.5, lambda x: x)
        params["p"] = pm.Deterministic("p", p, dims="participant")

        if model == "hybrid":
            mu_w = pm.Normal("mu_w", 0.0, 1.5)
            delta_w = pm.Normal("delta_w", 0.0, 1.0)
            sigma_u = pm.HalfNormal("sigma_u", 1.0)
            sigma_v = pm.HalfNormal("sigma_v", 0.5)
            z_u = pm.Normal("z_u", 0.0, 1.0, dims="participant")
            z_v = pm.Normal("z_v", 0.0, 1.0, dims="participant")
            eta_w = (
                mu_w
                + delta_w * c[None, :]
                + (sigma_u * z_u)[:, None]
                + (sigma_v * z_v)[:, None] * c[None, :]
            )
            w = pm.invlogit(eta_w)
        else:
            w = pt.zeros((params["p"].shape[0], 2)) + (1.0 if model == "mb" else 0.0)
        params["w"] = pm.Deterministic("w", w, dims=("participant", "condition"))

        def per_batch(x):  # (N,) or (N, 2) -> (B,), b = 2 * i + c
            return pt.repeat(x, 2) if x.ndim == 1 else x.flatten()

        names = ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p")
        if likelihood == "numba":
            theta_b = pt.stack([per_batch(params[k]) for k in names], axis=1)
            ll_b = _LoglikOp(stacked)(theta_b)[0]
        else:
            ll_b = _loglik_graph(inp, {k: per_batch(params[k]) for k in names}).sum(0)
        ll_pc = pm.Deterministic("ll_pc", ll_b.reshape((-1, 2)), dims=("participant", "condition"))
        pm.Potential("loglik", ll_pc.sum())
    return m


# --------------------------------------------------------------------------- fitting


def _theta_draws(post) -> np.ndarray:
    """Posterior -> theta (chain * draw, N, 2, 7) in agents order."""
    n_c, n_d = post.sizes["chain"], post.sizes["draw"]
    n_part = post.sizes["participant"]
    cols = []
    for name in ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p"):
        v = post[name].values.reshape(n_c * n_d, n_part, -1)
        cols.append(np.broadcast_to(v, (n_c * n_d, n_part, 2)))
    return np.stack(cols, -1)


def fit_hier(
    datasets,
    model: str = "hybrid",
    quick: bool = False,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    seed: int | None = None,
    target_accept: float = 0.9,
    beta1_by_condition: bool = False,
    trial_loglik: bool = False,
    progressbar: bool = False,
):
    """Sample the hierarchical model with NUTS and return an arviz InferenceData.

    ``quick=True`` caps draws / tune at 300 and chains at 2 (local runs). ``cores`` defaults to
    ``chains`` (parallel). The posterior holds the group parameters, per-participant
    ``alpha1, alpha2, lam, beta1, beta2, p`` and ``w`` (participant, condition), and, for
    ``model="hybrid"`` only, ``delta_w`` (plus ``delta_beta1`` with ``beta1_by_condition``);
    mf / mb fits have no ``delta_w``. ``idata.log_likelihood["y"]`` is the log-likelihood per
    (participant, condition) session, so ``az.loo(idata)`` / ``az.compare`` work and LOO is
    leave-one-session-out. With ``trial_loglik=True``, ``"y_trial"`` (..., trial) is added too
    (then pass ``var_name`` to az.loo).
    """
    if quick:
        draws, tune = min(draws, QUICK["draws"]), min(tune, QUICK["tune"])
        chains = min(chains, QUICK["chains"])
    m = build_model(datasets, model, beta1_by_condition=beta1_by_condition)
    with m:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores or chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=progressbar,
            compute_convergence_checks=False,
        )
    post = idata.posterior
    groups = {"y": post["ll_pc"]}
    if trial_loglik:
        stacked = stack_datasets(datasets)
        n_t, n_part, _ = stacked["a1"].shape
        inp = _scan_inputs(stacked)
        th = _theta_draws(post)
        ll = np.stack([_eval_loglik(t.reshape(-1, 7), inp) for t in th])  # (S, T, B)
        ll = ll.reshape(post.sizes["chain"], post.sizes["draw"], n_t, n_part, 2)
        groups["y_trial"] = xr.DataArray(
            ll.transpose(0, 1, 3, 4, 2),
            dims=("chain", "draw", "participant", "condition", "trial"),
            coords={
                "chain": post["chain"],
                "draw": post["draw"],
                "participant": post["participant"],
                "condition": post["condition"],
                "trial": np.arange(n_t),
            },
        )
    idata.add_groups(log_likelihood=xr.Dataset(groups))
    idata.posterior = post.drop_vars("ll_pc")
    return idata


def summarize_delta_w(idata, truth=None, hdi_prob: float = 0.95) -> dict:
    """Posterior mean / sd / HDI of delta_w, P(delta_w < 0), and the implied change in w at the
    group mean, invlogit(mu_w + delta_w) - invlogit(mu_w)."""
    post = idata.posterior
    d = post["delta_w"].values.ravel()
    lo, hi = az.hdi(post["delta_w"], hdi_prob=hdi_prob)["delta_w"].values
    mu = post["mu_w"].values.ravel()
    dw = 1 / (1 + np.exp(-(mu + d))) - 1 / (1 + np.exp(-mu))
    out = {
        "mean": float(d.mean()),
        "sd": float(d.std()),
        "hdi_prob": hdi_prob,
        "hdi_low": float(lo),
        "hdi_high": float(hi),
        "p_neg": float(np.mean(d < 0)),
        "dw_at_group_mean": float(dw.mean()),
    }
    if truth is not None:
        out["truth"] = float(truth)
        out["covers_truth"] = bool(lo <= truth <= hi)
    return out


def diagnostics(idata, var_names=None) -> dict:
    """Worst R-hat, min bulk / tail ESS over the key (group-level) parameters, divergences."""
    post = idata.posterior
    if var_names is None:
        key = ["delta_w", "mu_w", "sigma_u", "sigma_v", "delta_beta1"]
        key += [f"{s}_{p}" for p in SUBJECT_PARAMS for s in ("mu", "sd")]
        var_names = [v for v in key if v in post]
    rhat = az.rhat(post, var_names=var_names)
    ess_b = az.ess(post, var_names=var_names, method="bulk")
    ess_t = az.ess(post, var_names=var_names, method="tail")

    def worst(ds, fn):
        vals = {v: float(fn(ds[v].values)) for v in var_names}
        name = (max if fn is np.max else min)(vals, key=vals.get)
        return vals[name], name, vals

    r, r_name, r_all = worst(rhat, np.max)
    eb, eb_name, eb_all = worst(ess_b, np.min)
    et, et_name, _ = worst(ess_t, np.min)
    return {
        "var_names": var_names,
        "max_rhat": r,
        "max_rhat_var": r_name,
        "min_ess_bulk": eb,
        "min_ess_bulk_var": eb_name,
        "min_ess_tail": et,
        "min_ess_tail_var": et_name,
        "rhat": r_all,
        "ess_bulk": eb_all,
        "divergences": int(idata.sample_stats["diverging"].sum()),
    }


# --------------------------------------------------------------------------- simulation


def simulate_hier_dataset(
    n_participants: int,
    n_trials: int,
    delta_w: float,
    seed,
    *,
    mu_w: float = 0.0,
    sigma_u: float = 1.0,
    sigma_v: float = 0.3,
    delta_beta1: float = 0.0,
    model: str = "hybrid",
) -> dict:
    """Simulate a two-condition hierarchical dataset with a known delta_w.

    Group distributions match `recovery.sample_theta`'s realistic prior: alpha1, alpha2, lam =
    invlogit(N(0, 1)) (close to Beta(2, 2)); beta = exp(N(1.66, 0.53)) clipped to [2, 15] (the
    log-normal with the mean / sd of log Gamma(4, 1.5)); p ~ N(0.15, 0.2). w_ic =
    invlogit(mu_w + delta_w * c + u_i + v_i * c), u ~ N(0, sigma_u), v ~ N(0, sigma_v)
    (mu_w = 0, sigma_u = 1 matches w ~ Beta(2, 2)). beta1 in condition c is multiplied by
    exp(delta_beta1 * c). Each condition gets its own reward schedule. ``model="mf"`` / ``"mb"``
    fixes w = 0 / 1.

    Returns {"datasets": [{"no_pressure": data, "time_pressure": data}, ...], "truth": {...}}
    with truth["theta"] (N, 2, 7) and truth["schedules"] (N, 2, T, 2, 2).
    """
    rng = np.random.default_rng(seed)
    n = n_participants
    alphas = 1 / (1 + np.exp(-rng.normal(0.0, 1.0, size=(n, 3))))
    betas = np.clip(np.exp(rng.normal(1.66, 0.53, size=(n, 2))), 2.0, 15.0)
    p = rng.normal(0.15, 0.2, size=n)
    u = rng.normal(0.0, sigma_u, size=n)
    v = rng.normal(0.0, sigma_v, size=n)
    c = np.array([0.0, 1.0])
    w = 1 / (1 + np.exp(-(mu_w + delta_w * c[None, :] + u[:, None] + v[:, None] * c[None, :])))
    if model == "mf":
        w = np.zeros((n, 2))
    elif model == "mb":
        w = np.ones((n, 2))
    elif model != "hybrid":
        raise ValueError(f"unknown model {model!r}; use one of {MODELS}")
    theta = np.zeros((n, 2, 7))
    theta[:, :, 0:3] = alphas[:, None, :]
    theta[:, :, 3] = betas[:, 0, None] * np.exp(delta_beta1 * c)[None, :]
    theta[:, :, 4] = betas[:, 1, None]
    theta[:, :, 5] = w
    theta[:, :, 6] = p[:, None]
    schedules = np.zeros((n, 2, n_trials, 2, 2))
    datasets = []
    for i in range(n):
        entry = {}
        for ci, name in enumerate(CONDITIONS):
            schedules[i, ci] = make_reward_schedule(n_trials, rng)
            entry[name] = simulate(theta[i, ci], n_trials, rng, schedule=schedules[i, ci])
        datasets.append(entry)
    truth = {
        "theta": theta,
        "schedules": schedules,
        "delta_w": delta_w,
        "mu_w": mu_w,
        "sigma_u": sigma_u,
        "sigma_v": sigma_v,
        "delta_beta1": delta_beta1,
        "u": u,
        "v": v,
        "model": model,
    }
    return {"datasets": datasets, "truth": truth}


# --------------------------------------------------------------------------- recovery gate


def _plot_posteriors(runs, posteriors, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False)
    for ax, run, draws in zip(axes[0], runs, posteriors, strict=True):
        s = run["delta_w"]
        ax.hist(draws, bins=50, density=True, color="#6b8fb8", alpha=0.85)
        ax.axvline(s["truth"], color="k", lw=2, label=f"truth {s['truth']:g}")
        ax.axvspan(s["hdi_low"], s["hdi_high"], color="#e0a040", alpha=0.25, label="95% HDI")
        ax.axvline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"delta_w truth {s['truth']:g}, seed {run['seed']}")
        ax.set_xlabel("delta_w (logit scale)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def hier_recovery(
    out_dir,
    truths=(-0.8, 0.0),
    seeds=(0,),
    n_participants: int = 40,
    n_trials: int = 200,
    quick: bool = False,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
) -> dict:
    """TM-08 gate: simulate with each true delta_w (per seed), fit the hybrid model, check that
    the 95% HDI covers the truth, R-hat < 1.01, ESS > 400 and no divergences.

    Writes hier_recovery.json and delta_w_posteriors.png into `out_dir`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, posts = [], []
    for seed in seeds:
        for k, truth in enumerate(truths):
            sim = simulate_hier_dataset(n_participants, n_trials, truth, seed=[seed, k])
            t0 = time.perf_counter()
            idata = fit_hier(
                sim["datasets"], "hybrid", quick, draws=draws, tune=tune, chains=chains,
                seed=1000 * seed + k, target_accept=target_accept,
            )
            wall = time.perf_counter() - t0
            s = summarize_delta_w(idata, truth)
            d = diagnostics(idata)
            ok = (
                s["covers_truth"]
                and d["max_rhat"] < 1.01
                and d["min_ess_bulk"] > 400
                and d["min_ess_tail"] > 400
                and d["divergences"] == 0
            )
            runs.append({
                "truth": truth, "seed": seed, "delta_w": s, "diagnostics": d,
                "wall_time_s": round(wall, 1), "passed": bool(ok),
                "sampler": {"draws": idata.posterior.sizes["draw"],
                            "chains": idata.posterior.sizes["chain"],
                            "tune": min(tune, QUICK["tune"]) if quick else tune,
                            "target_accept": target_accept},
            })
            posts.append(idata.posterior["delta_w"].values.ravel())
            print(json.dumps({k: runs[-1][k] for k in ("truth", "seed", "wall_time_s", "passed")}
                             | {"mean": s["mean"], "hdi": [s["hdi_low"], s["hdi_high"]],
                                "max_rhat": d["max_rhat"], "min_ess": d["min_ess_bulk"],
                                "div": d["divergences"]}), flush=True)
    result = {
        "config": {"n_participants": n_participants, "n_trials_per_condition": n_trials,
                   "truths": list(truths), "seeds": list(seeds), "quick": quick,
                   "true_sigma_u": 1.0, "true_sigma_v": 0.3, "true_mu_w": 0.0},
        "runs": runs,
        "passed": all(r["passed"] for r in runs),
    }
    (out / "hier_recovery.json").write_text(json.dumps(result, indent=2))
    _plot_posteriors(runs, posts, out / "delta_w_posteriors.png")
    return result
