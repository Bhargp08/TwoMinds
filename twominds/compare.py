"""TM-09: hierarchical model comparison and time-pressure effect summaries.

Assumes `twominds.hier.fit_hier(datasets, model=..., quick=...)` returns an
`arviz.InferenceData` with `log_likelihood` stored. Only the **hybrid** fit is
expected to carry posterior `delta_w` (and `mu_w`); mf/mb fits are used solely
for `az.compare`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import arviz as az
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.special import expit

from twominds.agents import simulate
from twominds.analysis import stay_logit, stay_table

HDI_PROB = 0.95


def model_compare(idatas: Mapping[str, az.InferenceData]):
    """Compare hierarchical fits with PSIS-LOO (`az.compare`).

    `idatas` keys are model names (e.g. hybrid / mf / mb). Each InferenceData
    must include a `log_likelihood` group. Returns the arviz compare DataFrame
    (ELPD, SE, elpd_diff, dse, weight, …).
    """
    if len(idatas) < 2:
        raise ValueError("model_compare needs at least two InferenceData objects")
    return az.compare(dict(idatas), ic="loo")


def _flat(idata: az.InferenceData, name: str) -> np.ndarray:
    if name not in idata.posterior:
        raise KeyError(f"posterior variable '{name}' not found")
    return np.asarray(idata.posterior[name]).reshape(-1)


def _hdi_bounds(samples: np.ndarray, hdi_prob: float = HDI_PROB) -> tuple[float, float]:
    bounds = az.hdi(samples, hdi_prob=hdi_prob)
    # az.hdi on a 1d array returns an ndarray [low, high]
    return float(bounds[0]), float(bounds[1])


def summarize_delta_w(idata: az.InferenceData, hdi_prob: float = HDI_PROB) -> dict:
    """Summarize hybrid `delta_w`: mean, 95% HDI, P(delta_w < 0), group-mean Δw.

    Group-mean change in w uses `mu_w` on the logit scale:
    Δw = invlogit(mu_w + delta_w) − invlogit(mu_w), averaged over posterior draws.
    """
    delta = _flat(idata, "delta_w")
    if "mu_w" not in idata.posterior:
        raise KeyError("posterior variable 'mu_w' required for group-mean change in w")
    mu = _flat(idata, "mu_w")
    if mu.shape != delta.shape:
        raise ValueError("mu_w and delta_w must have the same number of draws")
    dw_prob = expit(mu + delta) - expit(mu)
    low, high = _hdi_bounds(delta, hdi_prob)
    return {
        "source_model": "hybrid",
        "mean": float(delta.mean()),
        "hdi_prob": hdi_prob,
        "hdi_low": low,
        "hdi_high": high,
        "p_delta_w_lt_0": float(np.mean(delta < 0)),
        "delta_w_group_mean": float(dw_prob.mean()),
        "delta_w_group_mean_hdi_low": _hdi_bounds(dw_prob, hdi_prob)[0],
        "delta_w_group_mean_hdi_high": _hdi_bounds(dw_prob, hdi_prob)[1],
    }


def summarize_beta1(idata: az.InferenceData, hdi_prob: float = HDI_PROB) -> dict:
    """Report condition effect on beta1 when `delta_beta1` is present.

    If only a shared `beta1` (or `mu_beta1`) exists, report that instead and set
    `condition_effect` to False.
    """
    if "delta_beta1" in idata.posterior:
        d = _flat(idata, "delta_beta1")
        low, high = _hdi_bounds(d, hdi_prob)
        out = {
            "condition_effect": True,
            "delta_beta1_mean": float(d.mean()),
            "delta_beta1_hdi_low": low,
            "delta_beta1_hdi_high": high,
            "p_delta_beta1_lt_0": float(np.mean(d < 0)),
        }
        if "beta1" in idata.posterior:
            b = _flat(idata, "beta1")
            out["beta1_mean"] = float(b.mean())
        return out
    name = "beta1" if "beta1" in idata.posterior else "mu_beta1"
    b = _flat(idata, name)
    low, high = _hdi_bounds(b, hdi_prob)
    return {
        "condition_effect": False,
        "beta1_mean": float(b.mean()),
        "beta1_hdi_low": low,
        "beta1_hdi_high": high,
        "note": f"shared {name}; no delta_beta1 in posterior",
    }


def stay_regression_by_condition(datasets: Mapping[str, dict | list]) -> dict:
    """Per-condition stay ~ reward * transition logistic regression (model-free check)."""
    return {str(cond): stay_logit(data) for cond, data in datasets.items()}


def _draw_theta(idata: az.InferenceData, rng: np.random.Generator) -> tuple:
    """Sample one hybrid θ from posterior draws (group-mean w from mu_w / delta_w).

    Missing optional vars fall back to mid-range defaults so PPC works with lean
    fake InferenceData in tests.
    """
    post = idata.posterior
    n = int(np.prod(post[next(iter(post.data_vars))].shape[:2]))
    idx = int(rng.integers(n))

    def pick(name, default):
        if name not in post:
            return default
        return float(np.asarray(post[name]).reshape(-1)[idx])

    mu_w = pick("mu_w", 0.0)
    delta_w = pick("delta_w", 0.0)
    # Return base params; caller chooses condition via c in {0,1}.
    alpha1 = pick("alpha1", pick("mu_alpha1", 0.5))
    alpha2 = pick("alpha2", pick("mu_alpha2", 0.5))
    lam = pick("lam", pick("mu_lam", 1.0))
    beta1 = pick("beta1", pick("mu_beta1", 5.0))
    beta2 = pick("beta2", pick("mu_beta2", 5.0))
    p = pick("p", pick("mu_p", 0.0))
    return (alpha1, alpha2, lam, beta1, beta2, mu_w, delta_w, p)


def plot_ppc_stay(
    idata: az.InferenceData,
    datasets: Mapping[str, dict | list],
    out_path,
    rng: np.random.Generator,
    n_ppc: int = 20,
    n_trials: int = 100,
) -> Path:
    """Posterior-predictive stay probabilities overlaid on observed stay tables.

    For each PPC draw, simulate one session per condition at group-mean
    w = invlogit(mu_w + delta_w * c) and plot mean stay ± band vs observed bars.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(datasets.keys())
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions), 3.4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions, strict=True):
        c = 0 if "no_pressure" in cond or cond.endswith("0") else 1
        obs = stay_table(datasets[cond])["stay"]
        ppc_tabs = []
        for _ in range(n_ppc):
            a1, a2, lam, b1, b2, mu_w, delta_w, p = _draw_theta(idata, rng)
            w = float(expit(mu_w + delta_w * c))
            sim = simulate((a1, a2, lam, b1, b2, w, p), n_trials, rng)
            ppc_tabs.append(stay_table(sim)["stay"])
        stack = np.stack(ppc_tabs, axis=0)  # (n_ppc, 2, 2)
        mean = np.nanmean(stack, axis=0)
        lo = np.nanpercentile(stack, 2.5, axis=0)
        hi = np.nanpercentile(stack, 97.5, axis=0)

        x = np.arange(2)
        width = 0.36
        ax.bar(x - width / 2, obs[1], width, label="Obs rewarded", color="#2f6f6a", alpha=0.85)
        ax.bar(x + width / 2, obs[0], width, label="Obs unrewarded", color="#c45c4a", alpha=0.85)
        ax.plot(x - width / 2, mean[1], "o", color="#1c2430", label="PPC mean")
        ax.plot(x + width / 2, mean[0], "o", color="#1c2430")
        ax.vlines(x - width / 2, lo[1], hi[1], color="#1c2430", linewidth=2)
        ax.vlines(x + width / 2, lo[0], hi[0], color="#1c2430", linewidth=2)
        ax.set_xticks(x, ["Common", "Rare"])
        ax.set_title(cond.replace("_", " "))
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("P(repeat stage-1 choice)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _compare_table_to_dict(table) -> dict:
    # arviz compare index = model names
    out = {}
    for name in table.index:
        row = table.loc[name]
        out[str(name)] = {
            "rank": int(row["rank"]),
            "elpd_loo": float(row["elpd_loo"]),
            "se": float(row["se"]),
            "elpd_diff": float(row["elpd_diff"]),
            "dse": float(row["dse"]),
            "weight": float(row["weight"]),
        }
    return out


def run_compare(
    idatas: Mapping[str, az.InferenceData],
    datasets: Mapping[str, dict | list],
    out_dir,
    rng: np.random.Generator,
    n_ppc: int = 20,
    n_trials: int = 100,
    hybrid_key: str = "hybrid",
) -> dict:
    """Run full TM-09 pipeline: compare, delta_w (hybrid only), stay logit, PPC figure.

    Writes `compare_summary.json` and `ppc_stay.png` under `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hybrid_key not in idatas:
        raise KeyError(f"idatas must include '{hybrid_key}' for delta_w summary")

    table = model_compare(idatas)
    delta = summarize_delta_w(idatas[hybrid_key])
    beta = summarize_beta1(idatas[hybrid_key])
    stay = stay_regression_by_condition(datasets)
    fig_path = plot_ppc_stay(
        idatas[hybrid_key],
        datasets,
        out_dir / "ppc_stay.png",
        rng=rng,
        n_ppc=n_ppc,
        n_trials=n_trials,
    )
    result = {
        "model_compare": _compare_table_to_dict(table),
        "delta_w": delta,
        "beta1": beta,
        "stay_logit": stay,
        "figures": {"ppc_stay": str(fig_path)},
    }
    (out_dir / "compare_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
