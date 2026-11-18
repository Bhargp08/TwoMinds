"""Parameter and model recovery on synthetic participants (TM-05 gate)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twominds.agents import simulate  # noqa: E402
from twominds.fit_mle import fit_mle  # noqa: E402

PARAM_NAMES = ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p")
MODEL_NAMES = ["mf", "mb", "hybrid"]


PRIORS = ("realistic", "wide")
GATE_PARAMS = ("beta1", "beta2", "alpha2")  # gated per person at r > GATE_THRESHOLD
GATE_THRESHOLD = 0.7
W_FLOOR = 0.6  # w is reported, not gated: sanity floor only (see docs/recovery/README.md)


def sample_theta(rng: np.random.Generator, prior: str = "realistic") -> np.ndarray:
    """Draw theta from plausible priors.

    alpha1, alpha2, lam, w ~ Beta(2, 2); p ~ Normal(0.15, 0.2). Betas depend on `prior`:
    - "realistic" (default): Gamma(4, 1.5) clipped to [2, 15], a realistic human range.
    - "wide" (the original spec prior): Gamma(3, 1.5) clipped to [0.5, 15]. Near-zero betas
      carry almost no information about w, which is why the default changed (TM-05 diagnosis).
    """
    alpha1, alpha2, lam, w = rng.beta(2, 2, size=4)
    if prior == "realistic":
        beta1, beta2 = np.clip(rng.gamma(4.0, 1.5, size=2), 2.0, 15.0)
    elif prior == "wide":
        beta1, beta2 = np.clip(rng.gamma(3.0, 1.5, size=2), 0.5, 15.0)
    else:
        raise ValueError(f"unknown prior {prior!r}; use one of {PRIORS}")
    p = rng.normal(0.15, 0.2)
    return np.array([alpha1, alpha2, lam, beta1, beta2, w, p])


def _map(fn, args, n_jobs):
    n_jobs = n_jobs or os.cpu_count() or 1
    if n_jobs == 1:
        return [fn(a) for a in args]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(fn, args, chunksize=1))


def _recover_one(args):
    seed_seq, n_trials, n_starts, prior = args
    rng = np.random.default_rng(seed_seq)
    theta = sample_theta(rng, prior)
    data = simulate(theta, n_trials, rng)
    return theta, fit_mle(data, n_starts, rng)["theta"]


def run_recovery(
    n_subjects=100, n_trials=200, seed=0, n_starts=20, n_jobs=None, prior="realistic"
) -> dict:
    """Simulate subjects from the priors, fit each with the hybrid model, return true and fit."""
    seeds = np.random.SeedSequence(seed).spawn(n_subjects)
    out = _map(_recover_one, [(s, n_trials, n_starts, prior) for s in seeds], n_jobs)
    return {"true": np.array([o[0] for o in out]), "fit": np.array([o[1] for o in out])}


def summarize_recovery(true, fit) -> dict:
    """Pearson r (true vs fitted) per parameter and the correlation matrix of fitted values."""
    true, fit = np.asarray(true), np.asarray(fit)
    r = {
        name: float(np.corrcoef(true[:, i], fit[:, i])[0, 1]) for i, name in enumerate(PARAM_NAMES)
    }
    return {"r": r, "fit_corr": np.corrcoef(fit.T)}


def confusion_matrix(generating, bics) -> np.ndarray:
    """Rows: generating model; columns: model with the lowest BIC."""
    bics = np.asarray(bics)
    cm = np.zeros((bics.shape[1], bics.shape[1]), dtype=int)
    for g, row in zip(np.asarray(generating), bics, strict=True):
        cm[g, int(np.argmin(row))] += 1
    return cm


def _model_one(args):
    seed_seq, gen, n_trials, n_starts, prior = args
    rng = np.random.default_rng(seed_seq)
    theta = sample_theta(rng, prior)
    if gen == 0:
        theta[5] = 0.0
    elif gen == 1:
        theta[5] = 1.0
    data = simulate(theta, n_trials, rng)
    return [fit_mle(data, n_starts, rng, model=m)["bic"] for m in MODEL_NAMES]


def model_recovery(
    n_per_model=30, n_trials=200, seed=0, n_starts=20, n_jobs=None, prior="realistic"
) -> dict:
    """Simulate from MF, MB and hybrid; fit all three; BIC confusion matrix."""
    seeds = np.random.SeedSequence(seed).spawn(3 * n_per_model)
    gens = [g for g in range(3) for _ in range(n_per_model)]
    bics = _map(
        _model_one,
        [(s, g, n_trials, n_starts, prior) for s, g in zip(seeds, gens, strict=True)],
        n_jobs,
    )
    return {"models": MODEL_NAMES, "confusion": confusion_matrix(gens, bics)}


def _plot_scatter(true, fit, r, path, n_trials):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for i, (ax, name) in enumerate(zip(axes.ravel(), PARAM_NAMES, strict=False)):
        ax.scatter(true[:, i], fit[:, i], s=14, alpha=0.7)
        lo, hi = min(true[:, i].min(), fit[:, i].min()), max(true[:, i].max(), fit[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name}  r = {r[name]:.2f}")
        ax.set_xlabel("true")
        ax.set_ylabel("fitted")
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"Parameter recovery, {n_trials} trials")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_matrix(m, xlabels, ylabels, path, title, fmt):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(
        m,
        cmap="RdBu_r" if fmt == "{:.2f}" else "Blues",
        vmin=-1 if fmt == "{:.2f}" else 0,
        vmax=1 if fmt == "{:.2f}" else None,
    )
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _sweep(out, prior, tag, n_subjects, trial_counts, n_starts, n_jobs, seed, gate_trials, n_model):
    """Parameter recovery at each trial count + model recovery for one prior; writes figures."""
    trials = {}
    for n_trials in trial_counts:
        res = run_recovery(n_subjects, n_trials, seed, n_starts, n_jobs, prior)
        s = summarize_recovery(res["true"], res["fit"])
        _plot_scatter(
            res["true"], res["fit"], s["r"], out / f"recovery_scatter_{n_trials}{tag}.png", n_trials
        )
        _plot_matrix(
            s["fit_corr"],
            PARAM_NAMES,
            PARAM_NAMES,
            out / f"fitted_param_corr_{n_trials}{tag}.png",
            f"Fitted-parameter correlations, {n_trials} trials ({prior} prior)",
            "{:.2f}",
        )
        trials[str(n_trials)] = {"r": s["r"], "fit_corr": s["fit_corr"].tolist()}
    mr = model_recovery(n_model, gate_trials, seed, n_starts, n_jobs, prior)
    _plot_matrix(
        mr["confusion"],
        MODEL_NAMES,
        MODEL_NAMES,
        out / f"model_confusion{tag}.png",
        f"BIC confusion ({prior} prior; rows: generating, cols: best fit)",
        "{:d}",
    )
    model = {
        "models": mr["models"],
        "confusion": mr["confusion"].tolist(),
        "n_trials": gate_trials,
    }
    return trials, model


def recovery_pipeline(
    out_dir,
    n_subjects=100,
    trial_counts=(200, 400),
    n_starts=20,
    n_jobs=None,
    seed=0,
    gate_trials=200,
    n_model_subjects=30,
    include_wide_prior=False,
) -> dict:
    """Run recovery (default realistic prior) at each trial count plus model recovery.

    Writes figures and recovery_summary.json to `out_dir`. With `include_wide_prior`, the
    original wide-beta prior is run too and reported under "wide_prior" (figures suffixed
    `_wide`).

    Gate (at `gate_trials`): beta1, beta2, alpha2 must have r > 0.7. w is reported with a
    sanity floor of r > 0.6 but is not gated per person: it is information-limited at this
    trial count, and the study question (group-level delta_w) is gated by TM-08.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = (n_subjects, trial_counts, n_starts, n_jobs, seed, gate_trials, n_model_subjects)
    summary = {
        "config": {
            "n_subjects": n_subjects,
            "n_starts": n_starts,
            "seed": seed,
            "n_model_subjects": n_model_subjects,
            "prior": "realistic",
        },
    }
    summary["trials"], summary["model_recovery"] = _sweep(out, "realistic", "", *args)
    if include_wide_prior:
        trials, model = _sweep(out, "wide", "_wide", *args)
        summary["wide_prior"] = {"trials": trials, "model_recovery": model}
    gate_r = summary["trials"][str(gate_trials)]["r"]
    params_ok = all(gate_r[k] > GATE_THRESHOLD for k in GATE_PARAMS)
    summary["gate"] = {
        "trials": gate_trials,
        "threshold": GATE_THRESHOLD,
        "params": {k: gate_r[k] for k in GATE_PARAMS},
        "w": {"r": gate_r["w"], "floor": W_FLOOR, "reported_not_gated": True},
        "passed": bool(params_ok and gate_r["w"] > W_FLOOR),
    }
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
