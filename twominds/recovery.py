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
GATE_PARAMS = ("w", "beta1", "beta2", "alpha2")
GATE_THRESHOLD = 0.7
MODEL_NAMES = ["mf", "mb", "hybrid"]


def sample_theta(rng: np.random.Generator) -> np.ndarray:
    """Draw theta from the plausible priors in the spec."""
    alpha1, alpha2, lam, w = rng.beta(2, 2, size=4)
    beta1, beta2 = np.clip(rng.gamma(3.0, 1.5, size=2), 0.5, 15.0)
    p = rng.normal(0.15, 0.2)
    return np.array([alpha1, alpha2, lam, beta1, beta2, w, p])


def _map(fn, args, n_jobs):
    n_jobs = n_jobs or os.cpu_count() or 1
    if n_jobs == 1:
        return [fn(a) for a in args]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(fn, args, chunksize=1))


def _recover_one(args):
    seed_seq, n_trials, n_starts = args
    rng = np.random.default_rng(seed_seq)
    theta = sample_theta(rng)
    data = simulate(theta, n_trials, rng)
    return theta, fit_mle(data, n_starts, rng)["theta"]


def run_recovery(n_subjects=100, n_trials=200, seed=0, n_starts=20, n_jobs=None) -> dict:
    """Simulate subjects from the priors, fit each with the hybrid model, return true and fit."""
    seeds = np.random.SeedSequence(seed).spawn(n_subjects)
    out = _map(_recover_one, [(s, n_trials, n_starts) for s in seeds], n_jobs)
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
    seed_seq, gen, n_trials, n_starts = args
    rng = np.random.default_rng(seed_seq)
    theta = sample_theta(rng)
    if gen == 0:
        theta[5] = 0.0
    elif gen == 1:
        theta[5] = 1.0
    data = simulate(theta, n_trials, rng)
    return [fit_mle(data, n_starts, rng, model=m)["bic"] for m in MODEL_NAMES]


def model_recovery(n_per_model=30, n_trials=200, seed=0, n_starts=20, n_jobs=None) -> dict:
    """Simulate from MF, MB and hybrid; fit all three; BIC confusion matrix."""
    seeds = np.random.SeedSequence(seed).spawn(3 * n_per_model)
    gens = [g for g in range(3) for _ in range(n_per_model)]
    bics = _map(
        _model_one, [(s, g, n_trials, n_starts) for s, g in zip(seeds, gens, strict=True)], n_jobs
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


def recovery_pipeline(
    out_dir,
    n_subjects=100,
    trial_counts=(200, 400),
    n_starts=20,
    n_jobs=None,
    seed=0,
    gate_trials=200,
    n_model_subjects=30,
) -> dict:
    """Run parameter recovery at each trial count plus model recovery; write figures and JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_subjects": n_subjects,
            "n_starts": n_starts,
            "seed": seed,
            "n_model_subjects": n_model_subjects,
        },
        "trials": {},
    }
    for n_trials in trial_counts:
        res = run_recovery(n_subjects, n_trials, seed, n_starts, n_jobs)
        s = summarize_recovery(res["true"], res["fit"])
        _plot_scatter(
            res["true"], res["fit"], s["r"], out / f"recovery_scatter_{n_trials}.png", n_trials
        )
        _plot_matrix(
            s["fit_corr"],
            PARAM_NAMES,
            PARAM_NAMES,
            out / f"fitted_param_corr_{n_trials}.png",
            f"Fitted-parameter correlations, {n_trials} trials",
            "{:.2f}",
        )
        summary["trials"][str(n_trials)] = {"r": s["r"], "fit_corr": s["fit_corr"].tolist()}
    mr = model_recovery(n_model_subjects, gate_trials, seed, n_starts, n_jobs)
    _plot_matrix(
        mr["confusion"],
        MODEL_NAMES,
        MODEL_NAMES,
        out / "model_confusion.png",
        "BIC confusion (rows: generating, cols: best fit)",
        "{:d}",
    )
    summary["model_recovery"] = {
        "models": mr["models"],
        "confusion": mr["confusion"].tolist(),
        "n_trials": gate_trials,
    }
    gate_r = summary["trials"][str(gate_trials)]["r"]
    summary["gate"] = {
        "trials": gate_trials,
        "threshold": GATE_THRESHOLD,
        "params": {k: gate_r[k] for k in GATE_PARAMS},
        "passed": bool(all(gate_r[k] > GATE_THRESHOLD for k in GATE_PARAMS)),
    }
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
