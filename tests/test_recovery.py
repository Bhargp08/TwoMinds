import json

import numpy as np
import pytest

from twominds.fit_mle import BOUNDS
from twominds.recovery import (
    PARAM_NAMES,
    confusion_matrix,
    recovery_pipeline,
    run_recovery,
    sample_theta,
    summarize_recovery,
)

# Conventions under test
# - sample_theta(rng) -> 7-vector (alpha1, alpha2, lam, beta1, beta2, w, p) from the spec priors.
# - run_recovery(n_subjects, n_trials, seed, n_starts, n_jobs) -> {"true": (n,7), "fit": (n,7)}.
# - summarize_recovery(true, fit) -> {"r": {param: pearson r}, "fit_corr": 7x7 fitted-value matrix}.
# - confusion_matrix(generating, bics) -> (m, m) counts; rows = generating model index,
#   columns = model with the lowest BIC; bics is (n, m).
# - recovery_pipeline(out_dir, n_subjects, trial_counts, n_starts, n_jobs, seed, gate_trials,
#   n_model_subjects) writes figures + recovery_summary.json into out_dir and returns the summary.


def test_param_names_order():
    assert PARAM_NAMES == ("alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p")


def test_sample_theta_shape_bounds_and_determinism():
    a = sample_theta(np.random.default_rng(0))
    b = sample_theta(np.random.default_rng(0))
    assert a.shape == (7,) and np.array_equal(a, b)
    for value, (lo, hi) in zip(a, BOUNDS, strict=True):
        assert lo <= value <= hi


def test_sample_theta_matches_spec_priors():
    rng = np.random.default_rng(1)
    th = np.array([sample_theta(rng) for _ in range(5000)])
    for i in (0, 1, 2, 5):  # Beta(2, 2): mean 0.5, sd sqrt(0.05)
        assert th[:, i].mean() == pytest.approx(0.5, abs=0.02)
        assert th[:, i].std() == pytest.approx(np.sqrt(0.05), abs=0.02)
    assert th[:, 3:5].min() >= 0.5 and th[:, 3:5].max() <= 15  # beta clipped
    assert th[:, 3:5].mean() > 3.5  # Gamma(3, 1.5) mean 4.5, clipped
    assert th[:, 6].mean() == pytest.approx(0.15, abs=0.02)
    assert th[:, 6].std() == pytest.approx(0.2, abs=0.02)


def test_summarize_recovery_perfect_and_independent():
    rng = np.random.default_rng(2)
    true = np.array([sample_theta(rng) for _ in range(50)])
    out = summarize_recovery(true, true.copy())
    assert all(out["r"][name] == pytest.approx(1.0) for name in PARAM_NAMES)
    corr = np.asarray(out["fit_corr"])
    assert corr.shape == (7, 7) and np.allclose(corr, corr.T) and np.allclose(np.diag(corr), 1)
    noise = summarize_recovery(true, rng.random(true.shape))
    assert all(abs(v) < 0.5 for v in noise["r"].values())


def test_confusion_matrix_counts_lowest_bic():
    generating = np.array([0, 0, 0, 1, 1, 2])
    bics = np.array(
        [[1, 5, 9], [1, 5, 9], [5, 1, 9], [5, 1, 9], [5, 1, 9], [9, 5, 1]], dtype=float
    )
    cm = confusion_matrix(generating, bics)
    assert np.array_equal(cm, [[2, 1, 0], [0, 2, 0], [0, 0, 1]])


def test_run_recovery_shapes_bounds_and_determinism():
    kw = dict(n_subjects=3, n_trials=60, seed=5, n_starts=1, n_jobs=1)
    a = run_recovery(**kw)
    b = run_recovery(**kw)
    assert a["true"].shape == (3, 7) and a["fit"].shape == (3, 7)
    assert np.array_equal(a["true"], b["true"]) and np.array_equal(a["fit"], b["fit"])
    for col, (lo, hi) in enumerate(BOUNDS):
        assert a["fit"][:, col].min() >= lo - 1e-9 and a["fit"][:, col].max() <= hi + 1e-9


def test_pipeline_end_to_end_small(tmp_path):
    summary = recovery_pipeline(
        tmp_path,
        n_subjects=10,
        trial_counts=(60,),
        n_starts=1,
        n_jobs=1,
        seed=0,
        gate_trials=60,
        n_model_subjects=3,
    )
    on_disk = json.loads((tmp_path / "recovery_summary.json").read_text())
    assert set(on_disk) == set(summary)
    assert set(on_disk["trials"]["60"]["r"]) == set(PARAM_NAMES)
    assert np.asarray(on_disk["trials"]["60"]["fit_corr"]).shape == (7, 7)
    cm = np.asarray(on_disk["model_recovery"]["confusion"])
    assert cm.shape == (3, 3) and cm.sum() == 9
    assert on_disk["model_recovery"]["models"] == ["mf", "mb", "hybrid"]
    assert set(on_disk["gate"]["params"]) == {"w", "beta1", "beta2", "alpha2"}
    assert isinstance(on_disk["gate"]["passed"], bool) and on_disk["gate"]["threshold"] == 0.7
    pngs = {p.name for p in tmp_path.glob("*.png")}
    assert {"recovery_scatter_60.png", "fitted_param_corr_60.png", "model_confusion.png"} <= pngs


@pytest.mark.slow
def test_full_recovery_gate(tmp_path):
    """TM-05 gate: r > 0.7 for w, beta1, beta2, alpha2 at 100 subjects x 200 trials."""
    summary = recovery_pipeline(tmp_path, n_subjects=100, trial_counts=(200, 400), seed=0)
    r200 = summary["trials"]["200"]["r"]
    for name in ("w", "beta1", "beta2", "alpha2"):
        assert r200[name] > 0.7, (name, r200)
    assert summary["gate"]["passed"]
