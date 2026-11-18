"""TM-09: model comparison + delta_w summary against fake InferenceData."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pytest

# ---- fake InferenceData helpers (hier.py stub contract) -------------------


def _fake_idata(
    *,
    seed: int,
    n_chains: int = 2,
    n_draws: int = 40,
    n_obs: int = 12,
    with_delta_w: bool = True,
    delta_w_mean: float = -0.8,
    mu_w_mean: float = 0.0,
    ll_offset: float = 0.0,
) -> az.InferenceData:
    """Tiny InferenceData via az.from_dict. Hybrid has delta_w; mf/mb do not."""
    rng = np.random.default_rng(seed)
    posterior: dict = {
        "mu_w": rng.normal(mu_w_mean, 0.15, size=(n_chains, n_draws)),
        "beta1": rng.normal(4.0, 0.4, size=(n_chains, n_draws)),
        "delta_beta1": rng.normal(-0.3, 0.2, size=(n_chains, n_draws)),
    }
    if with_delta_w:
        posterior["delta_w"] = rng.normal(delta_w_mean, 0.2, size=(n_chains, n_draws))
    log_lik = rng.normal(-1.0 + ll_offset, 0.3, size=(n_chains, n_draws, n_obs))
    return az.from_dict(posterior=posterior, log_likelihood={"obs": log_lik})


def _fake_model_fits() -> dict[str, az.InferenceData]:
    # hybrid slightly better ELPD via ll_offset; only hybrid carries delta_w
    return {
        "hybrid": _fake_idata(seed=1, with_delta_w=True, ll_offset=0.4),
        "mf": _fake_idata(seed=2, with_delta_w=False, ll_offset=0.0),
        "mb": _fake_idata(seed=3, with_delta_w=False, ll_offset=0.1),
    }


def _sim_condition_data(n_trials: int, w: float, seed: int) -> dict:
    from twominds.agents import simulate

    theta = (0.5, 0.5, 1.0, 5.0, 5.0, w, 0.0)
    return simulate(theta, n_trials, np.random.default_rng(seed))


# ---- import / compare ----------------------------------------------------


def test_compare_importable():
    from twominds.compare import model_compare, summarize_delta_w  # noqa: F401


def test_model_compare_returns_elpd_and_se():
    from twominds.compare import model_compare

    table = model_compare(_fake_model_fits())
    assert set(table.index) == {"hybrid", "mf", "mb"}
    for col in ("elpd_loo", "se", "elpd_diff", "dse", "weight"):
        assert col in table.columns
    assert np.isfinite(table["elpd_loo"]).all()


def test_summarize_delta_w_hybrid_only():
    from twominds.compare import summarize_delta_w

    fits = _fake_model_fits()
    # mf/mb must not be required for delta_w
    summary = summarize_delta_w(fits["hybrid"])
    assert "mean" in summary
    assert "hdi_low" in summary and "hdi_high" in summary
    assert summary["hdi_low"] < summary["hdi_high"]
    assert 0.0 <= summary["p_delta_w_lt_0"] <= 1.0
    assert "delta_w_group_mean" in summary  # change in w at group mean
    assert np.isfinite(summary["delta_w_group_mean"])
    # Negative delta_w_mean in fake → P(delta_w < 0) should be high
    assert summary["p_delta_w_lt_0"] > 0.8


def test_summarize_delta_w_rejects_missing_variable():
    from twominds.compare import summarize_delta_w

    with pytest.raises((KeyError, ValueError)):
        summarize_delta_w(_fake_idata(seed=9, with_delta_w=False))


def test_summarize_beta1_reports_condition_effect():
    from twominds.compare import summarize_beta1

    summary = summarize_beta1(_fake_model_fits()["hybrid"])
    assert "delta_beta1_mean" in summary
    assert "delta_beta1_hdi_low" in summary
    assert "delta_beta1_hdi_high" in summary
    assert np.isfinite(summary["delta_beta1_mean"])


def test_stay_regression_by_condition():
    from twominds.compare import stay_regression_by_condition

    datasets = {
        "no_pressure": _sim_condition_data(80, w=0.6, seed=10),
        "time_pressure": _sim_condition_data(80, w=0.3, seed=11),
    }
    out = stay_regression_by_condition(datasets)
    assert set(out) == {"no_pressure", "time_pressure"}
    for cond in out:
        assert "reward" in out[cond]["coef"]
        assert "interaction" in out[cond]["coef"]


def test_ppc_stay_overlay_writes_figure(tmp_path: Path):
    from twominds.compare import plot_ppc_stay

    datasets = {
        "no_pressure": _sim_condition_data(60, w=0.5, seed=20),
        "time_pressure": _sim_condition_data(60, w=0.3, seed=21),
    }
    out = tmp_path / "ppc_stay.png"
    path = plot_ppc_stay(
        _fake_model_fits()["hybrid"],
        datasets,
        out,
        rng=np.random.default_rng(0),
        n_ppc=5,
        n_trials=40,
    )
    assert path == out
    assert out.is_file() and out.stat().st_size > 0


def test_run_compare_writes_json_and_figure(tmp_path: Path):
    from twominds.compare import run_compare

    datasets = {
        "no_pressure": [_sim_condition_data(50, w=0.55, seed=30 + i) for i in range(3)],
        "time_pressure": [_sim_condition_data(50, w=0.25, seed=40 + i) for i in range(3)],
    }
    result = run_compare(
        _fake_model_fits(),
        datasets,
        out_dir=tmp_path,
        rng=np.random.default_rng(1),
        n_ppc=4,
        n_trials=30,
    )
    assert (tmp_path / "compare_summary.json").is_file()
    assert (tmp_path / "ppc_stay.png").is_file()
    assert "model_compare" in result
    assert "delta_w" in result
    assert "beta1" in result
    assert "stay_logit" in result
    # delta_w came from hybrid only
    assert result["delta_w"]["source_model"] == "hybrid"
