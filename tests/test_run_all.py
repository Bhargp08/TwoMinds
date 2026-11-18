"""TM-10: run_all pipeline + writeup placeholder filling (simulated data only)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "docs" / "writeup.md"


def _load_run_all():
    path = ROOT / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("twominds_run_all", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _fake_idata(seed: int, with_delta_w: bool, ll_offset: float = 0.0) -> az.InferenceData:
    rng = np.random.default_rng(seed)
    n_chains, n_draws, n_obs = 2, 30, 8
    posterior = {
        "mu_w": rng.normal(0.0, 0.15, size=(n_chains, n_draws)),
        "beta1": rng.normal(4.0, 0.3, size=(n_chains, n_draws)),
        "delta_beta1": rng.normal(-0.2, 0.15, size=(n_chains, n_draws)),
    }
    if with_delta_w:
        posterior["delta_w"] = rng.normal(-0.7, 0.2, size=(n_chains, n_draws))
    return az.from_dict(
        posterior=posterior,
        log_likelihood={"obs": rng.normal(-1.0 + ll_offset, 0.25, size=(n_chains, n_draws, n_obs))},
    )


def _fake_fit_hier(datasets, model="hybrid", quick=False):
    assert model in {"hybrid", "mf", "mb"}
    assert isinstance(datasets, dict)
    return _fake_idata(
        seed={"hybrid": 1, "mf": 2, "mb": 3}[model],
        with_delta_w=(model == "hybrid"),
        ll_offset={"hybrid": 0.3, "mf": 0.0, "mb": 0.05}[model],
    )


REQUIRED_PLACEHOLDERS = [
    "delta_w_mean",
    "delta_w_hdi_low",
    "delta_w_hdi_high",
    "p_delta_w_lt_0",
    "delta_w_group_mean",
    "elpd_hybrid",
    "elpd_mf",
    "elpd_mb",
    "stay_reward_no_pressure",
    "stay_interaction_time_pressure",
    "n_subjects",
    "n_trials_per_condition",
    "run_mode",
]


def test_writeup_skeleton_has_placeholders():
    text = WRITEUP.read_text()
    assert "not IRB" in text or "not IRB-approved" in text
    for key in REQUIRED_PLACEHOLDERS:
        assert "{{" + key + "}}" in text, key


def test_run_all_module_importable():
    mod = _load_run_all()
    assert hasattr(mod, "main") or hasattr(mod, "run_pipeline")
    assert hasattr(mod, "render_writeup")
    assert hasattr(mod, "build_placeholder_map")


def test_render_writeup_fills_all_placeholders(tmp_path: Path):
    mod = _load_run_all()
    summary = {
        "model_compare": {
            "hybrid": {
                "elpd_loo": -10.5,
                "se": 1.2,
                "rank": 0,
                "elpd_diff": 0.0,
                "dse": 0.0,
                "weight": 0.7,
            },
            "mf": {
                "elpd_loo": -12.0,
                "se": 1.1,
                "rank": 2,
                "elpd_diff": 1.5,
                "dse": 0.4,
                "weight": 0.1,
            },
            "mb": {
                "elpd_loo": -11.0,
                "se": 1.0,
                "rank": 1,
                "elpd_diff": 0.5,
                "dse": 0.3,
                "weight": 0.2,
            },
        },
        "delta_w": {
            "mean": -0.81,
            "hdi_low": -1.2,
            "hdi_high": -0.3,
            "p_delta_w_lt_0": 0.97,
            "delta_w_group_mean": -0.18,
        },
        "beta1": {
            "condition_effect": True,
            "delta_beta1_mean": -0.25,
            "delta_beta1_hdi_low": -0.6,
            "delta_beta1_hdi_high": 0.1,
        },
        "stay_logit": {
            "no_pressure": {"coef": {"reward": 0.4, "interaction": 0.2}, "se": {}},
            "time_pressure": {"coef": {"reward": 0.5, "interaction": 0.05}, "se": {}},
        },
        "figures": {"ppc_stay": "docs/figures/ppc_stay.png"},
    }
    meta = {
        "n_subjects": 4,
        "n_trials_per_condition": 30,
        "run_mode": "simulate-quick",
        "hier_recovery_note": "not run in quick mode",
    }
    mapping = mod.build_placeholder_map(summary, meta)
    rendered = mod.render_writeup(WRITEUP.read_text(), mapping)
    assert "{{" not in rendered
    assert "-0.81" in rendered or "-0.810" in rendered
    out = tmp_path / "writeup_filled.md"
    out.write_text(rendered)
    assert out.is_file()


def test_run_pipeline_simulate_quick(tmp_path: Path):
    mod = _load_run_all()
    out_dir = tmp_path / "out"
    result = mod.run_pipeline(
        out_dir=out_dir,
        simulate=True,
        quick=True,
        seed=0,
        n_subjects=3,
        n_trials=40,
        fit_hier=_fake_fit_hier,
        writeup_template=WRITEUP,
    )
    assert (out_dir / "compare_summary.json").is_file()
    assert (out_dir / "ppc_stay.png").is_file()
    assert (out_dir / "writeup_filled.md").is_file()
    filled = (out_dir / "writeup_filled.md").read_text()
    assert "{{" not in filled
    summary = json.loads((out_dir / "compare_summary.json").read_text())
    assert "delta_w" in summary and "model_compare" in summary
    assert result["n_subjects"] == 3
