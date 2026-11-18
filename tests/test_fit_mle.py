import numpy as np
import pytest

from twominds.agents import _replay, loglik, simulate
from twominds.fit_mle import BOUNDS, MODELS, fit_mle, fit_models

# Conventions under test
# - fit_mle(data, n_starts=20, rng=None, model="hybrid") -> dict with
#   theta (7-array, fixed params filled in), nll, aic, bic, converged (bool), n_free, n_obs, model.
# - model in MODELS = hybrid / mf / mb, each with and without perseveration "_nop" (p fixed 0).
#   mf fixes w = 0, mb fixes w = 1.
# - n_obs = number of answered choices (stage-1 plus stage-2); BIC = 2 nll + k ln n_obs.
# - fit_models(data, n_starts, rng) -> {model name: fit_mle result} for all six models.

THETA = (0.4, 0.6, 0.5, 5.0, 5.0, 0.6, 0.15)  # alpha1, alpha2, lam, beta1, beta2, w, p


@pytest.fixture(scope="module")
def long_data():
    return simulate(THETA, 2000, np.random.default_rng(0))


@pytest.fixture(scope="module")
def long_fit(long_data):
    return fit_mle(long_data, n_starts=4, rng=np.random.default_rng(1))


def with_missed(data, rng, frac=0.1):
    d = {k: np.array(v) for k, v in data.items()}
    n = len(d["a1"])
    m1 = rng.random(n) < frac
    m2 = (rng.random(n) < frac) & ~m1
    d["a1"][m1] = -1
    d["a2"][m2] = -1
    return d


# ---- fast loglik agrees with the reference replay -------------------------


def test_fast_loglik_matches_reference_replay_with_missed_trials():
    rng = np.random.default_rng(3)
    for _ in range(10):
        theta = (
            rng.random(), rng.random(), rng.random(),
            rng.uniform(0, 20), rng.uniform(0, 20), rng.random(), rng.uniform(-2, 2),
        )  # fmt: skip
        d = with_missed(simulate(theta, 150, rng), rng, 0.15)
        assert loglik(theta, d) == pytest.approx(_replay(theta, d)[0], abs=1e-10)


# ---- recovery on a long session (spec) ------------------------------------


def test_recovers_w_alpha2_and_beta1_on_2000_trials(long_fit):
    th = long_fit["theta"]
    assert abs(th[5] - THETA[5]) < 0.15  # w
    assert abs(th[1] - THETA[1]) < 0.15  # alpha2
    assert abs(th[3] - THETA[3]) < 1.5  # beta1


def test_fitted_nll_not_worse_than_nll_at_true_theta(long_data, long_fit):
    assert long_fit["nll"] <= -loglik(THETA, long_data) + 1e-6


def test_nll_in_result_matches_loglik_at_fitted_theta(long_data, long_fit):
    assert long_fit["nll"] == pytest.approx(-loglik(long_fit["theta"], long_data), abs=1e-8)


def test_fitted_theta_within_bounds(long_fit):
    th = long_fit["theta"]
    assert len(th) == 7
    for value, (lo, hi) in zip(th, BOUNDS, strict=True):
        assert lo - 1e-9 <= value <= hi + 1e-9


# ---- result bookkeeping ---------------------------------------------------


def test_aic_bic_and_flags(long_data, long_fit):
    k, n = long_fit["n_free"], long_fit["n_obs"]
    assert k == 7 and n == 4000
    assert long_fit["aic"] == pytest.approx(2 * long_fit["nll"] + 2 * k)
    assert long_fit["bic"] == pytest.approx(2 * long_fit["nll"] + k * np.log(n))
    assert isinstance(long_fit["converged"], bool)
    assert long_fit["model"] == "hybrid"


def test_n_obs_counts_only_answered_choices():
    rng = np.random.default_rng(5)
    d = {k: np.array(v) for k, v in simulate(THETA, 40, rng).items()}
    d["a1"][0] = -1  # missed stage 1: no choices counted
    d["a2"][1] = -1  # missed stage 2: only the stage-1 choice counted
    res = fit_mle(d, n_starts=1, rng=np.random.default_rng(0))
    assert res["n_obs"] == 2 * 40 - 2 - 1
    assert np.isfinite(res["nll"])


# ---- reduced models -------------------------------------------------------


@pytest.mark.parametrize(
    "model, n_free, fixed",
    [
        ("hybrid", 7, {}),
        ("hybrid_nop", 6, {6: 0.0}),
        ("mf", 6, {5: 0.0}),
        ("mb", 6, {5: 1.0}),
        ("mf_nop", 5, {5: 0.0, 6: 0.0}),
        ("mb_nop", 5, {5: 1.0, 6: 0.0}),
    ],
)
def test_reduced_models_fix_parameters(model, n_free, fixed):
    d = simulate(THETA, 120, np.random.default_rng(2))
    res = fit_mle(d, n_starts=1, rng=np.random.default_rng(0), model=model)
    assert res["n_free"] == n_free and res["model"] == model
    for idx, value in fixed.items():
        assert res["theta"][idx] == value


def test_models_constant_lists_the_six_models():
    assert set(MODELS) == {"hybrid", "hybrid_nop", "mf", "mb", "mf_nop", "mb_nop"}


def test_fit_models_returns_all_six():
    d = simulate(THETA, 80, np.random.default_rng(4))
    out = fit_models(d, n_starts=1, rng=np.random.default_rng(0))
    assert set(out) == set(MODELS)
    assert all(np.isfinite(v["bic"]) for v in out.values())


def test_bic_prefers_the_generating_system():
    # strongly model-based vs strongly model-free generators
    mb_theta = (0.5, 0.5, 0.5, 8.0, 8.0, 1.0, 0.0)
    mf_theta = (0.5, 0.5, 1.0, 8.0, 8.0, 0.0, 0.0)
    d_mb = simulate(mb_theta, 600, np.random.default_rng(10))
    d_mf = simulate(mf_theta, 600, np.random.default_rng(11))
    rng = np.random.default_rng(0)
    bic = {
        (g, m): fit_mle(d, n_starts=2, rng=rng, model=m)["bic"]
        for g, d in (("mb", d_mb), ("mf", d_mf))
        for m in ("mb_nop", "mf_nop")
    }
    assert bic["mb", "mb_nop"] < bic["mb", "mf_nop"]
    assert bic["mf", "mf_nop"] < bic["mf", "mb_nop"]


# ---- determinism ----------------------------------------------------------


def test_same_rng_seed_gives_identical_fit():
    d = simulate(THETA, 100, np.random.default_rng(6))
    a = fit_mle(d, n_starts=2, rng=np.random.default_rng(9))
    b = fit_mle(d, n_starts=2, rng=np.random.default_rng(9))
    assert np.array_equal(a["theta"], b["theta"]) and a["nll"] == b["nll"]
