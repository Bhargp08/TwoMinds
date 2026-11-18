"""TM-08 hierarchical Bayesian model (PyMC).

Conventions under test
- datasets: list over participants; each entry is {"no_pressure": data, "time_pressure": data}
  (io.load_session output) or a (c0, c1) pair, with data in the agents.simulate format.
- stack_datasets(datasets) -> dict of (T, N, 2) int arrays a1, s2, a2, r; shorter sessions are
  padded at the end with missed trials (a1 = a2 = -1).
- batch_loglik(theta, datasets, per_trial=False) -> (N, 2) totals ((N, 2, T) if per_trial),
  theta (N, 2, 7), computed by the same pytensor scan graph the PyMC model uses.
- build_model(datasets, model="hybrid"|"mf"|"mb", *, beta1_by_condition=False) -> pm.Model.
- fit_hier(datasets, model, quick, *, draws, tune, chains, cores, seed, target_accept,
  beta1_by_condition, trial_loglik) -> InferenceData with log_likelihood["y"]
  (chain, draw, participant, condition); "delta_w" in posterior for hybrid only.
- simulate_hier_dataset(n_participants, n_trials, delta_w, seed, ...) -> {"datasets", "truth"}.
"""

import arviz as az
import numpy as np
import pytest

from twominds import hier
from twominds.agents import loglik, simulate
from twominds.recovery import sample_theta

CONDS = hier.CONDITIONS


def _with_misses(data, rng, n_miss1=4, n_miss2=4):
    d = {k: np.array(v).copy() for k, v in data.items()}
    n = len(d["a1"])
    idx = rng.choice(n, n_miss1 + n_miss2, replace=False)
    m1, m2 = idx[:n_miss1], idx[n_miss1:]
    d["a1"][m1] = -1
    d["a2"][m1] = -1
    d["r"][m1] = 0
    d["a2"][m2] = -1
    d["r"][m2] = 0
    return d


def _random_datasets(n_part=4, n_trials=60, seed=0, unequal=False):
    rng = np.random.default_rng(seed)
    theta = np.zeros((n_part, 2, 7))
    datasets = []
    for i in range(n_part):
        entry = {}
        for c, name in enumerate(CONDS):
            th = sample_theta(rng)
            if i == 0:
                th[5] = 0.0  # pure MF
            elif i == 1:
                th[5] = 1.0  # pure MB
            theta[i, c] = th
            n = n_trials - 7 * c * int(unequal)
            entry[name] = _with_misses(simulate(th, n, rng), rng)
        datasets.append(entry)
    return theta, datasets


def test_stack_datasets_shapes_padding_and_input_forms():
    _, ds = _random_datasets(n_part=3, n_trials=30, unequal=True)
    st = hier.stack_datasets(ds)
    assert st["a1"].shape == (30, 3, 2)
    # time_pressure sessions are 7 shorter: padded with missed trials at the end
    assert np.all(st["a1"][-7:, :, 1] == -1) and np.all(st["a2"][-7:, :, 1] == -1)
    as_pairs = [(e["no_pressure"], e["time_pressure"]) for e in ds]
    as_int_keys = [{0: e["no_pressure"], 1: e["time_pressure"]} for e in ds]
    for alt in (as_pairs, as_int_keys):
        st2 = hier.stack_datasets(alt)
        for k in ("a1", "s2", "a2", "r"):
            assert np.array_equal(st[k], st2[k])
    with pytest.raises(ValueError):
        hier.stack_datasets([{"no_pressure": ds[0]["no_pressure"]}])


@pytest.mark.parametrize("unequal", [False, True])
def test_batch_loglik_matches_agents_loglik(unequal):
    theta, ds = _random_datasets(n_part=4, n_trials=60, seed=3, unequal=unequal)
    ll = hier.batch_loglik(theta, ds)
    assert ll.shape == (4, 2)
    for i in range(4):
        for c, name in enumerate(CONDS):
            assert abs(ll[i, c] - loglik(theta[i, c], ds[i][name])) < 1e-6


def test_batch_loglik_per_trial_sums_and_masks():
    theta, ds = _random_datasets(n_part=2, n_trials=40, seed=5, unequal=True)
    pt_ll = hier.batch_loglik(theta, ds, per_trial=True)
    assert pt_ll.shape == (2, 2, 40)
    assert np.allclose(pt_ll.sum(-1), hier.batch_loglik(theta, ds), atol=1e-10)
    st = hier.stack_datasets(ds)
    missed1 = st["a1"].transpose(1, 2, 0) < 0
    assert np.all(pt_ll[missed1] == 0.0)
    assert np.all(pt_ll[~missed1] < 0.0)


def test_batch_loglik_all_missed_is_zero():
    _, ds = _random_datasets(n_part=1, n_trials=20, seed=1)
    for name in CONDS:
        ds[0][name]["a1"][:] = -1
        ds[0][name]["a2"][:] = -1
    theta = np.tile(sample_theta(np.random.default_rng(0)), (1, 2, 1))
    assert np.all(hier.batch_loglik(theta, ds) == 0.0)


@pytest.fixture(scope="module")
def tiny():
    return hier.simulate_hier_dataset(3, 40, delta_w=-0.8, seed=11)


def test_build_model_variables_per_variant(tiny):
    ds = tiny["datasets"]
    names = {m: {v.name for v in hier.build_model(ds, m).free_RVs} for m in hier.MODELS}
    w_vars = {"mu_w", "delta_w", "sigma_u", "sigma_v", "z_u", "z_v"}
    assert w_vars <= names["hybrid"]
    assert not (w_vars & names["mf"]) and not (w_vars & names["mb"])
    for m in hier.MODELS:  # non-centered hierarchy for every per-participant parameter
        for p in ("alpha1", "alpha2", "lam", "beta1", "beta2", "p"):
            assert {f"mu_{p}", f"sd_{p}", f"z_{p}"} <= names[m]
    with_b = {v.name for v in hier.build_model(ds, beta1_by_condition=True).free_RVs}
    assert "delta_beta1" in with_b and "delta_beta1" not in names["hybrid"]
    with pytest.raises(ValueError):
        hier.build_model(ds, "nope")


def _eval_at_random_point(model, seed):
    rng = np.random.default_rng(seed)
    point = model.initial_point()
    point = {k: v + rng.normal(0, 0.5, size=np.shape(v)) for k, v in point.items()}
    names = ["alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p", "ll_pc"]
    fn = model.compile_fn([model[n] for n in names], inputs=model.value_vars, point_fn=True)
    return point, dict(zip(names, fn(point), strict=True))


@pytest.mark.parametrize(
    "model,b1c", [("hybrid", False), ("hybrid", True), ("mf", False), ("mb", False)]
)
def test_model_transforms_and_loglik_wiring(tiny, model, b1c):
    ds = tiny["datasets"]
    m = hier.build_model(ds, model, beta1_by_condition=b1c)
    point, v = _eval_at_random_point(m, seed=2)
    n = len(ds)
    for p in ("alpha1", "alpha2", "lam"):
        assert v[p].shape == (n,) and np.all((v[p] > 0) & (v[p] < 1))
    assert v["beta2"].shape == (n,) and np.all(v["beta2"] > 0)
    assert v["beta1"].shape == ((n, 2) if b1c else (n,)) and np.all(v["beta1"] > 0)
    assert v["w"].shape == (n, 2)
    if model == "mf":
        assert np.all(v["w"] == 0.0)
    elif model == "mb":
        assert np.all(v["w"] == 1.0)
    else:
        c = np.array([0.0, 1.0])
        eta = (
            point["mu_w"]
            + point["delta_w"] * c[None, :]
            + np.exp(point["sigma_u_log__"]) * point["z_u"][:, None]
            + np.exp(point["sigma_v_log__"]) * point["z_v"][:, None] * c[None, :]
        )
        assert np.allclose(v["w"], 1 / (1 + np.exp(-eta)))
    if b1c:
        ratio = v["beta1"][:, 1] / v["beta1"][:, 0]
        assert np.allclose(ratio, np.exp(point["delta_beta1"]))
    for i in range(n):
        for c, name in enumerate(CONDS):
            b1 = v["beta1"][i, c] if b1c else v["beta1"][i]
            th = [v["alpha1"][i], v["alpha2"][i], v["lam"][i], b1, v["beta2"][i], v["w"][i, c],
                  v["p"][i]]
            assert abs(v["ll_pc"][i, c] - loglik(th, ds[i][name])) < 1e-6


def test_simulate_hier_dataset_determinism_and_truth():
    a = hier.simulate_hier_dataset(5, 30, delta_w=-0.8, seed=4)
    b = hier.simulate_hier_dataset(5, 30, delta_w=-0.8, seed=4)
    c = hier.simulate_hier_dataset(5, 30, delta_w=-0.8, seed=5)
    assert len(a["datasets"]) == 5
    for i in range(5):
        for name in CONDS:
            assert len(a["datasets"][i][name]["a1"]) == 30
            assert np.array_equal(a["datasets"][i][name]["a1"], b["datasets"][i][name]["a1"])
    assert np.array_equal(a["truth"]["theta"], b["truth"]["theta"])
    assert not np.array_equal(a["truth"]["theta"], c["truth"]["theta"])
    t = a["truth"]
    assert t["delta_w"] == -0.8 and t["theta"].shape == (5, 2, 7)
    eta = t["mu_w"] + t["delta_w"] * np.array([0, 1]) + t["u"][:, None] + t["v"][:, None] * [0, 1]
    assert np.allclose(t["theta"][:, :, 5], 1 / (1 + np.exp(-eta)))
    # shared-across-condition parameters are identical across conditions
    for k in (0, 1, 2, 3, 4, 6):
        assert np.array_equal(t["theta"][:, 0, k], t["theta"][:, 1, k])
    # each condition has its own reward schedule
    assert t["schedules"].shape == (5, 2, 30, 2, 2)
    assert not np.allclose(t["schedules"][:, 0], t["schedules"][:, 1])


@pytest.fixture(scope="module")
def smoke_idata(tiny):
    return hier.fit_hier(
        tiny["datasets"], "hybrid", True, draws=40, tune=40, chains=2, cores=1, seed=1,
        trial_loglik=True,
    )


def test_fit_hier_smoke_structure(smoke_idata):
    idata = smoke_idata
    post = idata.posterior
    for v in ("delta_w", "mu_w", "sigma_u", "sigma_v", "w", "alpha1", "beta1"):
        assert v in post
    assert post["delta_w"].shape == (2, 40)
    ll = idata.log_likelihood
    assert ll["y"].dims == ("chain", "draw", "participant", "condition")
    assert ll["y"].shape == (2, 40, 3, 2)
    assert ll["y_trial"].shape == (2, 40, 3, 2, 40)
    assert np.allclose(ll["y_trial"].sum("trial"), ll["y"], atol=1e-8)
    assert "ll_pc" not in post
    loo = az.loo(idata, var_name="y")
    assert np.isfinite(loo.elpd_loo)
    s = hier.summarize_delta_w(idata, truth=-0.8)
    assert {"mean", "hdi_low", "hdi_high", "p_neg", "covers_truth"} <= set(s)
    assert s["hdi_low"] <= s["mean"] <= s["hdi_high"]
    d = hier.diagnostics(idata)
    assert {"max_rhat", "min_ess_bulk", "min_ess_tail", "divergences"} <= set(d)


def test_fit_hier_mf_has_no_delta_w(tiny):
    idata = hier.fit_hier(tiny["datasets"], "mf", draws=10, tune=10, chains=1, cores=1, seed=0)
    assert "delta_w" not in idata.posterior
    assert idata.log_likelihood["y"].shape == (1, 10, 3, 2)


@pytest.mark.slow
def test_delta_w_recovery_gate(tmp_path):
    res = hier.hier_recovery(tmp_path, truths=(-0.8, 0.0), n_participants=40, n_trials=200)
    for run in res["runs"]:
        assert run["delta_w"]["covers_truth"]
        assert run["diagnostics"]["max_rhat"] < 1.01
        assert run["diagnostics"]["min_ess_bulk"] > 400
