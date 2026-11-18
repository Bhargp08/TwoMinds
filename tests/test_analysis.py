import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from twominds.analysis import stay_effects, stay_logit, stay_table

from twominds.agents import simulate

# Conventions under test
# - stay_table(data) -> {"stay": (2, 2) array, "n": (2, 2) array}, indexed
#   [previous reward (0/1), previous transition (0 = common, 1 = rare)].
#   `data` is one session dict or a list of sessions (pooled). Empty cells are NaN in "stay".
# - A trial t >= 1 contributes iff a1[t] and a1[t-1] are answered and the previous trial's
#   stage 2 was answered (so its reward is known). Missed trials are never silently counted.
# - stay_effects(data) -> {"reward": ..., "interaction": ...} in stay-probability units:
#   reward = P(stay | rewarded) - P(stay | unrewarded) (pooled over transitions),
#   interaction = (stay[1,c] - stay[1,r]) - (stay[0,c] - stay[0,r]).
# - stay_logit(data) -> {"coef": {...}, "se": {...}} with keys intercept, reward, transition,
#   interaction; logistic regression stay ~ reward * transition, both coded +/-1
#   (reward: +1 rewarded; transition: +1 common).


def make_session(a1, r, transition, a2=None):
    a1 = np.asarray(a1)
    common = np.asarray([t == "common" for t in transition])
    s2 = np.where(common, 1 + a1, 2 - a1)
    s2 = np.where(a1 < 0, 1, s2)
    return {
        "a1": a1,
        "s2": s2,
        "a2": np.zeros(len(a1), dtype=int) if a2 is None else np.asarray(a2),
        "r": np.asarray(r),
        "transition": np.asarray(transition),
    }


def pair_sessions(cells):
    """One 2-trial session per observation -> exactly one stay observation each.

    cells: {(prev_reward, "common"/"rare"): (n_stay, n_switch)}
    """
    sessions = []
    for (rew, trans), (n_stay, n_switch) in cells.items():
        for k in range(n_stay + n_switch):
            second = 0 if k < n_stay else 1
            sessions.append(make_session([0, second], [rew, 0], [trans, "common"]))
    return sessions


CELLS = {
    (1, "common"): (8, 2),
    (1, "rare"): (4, 6),
    (0, "common"): (6, 4),
    (0, "rare"): (5, 5),
}

HAND = make_session(
    a1=[0, 0, 1, 1, 1, 0],
    r=[1, 0, 1, 1, 0, 0],
    transition=["common", "common", "rare", "common", "rare", "common"],
)
# Hand-computed stay observations (trial t >= 1: previous reward, previous transition, stay):
#  t=1: (1, common) stay      t=2: (0, common) switch   t=3: (1, rare) stay
#  t=4: (1, common) stay      t=5: (0, rare) switch


# ---- stay_table -----------------------------------------------------------


def test_stay_table_hand_computed():
    out = stay_table(HAND)
    assert np.array_equal(out["n"], [[1, 1], [2, 1]])
    assert np.allclose(out["stay"], [[0.0, 0.0], [1.0, 1.0]])


def test_stay_table_empty_cell_is_nan():
    sess = make_session([0, 0], [1, 0], ["common", "common"])
    out = stay_table(sess)
    assert out["n"][1, 0] == 1 and out["stay"][1, 0] == 1.0
    assert np.isnan(out["stay"][0, 1]) and out["n"][0, 1] == 0


def test_stay_table_pools_a_list_of_sessions():
    out = stay_table(pair_sessions(CELLS))
    assert np.array_equal(out["n"], [[10, 10], [10, 10]])
    assert np.allclose(out["stay"], [[0.6, 0.5], [0.8, 0.4]])


def test_stay_table_does_not_pair_trials_across_sessions():
    a = make_session([0, 1], [1, 1], ["common", "common"])
    b = make_session([0, 0], [0, 0], ["rare", "rare"])
    out = stay_table([a, b])
    assert out["n"].sum() == 2  # one pair per session, not 3


def test_stay_table_excludes_trials_around_a_missed_stage1():
    sess = {k: v.copy() for k, v in HAND.items()}
    sess["a1"][2] = -1  # drops t=2 (no choice) and t=3 (previous choice missing)
    out = stay_table(sess)
    # remaining: t=1 (1,c) stay, t=4 (1,c) stay, t=5 (0,rare) switch
    assert np.array_equal(out["n"], [[0, 1], [2, 0]])
    assert np.allclose(out["stay"][1, 0], 1.0) and np.allclose(out["stay"][0, 1], 0.0)


def test_stay_table_excludes_trial_after_a_missed_stage2():
    sess = {k: v.copy() for k, v in HAND.items()}
    sess["a2"][0] = -1  # reward on trial 0 is unknown -> t=1 excluded
    out = stay_table(sess)
    assert out["n"].sum() == 4
    assert np.array_equal(out["n"], [[1, 1], [1, 1]])


# ---- stay_effects ---------------------------------------------------------


def test_stay_effects_definitions_on_known_cells():
    eff = stay_effects(pair_sessions(CELLS))
    assert eff["reward"] == pytest.approx(0.05)  # (.8+.4)/2 - (.6+.5)/2
    assert eff["interaction"] == pytest.approx(0.30)  # (.8-.4) - (.6-.5)


# ---- stay_logit -----------------------------------------------------------


def test_stay_logit_matches_closed_form_for_saturated_model():
    out = stay_logit(pair_sessions(CELLS))["coef"]
    logit = {k: np.log(s / w) for k, (s, w) in CELLS.items()}
    x = {k: (1 if k[0] == 1 else -1, 1 if k[1] == "common" else -1) for k in CELLS}
    exp_int = np.mean(list(logit.values()))
    exp_rew = np.mean([x[k][0] * logit[k] for k in CELLS])
    exp_tr = np.mean([x[k][1] * logit[k] for k in CELLS])
    exp_ia = np.mean([x[k][0] * x[k][1] * logit[k] for k in CELLS])
    assert out["intercept"] == pytest.approx(exp_int, abs=1e-4)
    assert out["reward"] == pytest.approx(exp_rew, abs=1e-4)
    assert out["transition"] == pytest.approx(exp_tr, abs=1e-4)
    assert out["interaction"] == pytest.approx(exp_ia, abs=1e-4)


def test_stay_logit_returns_standard_errors():
    se = stay_logit(pair_sessions(CELLS))["se"]
    assert set(se) == {"intercept", "reward", "transition", "interaction"}
    assert all(v > 0 for v in se.values())


# ---- agent behaviour (spec reference sims: MF 0.34/0.02, MB 0.00/0.44, hybrid 0.23/0.21) ----

N_AGENTS, N_TRIALS = 100, 200


def simulate_agents(w, seed=0):
    # theta = (alpha1, alpha2, lam, beta1, beta2, w, p)
    theta = (0.5, 0.5, 1.0, 5.0, 5.0, w, 0.0)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(N_AGENTS)]
    return [simulate(theta, N_TRIALS, rng) for rng in rngs]


@pytest.fixture(scope="module")
def mf():
    return simulate_agents(0.0)


@pytest.fixture(scope="module")
def mb():
    return simulate_agents(1.0)


@pytest.fixture(scope="module")
def hybrid():
    return simulate_agents(0.5)


def test_model_free_agents_show_reward_effect_but_no_interaction(mf):
    eff = stay_effects(mf)
    assert eff["reward"] > 0.1
    assert abs(eff["interaction"]) < 0.05


def test_model_based_agents_show_interaction(mb):
    eff = stay_effects(mb)
    assert eff["interaction"] > 0.1
    tab = stay_table(mb)["stay"]
    assert tab[1, 1] < tab[1, 0]  # rewarded-rare stay < rewarded-common stay


def test_hybrid_agents_show_both_effects(hybrid):
    eff = stay_effects(hybrid)
    assert eff["reward"] > 0 and eff["interaction"] > 0


def test_logit_coefficients_have_expected_signs(mf, mb):
    assert stay_logit(mf)["coef"]["reward"] > 0.3
    assert abs(stay_logit(mf)["coef"]["interaction"]) < 0.15
    assert stay_logit(mb)["coef"]["interaction"] > 0.3


# ---- plotting script ------------------------------------------------------


def test_plot_stay_script_writes_figure(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_stay.py"
    subprocess.run(
        [sys.executable, str(script), "--out", str(tmp_path), "--n-agents", "5"],
        check=True,
        timeout=120,
    )
    pngs = list(Path(tmp_path).glob("*.png"))
    assert len(pngs) == 1 and pngs[0].stat().st_size > 1000
