"""Tests for the TM-02 hybrid agent (twominds/agents.py).

API under test:
    simulate(theta, n_trials, rng, schedule=None) -> dict with a1, s2, a2, r, transition,
        p_a1, p_a2 (probability of the chosen action on each trial)
    loglik(theta, data) -> float
    trace(theta, data) -> dict with per-trial values at the START of each trial (before the
        choice): q_mf (n, 2), q_mb (n, 2), q_net (n, 2), q2 (n, 2, 2) indexed [trial, s2-1, a2]

theta = (alpha1, alpha2, lam, beta1, beta2, w, p), as a tuple or array, in that order.

Data encoding (same as env.py): a1, a2 in {0, 1}; s2 in {1, 2}; r in {0, 1};
transition is "common" / "rare".

Missed trials: a1 == -1 means stage 1 was missed (the whole trial is missed; s2, a2 and r on
that trial are ignored). a1 >= 0 with a2 == -1 means stage 1 was answered but stage 2 was
missed (s2 and r on that trial are ignored).
"""

import numpy as np
import pytest

from twominds.agents import loglik, simulate, trace
from twominds.env import make_reward_schedule

LOWER = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
UPPER = np.array([1.0, 1.0, 1.0, 20.0, 20.0, 1.0, 2.0])

# Parameters for the hand-checked examples.
HAND_THETA = (0.5, 0.4, 0.6, 2.0, 3.0, 0.5, 0.3)  # alpha1, alpha2, lam, beta1, beta2, w, p


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def random_theta(rng):
    return rng.uniform(LOWER, UPPER)


def as_data(a1, s2, a2, r):
    return {
        "a1": np.array(a1, dtype=int),
        "s2": np.array(s2, dtype=int),
        "a2": np.array(a2, dtype=int),
        "r": np.array(r, dtype=float),
    }


# ---- simulate: basic contract --------------------------------------------


def test_simulate_keys_shapes_and_ranges():
    sim = simulate(HAND_THETA, 50, np.random.default_rng(0))
    for k in ("a1", "s2", "a2", "r", "transition", "p_a1", "p_a2"):
        assert len(sim[k]) == 50, k
    assert set(np.unique(sim["a1"])) <= {0, 1}
    assert set(np.unique(sim["a2"])) <= {0, 1}
    assert set(np.unique(sim["s2"])) <= {1, 2}
    assert set(np.unique(sim["r"])) <= {0, 1}
    # transition label is consistent with a1 -> s2 (action 0 commonly goes to state 1)
    common = np.asarray(sim["s2"]) == np.asarray(sim["a1"]) + 1
    assert np.array_equal(np.asarray(sim["transition"]) == "common", common)


def test_simulate_transition_rate_is_point_seven():
    sim = simulate(HAND_THETA, 4000, np.random.default_rng(1))
    assert abs(np.mean(np.asarray(sim["transition"]) == "common") - 0.7) < 0.03


def test_simulate_chosen_probs_strictly_between_0_and_1():
    rng = np.random.default_rng(2)
    for _ in range(5):
        sim = simulate(random_theta(rng), 100, rng)
        for k in ("p_a1", "p_a2"):
            p = np.asarray(sim[k])
            assert np.all(p > 0) and np.all(p < 1), k


def test_simulate_deterministic_given_rng_seed():
    a = simulate(HAND_THETA, 100, np.random.default_rng(7))
    b = simulate(HAND_THETA, 100, np.random.default_rng(7))
    c = simulate(HAND_THETA, 100, np.random.default_rng(8))
    for k in ("a1", "s2", "a2", "r", "p_a1", "p_a2"):
        assert np.array_equal(a[k], b[k]), k
    assert np.array_equal(a["transition"], b["transition"])
    assert any(not np.array_equal(a[k], c[k]) for k in ("a1", "s2", "a2", "r"))


def test_simulate_uses_shared_schedule():
    n = 200
    # State 1 always pays, state 2 never pays: r must equal (s2 == 1) exactly.
    sched = np.zeros((n, 2, 2))
    sched[:, 0, :] = 1.0
    sim = simulate(HAND_THETA, n, np.random.default_rng(3), schedule=sched)
    assert np.array_equal(np.asarray(sim["r"]), (np.asarray(sim["s2"]) == 1).astype(int))
    # Action 0 always pays, action 1 never pays: r must equal (a2 == 0).
    sched = np.zeros((n, 2, 2))
    sched[:, :, 0] = 1.0
    sim = simulate(HAND_THETA, n, np.random.default_rng(3), schedule=sched)
    assert np.array_equal(np.asarray(sim["r"]), (np.asarray(sim["a2"]) == 0).astype(int))


def test_simulate_accepts_env_schedule():
    sched = make_reward_schedule(100, np.random.default_rng(42))
    sim = simulate(HAND_THETA, 100, np.random.default_rng(0), schedule=sched)
    assert len(sim["r"]) == 100


# ---- 1. self-consistency -------------------------------------------------


def test_loglik_matches_simulated_probs_for_20_random_thetas():
    rng = np.random.default_rng(123)
    for _ in range(20):
        theta = random_theta(rng)
        sim = simulate(theta, 100, rng)
        expected = np.sum(np.log(sim["p_a1"]) + np.log(sim["p_a2"]))
        assert abs(loglik(theta, sim) - expected) < 1e-10


def test_loglik_accepts_tuple_or_array_theta():
    sim = simulate(HAND_THETA, 50, np.random.default_rng(4))
    assert loglik(HAND_THETA, sim) == loglik(np.array(HAND_THETA), sim)


# ---- 2. special cases ----------------------------------------------------


@pytest.mark.parametrize("w, key", [(0.0, "q_mf"), (1.0, "q_mb")])
def test_qnet_reduces_to_mf_or_mb(w, key):
    rng = np.random.default_rng(5)
    for _ in range(5):
        theta = random_theta(rng)
        theta[5] = w
        sim = simulate(theta, 100, rng)
        tr = trace(theta, sim)
        np.testing.assert_allclose(tr["q_net"], tr[key], rtol=0, atol=1e-12)


def test_trace_shapes_and_initial_values_are_zero():
    sim = simulate(HAND_THETA, 30, np.random.default_rng(6))
    tr = trace(HAND_THETA, sim)
    assert tr["q_mf"].shape == (30, 2)
    assert tr["q_mb"].shape == (30, 2)
    assert tr["q_net"].shape == (30, 2)
    assert tr["q2"].shape == (30, 2, 2)
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        assert np.all(tr[k][0] == 0), k


# ---- 3. hand-checked 3-trial example -------------------------------------
#
# theta: alpha1=0.5, alpha2=0.4, lam=0.6, beta1=2, beta2=3, w=0.5, p=0.3
# Trials (a1, s2, a2, r):
#   t1: (0, 1, 0, 1)   common
#   t2: (0, 2, 1, 1)   rare, repeats a1=0 -> perseveration bonus on action 0
#   t3: (1, 2, 0, 1)   common, switches a1 (bonus is still on action 0, the previous choice)
#
# Start of t1: everything 0. No perseveration on trial 1.
#   P(a1=0) = 0.5, P(a2=0 | s2=1) = 0.5
#   update: d1 = Q2[1,0] - QMF[0] = 0 - 0 = 0           -> QMF[0] = 0
#           d2 = r - Q2[1,0] = 1 - 0 = 1                -> Q2[1,0] = 0 + 0.4*1 = 0.4
#           QMF[0] += alpha1*lam*d2 = 0.5*0.6*1 = 0.3   -> QMF[0] = 0.3
# Start of t2: QMF = [0.3, 0], Q2 = [[0.4, 0], [0, 0]]  (rows = s2 1, 2)
#   max_a Q2 = [0.4, 0]
#   QMB[0] = 0.7*0.4 + 0.3*0 = 0.28,  QMB[1] = 0.3*0.4 + 0.7*0 = 0.12
#   QNET = 0.5*[0.28, 0.12] + 0.5*[0.3, 0] = [0.29, 0.06]
#   logits1 = 2*([0.29, 0.06] + 0.3*[1, 0]) = [1.18, 0.12] -> P(a1=0) = sigmoid(1.06)
#   s2=2: Q2[2,:] = [0, 0] -> P(a2=1) = 0.5
#   update: d1 = Q2[2,1] - QMF[0] = 0 - 0.3 = -0.3      -> QMF[0] = 0.3 + 0.5*(-0.3) = 0.15
#           d2 = 1 - Q2[2,1] = 1                        -> Q2[2,1] = 0.4
#           QMF[0] += 0.5*0.6*1 = 0.3                   -> QMF[0] = 0.45
# Start of t3: QMF = [0.45, 0], Q2 = [[0.4, 0], [0, 0.4]]
#   max_a Q2 = [0.4, 0.4] -> QMB = [0.4, 0.4]
#   QNET = 0.5*[0.4, 0.4] + 0.5*[0.45, 0] = [0.425, 0.2]
#   logits1 = 2*([0.425, 0.2] + 0.3*[1, 0]) = [1.45, 0.4] -> P(a1=1) = sigmoid(0.4 - 1.45)
#                                                                    = sigmoid(-1.05)
#   s2=2: logits2 = 3*[0, 0.4] = [0, 1.2] -> P(a2=0) = sigmoid(-1.2)
#
# loglik = log .5 + log .5 + log sigmoid(1.06) + log .5 + log sigmoid(-1.05) + log sigmoid(-1.2)
#        = -5.190258303193501


HAND_DATA = as_data(a1=[0, 0, 1], s2=[1, 2, 2], a2=[0, 1, 0], r=[1, 1, 1])


def test_hand_trace():
    tr = trace(HAND_THETA, HAND_DATA)
    np.testing.assert_allclose(tr["q_mf"], [[0, 0], [0.3, 0], [0.45, 0]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        tr["q_mb"], [[0, 0], [0.28, 0.12], [0.4, 0.4]], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        tr["q_net"], [[0, 0], [0.29, 0.06], [0.425, 0.2]], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        tr["q2"],
        [
            [[0, 0], [0, 0]],
            [[0.4, 0], [0, 0]],
            [[0.4, 0], [0, 0.4]],
        ],
        rtol=0,
        atol=1e-12,
    )


def test_hand_loglik():
    expected = (
        np.log(0.5)
        + np.log(0.5)
        + np.log(sigmoid(1.06))
        + np.log(0.5)
        + np.log(sigmoid(-1.05))
        + np.log(sigmoid(-1.2))
    )
    assert abs(expected - (-5.190258303193501)) < 1e-12
    assert abs(loglik(HAND_THETA, HAND_DATA) - expected) < 1e-12


# ---- 4. argmax sanity ----------------------------------------------------


def test_true_theta_beats_random_theta_on_average():
    theta_star = np.array([0.5, 0.5, 0.6, 5.0, 5.0, 0.5, 0.2])
    sims = [simulate(theta_star, 200, np.random.default_rng(s)) for s in range(10)]
    ll_star = np.mean([loglik(theta_star, d) for d in sims])
    rng = np.random.default_rng(99)
    for _ in range(5):
        theta = random_theta(rng)
        ll_rand = np.mean([loglik(theta, d) for d in sims])
        assert ll_star > ll_rand


# ---- 5. missed trials ----------------------------------------------------


def test_all_missed_session_has_zero_loglik():
    n = 20
    data = as_data(a1=[-1] * n, s2=[1] * n, a2=[-1] * n, r=[0] * n)
    assert loglik(HAND_THETA, data) == 0.0
    tr = trace(HAND_THETA, data)
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        assert np.all(tr[k] == 0), k


def test_missed_stage1_no_update_and_resets_perseveration():
    # t1 as in the hand example, t2 stage-1 missed (s2, a2, r are junk and must be ignored),
    # t3 = (0, 1, 0, 1).
    data = as_data(a1=[0, -1, 0], s2=[1, 2, 1], a2=[0, 1, 0], r=[1, 1, 1])
    tr = trace(HAND_THETA, data)
    # No update on t2: start of t3 == start of t2 == state after t1.
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        np.testing.assert_array_equal(tr[k][2], tr[k][1], err_msg=k)
    np.testing.assert_allclose(tr["q_mf"][2], [0.3, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(tr["q_net"][2], [0.29, 0.06], rtol=0, atol=1e-12)
    # t3: QNET = [0.29, 0.06], NO perseveration bonus (rep was reset by the miss):
    #   logits1 = 2*[0.29, 0.06] = [0.58, 0.12] -> P(a1=0) = sigmoid(0.46)
    #   (with the bonus it would have been sigmoid(1.06))
    #   s2=1: logits2 = 3*[0.4, 0] = [1.2, 0] -> P(a2=0) = sigmoid(1.2)
    expected = 2 * np.log(0.5) + np.log(sigmoid(0.46)) + np.log(sigmoid(1.2))
    assert abs(loglik(HAND_THETA, data) - expected) < 1e-12


def test_missed_stage2_counts_stage1_but_no_update():
    # t1 as in the hand example, t2 = a1=0 answered, stage 2 missed (s2, r junk), t3 = (0,1,0,1).
    # Reading of the spec: only a missed STAGE-1 response resets rep. On t2 the stage-1 choice
    # (a1=0) was made, so it is "the previous trial's stage-1 choice" for t3 and the
    # perseveration bonus carries over to t3.
    data = as_data(a1=[0, 0, 0], s2=[1, 2, 1], a2=[0, -1, 0], r=[1, 1, 1])
    tr = trace(HAND_THETA, data)
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        np.testing.assert_array_equal(tr[k][2], tr[k][1], err_msg=k)
    # t2: logits1 = 2*([0.29, 0.06] + 0.3*[1, 0]) = [1.18, 0.12] -> log sigmoid(1.06) counted,
    #     no stage-2 term.
    # t3: same values, rep from t2's a1=0 -> P(a1=0) = sigmoid(1.06); P(a2=0 | s2=1) = sigmoid(1.2)
    expected = (
        2 * np.log(0.5) + np.log(sigmoid(1.06)) + np.log(sigmoid(1.06)) + np.log(sigmoid(1.2))
    )
    assert abs(loglik(HAND_THETA, data) - expected) < 1e-12


def test_masking_a_trial_equals_dropping_its_update_in_a_long_session():
    # Masking trial k (a1 = -1) must leave values at trial k+1 equal to values at trial k.
    sim = simulate(HAND_THETA, 60, np.random.default_rng(10))
    data = {k: np.array(sim[k]) for k in ("a1", "s2", "a2", "r")}
    data["a1"][30] = -1
    tr = trace(HAND_THETA, data)
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        np.testing.assert_array_equal(tr[k][31], tr[k][30], err_msg=k)
    # Trials before the mask are unaffected.
    tr_full = trace(HAND_THETA, sim)
    for k in ("q_mf", "q_mb", "q_net", "q2"):
        np.testing.assert_array_equal(tr[k][:31], tr_full[k][:31], err_msg=k)


# ---- extra cheap checks --------------------------------------------------


def test_no_perseveration_on_trial_1():
    theta = (0.5, 0.5, 0.5, 20.0, 5.0, 0.5, 2.0)  # strong stickiness
    for seed in range(5):
        sim = simulate(theta, 1, np.random.default_rng(seed))
        assert sim["p_a1"][0] == 0.5
        assert sim["p_a2"][0] == 0.5
    data = as_data(a1=[1], s2=[2], a2=[0], r=[1])
    assert abs(loglik(theta, data) - 2 * np.log(0.5)) < 1e-12


def test_log_softmax_is_stable_for_large_beta_and_large_q():
    # r = 100 is outside the task's {0, 1} but drives Q to ~100 so beta*Q ~ 2000, which
    # overflows a naive exp. The agent then chooses the low-value options.
    n = 40
    theta = (1.0, 1.0, 1.0, 20.0, 20.0, 0.5, 2.0)
    a1 = [0] * (n // 2) + [1] * (n // 2)
    s2 = [1] * (n // 2) + [2] * (n // 2)
    a2 = [0] * (n // 2) + [1] * (n // 2)
    r = [100.0] * (n // 2) + [0.0] * (n // 2)
    # Underflow of exp(very negative) to 0 is fine; overflow / nan / log(0) are not.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        ll = loglik(theta, as_data(a1, s2, a2, r))
    assert np.isfinite(ll)
    assert ll < -1000  # the switch trials are extremely unlikely, but finite
