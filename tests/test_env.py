import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from twominds.env import TwoStepEnv, make_reward_schedule


def run_session(env, seed, policy_seed=0):
    """Play a full session with a seeded random policy; return per-step records."""
    prng = np.random.default_rng(policy_seed)
    obs, info = env.reset(seed=seed)
    steps = []
    done = False
    while not done:
        a = int(prng.integers(2))
        obs_next, r, terminated, truncated, info = env.step(a)
        steps.append((obs, a, obs_next, r, info))
        obs = obs_next
        done = terminated or truncated
    return steps


# ---- make_reward_schedule -------------------------------------------------


def test_schedule_shape():
    sched = make_reward_schedule(200, np.random.default_rng(0))
    assert sched.shape == (200, 2, 2)


def test_schedule_deterministic_given_rng_seed():
    a = make_reward_schedule(200, np.random.default_rng(7))
    b = make_reward_schedule(200, np.random.default_rng(7))
    c = make_reward_schedule(200, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_schedule_stays_in_bounds_even_with_large_sd():
    sched = make_reward_schedule(5000, np.random.default_rng(1), sd=0.3)
    assert sched.min() >= 0.25 and sched.max() <= 0.75


def test_schedule_initial_values_uniform_in_bounds():
    firsts = np.array(
        [make_reward_schedule(1, np.random.default_rng(s))[0] for s in range(2000)]
    ).ravel()
    assert firsts.min() >= 0.25 and firsts.max() <= 0.75
    assert abs(firsts.mean() - 0.5) < 0.01
    # Uniform(0.25, 0.75) has sd 0.5/sqrt(12) ~ 0.144
    assert abs(firsts.std() - 0.5 / np.sqrt(12)) < 0.01


def test_schedule_step_sd_matches_spec():
    # Start mid-range and keep walks short so reflection is rare; step sd ~ 0.025.
    sched = make_reward_schedule(20000, np.random.default_rng(2))
    steps = np.diff(sched, axis=0)
    assert abs(steps.std() - 0.025) < 0.003


def test_schedule_four_walks_are_independent():
    sched = make_reward_schedule(20000, np.random.default_rng(3))
    steps = np.diff(sched, axis=0).reshape(-1, 4)
    corr = np.corrcoef(steps.T)
    off_diag = corr[~np.eye(4, dtype=bool)]
    assert np.abs(off_diag).max() < 0.05


# ---- TwoStepEnv: gymnasium contract --------------------------------------


def test_check_env_passes():
    check_env(TwoStepEnv(n_trials=10), skip_render_check=True)


def test_spaces():
    env = TwoStepEnv()
    assert env.action_space.n == 2
    assert env.observation_space.n == 3


def test_defaults_match_spec():
    env = TwoStepEnv()
    assert env.n_trials == 200
    assert env.p_common == 0.7
    assert env.sd == 0.025
    assert env.bounds == (0.25, 0.75)


# ---- TwoStepEnv: task structure ------------------------------------------


def test_two_steps_per_trial_and_termination():
    env = TwoStepEnv(n_trials=15)
    obs, _ = env.reset(seed=0)
    assert obs == 0
    n_steps, terminated = 0, False
    while not terminated:
        obs_next, _, terminated, truncated, _ = env.step(0)
        assert not truncated
        n_steps += 1
        # stage 1 -> second-stage state; stage 2 -> back to s0
        if n_steps % 2 == 1:
            assert obs_next in (1, 2)
        else:
            assert obs_next == 0
    assert n_steps == 30


def test_stage1_gives_no_reward():
    env = TwoStepEnv(n_trials=5)
    env.reset(seed=0)
    _, r, terminated, _, _ = env.step(1)
    assert r == 0 and not terminated


def test_transition_mapping_and_labels():
    env = TwoStepEnv(n_trials=5000)
    steps = run_session(env, seed=0)
    stage1 = steps[0::2]
    for a in (0, 1):
        rows = [s for s in stage1 if s[1] == a]
        common_state = 1 if a == 0 else 2
        for _, _, s2, _, info in rows:
            expected = "common" if s2 == common_state else "rare"
            assert info["transition"] == expected
        frac_common = np.mean([s[4]["transition"] == "common" for s in rows])
        assert abs(frac_common - 0.7) < 0.03


def test_p_common_is_respected():
    env = TwoStepEnv(n_trials=4000, p_common=0.9)
    stage1 = run_session(env, seed=1)[0::2]
    frac = np.mean([s[4]["transition"] == "common" for s in stage1])
    assert abs(frac - 0.9) < 0.03


def test_info_exposes_reward_probs_matching_schedule():
    env = TwoStepEnv(n_trials=20, drift_seed=5)
    steps = run_session(env, seed=0)
    for t in range(20):
        probs = np.asarray(steps[2 * t + 1][4]["reward_probs"])
        assert probs.shape == (2, 2)
        assert np.allclose(probs, env.schedule[t])


def test_rewards_are_bernoulli_of_schedule():
    env = TwoStepEnv(n_trials=20000, drift_seed=3)
    steps = run_session(env, seed=0)
    stage2 = steps[1::2]
    rewards = np.array([s[3] for s in stage2])
    assert set(np.unique(rewards)) <= {0.0, 1.0}
    expected = np.array(
        [env.schedule[t, s[0] - 1, s[1]] for t, s in enumerate(stage2)]
    )
    # Sum of Bernoulli(p_t): compare total to expected total within ~4 sd
    sd = np.sqrt((expected * (1 - expected)).sum())
    assert abs(rewards.sum() - expected.sum()) < 4 * sd


# ---- determinism and shared schedules ------------------------------------


def test_same_seed_same_session():
    a = run_session(TwoStepEnv(n_trials=50), seed=11)
    b = run_session(TwoStepEnv(n_trials=50), seed=11)
    assert [(s[0], s[1], s[2], s[3]) for s in a] == [(s[0], s[1], s[2], s[3]) for s in b]


def test_drift_seed_fixes_schedule_independent_of_env_seed():
    e1 = TwoStepEnv(n_trials=100, drift_seed=42)
    e2 = TwoStepEnv(n_trials=100, drift_seed=42)
    e1.reset(seed=1)
    e2.reset(seed=2)
    assert np.array_equal(e1.schedule, e2.schedule)
    assert np.array_equal(e1.schedule, make_reward_schedule(100, np.random.default_rng(42)))


def test_drift_seed_schedule_survives_reset():
    env = TwoStepEnv(n_trials=100, drift_seed=42)
    env.reset(seed=1)
    first = env.schedule.copy()
    env.reset(seed=2)
    assert np.array_equal(env.schedule, first)


def test_without_drift_seed_schedule_follows_env_seed():
    env = TwoStepEnv(n_trials=100)
    env.reset(seed=1)
    s1 = env.schedule.copy()
    env.reset(seed=1)
    assert np.array_equal(env.schedule, s1)
    env.reset(seed=2)
    assert not np.array_equal(env.schedule, s1)


def test_step_before_reset_raises():
    env = TwoStepEnv(n_trials=5)
    with pytest.raises(Exception):  # noqa: B017
        env.step(0)
