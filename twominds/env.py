"""Two-step task (Daw et al., 2011) as a Gymnasium environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fold values into [lo, hi] by reflecting at the bounds (handles multiple bounces)."""
    width = hi - lo
    y = np.mod(x - lo, 2 * width)
    return lo + np.where(y > width, 2 * width - y, y)


def make_reward_schedule(
    n_trials: int,
    rng: np.random.Generator,
    sd: float = 0.025,
    bounds: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """Reward probabilities per trial: four Gaussian random walks with reflecting bounds.

    Returns an array of shape (n_trials, 2, 2) indexed [trial, stage-2 state - 1, action].
    """
    lo, hi = bounds
    p = np.empty((n_trials, 2, 2))
    p[0] = rng.uniform(lo, hi, size=(2, 2))
    for t in range(1, n_trials):
        p[t] = _reflect(p[t - 1] + rng.normal(0.0, sd, size=(2, 2)), lo, hi)
    return p


class TwoStepEnv(gym.Env):
    """Two-step task. Observation is the state index: 0 (stage 1), 1 or 2 (stage 2).

    Each trial is two steps: a stage-1 action moves to state 1 or 2 (reward 0), then a
    stage-2 action pays reward ~ Bernoulli(p[trial, state - 1, action]) and returns to state 0.
    Action 0 commonly leads to state 1; action 1 commonly leads to state 2.

    If `drift_seed` is given, the reward schedule is fixed by it (shared across resets and
    env seeds); otherwise it is drawn from the env's RNG at each reset.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_trials: int = 200,
        p_common: float = 0.7,
        sd: float = 0.025,
        bounds: tuple[float, float] = (0.25, 0.75),
        seed: int | None = None,
        drift_seed: int | None = None,
    ):
        super().__init__()
        self.n_trials = n_trials
        self.p_common = p_common
        self.sd = sd
        self.bounds = bounds
        self.drift_seed = drift_seed
        self._seed = seed
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.schedule: np.ndarray | None = None
        if drift_seed is not None:
            self.schedule = make_reward_schedule(
                n_trials, np.random.default_rng(drift_seed), sd, bounds
            )
        self._started = False
        self._state = 0
        self._trial = 0
        self._transition: str | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and self._seed is not None and self._started is False:
            seed = self._seed
        super().reset(seed=seed)
        if self.drift_seed is None:
            self.schedule = make_reward_schedule(
                self.n_trials, self.np_random, self.sd, self.bounds
            )
        self._started = True
        self._state = 0
        self._trial = 0
        self._transition = None
        return 0, {}

    def step(self, action):
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        action = int(action)
        probs = self.schedule[self._trial]
        if self._state == 0:
            common = self.np_random.random() < self.p_common
            common_state = 1 + action
            self._state = common_state if common else 3 - common_state
            self._transition = "common" if common else "rare"
            info = {"transition": self._transition, "reward_probs": probs.copy()}
            return self._state, 0.0, False, False, info
        reward = float(self.np_random.random() < probs[self._state - 1, action])
        info = {"transition": self._transition, "reward_probs": probs.copy()}
        self._state = 0
        self._trial += 1
        terminated = self._trial >= self.n_trials
        return 0, reward, terminated, False, info
