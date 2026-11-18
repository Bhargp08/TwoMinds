# TwoMinds Build Spec (for Claude Code)

Project: measure model-based (planning) vs model-free (habit) learning with the Daw et al. (2011) two-step task, fit RL agents to each person's choices, and test whether time pressure lowers the model-based weight `w`.

This is a student project with volunteer participants, not IRB-approved research.

## Ground rules (every unit)

- One unit = one branch `tm-XX-short-name` = one PR titled `TM-XX: ...`.
- **Tests first.** For any new behavior, write the failing tests, commit them, push, and STOP. Report the test list so Bharg can approve it. Implement only after he approves. Scaffold-only and docs-only units skip the pause.
- CI must be green: `uv run ruff check .` and `uv run pytest -q`. Python 3.10, managed with uv. Commit `uv.lock`.
- Keep it lean: no extra abstractions, no frameworks beyond the stack, no notebooks inside the package.
- Always seed RNGs (`np.random.default_rng(seed)`). Tests must be deterministic and run in under 60 s total, except tests marked `@pytest.mark.slow`, which CI skips by default.
- Never commit human data. `data/human/` and `data/raw/` are gitignored.
- Model: Sonnet implements. Use Opus for TM-02 (likelihood) and TM-08 (hierarchical model), and to check any math you are unsure of.
- End every unit with a report: PR link, CI status, tests added, any deviation from this spec, and open questions.

## Repo layout (target)

```
twominds/
  env.py          # TM-01 Gymnasium two-step env + reward drift
  agents.py       # TM-02 hybrid agent: simulate + log-likelihood (MF/MB are special cases)
  analysis.py     # TM-03 stay probabilities, logistic regression
  fit_mle.py      # TM-04 SciPy MLE, multi-start
  recovery.py     # TM-05 parameter recovery
  io.py           # TM-07 human data loader
  hier.py         # TM-08 PyMC hierarchical model
  compare.py      # TM-09 LOO/ELPD + time-pressure effect
experiment/       # TM-06 jsPsych task (Vite), adapted from Bhargp08/MindBench
scripts/          # one-command runners
docs/             # BUILD_SPEC.md, REVIEW_SPEC.md, PLAN.md, writeup
tests/
```

## The task (TM-01)

- Stage 1: state `s0`, actions `a1 ∈ {0,1}`.
- Transition: action 0 goes to state 1 with p = 0.7 (common), to state 2 with p = 0.3 (rare). Action 1 is the mirror (state 2 common).
- Stage 2: state `s2 ∈ {1,2}`, actions `a2 ∈ {0,1}`. Reward `r ∈ {0,1}` ~ Bernoulli(`p[s2, a2]`).
- Reward probabilities: four independent Gaussian random walks, sd 0.025 per trial, reflecting bounds [0.25, 0.75], initialized Uniform(0.25, 0.75). Expose a `drift_seed` so a human session and a simulation can share the same reward schedule.
- Default 200 trials per session.
- Gymnasium env: `TwoStepEnv(n_trials=200, p_common=0.7, sd=0.025, bounds=(0.25,0.75), seed=None)`. Observation = current state index (0, 1, 2). Two steps per trial. `info` includes `transition` ("common"/"rare") and the current reward probabilities. Must pass `gymnasium.utils.env_checker.check_env`.
- Pure-numpy helpers too: `make_reward_schedule(n_trials, rng)` → array (n_trials, 2, 2).

## The model (TM-02). Opus writes this.

One hybrid agent. Model-free = `w = 0`, model-based = `w = 1`.

Parameters `θ = (alpha1, alpha2, lam, beta1, beta2, w, p)`.

Values (all start at 0; `Q2` is shared by the MF and MB systems):

- `Q_MF(a1)`: stage-1 model-free values (2).
- `Q2(s2, a2)`: stage-2 values (2×2).
- `Q_MB(a1) = Σ_s T(s | a1) · max_a Q2(s, a)`, with `T` fixed at the true 0.7/0.3 (the agent knows the structure).
- `Q_net(a1) = w · Q_MB(a1) + (1 − w) · Q_MF(a1)`.

Choice:

- `P(a1) = softmax(beta1 · [Q_net(a1) + p · rep(a1)])`, where `rep(a1) = 1` if `a1` equals the previous trial's stage-1 choice, else 0. No perseveration on trial 1.
- `P(a2 | s2) = softmax(beta2 · Q2(s2, ·))`.

Learning, after each trial:

- `δ1 = Q2(s2, a2) − Q_MF(a1)`, `Q_MF(a1) += alpha1 · δ1`
- `δ2 = r − Q2(s2, a2)`, `Q2(s2, a2) += alpha2 · δ2`
- `Q_MF(a1) += alpha1 · lam · δ2`

Unchosen values do not decay.

Missed trials (no response before the deadline): the trial contributes nothing to the likelihood and causes no value update. A missed stage-1 response also resets `rep`, so the next trial has no perseveration bonus. If stage 1 was answered but stage 2 was missed, count the stage-1 log-prob but do no updates.

Bounds for fitting: `alpha1, alpha2, lam, w ∈ [0,1]`; `beta1, beta2 ∈ [0, 20]`; `p ∈ [−2, 2]`.

API:

```python
simulate(theta, n_trials, rng, schedule=None) -> dict of arrays
    # a1, s2, a2, r, transition, plus p_a1 and p_a2 (prob of the chosen action) per trial
loglik(theta, data) -> float   # sum of log P(a1) + log P(a2); vectorize later only if needed
```

Use a numerically stable log-softmax (subtract the max).

Required tests:

1. **Self-consistency:** `loglik(theta, sim)` equals `sum(log p_a1 + log p_a2)` recorded during `simulate` to 1e-10, for 20 random θ.
2. **Special cases:** with `w = 0`, `Q_net == Q_MF` every trial; with `w = 1`, `Q_net == Q_MB`.
3. **Hand-checked trace:** a 3-trial hand-computed example (write the numbers in the test) matches to 1e-12.
4. **Argmax sanity:** for data simulated at θ*, the mean `loglik(θ*)` over seeds beats `loglik` at a random θ.
5. **Missed trials:** an all-missed session gives `loglik == 0`; masking one trial changes the update sequence exactly as specified.

## Stay probabilities (TM-03)

- `stay_table(data)` returns P(repeat a1) split by previous reward (0/1) × previous transition (common/rare).
- Logistic regression `stay ~ reward * transition`, coded ±1. Fixed effects across simulated agents are fine; use statsmodels or a small SciPy MLE (statsmodels is OK to add).
- Definitions in stay-prob units: reward effect = mean stay after reward − mean stay after no reward; interaction = (stay[rew,common] − stay[rew,rare]) − (stay[unrew,common] − stay[unrew,rare]).
- Tests, with 100 agents × 200 trials, fixed seed (a reference simulation gave MF reward 0.34 / interaction 0.02, MB reward 0.00 / interaction 0.44, hybrid 0.23 / 0.21):
  - Pure MF (w=0, beta1=beta2=5, alpha1=alpha2=0.5, lam=1, p=0): reward main effect > 0.1 in stay prob, reward × transition interaction near 0 (|interaction| < 0.05 in stay prob units).
  - Pure MB (w=1, same other params): reward × transition interaction > 0.1, and stay after rewarded-rare < stay after rewarded-common.
  - Hybrid (w=0.5): both effects positive.
- `scripts/plot_stay.py` produces the classic 3-panel bar figure (MF, MB, hybrid) into `docs/figures/`.

## MLE (TM-04)

- `fit_mle(data, n_starts=20, rng)`: L-BFGS-B on the negative log-likelihood within the bounds above, random starts, keep the best. Return params, NLL, BIC, AIC, a convergence flag.
- Also fit the reduced models: MF (w fixed 0), MB (w fixed 1), with and without `p`.
- Tests: on a 2000-trial simulated session at a known θ, recovered `w`, `beta1` and `alpha2` are within ±0.15 (β within ±1.5). Fitted NLL ≤ NLL at the true θ + 1e-6.

## Parameter recovery (TM-05). Gate before any human data.

- 100 synthetic participants, 200 trials each (same as the human task). θ drawn from plausible priors: alpha ~ Beta(2,2), lam ~ Beta(2,2), beta ~ Gamma(shape 4, scale 1.5) clipped to [2, 15] (a realistic human range; see the deviation note below), w ~ Beta(2,2), p ~ Normal(0.15, 0.2).
- **Deviation (agreed after the TM-05 diagnosis):** the original prior was beta ~ Gamma(shape 3, scale 1.5) clipped to [0.5, 15]. It is kept as an option (`prior="wide"`) and reported alongside the default in `docs/recovery/`, but near-zero betas carry no information about w (r(w) = 0.57 at 200 trials), so the default is now Gamma(4, 1.5) clipped to [2, 15].
- Fit each with `fit_mle`. Report the Pearson r (true vs fitted) for every parameter, a scatter grid, and the parameter-parameter correlation matrix of the fitted values (to show trade-offs, e.g. beta vs alpha).
- **Target (revised):** r > 0.7 for `beta1`, `beta2`, `alpha2`. `w` is reported with a sanity floor of r > 0.6 at 200 trials, not gated per person: it is information-limited (SE ≈ 0.22 at 200 trials) and the study question is the group-level `delta_w`, which TM-08's simulated recovery gates. `lam` and `alpha1` are reported, not gated. Rationale in `docs/recovery/README.md`.
- Also run 400 trials (two sessions) to show how trial count affects recovery of `w`, which matters for the time-pressure design.
- Model recovery: simulate from MF, MB and hybrid, fit all three, and report a BIC confusion matrix.
- Output: `docs/recovery/` with figures plus `recovery_summary.json`. A slow test (`@pytest.mark.slow`) runs the full sweep; a fast test runs 10 participants and checks the pipeline end to end.
- If the gate fails, stop and report. Do not move to human data. (The binding gate before any human fits is TM-08's simulated `delta_w` recovery.)

## jsPsych task (TM-06)

- Reuse MindBench conventions: jsPsych 8, Vite, DataPipe via `@jspsych/extension-pipe`, mock-save mode when no DataPipe ID is set, anonymous participant codes (random, e.g. `TM-7F3K`), no names or identifying data.
- Flow: consent screen (student project, not IRB research, voluntary, anonymous, can quit at any time and ask for deletion by code) → instructions with a practice block that teaches the transition structure (spaceship/planet or rocket/alien cover story) → a comprehension quiz that must be passed → two blocks of 200 trials each, one **No pressure** and one **Time pressure**, order counterbalanced by participant code → debrief.
- Deadlines: No pressure = 3000 ms per stage. Time pressure = 1000 ms for stage 1 and 1500 ms for stage 2. Make these config constants because the pilot will tune them. Missed responses show "Too slow" and are recorded as missed (`a1 = null`).
- Reward schedules come from fixed seeds per block, generated by the same algorithm as `env.py`, and saved with the data.
- Per-trial data: participant code, block, condition, trial index, a1, s2, a2, r, rt1, rt2, transition, reward probs, missed flags.
- Tests: Vitest unit tests for the schedule generator (it must match the Python generator for the same seed, via a shared JSON fixture) and the counterbalancing; one headless smoke run of a short session in mock-save mode.
- Note: 200 trials per condition (revised from 150) matches the recovery setting; TM-05 showed w is information-limited per person at this length, so the group-level analysis (TM-08) carries the inference.

## Data loader (TM-07)

- `io.load_session(path)` turns a DataPipe/mock-save CSV into the `data` dict that `loglik` takes, one per condition. Missed trials are masked as specified in TM-02.
- Exclusion rules (documented in the README, applied in code): more than 20% missed trials in a block, or stage-2 choice at chance (does not pick the higher-reward option more than 55% of the time) are flagged. Report them; do not silently drop.
- Test against a fixture made by the TM-06 smoke run.

## Hierarchical Bayesian model (TM-08). Opus writes this.

- PyMC + `pytensor.scan` likelihood that reproduces `loglik` (test: match to 1e-6 at fixed θ).
- Parameters go through transforms: probit or logit for [0,1] parameters, softplus/log for β, identity for `p`.
- Per participant `i` and condition `c ∈ {0 = no pressure, 1 = pressure}`:
  `w_ic = invlogit(mu_w + delta_w · c + u_i + v_i · c)`, with `u_i, v_i ~ Normal(0, sigma)` (non-centered). The other parameters are shared across conditions per participant (start simple; a later option is to let `beta1` vary by condition too, since time pressure may add noise rather than lower planning, which is a confound to report).
- Weakly informative priors; document them in the docstring.
- Sampling: NUTS, 4 chains, check R-hat < 1.01, ESS > 400, and no divergences (or explain them).
- **Recovery first:** run it on simulated data with a known `delta_w` (e.g. −0.8 on the logit scale) and on data with `delta_w = 0`, and show the 95% HDI covers the truth. This is the **binding gate before any human fits**: the 95% HDI must cover the truth in both cases.
- Mac note: Bharg has an Intel MacBook. Keep a small `--quick` mode (fewer draws) for local runs.

## Model comparison and the answer (TM-09)

- Compare hybrid vs MF vs MB (each hierarchical) with `arviz.loo` / `az.compare`, ELPD and SE.
- Main result: the posterior for `delta_w` (and the implied change in `w` at the group mean), with the 95% HDI and P(delta_w < 0). Also report the effect on `beta1`, and the stay-probability logistic regression per condition as a model-free check.
- Posterior predictive check: simulate from posterior draws and overlay stay probabilities on the human ones.

## Pilot and write-up (TM-10)

- `scripts/run_all.py` reruns everything from raw data to figures with one command.
- `docs/writeup.md`: question, task, models, recovery appendix (TM-05 figures plus TM-08 simulated recovery), human results with the HDI, limitations (small volunteer sample, no IRB, time pressure vs noise confound, trial count).
- Deliverables: task link (GitHub Pages or Vercel), the `twominds` library, the write-up.

## Stretch (only after TM-10)

- Successor representation agent.
- Meta-RL LSTM agent (Stable-Baselines3 / sb3-contrib RecurrentPPO) trained on `TwoStepEnv`, with stay probabilities compared to humans.
- Steinmetz et al. 2019 mouse data: look for value-like signals.
