# TwoMinds

Measure model-based (planning) vs model-free (habit) learning with the Daw et al. (2011) two-step task,
fit RL agents to each person's choices, and test whether time pressure lowers the model-based weight `w`.

Student project with volunteer participants; not IRB-approved research.

See [docs/BUILD_SPEC.md](docs/BUILD_SPEC.md) for the build plan.

## Dev

```bash
uv sync
uv run ruff check .
uv run pytest -q
```

Tests marked `@pytest.mark.slow` are skipped by default; run them with `uv run pytest -m slow`.

### Experiment (jsPsych / TM-06)

```bash
cd experiment
npm ci
npm test
npm run smoke   # short mock-save session → tests/fixtures/smoke_session.csv
npm run dev     # local task; mock-save when VITE_DATAPIPE_EXPERIMENT_ID is unset
```

Reward schedules are exported from Python (`experiment/scripts/export_schedules.py`) and shipped as fixed JSON — JS does not regenerate numpy RNG draws.

### Full pipeline (TM-10)

```bash
uv run python scripts/run_all.py --simulate --quick
# writes compare_summary.json, ppc_stay.png, writeup_filled.md under docs/run/
```

`docs/writeup.md` is the skeleton with `{{placeholders}}`; `run_all.py` fills them from JSON (no hand-typed numbers). Hierarchical fits come from `twominds.hier.fit_hier` (TM-08).

### Human data loader (TM-07)

`twominds.io.load_session(path)` turns a mock-save / DataPipe CSV into `{condition: data}` for `twominds.agents.loglik`. Missed trials are masked as in TM-02 (`a1 = -1` / `a2 = -1`).

**Exclusion rules** (flagged on each condition via `excluded` / `exclusion_reasons`; never dropped silently):

1. More than 20% of trials in the block are missed (stage 1 or stage 2).
2. Among answered stage-2 trials, the higher-reward option is chosen ≤ 55% of the time.
