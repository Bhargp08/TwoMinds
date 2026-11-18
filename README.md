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
