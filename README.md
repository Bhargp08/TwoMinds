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
