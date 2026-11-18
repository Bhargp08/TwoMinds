"""Load human two-step session CSVs (DataPipe / mock-save) for fitting.

Exclusion rules (also in README): a block is *flagged* (never silently dropped) when
(1) more than 20% of trials are missed (stage 1 or stage 2), or
(2) among answered stage-2 trials, the higher-reward option is chosen ≤ 55% of the time.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

MISS_RATE_EXCLUDE = 0.20
BETTER_CHOICE_EXCLUDE = 0.55


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _int_or_miss(v, missed: bool) -> int:
    if missed:
        return -1
    if v is None or str(v).strip() == "":
        return -1
    return int(float(v))


def _reward_or_zero(v, missed: bool) -> int:
    if missed:
        return 0
    if v is None or str(v).strip() == "":
        return 0
    return int(float(v))


def _s2_or_placeholder(v, missed_a1: bool) -> int:
    """s2 is unused when stage 1 was missed; keep a valid placeholder for array typing."""
    if missed_a1:
        return 1
    if v is None or str(v).strip() == "":
        return 1
    return int(float(v))


def _better_choice_rate(rows: list[dict]) -> float | None:
    """Fraction of answered stage-2 trials where the higher-p option was chosen."""
    hits = 0
    n = 0
    for row in rows:
        if _as_bool(row.get("missed_a1")) or _as_bool(row.get("missed_a2")):
            continue
        s2 = _int_or_miss(row.get("s2"), False)
        a2 = _int_or_miss(row.get("a2"), False)
        if a2 < 0 or s2 not in (1, 2):
            continue
        if s2 == 1:
            p0, p1 = float(row["p_s1_a0"]), float(row["p_s1_a1"])
        else:
            p0, p1 = float(row["p_s2_a0"]), float(row["p_s2_a1"])
        better = 0 if p0 >= p1 else 1
        hits += int(a2 == better)
        n += 1
    if n == 0:
        return None
    return hits / n


def _evaluate_exclusions(rows: list[dict]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    n = len(rows)
    if n == 0:
        return True, ["empty block"]
    n_miss = sum(
        1
        for row in rows
        if _as_bool(row.get("missed_a1")) or _as_bool(row.get("missed_a2"))
    )
    miss_rate = n_miss / n
    if miss_rate > MISS_RATE_EXCLUDE:
        reasons.append(f"missed trials {miss_rate:.1%} > {MISS_RATE_EXCLUDE:.0%}")
    better_rate = _better_choice_rate(rows)
    if better_rate is not None and better_rate <= BETTER_CHOICE_EXCLUDE:
        reasons.append(
            f"stage-2 better-option rate {better_rate:.1%} ≤ {BETTER_CHOICE_EXCLUDE:.0%}"
        )
    return bool(reasons), reasons


def _rows_to_data(rows: list[dict]) -> dict:
    """Convert CSV rows for one condition into loglik arrays + exclusion flags.

    Missed-trial encoding matches TM-02 / agents.loglik:
    - stage-1 miss → a1 = -1 (and a2 = -1); no likelihood, no update, rep resets
    - stage-2 miss → a1 kept, a2 = -1; stage-1 log-prob counts, no update
    """
    a1, s2, a2, r = [], [], [], []
    transition = []
    for row in rows:
        miss1 = _as_bool(row.get("missed_a1"))
        miss2 = _as_bool(row.get("missed_a2")) or miss1
        a1.append(_int_or_miss(row.get("a1"), miss1))
        s2.append(_s2_or_placeholder(row.get("s2"), miss1))
        a2.append(_int_or_miss(row.get("a2"), miss2))
        r.append(_reward_or_zero(row.get("r"), miss1 or miss2))
        transition.append(str(row.get("transition") or ""))
    excluded, reasons = _evaluate_exclusions(rows)
    return {
        "a1": np.asarray(a1, dtype=int),
        "s2": np.asarray(s2, dtype=int),
        "a2": np.asarray(a2, dtype=int),
        "r": np.asarray(r, dtype=int),
        "transition": np.asarray(transition),
        "excluded": excluded,
        "exclusion_reasons": reasons,
    }


def load_session(path) -> dict[str, dict]:
    """Load a DataPipe/mock-save CSV into `{condition: data}` for `agents.loglik`.

    Excluded blocks are kept and flagged via `excluded` / `exclusion_reasons`
    on each condition dict — never dropped silently.
    """
    path = Path(path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "condition" not in reader.fieldnames:
            raise ValueError(f"{path} has no 'condition' column")
        all_rows = list(reader)

    by_condition: dict[str, list[dict]] = {}
    for row in all_rows:
        cond = str(row["condition"])
        by_condition.setdefault(cond, []).append(row)

    out: dict[str, dict] = {}
    for condition, rows in by_condition.items():
        rows_sorted = sorted(
            rows,
            key=lambda r: (int(float(r.get("block") or 0)), int(float(r.get("trial") or 0))),
        )
        out[condition] = _rows_to_data(rows_sorted)
    return out
