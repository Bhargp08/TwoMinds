#!/usr/bin/env python3
"""One-command pipeline: simulated or raw sessions → figures + filled write-up.

Does not call into twominds.hier at import time. Pass ``fit_hier`` (tests) or
import it inside ``run_pipeline`` when fitting for real.

Examples:
  uv run python scripts/run_all.py --simulate --quick
  uv run python scripts/run_all.py --simulate --out docs/run
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from twominds.agents import simulate
from twominds.compare import run_compare

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRITEUP = ROOT / "docs" / "writeup.md"

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def simulate_datasets(
    n_subjects: int,
    n_trials: int,
    seed: int,
    w_no: float = 0.55,
    w_pressure: float = 0.30,
) -> dict[str, list[dict]]:
    """Mock two-condition sessions (no human data)."""
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(2 * n_subjects)
    no_p, tp = [], []
    for i in range(n_subjects):
        theta_no = (0.5, 0.5, 1.0, 5.0, 5.0, w_no, 0.1)
        theta_tp = (0.5, 0.5, 1.0, 4.0, 5.0, w_pressure, 0.1)
        no_p.append(simulate(theta_no, n_trials, np.random.default_rng(child[2 * i])))
        tp.append(simulate(theta_tp, n_trials, np.random.default_rng(child[2 * i + 1])))
    return {"no_pressure": no_p, "time_pressure": tp}


def load_datasets_from_dir(data_dir: Path) -> dict[str, list[dict]]:
    """Load every CSV in ``data_dir`` via ``io.load_session`` (mock-save format)."""
    from twominds.io import load_session

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {data_dir}")
    buckets: dict[str, list[dict]] = {}
    for path in files:
        session = load_session(path)
        for cond, data in session.items():
            # Strip exclusion metadata keys that are not arrays for simulate/stay
            clean = {k: v for k, v in data.items() if k in {"a1", "s2", "a2", "r", "transition"}}
            buckets.setdefault(cond, []).append(clean)
    return buckets


def _fmt(x, digits: int = 3) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def build_placeholder_map(summary: Mapping, meta: Mapping) -> dict[str, str]:
    """Map writeup ``{{keys}}`` → strings from compare_summary.json + run meta."""
    mc = summary["model_compare"]
    dw = summary["delta_w"]
    b1 = summary["beta1"]
    stay = summary["stay_logit"]
    figs = summary.get("figures", {})

    def elpd(model: str, field: str):
        return _fmt(mc[model][field])

    mapping = {
        "n_subjects": _fmt(meta["n_subjects"]),
        "n_trials_per_condition": _fmt(meta["n_trials_per_condition"]),
        "run_mode": str(meta["run_mode"]),
        "hier_recovery_note": str(meta.get("hier_recovery_note", "see docs/recovery/")),
        "elpd_hybrid": elpd("hybrid", "elpd_loo"),
        "se_hybrid": elpd("hybrid", "se"),
        "rank_hybrid": elpd("hybrid", "rank"),
        "elpd_mf": elpd("mf", "elpd_loo"),
        "se_mf": elpd("mf", "se"),
        "rank_mf": elpd("mf", "rank"),
        "elpd_mb": elpd("mb", "elpd_loo"),
        "se_mb": elpd("mb", "se"),
        "rank_mb": elpd("mb", "rank"),
        "delta_w_mean": _fmt(dw["mean"]),
        "delta_w_hdi_low": _fmt(dw["hdi_low"]),
        "delta_w_hdi_high": _fmt(dw["hdi_high"]),
        "p_delta_w_lt_0": _fmt(dw["p_delta_w_lt_0"]),
        "delta_w_group_mean": _fmt(dw["delta_w_group_mean"]),
        "beta1_condition_effect": _fmt(b1.get("condition_effect", False)),
        "delta_beta1_mean": _fmt(b1.get("delta_beta1_mean", "n/a")),
        "delta_beta1_hdi_low": _fmt(b1.get("delta_beta1_hdi_low", "n/a")),
        "delta_beta1_hdi_high": _fmt(b1.get("delta_beta1_hdi_high", "n/a")),
        "stay_reward_no_pressure": _fmt(stay["no_pressure"]["coef"]["reward"]),
        "stay_interaction_no_pressure": _fmt(stay["no_pressure"]["coef"]["interaction"]),
        "stay_reward_time_pressure": _fmt(stay["time_pressure"]["coef"]["reward"]),
        "stay_interaction_time_pressure": _fmt(stay["time_pressure"]["coef"]["interaction"]),
        "ppc_stay_figure": str(figs.get("ppc_stay", "ppc_stay.png")),
    }
    return mapping


def render_writeup(template: str, mapping: Mapping[str, str]) -> str:
    """Replace ``{{key}}`` placeholders; error if any remain unknown."""

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in mapping:
            raise KeyError(f"no value for writeup placeholder {{{{{key}}}}}")
        return mapping[key]

    rendered = PLACEHOLDER_RE.sub(repl, template)
    leftover = PLACEHOLDER_RE.findall(rendered)
    if leftover:
        raise ValueError(f"unfilled placeholders: {leftover}")
    return rendered


def run_pipeline(
    *,
    out_dir: Path,
    simulate: bool = True,
    quick: bool = False,
    seed: int = 0,
    n_subjects: int = 20,
    n_trials: int = 150,
    data_dir: Path | None = None,
    fit_hier: Callable | None = None,
    writeup_template: Path = DEFAULT_WRITEUP,
) -> dict:
    """Run simulate/load → hierarchical fits → compare → fill writeup.

    ``fit_hier`` defaults to ``twominds.hier.fit_hier`` (TM-08). Tests inject a stub.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if quick:
        n_subjects = min(n_subjects, 4)
        n_trials = min(n_trials, 40)

    if simulate:
        datasets = simulate_datasets(n_subjects, n_trials, seed)
        run_mode = "simulate-quick" if quick else "simulate"
    else:
        if data_dir is None:
            raise ValueError("data_dir is required unless --simulate")
        datasets = load_datasets_from_dir(Path(data_dir))
        n_subjects = len(next(iter(datasets.values())))
        n_trials = len(next(iter(datasets.values()))[0]["a1"])
        run_mode = "raw-quick" if quick else "raw"

    if fit_hier is None:
        from twominds.hier import fit_hier as fit_hier  # TM-08; not imported at module top

    idatas = {
        "hybrid": fit_hier(datasets, model="hybrid", quick=quick),
        "mf": fit_hier(datasets, model="mf", quick=quick),
        "mb": fit_hier(datasets, model="mb", quick=quick),
    }

    summary = run_compare(
        idatas,
        datasets,
        out_dir=out_dir,
        rng=np.random.default_rng(seed + 1),
        n_ppc=5 if quick else 20,
        n_trials=n_trials,
    )
    # Also drop PPC under figures/ for the writeup path convention
    ppc_src = out_dir / "ppc_stay.png"
    ppc_dst = figures / "ppc_stay.png"
    if ppc_src.is_file():
        ppc_dst.write_bytes(ppc_src.read_bytes())
        summary["figures"]["ppc_stay"] = str(ppc_dst.relative_to(out_dir))

    meta = {
        "n_subjects": n_subjects,
        "n_trials_per_condition": n_trials,
        "run_mode": run_mode,
        "hier_recovery_note": (
            "skipped in --quick; see TM-08 recovery when available"
            if quick
            else "see TM-08 hierarchical recovery outputs when available"
        ),
    }
    mapping = build_placeholder_map(summary, meta)
    filled = render_writeup(Path(writeup_template).read_text(), mapping)
    filled_path = out_dir / "writeup_filled.md"
    filled_path.write_text(filled)

    # Refresh compare_summary with meta for provenance
    summary["meta"] = meta
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    return {
        "n_subjects": n_subjects,
        "n_trials": n_trials,
        "out_dir": str(out_dir),
        "writeup": str(filled_path),
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--simulate", action="store_true", help="use simulated sessions (no human data)"
    )
    ap.add_argument("--data", type=Path, default=None, help="directory of mock-save CSVs")
    ap.add_argument("--quick", action="store_true", help="fewer subjects/trials/PPC draws")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-subjects", type=int, default=20)
    ap.add_argument("--n-trials", type=int, default=150)
    ap.add_argument("--writeup", type=Path, default=DEFAULT_WRITEUP)
    args = ap.parse_args(argv)

    if not args.simulate and args.data is None:
        ap.error("pass --simulate or --data DIR")

    result = run_pipeline(
        out_dir=args.out,
        simulate=args.simulate,
        quick=args.quick,
        seed=args.seed,
        n_subjects=args.n_subjects,
        n_trials=args.n_trials,
        data_dir=args.data,
        writeup_template=args.writeup,
    )
    print(f"wrote {result['writeup']}")
    print(f"summary → {Path(result['out_dir']) / 'compare_summary.json'}")


if __name__ == "__main__":
    main()
