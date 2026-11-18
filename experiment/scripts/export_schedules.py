#!/usr/bin/env python3
"""Export make_reward_schedule outputs for JS.

JS cannot reproduce numpy's RNG, so we ship full schedules per seed as JSON
fixtures rather than regenerating them in the browser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from twominds.env import make_reward_schedule  # noqa: E402

SEEDS = [0, 1, 42, 101, 202]
N_TRIALS = 150


def main() -> None:
    fixture_dir = ROOT / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, dict] = {}
    for seed in SEEDS:
        sched = make_reward_schedule(N_TRIALS, np.random.default_rng(seed))
        payload = {
            "seed": seed,
            "n_trials": N_TRIALS,
            "sd": 0.025,
            "bounds": [0.25, 0.75],
            "note": (
                "Full schedule exported from numpy RNG; JS ships this rather than "
                "regenerating (numpy RNG is not portable)."
            ),
            "schedule": sched.tolist(),
        }
        path = fixture_dir / f"schedule_seed{seed}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        bundle[str(seed)] = payload
        print(f"wrote {path}")
    bundled = ROOT / "experiment" / "src" / "lib" / "schedules.json"
    bundled.parent.mkdir(parents=True, exist_ok=True)
    bundled.write_text(json.dumps(bundle) + "\n")
    print(f"wrote {bundled}")


if __name__ == "__main__":
    main()
