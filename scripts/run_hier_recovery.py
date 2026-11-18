"""TM-08 gate: recover delta_w on simulated hierarchical data; writes docs/hier/.

    uv run python scripts/run_hier_recovery.py            # full gate (4 chains, 1000 + 1000)
    uv run python scripts/run_hier_recovery.py --quick    # local smoke (2 chains, 300 + 300)
"""

import argparse

from twominds.hier import hier_recovery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/hier")
    ap.add_argument("--truths", type=float, nargs="+", default=[-0.8, 0.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-participants", type=int, default=40)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    res = hier_recovery(
        a.out,
        truths=tuple(a.truths),
        seeds=tuple(a.seeds),
        n_participants=a.n_participants,
        n_trials=a.n_trials,
        quick=a.quick,
        draws=a.draws,
        tune=a.tune,
        chains=a.chains,
        target_accept=a.target_accept,
    )
    print("gate passed:", res["passed"])


if __name__ == "__main__":
    main()
