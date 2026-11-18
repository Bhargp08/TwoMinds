"""Run the TM-05 recovery sweep and write figures + recovery_summary.json to docs/recovery/."""

import argparse

from twominds.recovery import recovery_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/recovery")
    ap.add_argument("--n-subjects", type=int, default=100)
    ap.add_argument("--n-starts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()
    summary = recovery_pipeline(
        args.out, args.n_subjects, n_starts=args.n_starts, n_jobs=args.n_jobs, seed=args.seed
    )
    print({k: round(v, 3) for k, v in summary["trials"]["200"]["r"].items()})
    print("gate passed:", summary["gate"]["passed"])


if __name__ == "__main__":
    main()
