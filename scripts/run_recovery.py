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
        args.out,
        args.n_subjects,
        n_starts=args.n_starts,
        n_jobs=args.n_jobs,
        seed=args.seed,
        include_wide_prior=True,
    )
    for label, trials in (
        ("realistic", summary["trials"]),
        ("wide", summary["wide_prior"]["trials"]),
    ):
        for n, v in trials.items():
            print(label, n, {k: round(x, 2) for k, x in v["r"].items()})
    print("confusion (realistic):", summary["model_recovery"]["confusion"])
    print("confusion (wide):", summary["wide_prior"]["model_recovery"]["confusion"])
    print("gate:", summary["gate"])


if __name__ == "__main__":
    main()
