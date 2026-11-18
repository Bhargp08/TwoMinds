"""Classic 3-panel stay-probability figure (model-free, model-based, hybrid agents)."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from twominds.agents import simulate  # noqa: E402
from twominds.analysis import stay_table  # noqa: E402

AGENTS = {"Model-free (w=0)": 0.0, "Model-based (w=1)": 1.0, "Hybrid (w=0.5)": 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--n-agents", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (name, w) in zip(axes, AGENTS.items(), strict=True):
        theta = (0.5, 0.5, 1.0, 5.0, 5.0, w, 0.0)
        seeds = np.random.SeedSequence(args.seed).spawn(args.n_agents)
        rngs = [np.random.default_rng(s) for s in seeds]
        tab = stay_table([simulate(theta, args.n_trials, rng) for rng in rngs])["stay"]
        x = np.arange(2)
        ax.bar(x - 0.18, tab[1], 0.36, label="Rewarded")
        ax.bar(x + 0.18, tab[0], 0.36, label="Unrewarded")
        ax.set_xticks(x, ["Common", "Rare"])
        ax.set_title(name)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("P(repeat stage-1 choice)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stay_probabilities.png", dpi=150)


if __name__ == "__main__":
    main()
