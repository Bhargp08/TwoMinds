# TwoMinds plan / status

- **TM-08 (hierarchical Bayesian model): delta_w gate PASSED.** Simulated recovery with
  40 participants x 2 conditions x 200 trials, 4 chains x 1000 draws (1000 tune),
  target_accept 0.9. The 95% HDI of `delta_w` covers the truth in all 4 runs (truth -0.8 and
  0, seeds 0 and 1), with R-hat <= 1.009, bulk ESS >= 485 and no divergences on the group
  parameters. See `docs/hier/hier_recovery.json` and `docs/hier/delta_w_posteriors.png`.
  Human fits may proceed.
