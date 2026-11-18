# TwoMinds write-up

Student project with volunteer participants; **not IRB-approved research**.

Numbers in `{{double_braces}}` are filled by `scripts/run_all.py` from JSON outputs — do not hand-type results.

## Question

Does time pressure lower the model-based weight \(w\) in the Daw et al. (2011) two-step task?

## Task

Two-step Markov task with drifting reward probabilities. Each participant completes two blocks (no pressure / time pressure), order counterbalanced. Trial counts per condition in this run: **{{n_trials_per_condition}}**. Number of simulated or loaded sessions: **{{n_subjects}}**.

## Models

Hybrid RL agent (Daw et al.): parameters \((\alpha_1, \alpha_2, \lambda, \beta_1, \beta_2, w, p)\). Model-free is \(w=0\); model-based is \(w=1\). Hierarchical fits (TM-08) for hybrid, MF, and MB; time pressure enters hybrid \(w\) as \(\mathrm{logit}(w_{ic}) = \mu_w + \delta_w\, c + \cdots\).

## Recovery appendix

Parameter recovery (TM-05) lives under `docs/recovery/` (scatter grids, confusion matrix, `recovery_summary.json`). Hierarchical recovery (TM-08) should be attached when available: **{{hier_recovery_note}}**.

## Model comparison

| Model | ELPD (LOO) | SE | Rank |
|-------|------------|----|------|
| hybrid | {{elpd_hybrid}} | {{se_hybrid}} | {{rank_hybrid}} |
| mf | {{elpd_mf}} | {{se_mf}} | {{rank_mf}} |
| mb | {{elpd_mb}} | {{se_mb}} | {{rank_mb}} |

## Time-pressure effect on \(w\)

- Posterior mean \(\delta_w\): **{{delta_w_mean}}**
- 95% HDI: **[{{delta_w_hdi_low}}, {{delta_w_hdi_high}}]**
- \(P(\delta_w < 0)\): **{{p_delta_w_lt_0}}**
- Group-mean change in \(w\) (pressure − no pressure): **{{delta_w_group_mean}}**

## Effect on \(\beta_1\)

- Condition effect present: **{{beta1_condition_effect}}**
- \(\delta_{\beta_1}\) mean (if any): **{{delta_beta1_mean}}**
- 95% HDI: **[{{delta_beta1_hdi_low}}, {{delta_beta1_hdi_high}}]**

## Stay-probability check (model-free)

Per-condition logistic regression `stay ~ reward * transition` (±1 coding):

- No pressure — reward coef: **{{stay_reward_no_pressure}}**; interaction: **{{stay_interaction_no_pressure}}**
- Time pressure — reward coef: **{{stay_reward_time_pressure}}**; interaction: **{{stay_interaction_time_pressure}}**

Posterior predictive stay overlay: `{{ppc_stay_figure}}`.

## Limitations

- Small volunteer sample; not IRB-reviewed.
- Time pressure may increase choice noise (\(\beta_1\)) rather than (or in addition to) lowering planning weight \(w\) — a confound to report.
- Trial count per condition ({{n_trials_per_condition}}) is below the TM-05 recovery default of 200; recovery of \(w\) may be limited.
- This filled write-up used mode: **{{run_mode}}**.
