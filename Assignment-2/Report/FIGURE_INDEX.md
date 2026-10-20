# PA2 figure inventory (engineering only)

All 25 saved PNGs are retained in `Report/figures/`. Required quantitative evidence is in the main report as figures or tables; the appendix provides supporting/redundant plots. Report interpretation, examples, and captions must be written by the student.

| Figure | In PDF | Meaning |
|---|---|---|
| `01_dpo_beta_accuracy.png` | Appendix | DPO short-run beta accuracy |
| `02_dpo_length_strata.png` | Main | DPO fixed response-length strata |
| `03_ppo_reward_trajectory.png` | Catalog only | PPO reward also in combined main grid 21 |
| `04_ppo_kl_trajectory.png` | Catalog only | PPO sampled KL also in grid 21 |
| `05_ppo_clip_cached.png` | Appendix | PPO cached affected-token fraction |
| `06_grpo_group_size.png` | Appendix | GRPO informative groups K=2/4/8 |
| `07_grpo_normalization_length.png` | Appendix | GRPO vs Dr-GRPO length-conditioned loss contributions |
| `08_safety_ai_label_distribution.png` | Appendix | Task 4 overall AI judge five-label distribution |
| `10_task5_math_accuracy.png` | Appendix | GSM8K and SVAMP exact accuracy |
| `11_task5_diagnostic.png` | Appendix | Task 5 clean-answer better preference by perturbation |
| `12_safety_text_overlap.png` | Appendix | Safety exact response-text overlap |
| `13_ppo_four_diagnostics.png` | Catalog only | Alternative PPO diagnostic view, grid 21 in main |
| `14_ppo_policy_critic.png` | Catalog only | Alternative PPO optimizer view, grid 22 in main |
| `15_ppo_clip_fork_diagnostics.png` | Appendix | Short clipping forks stability |
| `16_grpo_reward_grad.png` | Catalog only | Alternative GRPO diagnostics, main grid 23 |
| `17_grpo_loss_entropy_length.png` | Catalog only | Alternative GRPO diagnostics, main grid 24 |
| `18_grpo_difficulty_heatmap.png` | Appendix | GRPO groups by reward-quartile difficulty |
| `19_safety_category_heatmap.png` | Appendix | SAFE_ANSWER category classification map |
| `20_task5_pairwise_ties.png` | Appendix | Task 5 GSM8K judge tie rates |
| `21_ppo_reward_critic_grid.png` | Main | PPO continuation reward KL critic surprisal |
| `22_ppo_optimization_grid.png` | Main | PPO policy loss gradient length |
| `23_grpo_signal_grid.png` | Main | GRPO reward KL group informativeness |
| `24_grpo_optimization_grid.png` | Main | GRPO policy loss gradient surprisal length |
| `25_safety_full_label_distribution.png` | Main | All Task4 categorical classifications for four policies |
| `26_ppo_clip_trajectory.png` | Appendix | Zero on-policy clip fraction over 20 updates |

## Safety classification evidence

- Main `25_safety_full_label_distribution.png`: 18 XSTest categories by 4 policies, automated-judge labels; OVER_REFUSAL and AMBIGUOUS are zero across categories under the fixed judge.
- Appendix `19_safety_category_heatmap.png`: safe-answer category map, a complementary view rather than a replacement for the full distribution.
- Main `tables/manual-audit.tex`: 5x5 AI-row/human-column confusion matrix with 240 manual judgments and conditional color shading. Human label accuracy is not independently verified here.
- Source `data/14_safety_full_category_distribution.csv` and `data/15_manual_audit.json` record the evidence used. `scripts/verify_assets.py` checks totals and figures.

## Metrics caution

`Entropy` in the original saved PPO/GRPO figures corresponds to sampled-token surprisal (`-mean(sampled log p)`) and should not be represented as exact full-token distribution entropy. The on-policy PPO clip fraction is zero for every standard update while separate cached-rollout affected-token rates are nonzero. Four GRPO update slots had zero eligible tokens. Do not compare the full-epoch DPO standard to the short beta forks as matched training budgets.

**Final report:** check the instructor's appendix page-count policy, fit all required evidence in 8 main pages, independently verify manual labels, and select genuine qualitative examples from saved JSONL. No example quotations or final report prose were generated here.
