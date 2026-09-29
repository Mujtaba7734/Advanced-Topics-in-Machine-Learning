# PA2 Instructions

This is Programming Assignment 2 on LLM post-training.

## Assignment rules
- Tasks 1-3 contain incomplete scaffolding and one deliberate algorithmic defect in each core objective.
- Identify and explain any suspected objective defect before modifying it.
- Preserve fixed prompt IDs, dataset indices, seeds, checkpoints, caches, decoding settings, and evaluation procedures unless the assignment explicitly requires a change.
- Do not modify course-provided cached rollouts, checkpoints, or fixed evaluation data.
- Keep ablation budgets and comparison conditions matched exactly as specified.

## Compute
- Do not launch model downloads, long training, or GPU experiments without explicit approval.
- Prefer static checks, CPU checks, and tiny smoke tests before full experiments.
- Keep commands reproducible and independent of hidden notebook state.

## PA2 report policy
- Do not generate prose intended for the submitted PDF report.
- Coding, debugging, implementation explanation, metrics, analysis tooling, and plots are allowed.

## Code changes
- Avoid unnecessary refactoring of starter infrastructure.
- Any change to DPO, PPO, or GRPO objective code must include a mathematical explanation of why the original implementation was incorrect.
