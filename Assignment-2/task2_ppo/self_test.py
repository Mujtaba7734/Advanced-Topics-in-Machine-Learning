from __future__ import annotations

import math
import torch

from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def assert_close(actual: float, expected: float, name: str, tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main() -> None:
    # Clipped PPO surrogate: positive and negative advantages must both use min().
    old = torch.zeros(1, 3)
    ratios = torch.tensor([[1.5, 0.5, 1.5]])
    new = ratios.log()
    advantage = torch.tensor([[1.0, 1.0, -1.0]])
    mask = torch.ones_like(advantage)
    loss, observed_ratio, clip_fraction = ppo_policy_loss(
        new, old, advantage, mask, eps=0.2
    )
    expected_objective = torch.tensor([[1.2, 0.5, -1.5]]).mean().item()
    assert_close(float(loss), -expected_objective, "PPO clipped loss")
    assert_close(float(clip_fraction), 1.0, "PPO clip fraction")
    if not torch.allclose(observed_ratio, ratios, atol=1e-6, rtol=1e-6):
        raise AssertionError("PPO probability ratios are incorrect")

    # Terminal learned reward + tokenwise sampled-KL shaping.
    task_reward = torch.tensor([2.0])
    policy_logp = torch.tensor([[-1.0, -2.0, -3.0]])
    ref_logp = torch.tensor([[-1.5, -1.5, -2.5]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])
    rewards = shaped_rewards(
        task_reward, policy_logp, ref_logp, response_mask, beta_kl=0.1
    )
    expected_rewards = torch.tensor([[-0.05, 0.05, 2.05]])
    if not torch.allclose(rewards, expected_rewards, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"KL-shaped rewards incorrect: {rewards}")

    # With zero values and gamma=lambda=1, GAE is the reward-to-go.
    values = torch.zeros_like(rewards)
    advantages, returns = compute_gae(
        rewards, values, response_mask, gamma=1.0, lam=1.0
    )
    expected_advantages = torch.tensor([[2.05, 2.10, 2.05]])
    if not torch.allclose(advantages, expected_advantages, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"GAE incorrect: {advantages}")
    if not torch.allclose(returns, expected_advantages, atol=1e-6, rtol=1e-6):
        raise AssertionError("Returns should equal advantages when values are zero")

    normalized = normalize_advantages(advantages, response_mask)
    valid = normalized[response_mask.bool()]
    assert_close(float(valid.mean()), 0.0, "normalized advantage mean", tol=1e-5)
    assert_close(float(valid.std(unbiased=False)), 1.0, "normalized advantage std", tol=1e-5)

    value_loss = value_mse_loss(
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[2.0, 2.0, 1.0]]),
        response_mask,
    )
    assert_close(float(value_loss), 5.0 / 3.0, "masked value MSE")

    print("TASK 2 PPO MATH SELF-TEST: PASS")
    print("  clipped surrogate: PASS")
    print("  KL-shaped reward:  PASS")
    print("  GAE/returns:       PASS")
    print("  advantage norm:    PASS")
    print("  value MSE:         PASS")


if __name__ == "__main__":
    main()
