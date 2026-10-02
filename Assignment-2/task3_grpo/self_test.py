from __future__ import annotations

import math

import torch

from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


def assert_close(actual, expected, message, atol=1e-6):
    if not math.isclose(
        float(actual),
        float(expected),
        abs_tol=atol,
        rel_tol=atol,
    ):
        raise AssertionError(
            f"{message}: got {float(actual)}, expected {float(expected)}"
        )


def test_group_relative_advantages():
    rewards = torch.tensor([1.0, 3.0, 10.0, 14.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    advantages = group_relative_advantages(rewards, group_ids)

    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    if not torch.allclose(
        advantages,
        expected,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "Within-prompt GRPO advantages incorrect: "
            f"{advantages.tolist()}"
        )

    shifted = torch.tensor([1.0, 3.0, 1000.0, 1004.0])
    shifted_adv = group_relative_advantages(shifted, group_ids)
    if not torch.allclose(
        shifted_adv[:2],
        advantages[:2],
        atol=2e-6,
        rtol=2e-6,
    ):
        raise AssertionError(
            "Advantages leak reward scale across unrelated prompt groups"
        )

    flat = torch.tensor([2.0, 2.0, 5.0, 5.0])
    flat_adv = group_relative_advantages(flat, group_ids)
    if not torch.equal(flat_adv, torch.zeros_like(flat_adv)):
        raise AssertionError(
            "Uninformative groups should have zero advantage"
        )


def test_clipped_objective_and_normalization():
    old = torch.zeros((2, 4))
    ref = torch.zeros((2, 4))
    new = torch.zeros((2, 4))
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    canonical, canonical_diag = grpo_policy_loss(
        new,
        old,
        advantages,
        mask,
        ref,
        eps=0.2,
        beta=0.0,
        loss_type="grpo",
        max_completion_length=4,
    )
    dr, dr_diag = grpo_policy_loss(
        new,
        old,
        advantages,
        mask,
        ref,
        eps=0.2,
        beta=0.0,
        loss_type="dr_grpo",
        max_completion_length=4,
    )

    assert_close(
        canonical,
        0.0,
        "Canonical GRPO normalization",
    )
    assert_close(
        dr,
        0.25,
        "Dr-GRPO constant normalization",
    )
    assert_close(
        canonical_diag["sampled_kl"],
        0.0,
        "Zero KL at policy=reference",
    )
    assert_close(
        dr_diag["sampled_kl"],
        0.0,
        "Zero KL at policy=reference",
    )

    ratio = torch.tensor([[1.5], [0.5]])
    new_clip = ratio.log()
    old_clip = torch.zeros_like(new_clip)
    ref_clip = new_clip.clone()
    adv_clip = torch.tensor([1.0, -1.0])
    mask_clip = torch.ones_like(new_clip)

    loss, diag = grpo_policy_loss(
        new_clip,
        old_clip,
        adv_clip,
        mask_clip,
        ref_clip,
        eps=0.2,
        beta=0.0,
        loss_type="grpo",
        max_completion_length=1,
    )
    assert_close(
        loss,
        -0.2,
        "Clipped GRPO surrogate",
    )
    assert_close(
        diag["clip_fraction"],
        1.0,
        "GRPO clip fraction",
    )


def test_truncation_mask():
    mask = torch.ones((3, 4))
    truncated = [False, True, False]
    masked = mask_truncated_sequences(mask, truncated)

    if masked[1].sum().item() != 0:
        raise AssertionError(
            "Truncated completion was not fully masked"
        )
    if (
        masked[0].sum().item() != 4
        or masked[2].sum().item() != 4
    ):
        raise AssertionError(
            "Non-truncated completion was incorrectly masked"
        )


def main():
    test_group_relative_advantages()
    test_clipped_objective_and_normalization()
    test_truncation_mask()

    print("TASK 3 GRPO MATH SELF-TEST: PASS")
    print("  within-prompt advantages: PASS")
    print("  uninformative groups:     PASS")
    print("  clipped surrogate:        PASS")
    print("  GRPO vs Dr-GRPO norm:     PASS")
    print("  max-length masking:       PASS")


if __name__ == "__main__":
    main()
