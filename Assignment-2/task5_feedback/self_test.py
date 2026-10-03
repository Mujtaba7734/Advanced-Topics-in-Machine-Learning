from __future__ import annotations

import math

from task5_feedback.rlvr import (
    exact_reward,
    extract_designated_final,
    numerically_equal,
)
from task5_feedback.score_perturbations import _rates


def assert_close(actual, expected, name, tol=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{name}: expected {expected}, got {actual}")


def main():
    # The verifier must use only the designated #### field and must use the
    # final designated field when more than one appears.
    if extract_designated_final("Reasoning says 42, but no final marker.") is not None:
        raise AssertionError("Intermediate numbers must not count as designated finals")

    if extract_designated_final("#### 1,234") != "1234":
        raise AssertionError("Comma-formatted designated final was parsed incorrectly")

    if extract_designated_final("#### 5\nrevision\n#### -3.5") != "-3.5":
        raise AssertionError("Verifier must use the last designated final")

    if not numerically_equal("3.0", "3"):
        raise AssertionError("Numeric equality failed for equivalent numbers")

    assert_close(
        exact_reward("work\n#### 42", "42"),
        1.0,
        "exact verifier positive",
    )
    assert_close(
        exact_reward("42 appears in reasoning only", "42"),
        0.0,
        "exact verifier ignores distractor",
    )
    assert_close(
        exact_reward("#### 42\nthen corrected\n#### 41", "42"),
        0.0,
        "exact verifier uses final designated answer",
    )

    rates = _rates(["A", "A", "B", "TIE"])
    assert_close(rates["better_response_rate"], 0.5, "better-response rate")
    assert_close(rates["tie_rate"], 0.25, "tie rate")
    assert_close(rates["wrong_preference_rate"], 0.25, "wrong-preference rate")
    assert_close(rates["pairwise_score"], 0.625, "pairwise score")

    print("TASK 5 FEEDBACK SELF-TEST: PASS")
    print("  designated-final parser: PASS")
    print("  exact verifier:          PASS")
    print("  distractor robustness:   PASS")
    print("  pairwise rate metrics:   PASS")


if __name__ == "__main__":
    main()
