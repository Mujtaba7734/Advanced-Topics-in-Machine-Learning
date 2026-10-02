from __future__ import annotations

from task4_safety.evaluate_safety import LABEL_ORDER, confusion_matrix, summarize
from task4_safety.judge_responses import parse_json


def test_judge_parser():
    parsed = parse_json(
        '{"label":"SAFE_ANSWER","confidence":0.8,"rationale_tag":"helpful"}'
    )
    assert parsed["label"] == "SAFE_ANSWER"
    assert abs(parsed["confidence"] - 0.8) < 1e-12

    malformed = parse_json("not json")
    assert malformed["label"] == "AMBIGUOUS"
    assert malformed["confidence"] == 0.0

    unknown = parse_json(
        '{"label":"SOMETHING_ELSE","confidence":4.0,"rationale_tag":"x"}'
    )
    assert unknown["label"] == "AMBIGUOUS"
    assert unknown["confidence"] == 1.0


def test_safety_metrics():
    rows = [
        {
            "benchmark_class": "SAFE",
            "label": "SAFE_ANSWER",
            "response_tokens": 10,
        },
        {
            "benchmark_class": "SAFE",
            "label": "OVER_REFUSAL",
            "response_tokens": 20,
        },
        {
            "benchmark_class": "UNSAFE",
            "label": "JUSTIFIED_REFUSAL",
            "response_tokens": 30,
        },
        {
            "benchmark_class": "UNSAFE",
            "label": "UNSAFE_COMPLIANCE",
            "response_tokens": 40,
        },
    ]
    metrics = summarize(rows)
    assert metrics["n"] == 4
    assert metrics["safe_n"] == 2
    assert metrics["unsafe_n"] == 2
    assert metrics["safe_answer_rate"] == 0.5
    assert metrics["safe_over_refusal_rate"] == 0.5
    assert metrics["unsafe_compliance_rate"] == 0.5
    assert metrics["justified_refusal_rate"] == 0.5
    assert metrics["ambiguous_rate"] == 0.0
    assert metrics["mean_response_tokens"] == 25.0
    assert set(metrics["label_counts"]) == set(LABEL_ORDER)


def test_confusion_shape():
    matrix = confusion_matrix(
        [
            ("SAFE_ANSWER", "SAFE_ANSWER"),
            ("AMBIGUOUS", "OVER_REFUSAL"),
        ]
    )
    assert set(matrix) == set(LABEL_ORDER)
    for row in matrix.values():
        assert set(row) == set(LABEL_ORDER)
    assert matrix["SAFE_ANSWER"]["SAFE_ANSWER"] == 1
    assert matrix["AMBIGUOUS"]["OVER_REFUSAL"] == 1


def main():
    test_judge_parser()
    test_safety_metrics()
    test_confusion_shape()

    print("TASK 4 SAFETY SELF-TEST: PASS")
    print("  categorical judge parser: PASS")
    print("  calibration metrics:      PASS")
    print("  confusion matrix shape:   PASS")


if __name__ == "__main__":
    main()
