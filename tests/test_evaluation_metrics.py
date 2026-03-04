from src.evaluation.metrics import (
    AnswerQualitySample,
    RoutingSampleResult,
    plan_quality_rate,
    reference_inclusion_rate,
    routing_accuracy,
)


def test_routing_accuracy() -> None:
    samples = [
        RoutingSampleResult(expected_route="resume_only", predicted_route="resume_only"),
        RoutingSampleResult(expected_route="full", predicted_route="plan_only"),
        RoutingSampleResult(expected_route="interview_only", predicted_route="interview_only"),
    ]
    assert routing_accuracy(samples) == 2 / 3


def test_reference_inclusion_rate() -> None:
    samples = [
        AnswerQualitySample(references=["a"], two_week_plan=["1", "2", "3", "4"]),
        AnswerQualitySample(references=[], two_week_plan=["1", "2", "3", "4"]),
    ]
    assert reference_inclusion_rate(samples) == 0.5


def test_plan_quality_rate() -> None:
    samples = [
        AnswerQualitySample(references=["a"], two_week_plan=["1", "2", "3", "4"]),
        AnswerQualitySample(references=["a"], two_week_plan=["1", "2"]),
    ]
    assert plan_quality_rate(samples) == 0.5

