"""Evaluation helpers for differentiation metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingSampleResult:
    expected_route: str
    predicted_route: str


@dataclass(frozen=True)
class AnswerQualitySample:
    references: list[str]
    two_week_plan: list[str]


def routing_accuracy(samples: list[RoutingSampleResult]) -> float:
    """Return Top-1 routing accuracy in [0, 1]."""
    if not samples:
        return 0.0
    correct = sum(
        1
        for item in samples
        if item.expected_route.strip().lower() == item.predicted_route.strip().lower()
    )
    return correct / len(samples)


def reference_inclusion_rate(samples: list[AnswerQualitySample], min_references: int = 1) -> float:
    """Return ratio of answers that include at least min_references refs."""
    if not samples:
        return 0.0
    included = sum(1 for item in samples if len(item.references) >= min_references)
    return included / len(samples)


def plan_quality_rate(samples: list[AnswerQualitySample], min_plan_items: int = 4) -> float:
    """Return ratio of answers that include enough actionable plan items."""
    if not samples:
        return 0.0
    qualified = sum(1 for item in samples if len(item.two_week_plan) >= min_plan_items)
    return qualified / len(samples)

