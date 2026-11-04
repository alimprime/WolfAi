"""Эквивалентные формулы, Парето и тот же бюджет PySR."""

from datetime import datetime, timezone

import pytest

from wolfai.research.formula import FormulaError, parse_formula, tree_depth
from wolfai.research.pysr_engine import LIGHT_LIMITS, discover, regressor_settings
from wolfai.research.symbolic_strategy import (
    canonical_expression,
    examination_count,
    extra_regressor_options,
    pareto_front,
    screen_equations,
    ScoredExpression,
)

UTC = timezone.utc


def test_commutative_expressions_share_one_canonical_form() -> None:
    left = canonical_expression("sigmoid(log_return + momentum_7)")
    right = canonical_expression("sigmoid(momentum_7 + log_return)")
    assert left == right
    assert canonical_expression("sigmoid(log_return * momentum_7)") == canonical_expression(
        "sigmoid(momentum_7 * log_return)"
    )
    assert canonical_expression("sigmoid(log_return - momentum_7)") == canonical_expression(
        "sigmoid(log_return + -momentum_7)"
    )
    assert canonical_expression("sigmoid(log_return * 1)") == canonical_expression("sigmoid(log_return)")
    parse_formula(left)


def test_reassociation_stays_inside_the_grammar() -> None:
    expression = (
        "sigmoid(((log_return / -0.2224) + "
        "abs(((log_return + momentum_7) + (log_return + momentum_7)))))"
    )
    rendered = canonical_expression(expression)
    tree = parse_formula(rendered)
    assert tree_depth(tree) <= 6
    swapped = (
        "sigmoid(((log_return / -0.2224) + "
        "abs(((momentum_7 + log_return) + (momentum_7 + log_return)))))"
    )
    assert rendered == canonical_expression(swapped)


def test_tiny_divisor_is_rejected_before_evaluation() -> None:
    with pytest.raises(FormulaError, match="малое"):
        canonical_expression("sigmoid(log_return / 0.0000001)")


def test_screen_collapses_equivalents_and_keeps_pareto() -> None:
    assembly = screen_equations(
        [
            "log_return + momentum_7",
            "momentum_7 + log_return",
            "cos(log_return)",
            "volatility_7",
            "__import__('os').system('echo hi')",
        ],
        [0.40, 0.20, 0.01, 0.50, 0.01],
        feature_names=("log_return", "momentum_7", "volatility_7"),
        horizon_days=1,
        dataset_version="synthetic",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
        max_candidates=5,
    )
    assert len(assembly.hypotheses) == 2
    assert any(item.reason == "математически та же формула" for item in assembly.rejected)
    assert any("неизвестная функция" in item.reason or "запрещ" in item.reason for item in assembly.rejected)
    expressions = [item.expression for item in assembly.hypotheses]
    assert all(parse_formula(expression) for expression in expressions)
    assert assembly.pareto_size == 2
    assert assembly.examined == 5


def test_dominated_expression_leaves_the_front() -> None:
    front = pareto_front((
        ScoredExpression("sigmoid(log_return)", 3, 0.50),
        ScoredExpression("sigmoid(momentum_7)", 5, 0.40),
        ScoredExpression("sigmoid(volatility_7)", 7, 0.45),
    ))
    assert [item.expression for item in front] == ["sigmoid(log_return)", "sigmoid(momentum_7)"]


def test_improved_options_do_not_buy_more_iterations() -> None:
    extra = extra_regressor_options()
    assert "niterations" not in extra
    assert extra["warmup_maxsize_by"] == 0.5
    baseline = regressor_settings("baseline", 1729)
    improved = regressor_settings("improved", 1729)
    assert baseline["niterations"] == improved["niterations"] == LIGHT_LIMITS["niterations"] == 4
    assert improved["populations"] == LIGHT_LIMITS["populations"]
    assert improved["procs"] == 1
    assert improved["timeout_in_seconds"] <= LIGHT_LIMITS["timeout_seconds"]
    assert examination_count(5) == 4 * 3 * 16
    with pytest.raises(ValueError, match="стратегия"):
        discover([], [], dataset_version="v", horizon_days=1, strategy="larger")
