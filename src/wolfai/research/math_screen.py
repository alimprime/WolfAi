"""Проверка формулы до вычисления. Текст модели не исполняется как код.

Допустимы только операции текущего грамматического движка. Калибровка
по диагностическому 2021 году не подбирается.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from wolfai.research.formula import (
    DIVISION_EPSILON,
    FEATURES,
    MAX_DEPTH,
    MAX_NODES,
    Binary,
    Const,
    EvalState,
    Expr,
    FormulaError,
    Unary,
    evaluate,
    features_used,
    node_count,
    parse_formula,
    tree_depth,
)
from wolfai.research.generator import expression_is_safe

EXPLICIT_SIGMOID = "explicit_sigmoid"
VERIFIER_ID = "verification-desk"


@dataclass(frozen=True)
class MathReport:
    accepted: bool
    code: str
    reason: str
    depth: int = 0
    nodes: int = 0
    protected_divisions: int = 0
    probabilities: tuple[float, ...] = ()


def screen_formula(expression: str, available: Sequence[str]) -> MathReport:
    """AST, конечность, диапазон вероятности, деление, сложность и признаки."""
    if not expression_is_safe(expression):
        return MathReport(False, "unsafe_math", "выражение содержит запрещённый символ или вызов")
    try:
        tree = parse_formula(expression)
    except FormulaError as error:
        return MathReport(False, _code_for_formula(str(error)), str(error))
    if tree_depth(tree) > MAX_DEPTH or node_count(tree) > MAX_NODES:
        return MathReport(False, "too_complex", "выражение глубже или длиннее допустимого")
    division = _bad_constant_division(tree)
    if division is not None:
        return MathReport(False, "unsafe_math", division, depth=tree_depth(tree), nodes=node_count(tree))
    used = features_used(tree)
    unknown = sorted(name for name in used if name not in FEATURES)
    if unknown:
        return MathReport(False, "unknown_feature", f"неизвестный признак {unknown[0]}")
    missing = sorted(name for name in used if name not in set(available))
    if missing:
        return MathReport(False, "missing_feature", f"признак {missing[0]} недоступен в переданном наборе")
    state = EvalState()
    probabilities: list[float] = []
    try:
        for probe in _probes():
            probabilities.append(evaluate(tree, probe, state))
    except FormulaError as error:
        return MathReport(False, "unsafe_math", str(error), depth=tree_depth(tree), nodes=node_count(tree))
    return MathReport(
        True,
        "ok",
        "формула прошла независимую проверку",
        depth=tree_depth(tree),
        nodes=node_count(tree),
        protected_divisions=state.protected_divisions,
        probabilities=tuple(probabilities),
    )


def screen_score(expression: str, calibration: str, available: Sequence[str]) -> MathReport:
    """Score становится вероятностью только через заранее заданный sigmoid."""
    if calibration != EXPLICIT_SIGMOID or "2021" in calibration:
        return MathReport(
            False,
            "calibration_refused",
            "калибровка не задана заранее и не подбирается по 2021 году",
        )
    if "sigmoid" in expression:
        return screen_formula(expression, available)
    return screen_formula(f"sigmoid({expression})", available)


def score_synthetic(expression: str) -> dict[str, object]:
    """Синтетические точки. Результат не является рыночным преимуществом."""
    report = screen_formula(expression, FEATURES)
    if not report.accepted:
        raise FormulaError(report.reason)
    tree = parse_formula(expression)
    brier_sum = 0.0
    base_sum = 0.0
    observations = 40
    for index in range(observations):
        features = _synthetic_features(index)
        label = 1 if features["momentum_7"] > 0 else 0
        probability = evaluate(tree, features)
        brier_sum += (probability - label) ** 2
        base_sum += (0.5 - label) ** 2
    return {
        "period": "synthetic",
        "observations": observations,
        "brier": round(brier_sum / observations, 6),
        "constant_half_brier": round(base_sum / observations, 6),
        "claim_status": "not_confirmed",
        "validated_law": False,
        "advantage_claim": False,
        "btc_claim": False,
        "graded_by": VERIFIER_ID,
        "fitted_on_2021": False,
        "uses_data_after_2021": False,
    }


def _bad_constant_division(tree: Expr) -> str | None:
    found: list[str] = []
    _walk_division(tree, found)
    if found:
        return found[0]
    return None


def _walk_division(tree: Expr, found: list[str]) -> None:
    if isinstance(tree, Binary):
        if tree.op == "div" and isinstance(tree.right, Const) and abs(tree.right.value) < DIVISION_EPSILON:
            found.append("деление на константу меньше 1e-6")
        _walk_division(tree.left, found)
        _walk_division(tree.right, found)
    elif isinstance(tree, Unary):
        _walk_division(tree.arg, found)


def _probes() -> tuple[dict[str, float], ...]:
    zero = {name: 0.0 for name in FEATURES}
    positive = {name: 0.2 for name in FEATURES}
    negative = {name: -0.2 for name in FEATURES}
    tiny = {name: 1e-8 for name in FEATURES}
    return (zero, positive, negative, tiny)


def _synthetic_features(index: int) -> dict[str, float]:
    features: dict[str, float] = {}
    for offset, name in enumerate(FEATURES):
        features[name] = math.sin((index + 1) * (offset + 1) / 7.0) * 0.25
    features["volatility_7"] = abs(features["volatility_7"]) + 0.05
    features["volatility_30"] = abs(features["volatility_30"]) + 0.05
    return features


def _code_for_formula(message: str) -> str:
    if "признак" in message and "неизвест" in message:
        return "unknown_feature"
    if "глуб" in message or "много операций" in message or "длиннее" in message:
        return "too_complex"
    return "unsafe_math"
