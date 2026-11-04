"""Отбор формул при том же бюджете PySR.

Сначала простые выражения, затем более длинные. Итерации, популяции
и число процессов не увеличиваются. Текст формулы не исполняется.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wolfai.research.formula import (
    DIVISION_EPSILON,
    MAX_COEFFICIENT,
    Binary,
    Const,
    Expr,
    Feature,
    FormulaError,
    Unary,
    features_used,
    node_count,
    parse_formula,
)
from wolfai.research.formula_edit import render, skeleton
from wolfai.research.spec import ORIGIN_PYSR, ResearchHypothesis, bind_hypothesis
from wolfai.research.translate import equation_to_expression

STRATEGY_BASELINE = "baseline"
STRATEGY_IMPROVED = "improved"
STRATEGIES = frozenset({STRATEGY_BASELINE, STRATEGY_IMPROVED})
WARMUP_MAXSIZE_BY = 0.5
POPULATION_SLOTS_NOTE = (
    "K для PySR — максимум из числа возвращённых уравнений и произведения "
    "итераций, популяций и размера популяции. Это занятые слоты поиска, "
    "не число независимых испытаний и не порог значимости."
)


@dataclass(frozen=True)
class ScoredExpression:
    expression: str
    complexity: int
    loss: float


@dataclass(frozen=True)
class RejectedFormula:
    equation: str
    reason: str


@dataclass(frozen=True)
class SearchAssembly:
    hypotheses: tuple[ResearchHypothesis, ...]
    rejected: tuple[RejectedFormula, ...]
    unique_structures: int
    screened: int
    pareto_size: int
    examined: int
    selector: str


def extra_regressor_options() -> dict[str, object]:
    """Ограничения поверх лёгкого бюджета. Число итераций сюда не входит."""
    return {
        "warmup_maxsize_by": WARMUP_MAXSIZE_BY,
        "constraints": {"/": (-1, 3), "tanh": 6, "abs": 6},
        "nested_constraints": {
            "tanh": {"tanh": 0, "abs": 0},
            "abs": {"abs": 0, "tanh": 1},
            "/": {"/": 1},
        },
    }


def examination_count(returned_equations: int) -> int:
    """Сколько гипотез учитывать после символического поиска."""
    from wolfai.research.pysr_engine import LIGHT_LIMITS

    if returned_equations < 0:
        raise ValueError("число уравнений не может быть отрицательным")
    slots = (
        int(LIGHT_LIMITS["niterations"])
        * int(LIGHT_LIMITS["populations"])
        * int(LIGHT_LIMITS["population_size"])
    )
    return max(returned_equations, slots)


def canonical_expression(expression: str) -> str:
    """Одинаковый текст для перестановок и нулевых слагаемых.

    Коэффициенты округляются до 0.0001, как при привязке гипотезы.
    Деление на число меньше 1e-6 отклоняется до вычисления.
    """
    normalized = normalize(parse_formula(expression))
    rendered = render(normalized)
    parse_formula(rendered)
    return rendered


def normalize(node: Expr) -> Expr:
    if isinstance(node, Feature):
        return node
    if isinstance(node, Const):
        return _const_node(node.value)
    if isinstance(node, Unary):
        argument = normalize(node.arg)
        if node.op == "neg":
            return _negate(argument)
        return Unary(node.op, argument)
    left = normalize(node.left)
    right = normalize(node.right)
    if node.op == "sub":
        return _add(left, _negate(right))
    if node.op == "add":
        return _add(left, right)
    if node.op == "mul":
        return _mul(left, right)
    if node.op == "div":
        return _div(left, right)
    raise FormulaError(f"неизвестная операция {node.op}")


def pareto_front(items: Sequence[ScoredExpression]) -> tuple[ScoredExpression, ...]:
    """Недоминируемые точки: меньше сложность и меньше потеря — лучше."""
    front: list[ScoredExpression] = []
    for item in items:
        if any(_dominates(other, item) for other in items if other is not item):
            continue
        if any(existing.expression == item.expression for existing in front):
            continue
        front.append(item)
    front.sort(key=lambda item: (item.complexity, item.loss, item.expression))
    return tuple(front)


def screen_equations(
    equations: Sequence[str],
    losses: Sequence[float] | None,
    *,
    feature_names: tuple[str, ...],
    horizon_days: int,
    dataset_version: str,
    created_at: datetime,
    max_candidates: int,
) -> SearchAssembly:
    """Канонический вид, Парето и грамматика. Отклонённый текст не исполняется."""
    if max_candidates < 1:
        raise ValueError("нужен хотя бы один кандидат")
    if losses is not None and len(losses) != len(equations):
        raise ValueError("потери должны совпадать с уравнениями")
    rejected: list[RejectedFormula] = []
    scored: list[ScoredExpression] = []
    seen: set[str] = set()
    for index, equation in enumerate(equations):
        try:
            rendered = canonical_expression(equation_to_expression(equation, feature_names))
        except FormulaError as error:
            rejected.append(RejectedFormula(equation, str(error)[:200]))
            continue
        if rendered in seen:
            rejected.append(RejectedFormula(equation, "математически та же формула"))
            continue
        seen.add(rendered)
        loss = 0.0 if losses is None else float(losses[index])
        scored.append(ScoredExpression(rendered, node_count(parse_formula(rendered)), loss))
    chosen = pareto_front(scored) if losses is not None else tuple(
        sorted(scored, key=lambda item: (item.complexity, item.expression))
    )
    limited = chosen[:max_candidates]
    hypotheses: list[ResearchHypothesis] = []
    for index, item in enumerate(limited):
        tree = parse_formula(item.expression)
        hypotheses.append(bind_hypothesis(
            id=f"pysr-{index}",
            origin=ORIGIN_PYSR,
            expression=item.expression,
            features=tuple(sorted(features_used(tree))),
            horizon_days=horizon_days,
            dataset_version=dataset_version,
            created_at=created_at,
            source_expression=item.expression,
        ))
    structures = {skeleton(parse_formula(item.expression)) for item in scored}
    return SearchAssembly(
        hypotheses=tuple(hypotheses),
        rejected=tuple(rejected),
        unique_structures=len(structures),
        screened=len(scored),
        pareto_size=len(chosen),
        examined=len(equations),
        selector="pareto" if losses is not None else "complexity",
    )


def _dominates(left: ScoredExpression, right: ScoredExpression) -> bool:
    if left.expression == right.expression:
        return False
    not_worse = left.complexity <= right.complexity and left.loss <= right.loss
    strictly_better = left.complexity < right.complexity or left.loss < right.loss
    return not_worse and strictly_better


def _add(left: Expr, right: Expr) -> Expr:
    parts = _flatten(left, "add") + _flatten(right, "add")
    constant = 0.0
    others: list[Expr] = []
    for part in parts:
        value = _const_value(part)
        if value is not None:
            constant += value
        else:
            others.append(part)
    if abs(constant) > 1e-12:
        others.append(_const_node(constant))
    if not others:
        return _const_node(0.0)
    others.sort(key=render)
    return _fold("add", others)


def _mul(left: Expr, right: Expr) -> Expr:
    parts = _flatten(left, "mul") + _flatten(right, "mul")
    constant = 1.0
    others: list[Expr] = []
    for part in parts:
        value = _const_value(part)
        if value is not None:
            constant *= value
        else:
            others.append(part)
    if abs(constant) <= 1e-12:
        return _const_node(0.0)
    if abs(constant - 1.0) > 1e-12:
        others.append(_const_node(constant))
    if not others:
        return _const_node(constant)
    others.sort(key=render)
    return _fold("mul", others)


def _div(left: Expr, right: Expr) -> Expr:
    denominator = _const_value(right)
    numerator = _const_value(left)
    if denominator is not None and abs(denominator) < DIVISION_EPSILON:
        raise FormulaError("деление на слишком малое число")
    if denominator is not None and numerator is not None:
        return _const_node(numerator / denominator)
    if denominator is not None and abs(denominator - 1.0) <= 1e-12:
        return left
    return Binary("div", left, right)


def _flatten(node: Expr, op: str) -> list[Expr]:
    if isinstance(node, Binary) and node.op == op:
        return _flatten(node.left, op) + _flatten(node.right, op)
    return [node]


def _fold(op: str, nodes: Sequence[Expr]) -> Expr:
    """Сбалансированное дерево. Левая цепочка увеличивала бы глубину сверх грамматики."""
    items = list(nodes)
    if len(items) == 1:
        return items[0]
    while len(items) > 1:
        paired: list[Expr] = []
        index = 0
        while index < len(items):
            if index + 1 == len(items):
                paired.append(items[index])
            else:
                paired.append(Binary(op, items[index], items[index + 1]))
            index += 2
        items = paired
    return items[0]


def _negate(node: Expr) -> Expr:
    if isinstance(node, Unary) and node.op == "neg":
        return node.arg
    value = _const_value(node)
    if value is not None:
        return _const_node(-value)
    return Unary("neg", node)


def _const_value(node: Expr) -> float | None:
    if isinstance(node, Const):
        return node.value
    if isinstance(node, Unary) and node.op == "neg" and isinstance(node.arg, Const):
        return -node.arg.value
    return None


def _const_node(value: float) -> Expr:
    if not math.isfinite(value):
        raise FormulaError("коэффициент не является конечным числом")
    rounded = round(float(value), 4)
    if abs(rounded) > MAX_COEFFICIENT:
        raise FormulaError("коэффициент вне [-10, 10]")
    if rounded < 0:
        return Unary("neg", Const(-rounded))
    return Const(rounded if rounded != 0 else 0.0)
