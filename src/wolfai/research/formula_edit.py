"""Сборка и разбор дерева формулы без изменения грамматики."""

from __future__ import annotations

import math

from wolfai.research.formula import (
    MAX_COEFFICIENT,
    Binary,
    Const,
    Expr,
    Feature,
    FormulaError,
    Unary,
)


def list_coefficients(tree: Expr) -> tuple[float, ...]:
    found: list[float] = []
    _collect(tree, found)
    return tuple(found)


def replace_coefficients(tree: Expr, values: tuple[float, ...]) -> Expr:
    index = 0

    def walk(node: Expr) -> Expr:
        nonlocal index
        if isinstance(node, Unary) and node.op == "neg" and isinstance(node.arg, Const):
            value = _take(values, index)
            index += 1
            return _number(value)
        if isinstance(node, Const):
            value = _take(values, index)
            index += 1
            return _number(value)
        if isinstance(node, Feature):
            return node
        if isinstance(node, Unary):
            return Unary(node.op, walk(node.arg))
        return Binary(node.op, walk(node.left), walk(node.right))

    rebuilt = walk(tree)
    if index != len(values):
        raise FormulaError("число коэффициентов не совпадает с выражением")
    return rebuilt


def skeleton(tree: Expr) -> tuple[object, ...]:
    """Структура без числовых значений. Знак коэффициента в структуру не входит."""
    if isinstance(tree, Unary) and tree.op == "neg" and isinstance(tree.arg, Const):
        return ("C",)
    if isinstance(tree, Const):
        return ("C",)
    if isinstance(tree, Feature):
        return ("F", tree.name)
    if isinstance(tree, Unary):
        return ("U", tree.op, skeleton(tree.arg))
    return ("B", tree.op, skeleton(tree.left), skeleton(tree.right))


def render(tree: Expr) -> str:
    if isinstance(tree, Const):
        return _format_positive(tree.value)
    if isinstance(tree, Feature):
        return tree.name
    if isinstance(tree, Unary):
        if tree.op == "neg":
            return f"-{_render_factor(tree.arg)}"
        return f"{tree.op}({render(tree.arg)})"
    symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[tree.op]
    return f"({render(tree.left)} {symbol} {render(tree.right)})"


def _collect(tree: Expr, found: list[float]) -> None:
    if isinstance(tree, Unary) and tree.op == "neg" and isinstance(tree.arg, Const):
        found.append(-tree.arg.value)
        return
    if isinstance(tree, Const):
        found.append(tree.value)
        return
    if isinstance(tree, Feature):
        return
    if isinstance(tree, Unary):
        _collect(tree.arg, found)
        return
    _collect(tree.left, found)
    _collect(tree.right, found)


def _take(values: tuple[float, ...], index: int) -> float:
    if index >= len(values):
        raise FormulaError("число коэффициентов не совпадает с выражением")
    return values[index]


def _number(value: float) -> Expr:
    if not math.isfinite(value):
        raise FormulaError("коэффициент не является конечным числом")
    rounded = round(float(value), 4)
    if abs(rounded) > MAX_COEFFICIENT:
        raise FormulaError("коэффициент вне [-10, 10]")
    if rounded < 0:
        return Unary("neg", Const(-rounded))
    return Const(rounded)


def _format_positive(value: float) -> str:
    if value < 0:
        raise FormulaError("внутренний коэффициент отрицателен")
    rounded = round(value, 4)
    if rounded > MAX_COEFFICIENT:
        raise FormulaError("коэффициент вне [-10, 10]")
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _render_factor(tree: Expr) -> str:
    if isinstance(tree, Binary):
        return f"({render(tree)})"
    return render(tree)
