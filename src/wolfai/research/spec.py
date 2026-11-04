"""Общий контракт гипотезы. Текст источника не исполняется.

Qwen, PySR и Optuna отдают одно и то же представление. Рабочий статус
`active` отсюда не назначается. Выражение обязано пройти `parse_formula`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wolfai.research.formula import FormulaError, features_used, parse_formula
from wolfai.research.formula_edit import list_coefficients

DISCOVERY_VERSION = "2.3.0"
ORIGIN_QWEN = "qwen"
ORIGIN_PYSR = "pysr"
ORIGIN_OPTUNA = "optuna"
ORIGINS = frozenset({ORIGIN_QWEN, ORIGIN_PYSR, ORIGIN_OPTUNA})


@dataclass(frozen=True)
class ResearchHypothesis:
    """Гипотеза с происхождением. В лабораторию уходит со статусом proposed."""

    id: str
    origin: str
    expression: str
    features: tuple[str, ...]
    horizon_days: int
    parameters: tuple[float, ...]
    dataset_version: str
    created_at: datetime
    source_expression: str | None = None
    status: str = "proposed"


def as_engine(hypothesis: ResearchHypothesis):
    """Гипотеза лаборатории 2.1. Статус всегда proposed."""
    from wolfai.research.engine import Hypothesis

    return Hypothesis(
        id=hypothesis.id,
        expression=hypothesis.expression,
        features=hypothesis.features,
        horizon_days=hypothesis.horizon_days,
        created_at=hypothesis.created_at,
        dataset_version=hypothesis.dataset_version,
        status="proposed",
    )


def bind_hypothesis(
    *,
    id: str,
    origin: str,
    expression: str,
    features: tuple[str, ...] | list[str],
    horizon_days: int,
    dataset_version: str,
    created_at: datetime,
    parameters: tuple[float, ...] | None = None,
    source_expression: str | None = None,
    status: str = "proposed",
) -> ResearchHypothesis:
    if status == "active":
        raise ValueError("рабочая модель не назначается источником гипотезы")
    if origin not in ORIGINS:
        raise ValueError("неизвестное происхождение гипотезы")
    if not str(id).strip():
        raise ValueError("id гипотезы пуст")
    if not str(dataset_version).strip():
        raise ValueError("версия датасета пуста")
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    names = tuple(features)
    tree = parse_formula(expression)
    if features_used(tree) != set(names):
        raise FormulaError("набор признаков в формуле не совпадает с полями features")
    found = list_coefficients(tree)
    if parameters is None:
        parameters = found
    elif len(parameters) != len(found) or any(
        abs(left - right) > 1e-4 for left, right in zip(parameters, found, strict=True)
    ):
        raise FormulaError("параметры не совпадают с коэффициентами выражения")
    return ResearchHypothesis(
        id=str(id).strip(),
        origin=origin,
        expression=expression,
        features=names,
        horizon_days=horizon_days,
        parameters=tuple(round(value, 4) for value in parameters),
        dataset_version=str(dataset_version),
        created_at=created_at,
        source_expression=source_expression,
        status="proposed",
    )
