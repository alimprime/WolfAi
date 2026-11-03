"""Хронологические периоды. Границы заданы до расчёта метрик."""

from __future__ import annotations

from datetime import date
from enum import Enum


class Period(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


# Holdout не участвует в метриках, подборе формул и выборе модели.
PERIOD_BOUNDS: dict[Period, tuple[date, date]] = {
    Period.TRAINING: (date(2019, 1, 1), date(2022, 12, 31)),
    Period.VALIDATION: (date(2023, 1, 1), date(2024, 12, 31)),
    Period.HOLDOUT: (date(2025, 1, 1), date(2026, 9, 30)),
}

EVALUATED_PERIODS: tuple[Period, ...] = (Period.TRAINING, Period.VALIDATION)

# Внутренние границы поиска формул. Они лежат внутри training и не используют 2023–2024.
COEFFICIENT_FIT_BOUNDS: tuple[date, date] = (date(2019, 1, 1), date(2021, 12, 31))
INNER_CHECK_BOUNDS: tuple[date, date] = (date(2022, 1, 1), date(2022, 12, 31))


def period_bounds(period: Period) -> tuple[date, date]:
    return PERIOD_BOUNDS[period]


def refuse_holdout(period: Period) -> None:
    """Останавливает расчёт, если период — закрытый тест."""
    if period is Period.HOLDOUT:
        start, end = period_bounds(period)
        raise ValueError(
            "holdout "
            f"{start.isoformat()}..{end.isoformat()} закрыт: "
            "метрики, подбор формул и выбор модели на нём не считаются"
        )
