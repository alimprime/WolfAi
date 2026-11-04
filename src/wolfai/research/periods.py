"""Окна поиска. Holdout закрыт, финальная валидация не входит в подбор.

2019–2020 — подбор. 2021 — внутренняя хронологическая проверка.
2022 — ограниченная обратная связь и не независимый тест.
2023–2024 считает только замороженный кандидат. 2025–2026 не читается.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

from wolfai.data.gateway import HOLDOUT_START
from wolfai.forecasting.splits import INNER_CHECK_BOUNDS, Period, period_bounds
from wolfai.research.engine import ExperimentRow

SEARCH_TRAIN = (date(2019, 1, 1), date(2020, 12, 31))
SEARCH_CHECK = (date(2021, 1, 1), date(2021, 12, 31))
FEEDBACK = INNER_CHECK_BOUNDS
FINAL = period_bounds(Period.VALIDATION)
FEEDBACK_NOTE = "2022 уже использовался ранее и не является независимым тестом"


def chronological_rows(
    rows: Sequence[ExperimentRow],
    start: date,
    end: date,
    horizon: int,
) -> list[ExperimentRow]:
    """Точки, чьи решение и уже опубликованная метка лежат внутри окна.

    Для горизонта 7 дней шаг равен горизонту, а цель, которая публикуется
    уже в следующем окне, в выборку не входит.
    """
    if horizon not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    if start >= HOLDOUT_START or end >= HOLDOUT_START:
        raise ValueError("holdout закрыт")
    chosen: list[ExperimentRow] = []
    next_day: date | None = None
    for row in rows:
        if row.decision_date < start or row.decision_date > end:
            continue
        if row.target_date < start or row.target_date > end:
            continue
        if row.label_available > end:
            continue
        if row.label_available <= row.decision_available:
            raise ValueError("метка доступна не позже решения")
        if next_day is not None and row.decision_date < next_day:
            continue
        chosen.append(row)
        next_day = row.decision_date + timedelta(days=horizon)
    return chosen


def assert_search_window(start: date, end: date) -> None:
    """Подбор коэффициентов не смотрит на 2022, 2023–2024 и holdout."""
    if (start, end) not in {SEARCH_TRAIN, SEARCH_CHECK}:
        raise ValueError("подбор разрешён только на внутренних окнах 2019–2021")
