"""Единый вход дневного ряда в признаки и исследование.

Любой расчёт по дневным свечам проходит здесь: непрерывные сутки UTC,
известное available_at и граница holdout. Пропуск не заполняется.
Неизвестное время доступности не заменяется временем события.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from wolfai.data.availability import DataAvailability, candle_availability, require_usable
from wolfai.data.binance import validate_daily_candles
from wolfai.data.candles import Candle
from wolfai.forecasting.splits import Period, period_bounds

HOLDOUT_START = period_bounds(Period.HOLDOUT)[0]


def admit_observations(
    records: Sequence[DataAvailability],
    as_of: datetime,
) -> list[DataAvailability]:
    """Записи, которые уже можно использовать в as_of.

    Пустой available_at остаётся причиной отказа. event_time его не заполняет.
    """
    return require_usable(records, as_of)


def admit_daily_research(candles: list[Candle]) -> list[Candle]:
    """Проверяет ряд до того, как по нему считают признаки.

    Свеча holdout отклоняет весь переданный список: её close в расчёт не входит.
    """
    if not candles:
        raise ValueError("исследовательский ряд пуст")
    for index, candle in enumerate(candles):
        if candle.timestamp.tzinfo is None or candle.timestamp.utcoffset() is None:
            raise ValueError(f"свеча {index}: timestamp без часового пояса")
        # Дата holdout проверяется по UTC до разбора цен этой свечи.
        utc_day = candle.timestamp.astimezone(timezone.utc).date()
        if utc_day >= HOLDOUT_START:
            raise ValueError(
                f"свеча {index}: дата {utc_day.isoformat()} попадает в holdout"
            )
    validate_daily_candles(candles)
    records = [candle_availability(candle, "ohlcv") for candle in candles]
    as_of = records[-1].available_at
    if as_of is None:
        raise ValueError("available_at неизвестен; event_time не заменяет время публикации")
    admit_observations(records, as_of)
    return candles


def require_calendar_days(candles: list[Candle], start: int, end: int, days: int) -> None:
    """Интервал из `days` баров должен занимать ровно `days` суток UTC."""
    if days < 1 or start < 0 or end >= len(candles) or end <= start:
        raise ValueError("интервал признаков выходит за ряд")
    span = candles[end].timestamp - candles[start].timestamp
    if end - start != days or span != timedelta(days=days):
        raise ValueError(
            f"интервал из {end - start} баров не равен {days} календарным суткам"
        )
