"""Случайное блуждание без сноса: медиана равна последнему известному close."""

from __future__ import annotations

from wolfai.data.candles import Candle
from wolfai.forecasting.baseline import (
    DEFAULT_QUANTILES,
    Forecast,
    _quantile,
    decision_time,
    known_horizon_returns,
)


class RandomWalkBaseline:
    """Контроль без оценки дрифта.

    Точечный прогноз — последний close. Границы интервала — квантили прошлых
    доходностей того же горизонта после вычитания их медианы, перенесённые
    на этот close. Ширина берётся из истории, снос не оценивается.
    """

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        if not quantiles or any(not 0 < level < 1 for level in quantiles):
            raise ValueError("квантили должны быть в интервале (0, 1)")
        self.quantiles = tuple(sorted(set(quantiles)))

    def forecast(self, candles: list[Candle], horizon_days: int) -> Forecast:
        if horizon_days < 1:
            raise ValueError("горизонт должен быть не меньше 1 дня")
        as_of = decision_time(candles)
        closes = [candle.close for candle in candles]
        returns = known_horizon_returns(candles, horizon_days, as_of)
        if len(returns) < 2:
            raise ValueError("недостаточно истории для распределения доходностей")
        center = _quantile(returns, 0.5)
        centered = [value - center for value in returns]
        last_close = closes[-1]
        levels = {
            level: last_close * (1 + _quantile(centered, level)) for level in self.quantiles
        }
        if 0.5 in levels:
            levels[0.5] = last_close
        return Forecast(
            horizon_days=horizon_days,
            as_of=as_of,
            last_close=last_close,
            median=last_close,
            quantiles=levels,
        )
