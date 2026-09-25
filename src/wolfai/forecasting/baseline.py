"""Эмпирический квантильный baseline по историческим доходностям."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wolfai.data.candles import Candle

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class Forecast:
    """Сценарии цены. Это оценка по прошлому ряду, не гарантия будущей цены."""

    horizon_days: int
    as_of: datetime
    last_close: float
    median: float
    quantiles: dict[float, float]


@dataclass(frozen=True)
class WalkForwardResult:
    horizon_days: int
    observations: int
    mean_absolute_error: float
    mean_absolute_percentage_error: float


class HistoricalQuantileBaseline:
    """Медиана и квантили цены из распределения прошлых доходностей горизонта."""

    def __init__(self, quantiles: tuple[float, ...] = DEFAULT_QUANTILES) -> None:
        if not quantiles or any(not 0 < level < 1 for level in quantiles):
            raise ValueError("квантили должны быть в интервале (0, 1)")
        self.quantiles = tuple(sorted(set(quantiles)))

    def forecast(self, candles: list[Candle], horizon_days: int) -> Forecast:
        if horizon_days < 1:
            raise ValueError("горизонт должен быть не меньше 1 дня")
        closes = [candle.close for candle in candles]
        returns = _horizon_returns(closes, horizon_days)
        if len(returns) < 2:
            raise ValueError("недостаточно истории для распределения доходностей")
        last_close = closes[-1]
        levels = {level: last_close * (1 + _quantile(returns, level)) for level in self.quantiles}
        median = levels.get(0.5, last_close * (1 + _quantile(returns, 0.5)))
        return Forecast(
            horizon_days=horizon_days,
            as_of=candles[-1].timestamp,
            last_close=last_close,
            median=median,
            quantiles=levels,
        )


def walk_forward(
    candles: list[Candle],
    horizon_days: int,
    *,
    min_history: int = 30,
    model: HistoricalQuantileBaseline | None = None,
) -> WalkForwardResult:
    """Ошибки медианы на отложенных по времени точках. Будущее в обучение не входит."""
    if min_history < horizon_days + 2:
        raise ValueError("min_history слишком мал для выбранного горизонта")
    baseline = model or HistoricalQuantileBaseline()
    errors: list[float] = []
    percentage_errors: list[float] = []
    for end in range(min_history, len(candles) - horizon_days):
        prediction = baseline.forecast(candles[:end], horizon_days)
        actual = candles[end + horizon_days - 1].close
        errors.append(abs(prediction.median - actual))
        percentage_errors.append(abs(prediction.median - actual) / actual)
    if not errors:
        raise ValueError("ряд короче, чем нужно для хронологической проверки")
    return WalkForwardResult(
        horizon_days=horizon_days,
        observations=len(errors),
        mean_absolute_error=sum(errors) / len(errors),
        mean_absolute_percentage_error=sum(percentage_errors) / len(percentage_errors),
    )


def _horizon_returns(closes: list[float], horizon_days: int) -> list[float]:
    return [
        closes[index + horizon_days] / closes[index] - 1
        for index in range(len(closes) - horizon_days)
    ]


def _quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = level * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
