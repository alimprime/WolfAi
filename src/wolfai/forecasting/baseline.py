"""Эмпирический квантильный baseline по историческим доходностям."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol

from wolfai.data.availability import DataAvailability, require_usable
from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import Candle

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
INTERVAL_ALPHA = 0.2


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
    directional_accuracy: float | None
    directional_calls: int
    interval_coverage: float | None
    mean_interval_width: float | None
    mean_interval_score: float | None


class ForecastModel(Protocol):
    """Модель, которая строит сценарий только по уже переданным свечам."""

    def forecast(self, candles: list[Candle], horizon_days: int) -> Forecast:
        """Прогноз на horizon_days закрытий вперёд от последнего известного close."""


class HistoricalQuantileBaseline:
    """Медиана и квантили цены из распределения прошлых доходностей горизонта."""

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
        last_close = closes[-1]
        levels = {
            level: last_close * (1 + _quantile(returns, level)) for level in self.quantiles
        }
        median = levels.get(0.5, last_close * (1 + _quantile(returns, 0.5)))
        return Forecast(
            horizon_days=horizon_days,
            as_of=as_of,
            last_close=last_close,
            median=median,
            quantiles=levels,
        )


def decision_time(candles: list[Candle]) -> datetime:
    """Момент, когда известен close последней свечи и всех более ранних.

    Список и есть информационное множество: свеча с более поздним close в него
    не входит. Метка открытия последней свечи этим моментом не является:
    close доступен в следующую полночь UTC. Шаг между открытиями должен быть
    ровно один календарный день, иначе горизонт в сутках смешивается с числом
    баров. Пропущенная свеча не восстанавливается.
    """
    if not candles:
        raise ValueError("ряд свечей пуст")
    as_of = daily_bar_available_at(candles[-1])
    previous: datetime | None = None
    visible: list[DataAvailability] = []
    for index, candle in enumerate(candles):
        available = daily_bar_available_at(candle)
        visible.append(DataAvailability(
            event_time=candle.timestamp,
            available_at=available,
            source="ohlcv",
        ))
        if available > as_of:
            raise ValueError(f"свеча {index}: close ещё не известен в момент прогноза")
        if previous is not None and candle.timestamp <= previous:
            raise ValueError(f"свеча {index}: история прогноза не упорядочена")
        if previous is not None and candle.timestamp - previous != timedelta(days=1):
            raise ValueError(f"свеча {index}: пропуск или нарушение дневного шага")
        previous = candle.timestamp
    require_usable(visible, as_of)
    return as_of


def walk_forward(
    candles: list[Candle],
    horizon_days: int,
    *,
    min_history: int = 30,
    model: ForecastModel | None = None,
    decision_from: date | None = None,
    decision_to: date | None = None,
    target_from: date | None = None,
    target_to: date | None = None,
) -> WalkForwardResult:
    """Ошибки медианы на отложенных по времени точках.

    История среза заканчивается последним известным close. Цель — close через
    horizon_days баров после него. Этот close к моменту as_of ещё не известен.
    """
    if min_history < horizon_days + 2:
        raise ValueError("min_history слишком мал для выбранного горизонта")
    baseline = model or HistoricalQuantileBaseline()
    bounded = any(
        bound is not None for bound in (decision_from, decision_to, target_from, target_to)
    )
    if bounded:
        for previous, candle in zip(candles, candles[1:]):
            if candle.timestamp <= previous.timestamp:
                raise ValueError("ряд для окна должен быть упорядочен")
    absolute_errors: list[float] = []
    percentage_errors: list[float] = []
    directional_hits = 0
    directional_calls = 0
    covered = 0
    widths: list[float] = []
    scores: list[float] = []
    intervals = True
    last_origin = len(candles) - horizon_days
    for end in range(min_history, last_origin + 1):
        history = candles[:end]
        last = history[-1]
        decision_day = last.timestamp.date()
        if decision_to is not None and decision_day > decision_to:
            break
        if not _inside(decision_day, decision_from, decision_to):
            continue
        target = candles[end - 1 + horizon_days]
        if not _inside(target.timestamp.date(), target_from, target_to):
            continue
        prediction = baseline.forecast(history, horizon_days)
        if prediction.as_of != daily_bar_available_at(last):
            raise RuntimeError("as_of не совпал со временем доступности последнего close")
        if daily_bar_available_at(target) <= prediction.as_of:
            raise RuntimeError("целевой close уже известен в момент прогноза")
        if target.timestamp <= last.timestamp:
            raise RuntimeError("цель не позже последнего известного close")
        error = abs(prediction.median - target.close)
        absolute_errors.append(error)
        percentage_errors.append(error / target.close)
        predicted_move = prediction.median - prediction.last_close
        actual_move = target.close - prediction.last_close
        if predicted_move != 0.0 and actual_move != 0.0:
            directional_calls += 1
            if (predicted_move > 0) == (actual_move > 0):
                directional_hits += 1
        lower = prediction.quantiles.get(0.1)
        upper = prediction.quantiles.get(0.9)
        if lower is None or upper is None:
            intervals = False
            continue
        if upper < lower:
            raise ValueError("верх вероятностного интервала ниже нижней границы")
        widths.append(upper - lower)
        scores.append(_interval_score(target.close, lower, upper))
        if lower <= target.close <= upper:
            covered += 1
    if not absolute_errors:
        raise ValueError("ряд короче, чем нужно для хронологической проверки")
    count = len(absolute_errors)
    coverage = None if not intervals else covered / count
    mean_width = None if not intervals else sum(widths) / count
    mean_score = None if not intervals else sum(scores) / count
    accuracy = None if directional_calls == 0 else directional_hits / directional_calls
    return WalkForwardResult(
        horizon_days=horizon_days,
        observations=count,
        mean_absolute_error=sum(absolute_errors) / count,
        mean_absolute_percentage_error=sum(percentage_errors) / count,
        directional_accuracy=accuracy,
        directional_calls=directional_calls,
        interval_coverage=coverage,
        mean_interval_width=mean_width,
        mean_interval_score=mean_score,
    )


def _inside(day: date, start: date | None, end: date | None) -> bool:
    if start is not None and day < start:
        return False
    if end is not None and day > end:
        return False
    return True


def _interval_score(actual: float, lower: float, upper: float) -> float:
    """Winkler score центрального интервала 10–90% (alpha = 0.2)."""
    score = upper - lower
    penalty = 2 / INTERVAL_ALPHA
    if actual < lower:
        return score + penalty * (lower - actual)
    if actual > upper:
        return score + penalty * (actual - upper)
    return score


def known_horizon_returns(
    candles: list[Candle],
    horizon_days: int,
    as_of: datetime,
) -> list[float]:
    """Доходности горизонта, чей конечный close уже известен к as_of.

    Пара close[i + h] / close[i] - 1 входит в распределение только если оба
    close доступны не позже as_of и между открытиями ровно h календарных суток.
    Более поздний close пропускается. Дыра внутри уже известного окна не
    заменяется соседней свечой.
    """
    if horizon_days < 1:
        raise ValueError("горизонт должен быть не меньше 1 дня")
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of без часового пояса")
    returns: list[float] = []
    for index in range(len(candles) - horizon_days):
        start = candles[index]
        end = candles[index + horizon_days]
        if daily_bar_available_at(end) > as_of or daily_bar_available_at(start) > as_of:
            continue
        if end.timestamp - start.timestamp != timedelta(days=horizon_days):
            raise ValueError(
                f"свеча {index + horizon_days}: пропуск или нарушение дневного шага"
            )
        returns.append(end.close / start.close - 1)
    return returns


def _quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = level * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
