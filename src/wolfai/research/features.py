"""Признаки только из свечей, уже известных на момент решения."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from wolfai.data.binance import validate_daily_candles
from wolfai.data.candles import Candle, parse_utc_timestamp
from wolfai.data.gateway import admit_daily_research, require_calendar_days
from wolfai.forecasting.splits import Period, period_bounds, refuse_holdout

LOOKBACK = 30
MIN_STANDARDIZE = 60
STANDARDIZE_CLIP = 8.0
STD_FLOOR = 1e-8
HOLDOUT_START = period_bounds(Period.HOLDOUT)[0]
FEATURE_NAMES = (
    "log_return",
    "momentum_7",
    "momentum_30",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "range_position",
)
FEATURE_SCHEMA_VERSION = "ohlcv-daily-v1"


@dataclass(frozen=True)
class FeatureRow:
    """Одна точка решения. y_up — метка следующего close, не признак."""

    index: int
    decision_date: date
    target_date: date
    features: dict[str, float]
    y_up: int
    baseline_up_frequency: float


def load_research_candles(path: str | Path) -> list[Candle]:
    """Читает CSV до первой свечи holdout и не разбирает её цены."""
    file_path = Path(path)
    candles: list[Candle] = []
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "timestamp" not in reader.fieldnames:
            raise ValueError(f"{file_path}: нет колонки timestamp")
        for row in reader:
            timestamp, _shifted = parse_utc_timestamp((row.get("timestamp") or "").strip())
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"свеча {len(candles)}: timestamp без часового пояса")
            if timestamp.date() >= HOLDOUT_START:
                break
            candles.append(Candle(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    assert_research_series(candles)
    validate_daily_candles(candles)
    return candles


def assert_research_series(candles: list[Candle]) -> None:
    """Ряд исследования непрерывен по суткам UTC и не содержит holdout."""
    admit_daily_research(candles)


def feature_vector(
    candles: list[Candle],
    index: int,
    *,
    _admitted: bool = False,
) -> dict[str, float]:
    """Признаки бара index. Свечи после него в расчёт не входят.

    Прямой вызов сам проводит ряд через общий вход. Внутренний проход
    `build_rows` передаёт `_admitted` только после этой проверки.
    """
    if not _admitted:
        admit_daily_research(candles)
    return features_from_closed_window(candles, index)


def features_from_closed_window(candles: list[Candle], index: int) -> dict[str, float]:
    """Признаки бара index. Свечи после него в расчёт не входят.

    Holdout здесь не открывается и не проверяется: исследовательский вход
    по-прежнему `feature_vector`. У этого вызова нет метки будущего close.
    """
    if index < LOOKBACK:
        raise ValueError("недостаточно истории для признаков")
    require_calendar_days(candles, index - 1, index, 1)
    require_calendar_days(candles, index - 7, index, 7)
    require_calendar_days(candles, index - 30, index, 30)
    history = candles[: index + 1]
    closes = [candle.close for candle in history]
    latest = closes[-1]
    previous = closes[-2]
    log_returns = [
        math.log(closes[cursor] / closes[cursor - 1]) for cursor in range(1, len(closes))
    ]
    low = min(candle.low for candle in history[-LOOKBACK:])
    high = max(candle.high for candle in history[-LOOKBACK:])
    span = high - low
    volume_previous = history[-2].volume
    if volume_previous > 0:
        volume_change = history[-1].volume / volume_previous - 1
    else:
        volume_change = 0.0
    return {
        "log_return": log_returns[-1],
        "momentum_7": latest / closes[-8] - 1,
        "momentum_30": latest / closes[-31] - 1,
        "volatility_7": _stdev(log_returns[-7:]),
        "volatility_30": _stdev(log_returns[-30:]),
        "volume_change": volume_change,
        "range_position": 0.5 if span == 0 else (latest - low) / span,
    }


def direction_label(candles: list[Candle], index: int, *, _admitted: bool = False) -> int:
    """1, если следующий close выше последнего известного."""
    if not _admitted:
        admit_daily_research(candles)
    if index < 0 or index + 1 >= len(candles):
        raise ValueError("нет следующего close")
    require_calendar_days(candles, index, index + 1, 1)
    return 1 if candles[index + 1].close > candles[index].close else 0


def historical_up_frequency(
    candles: list[Candle],
    index: int,
    *,
    _admitted: bool = False,
) -> float:
    """Доля уже известных однодневных ростов. Это вероятность знака baseline."""
    if not _admitted:
        admit_daily_research(candles)
    history = candles[: index + 1]
    ups = sum(history[cursor].close > history[cursor - 1].close for cursor in range(1, len(history)))
    return ups / (len(history) - 1)


def build_rows(candles: list[Candle]) -> list[FeatureRow]:
    assert_research_series(candles)
    rows: list[FeatureRow] = []
    for index in range(LOOKBACK, len(candles) - 1):
        rows.append(FeatureRow(
            index=index,
            decision_date=candles[index].timestamp.date(),
            target_date=candles[index + 1].timestamp.date(),
            features=feature_vector(candles, index, _admitted=True),
            y_up=direction_label(candles, index, _admitted=True),
            baseline_up_frequency=historical_up_frequency(candles, index, _admitted=True),
        ))
    return rows


@dataclass(frozen=True)
class StandardizedRow:
    """Признаки, нормированные только по строкам строго раньше этой."""

    decision_date: date
    target_date: date
    decision_available: date
    label_available: date
    y_up: int
    raw: dict[str, float]
    standardized: dict[str, float]
    baseline_up_frequency: float


def standardize_rows(rows: list[FeatureRow]) -> list[StandardizedRow]:
    """z-оценка по прошлой истории. Будущие строки параметры не двигают."""
    counts = {name: 0 for name in FEATURE_NAMES}
    means = {name: 0.0 for name in FEATURE_NAMES}
    squares = {name: 0.0 for name in FEATURE_NAMES}
    standardized_rows: list[StandardizedRow] = []
    for row in rows:
        if min(counts.values()) >= MIN_STANDARDIZE:
            standardized_rows.append(StandardizedRow(
                decision_date=row.decision_date,
                target_date=row.target_date,
                decision_available=row.decision_date + timedelta(days=1),
                label_available=row.target_date + timedelta(days=1),
                y_up=row.y_up,
                raw=dict(row.features),
                standardized=_standardize_one(row.features, counts, means, squares),
                baseline_up_frequency=row.baseline_up_frequency,
            ))
        _update_moments(row.features, counts, means, squares)
    for item in standardized_rows:
        if item.label_available <= item.decision_available:
            raise ValueError("метка созревает не позже признаков")
        if item.target_date >= HOLDOUT_START or item.decision_date >= HOLDOUT_START:
            raise ValueError("стандартизация получила дату holdout")
    return standardized_rows


def rows_between(rows: list[StandardizedRow], start: date, end: date) -> list[StandardizedRow]:
    """Решения и уже созревшие метки внутри окна."""
    if start >= HOLDOUT_START or end >= HOLDOUT_START:
        raise ValueError("окно захватывает holdout")
    return [
        row for row in rows
        if start <= row.decision_date <= end and start <= row.target_date <= end
    ]


def rows_for_period(rows: list[FeatureRow], period: Period) -> list[FeatureRow]:
    refuse_holdout(period)
    start, end = period_bounds(period)
    return [
        row for row in rows
        if start <= row.decision_date <= end and start <= row.target_date <= end
    ]


def _standardize_one(
    features: dict[str, float],
    counts: dict[str, int],
    means: dict[str, float],
    squares: dict[str, float],
) -> dict[str, float]:
    scaled: dict[str, float] = {}
    for name in FEATURE_NAMES:
        value = features.get(name, 0.0)
        count = counts[name]
        if not math.isfinite(value) or count < 2:
            scaled[name] = 0.0
            continue
        variance = squares[name] / (count - 1)
        deviation = math.sqrt(variance) if variance > 0 else 0.0
        if deviation < STD_FLOOR:
            standardized = 0.0
        else:
            standardized = (value - means[name]) / deviation
        if not math.isfinite(standardized):
            standardized = 0.0
        scaled[name] = max(-STANDARDIZE_CLIP, min(STANDARDIZE_CLIP, standardized))
    return scaled


def _update_moments(
    features: dict[str, float],
    counts: dict[str, int],
    means: dict[str, float],
    squares: dict[str, float],
) -> None:
    for name in FEATURE_NAMES:
        value = features.get(name, 0.0)
        if not math.isfinite(value):
            continue
        counts[name] += 1
        delta = value - means[name]
        means[name] += delta / counts[name]
        squares[name] += delta * (value - means[name])


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("для волатильности нужны хотя бы два наблюдения")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
