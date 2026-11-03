"""Общая матрица признаков. Пропуск остаётся None и не становится нулём."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import Candle
from wolfai.intelligence.align import SeriesIndex
from wolfai.intelligence.institutions.etf import ETF_SERIES, etf_value
from wolfai.intelligence.market.derivatives import FUNDING_SERIES, OI_SERIES
from wolfai.intelligence.schema import Observation
from wolfai.research.features import (
    FEATURE_NAMES,
    MIN_STANDARDIZE,
    STANDARDIZE_CLIP,
    STD_FLOOR,
    FeatureRow,
    assert_research_series,
    build_rows,
)

PRICE_FEATURES = FEATURE_NAMES
DERIVATIVE_FEATURES = ("funding_rate", "oi_log_change_1d")
MACRO_ETF_FEATURES = ("effr", "cpi_index", "vix", "etf_net_flow_usd_million")
MACRO_SERIES = {
    "effr": "nyfed.effr",
    "cpi_index": "fred.CPIAUCSL",
    "vix": "fred.VIXCLS",
}
FUNDING_MAX_AGE = timedelta(hours=24)
OI_GAP_MIN = timedelta(hours=18)
OI_GAP_MAX = timedelta(hours=30)


@dataclass(frozen=True)
class MarketRow:
    """Точка решения. outcome_* не входит в словарь raw."""

    decision_date: date
    target_date: date
    as_of: datetime
    y_up: int
    baseline_up_frequency: float
    raw: dict[str, float | None]
    outcome_1d: float | None
    outcome_7d: float | None
    outcome_1d_available_at: datetime | None
    outcome_7d_available_at: datetime | None


@dataclass(frozen=True)
class ScaledRow:
    decision_date: date
    target_date: date
    as_of: datetime
    y_up: int
    baseline_up_frequency: float
    raw: dict[str, float | None]
    standardized: dict[str, float]


def build_market_rows(candles: list[Candle], observations: list[Observation]) -> list[MarketRow]:
    """Признаки цены и только строгие внешние значения, видимые на as_of."""
    assert_research_series(candles)
    base_rows = build_rows(candles)
    index = SeriesIndex(observations)
    rows: list[MarketRow] = []
    for base in base_rows:
        as_of = daily_bar_available_at(candles[base.index])
        raw = _external(index, as_of, base)
        outcome_1d, outcome_1d_at = _outcome(candles, base.index, 1)
        outcome_7d, outcome_7d_at = _outcome(candles, base.index, 7)
        rows.append(MarketRow(
            decision_date=base.decision_date,
            target_date=base.target_date,
            as_of=as_of,
            y_up=base.y_up,
            baseline_up_frequency=base.baseline_up_frequency,
            raw=raw,
            outcome_1d=outcome_1d,
            outcome_7d=outcome_7d,
            outcome_1d_available_at=outcome_1d_at,
            outcome_7d_available_at=outcome_7d_at,
        ))
    _reject_holdout(rows)
    return rows


def finite_feature_names(rows: list[MarketRow], names: tuple[str, ...]) -> tuple[str, ...]:
    """Имена, у которых есть хотя бы одно наблюдение. Пустая колонка не включается."""
    return tuple(name for name in names if any(_finite(row.raw.get(name)) for row in rows))


def rows_with_features(rows: list[MarketRow], names: tuple[str, ...]) -> list[MarketRow]:
    return [row for row in rows if all(_finite(row.raw.get(name)) for name in names)]


def standardize(rows: list[MarketRow], names: tuple[str, ...]) -> list[ScaledRow]:
    """z-оценка по строкам строго раньше текущей. Будущие строки моменты не двигают."""
    if not names:
        raise ValueError("нет признаков для нормировки")
    counts = {name: 0 for name in names}
    means = {name: 0.0 for name in names}
    squares = {name: 0.0 for name in names}
    scaled: list[ScaledRow] = []
    for row in rows:
        if min(counts.values()) >= MIN_STANDARDIZE:
            scaled.append(ScaledRow(
                decision_date=row.decision_date,
                target_date=row.target_date,
                as_of=row.as_of,
                y_up=row.y_up,
                baseline_up_frequency=row.baseline_up_frequency,
                raw=dict(row.raw),
                standardized=_scale(row.raw, names, counts, means, squares),
            ))
        _update(row.raw, names, counts, means, squares)
    return scaled


def between(rows: list[ScaledRow], start: date, end: date) -> list[ScaledRow]:
    return [
        row for row in rows
        if start <= row.decision_date <= end and start <= row.target_date <= end
    ]


def _external(index: SeriesIndex, as_of: datetime, base: FeatureRow) -> dict[str, float | None]:
    raw: dict[str, float | None] = {name: base.features[name] for name in PRICE_FEATURES}
    funding = index.latest_strict(FUNDING_SERIES, as_of)
    if funding is None or funding.published_at is None or as_of - funding.published_at > FUNDING_MAX_AGE:
        raw["funding_rate"] = None
    else:
        raw["funding_rate"] = funding.value
    current = index.latest_strict(OI_SERIES, as_of)
    previous = index.latest_strict(OI_SERIES, as_of - timedelta(days=1))
    raw["oi_log_change_1d"] = _oi_change(current, previous)
    for name, series_id in MACRO_SERIES.items():
        point = index.latest_strict(series_id, as_of)
        raw[name] = None if point is None else point.value
    etf_point = index.latest_strict(ETF_SERIES, as_of)
    etf_raw = None if etf_point is None else etf_point.value
    raw["etf_net_flow_usd_million"] = etf_value(as_of, etf_raw)
    return raw


def _oi_change(current: Observation | None, previous: Observation | None) -> float | None:
    if current is None or previous is None or current.value is None or previous.value is None:
        return None
    gap = current.event_at - previous.event_at
    if not OI_GAP_MIN <= gap <= OI_GAP_MAX or current.value <= 0 or previous.value <= 0:
        return None
    return math.log(current.value / previous.value)


def _outcome(candles: list[Candle], index: int, horizon: int) -> tuple[float | None, datetime | None]:
    target = index + horizon
    if target >= len(candles):
        return None, None
    change = candles[target].close / candles[index].close - 1
    return change, daily_bar_available_at(candles[target])


def _scale(
    raw: dict[str, float | None],
    names: tuple[str, ...],
    counts: dict[str, int],
    means: dict[str, float],
    squares: dict[str, float],
) -> dict[str, float]:
    scaled: dict[str, float] = {}
    for name in names:
        value = raw.get(name)
        count = counts[name]
        if value is None or not math.isfinite(value) or count < 2:
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


def _update(
    raw: dict[str, float | None],
    names: tuple[str, ...],
    counts: dict[str, int],
    means: dict[str, float],
    squares: dict[str, float],
) -> None:
    for name in names:
        value = raw.get(name)
        if value is None or not math.isfinite(value):
            continue
        counts[name] += 1
        delta = value - means[name]
        means[name] += delta / counts[name]
        squares[name] += delta * (value - means[name])


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _reject_holdout(rows: list[MarketRow]) -> None:
    from wolfai.forecasting.splits import Period, period_bounds

    start = period_bounds(Period.HOLDOUT)[0]
    for row in rows:
        if row.decision_date >= start or row.target_date >= start:
            raise ValueError("матрица признаков получила дату holdout")
