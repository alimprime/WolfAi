"""Метки горизонтов 1, 3 и 7 дней с явным временем решения и метки.

Обучение берёт только строки, у которых target_time не позже
2020-12-31 23:59:59 UTC. Диагностика 2021 берёт прогнозы с decision_time
в 2021 году и target_time строго раньше 2022-01-01. Между окнами остаётся
зазор: метка, созревшая уже в 2021, в обучение не входит, а решение,
сделанное ещё в 2020, не становится диагностикой 2021.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wolfai.data.candles import Candle
from wolfai.data.gateway import admit_daily_research, require_calendar_days
from wolfai.research.features import LOOKBACK
from wolfai.research.ohlcv_features import ALL_FEATURES, causal_features
from wolfai.research.quality_contract import DIAGNOSTIC_HORIZONS, FORBIDDEN_FROM
from wolfai.research.time_alignment import (
    bar_open_time,
    decision_time,
    dev_forecast,
    direction,
    target_known_time,
    train_label_closed,
)


@dataclass(frozen=True)
class AlignedRow:
    index: int
    open_time: datetime
    decision_time: datetime
    target_open_time: datetime
    target_time: datetime
    horizon_days: int
    y_up: int
    features: dict[str, float | None]
    exclusion: str | None
    role: str | None


def build_aligned_rows(candles: list[Candle], horizon_days: int) -> list[AlignedRow]:
    """Строки решения. Цена с датой открытия 2022-01-01 и позже запрещена."""
    if horizon_days not in DIAGNOSTIC_HORIZONS:
        raise ValueError("горизонт диагностики должен быть 1, 3 или 7 дней")
    for candle in candles:
        if candle.timestamp.date() >= FORBIDDEN_FROM:
            raise ValueError("цена с 2022-01-01 не входит в ряд")
    admit_daily_research(candles)
    rows: list[AlignedRow] = []
    last = len(candles) - horizon_days
    for index in range(LOOKBACK, last):
        require_calendar_days(candles, index, index + horizon_days, horizon_days)
        current = candles[index]
        target = candles[index + horizon_days]
        features, reason = causal_features(candles, index)
        made = decision_time(current)
        known = target_known_time(target)
        role = _role(made, known)
        rows.append(AlignedRow(
            index=index,
            open_time=bar_open_time(current),
            decision_time=made,
            target_open_time=bar_open_time(target),
            target_time=known,
            horizon_days=horizon_days,
            y_up=direction(current.close, target.close),
            features=features,
            exclusion=reason,
            role=role,
        ))
    return rows


def complete_rows(rows: list[AlignedRow]) -> list[AlignedRow]:
    """Строки, где все сравниваемые признаки конечны. Пропуск не заполняется."""
    chosen: list[AlignedRow] = []
    for row in rows:
        if row.exclusion is not None:
            continue
        if _complete(row):
            chosen.append(row)
    return chosen


def rows_by_role(rows: list[AlignedRow], role: str) -> list[AlignedRow]:
    return [row for row in rows if row.role == role]


def _role(made: datetime, known: datetime) -> str | None:
    if train_label_closed(known):
        return "train"
    if dev_forecast(made, known):
        return "dev"
    return None


def _complete(row: AlignedRow) -> bool:
    for name in ALL_FEATURES:
        value = row.features.get(name)
        if value is None:
            return False
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if value != value or value in (float("inf"), float("-inf")):
            return False
    return True
