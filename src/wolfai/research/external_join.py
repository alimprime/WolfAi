"""Признаки деривативов только из записей, уже доступных к решению.

Пропуск остаётся пустым. Следующий снимок не подставляется вместо пропущенного.
Сравнение funding и open interest за 2022 год здесь не повторяется.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta

from wolfai.research.source_record import CUTOFF, MetricRecord, metric_record, usable_at

FUNDING_METRIC = "binance_vision.um.BTCUSDT.funding_rate"
OI_METRIC = "binance_vision.um.BTCUSDT.open_interest"
FUNDING_BASIS = "calc_time — момент, когда ставка уже стоит на бирже, а не время открытия свечи"
OI_BASIS = "create_time снимка open interest — момент, когда снимок уже опубликован биржей"
FUNDING_MAX_AGE = timedelta(hours=24)
OI_GAP_MIN = timedelta(hours=18)
OI_GAP_MAX = timedelta(hours=30)


def load_strict_derivatives(database: str) -> list[MetricRecord]:
    """Строгие снимки с событием и публикацией раньше 2022-01-01. value позже этой даты не читается."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT source, series_id, event_at, published_at, ingested_at, value, source_version
            FROM observations
            WHERE series_id IN (?, ?)
              AND strict_backtest = 1
              AND value IS NOT NULL
              AND published_at IS NOT NULL
              AND published_at < ?
              AND event_at < ?
            ORDER BY series_id, event_at
            """,
            (FUNDING_METRIC, OI_METRIC, "2022-01-01", "2022-01-01"),
        ).fetchall()
    finally:
        connection.close()
    records: list[MetricRecord] = []
    for source, metric, event_at, published_at, ingested_at, value, version in rows:
        event = _utc(event_at)
        published = _utc(published_at)
        ingested = _utc(ingested_at)
        if event >= CUTOFF or published >= CUTOFF:
            continue
        basis = FUNDING_BASIS if metric == FUNDING_METRIC else OI_BASIS
        records.append(metric_record(
            source=str(source),
            metric_name=str(metric),
            event_time=event,
            published_at=published,
            available_at=published,
            ingested_at=ingested,
            value=float(value),
            source_version=str(version),
            availability_basis=basis,
        ))
    return records


def derivative_features(records: list[MetricRecord], as_of: datetime) -> dict[str, float] | None:
    """Последняя ставка, её изменение и суточный log-change open interest.

    None означает, что на эту дату группа деривативов неполная.
    """
    visible = usable_at(records, as_of)
    funding = [item for item in visible if item.metric_name == FUNDING_METRIC and item.available_at is not None]
    oi = [item for item in visible if item.metric_name == OI_METRIC and item.value is not None and item.value > 0]
    funding.sort(key=lambda item: item.event_time)
    oi.sort(key=lambda item: item.event_time)
    if len(funding) < 2 or len(oi) < 2:
        return None
    latest = funding[-1]
    previous = funding[-2]
    if latest.available_at is None or as_of - latest.available_at > FUNDING_MAX_AGE:
        return None
    if latest.value is None or previous.value is None:
        return None
    current = oi[-1]
    prior = oi[-2]
    gap = current.event_time - prior.event_time
    if not OI_GAP_MIN <= gap <= OI_GAP_MAX or current.value is None or prior.value is None:
        return None
    if current.value <= 0 or prior.value <= 0:
        return None
    return {
        "funding_rate": latest.value,
        "funding_change": latest.value - previous.value,
        "oi_log_change": math.log(current.value / prior.value),
    }


def _utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("метка внешнего ряда должна быть в UTC")
    return parsed.astimezone(parsed.tzinfo)
