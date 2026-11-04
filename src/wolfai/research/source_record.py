"""Запись внешнего показателя. Четыре времени не подставляются друг вместо друга.

`event_time` — когда произошло событие.
`published_at` — когда источник опубликовал значение.
`available_at` — когда его уже можно использовать в прогнозе.
`ingested_at` — когда система его загрузила.

Без явно указанного основания `available_at` запись не годится для прогноза,
даже если остальные метки заполнены.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

QUALITY_READY = "forecast_ready"
QUALITY_BLOCKED = "forecast_blocked"
CUTOFF = datetime.fromisoformat("2022-01-01T00:00:00+00:00")


@dataclass(frozen=True)
class MetricRecord:
    source: str
    metric_name: str
    event_time: datetime
    published_at: datetime | None
    available_at: datetime | None
    ingested_at: datetime
    value: float | None
    quality_status: str
    source_version: str
    availability_basis: str

    def __post_init__(self) -> None:
        for name, moment in (
            ("event_time", self.event_time),
            ("ingested_at", self.ingested_at),
        ):
            _aware(moment, name)
        if self.published_at is not None:
            _aware(self.published_at, "published_at")
        if self.available_at is not None:
            _aware(self.available_at, "available_at")
        if not self.source.strip() or not self.metric_name.strip() or not self.source_version.strip():
            raise ValueError("source, metric_name и source_version обязательны")
        if self.quality_status not in {QUALITY_READY, QUALITY_BLOCKED}:
            raise ValueError("неизвестный quality_status")
        if self.quality_status == QUALITY_READY:
            if self.available_at is None or self.published_at is None or self.value is None:
                raise ValueError("forecast_ready требует published_at, available_at и value")
            if not self.availability_basis.strip():
                raise ValueError("forecast_ready требует основание available_at")
            if self.available_at < self.published_at:
                raise ValueError("available_at раньше published_at")
            if not math.isfinite(self.value):
                raise ValueError("value не конечен")


def metric_record(
    *,
    source: str,
    metric_name: str,
    event_time: datetime,
    published_at: datetime | None,
    available_at: datetime | None,
    ingested_at: datetime,
    value: float | None,
    source_version: str,
    availability_basis: str = "",
) -> MetricRecord:
    """Собирает запись. Пустое основание не превращает event_time в available_at."""
    number_ok = (
        value is not None
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
    ready = (
        published_at is not None
        and available_at is not None
        and number_ok
        and bool(availability_basis.strip())
        and available_at >= published_at
    )
    if available_at is not None and published_at is not None and available_at < published_at:
        raise ValueError("available_at раньше published_at")
    return MetricRecord(
        source=source,
        metric_name=metric_name,
        event_time=event_time,
        published_at=published_at,
        available_at=available_at if ready else None,
        ingested_at=ingested_at,
        value=value if ready else None,
        quality_status=QUALITY_READY if ready else QUALITY_BLOCKED,
        source_version=source_version,
        availability_basis=availability_basis.strip(),
    )


def usable_at(records: Sequence[MetricRecord], as_of: datetime) -> list[MetricRecord]:
    """Только forecast_ready с available_at и event_time не позже решения."""
    _aware(as_of, "as_of")
    chosen: list[MetricRecord] = []
    for record in records:
        if record.quality_status != QUALITY_READY or record.available_at is None:
            continue
        if record.available_at <= as_of and record.event_time <= as_of:
            chosen.append(record)
    return chosen


def reject_closed_window(moment: datetime) -> None:
    """Цена и внешнее значение с меткой 2022-01-01 и позже в этап не входят."""
    _aware(moment, "moment")
    if moment >= CUTOFF:
        raise ValueError("данные с 2022-01-01 закрыты")


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} должен быть в UTC")
