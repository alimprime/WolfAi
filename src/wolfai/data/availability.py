"""Минимальные метаданные доступности наблюдения.

event_time — когда событие произошло.
available_at — когда информацию уже можно использовать в прогнозе.
source — имя источника.

Неизвестное время публикации остаётся None. event_time его не заменяет.
На момент решения t допустима только запись с известным available_at <= t.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import Candle
from wolfai.data.provenance import DataProvenance


@dataclass(frozen=True)
class DataAvailability:
    """Три поля, без которых наблюдение нельзя ставить на ось решения."""

    event_time: datetime
    available_at: datetime | None
    source: str

    def __post_init__(self) -> None:
        _require_aware(self.event_time, "event_time")
        if self.available_at is not None:
            _require_aware(self.available_at, "available_at")
        if not self.source.strip():
            raise ValueError("source пуст")

    @classmethod
    def unpublished(cls, event_time: datetime, source: str) -> DataAvailability:
        """Время публикации неизвестно и не заполняется временем события."""
        return cls(event_time=event_time, available_at=None, source=source)


def usable_at(records: Sequence[DataAvailability], as_of: datetime) -> list[DataAvailability]:
    """Записи с известным available_at <= as_of, в исходном порядке.

    Запись без available_at пропускается. Её event_time границей не становится.
    """
    _require_aware(as_of, "as_of")
    return [
        record
        for record in records
        if record.available_at is not None and record.available_at <= as_of
    ]


def require_usable(records: Sequence[DataAvailability], as_of: datetime) -> list[DataAvailability]:
    """Возвращает records, если каждую запись уже можно использовать в as_of.

    Отсутствующее available_at — ошибка, а не повод подставить event_time.
    """
    _require_aware(as_of, "as_of")
    missing = sum(record.available_at is None for record in records)
    if missing:
        raise ValueError(
            f"нет available_at у {missing} записей; event_time не заменяет время публикации"
        )
    future = sum(
        record.available_at is not None and record.available_at > as_of for record in records
    )
    if future:
        raise ValueError(f"есть данные с available_at после {as_of.isoformat()}: {future}")
    return list(records)


def from_provenance(record: DataProvenance) -> DataAvailability:
    """Доступность внешнего факта — это published_at, не event_time."""
    return DataAvailability(
        event_time=record.event_time,
        available_at=record.published_at,
        source=record.source,
    )


def candle_availability(candle: Candle, source: str) -> DataAvailability:
    """Close дневной свечи доступен в следующую полночь UTC, не в момент открытия.

    Если свеча не лежит на полуночной сетке, время доступности не выводится
    из event_time и вызов завершается ошибкой.
    """
    return DataAvailability(
        event_time=candle.timestamp,
        available_at=daily_bar_available_at(candle),
        source=source,
    )


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} без часового пояса")
