"""Происхождение наблюдений и запрет заглядывания в будущее.

Внешний факт участвует в историческом тесте на момент решения t только если
его published_at не позже t. event_time говорит, когда событие произошло,
ingested_at — когда файл оказался на диске WolfAi. Ни одно из этих двух полей
не заменяет время публикации: архив 2024 года, скачанный позже, был публичен
в момент published_at, а правка ряда, вышедшая после t, в этот тест не входит.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DataProvenance:
    """Время события, публикации и загрузки плюс источник и достоверность.

    confidence лежит в закрытом интервале от 0 до 1. source_version — checksum
    выгрузки или иной неизменяемый идентификатор версии. ingested_at не может
    быть раньше published_at.
    """

    event_time: datetime
    published_at: datetime
    ingested_at: datetime
    source: str
    source_version: str
    confidence: float

    def __post_init__(self) -> None:
        _require_aware(self.event_time, "event_time")
        _require_aware(self.published_at, "published_at")
        _require_aware(self.ingested_at, "ingested_at")
        if self.ingested_at < self.published_at:
            raise ValueError("ingested_at раньше published_at")
        if not self.source.strip():
            raise ValueError("source пуст")
        if not self.source_version.strip():
            raise ValueError("source_version пуст")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence должен быть конечным числом от 0 до 1")


def visible_at(records: Sequence[DataProvenance], as_of: datetime) -> list[DataProvenance]:
    """Записи, уже опубликованные к моменту as_of, в исходном порядке."""
    _require_aware(as_of, "as_of")
    return [record for record in records if record.published_at <= as_of]


def require_visible(records: Sequence[DataProvenance], as_of: datetime) -> list[DataProvenance]:
    """Возвращает records, если каждая запись была опубликована к as_of.

    Исторический прогон, который уже должен быть обрезан по времени, останавливается,
    когда в выборке есть публикация из будущего.
    """
    _require_aware(as_of, "as_of")
    hidden = sum(record.published_at > as_of for record in records)
    if hidden:
        raise ValueError(f"есть данные, опубликованные после {as_of.isoformat()}: {hidden}")
    return list(records)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} без часового пояса")
