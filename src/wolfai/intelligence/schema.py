"""Общая запись внешнего наблюдения.

Поля времени совпадают с контрактом DataProvenance. value=None означает
«недоступно». Настоящий измеренный ноль допустим и не используется как заполнитель.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from wolfai.data.provenance import DataProvenance

USE_STRICT = "strict_backtest"
USE_DESCRIPTIVE = "descriptive"
USE_FORWARD = "forward_monitoring"
USE_CLASSES = frozenset({USE_STRICT, USE_DESCRIPTIVE, USE_FORWARD})

CATEGORIES = frozenset({
    "exchange_flow",
    "exchange_reserve",
    "miner",
    "network",
    "derivatives",
    "macro",
    "etf",
    "news",
    "whale_transfer",
})


class SourceUnavailable(RuntimeError):
    """Источник не отдал пригодный ряд. Подставлять значения нельзя."""


def parse_utc(value: str) -> datetime:
    """Разбирает ISO-время, включая наносекунды и суффикс Z."""
    text = value.strip()
    if not text:
        raise ValueError("пустая метка времени")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        split_at = len(tail)
        for index, char in enumerate(tail):
            if char in "+-" and index > 0:
                split_at = index
                break
        fraction = "".join(char for char in tail[:split_at] if char.isdigit())
        offset = tail[split_at:] or "+00:00"
        text = f"{head}.{(fraction + '000000')[:6]}{offset}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("метка времени без часового пояса")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class Observation:
    """Один факт из внешнего источника.

    strict_backtest истинен только вместе с точным published_at и конечным value.
    Поздняя ревизия — отдельная запись с тем же event_at и новым source_version,
    а не перезапись прежней строки.
    """

    series_id: str
    event_at: datetime
    published_at: datetime | None
    ingested_at: datetime
    source: str
    source_version: str
    confidence: float
    value: float | None
    unit: str
    category: str
    use_class: str
    publication_time_exact: bool
    strict_backtest: bool
    notes: str
    entity: str | None = None

    def __post_init__(self) -> None:
        _aware(self.event_at, "event_at")
        _aware(self.ingested_at, "ingested_at")
        if self.published_at is not None:
            _aware(self.published_at, "published_at")
            if self.ingested_at < self.published_at:
                raise ValueError("ingested_at раньше published_at")
        if self.publication_time_exact and self.published_at is None:
            raise ValueError("точное время публикации не задано")
        if not self.series_id.strip():
            raise ValueError("series_id пуст")
        if not self.source.strip() or not self.source_version.strip():
            raise ValueError("source или source_version пуст")
        if not self.unit.strip() or not self.notes.strip():
            raise ValueError("unit или notes пуст")
        if self.category not in CATEGORIES:
            raise ValueError(f"неизвестная категория {self.category}")
        if self.use_class not in USE_CLASSES:
            raise ValueError(f"неизвестный use_class {self.use_class}")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence должен быть конечным числом от 0 до 1")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("value не конечен")
        if self.strict_backtest:
            if self.use_class != USE_STRICT:
                raise ValueError("strict_backtest требует use_class strict_backtest")
            if not self.publication_time_exact or self.published_at is None or self.value is None:
                raise ValueError("strict_backtest требует точное published_at и value")

    def as_provenance(self) -> DataProvenance:
        if self.published_at is None:
            raise ValueError("нет published_at, запись нельзя поставить на ось публикации")
        return DataProvenance(
            event_time=self.event_at,
            published_at=self.published_at,
            ingested_at=self.ingested_at,
            source=self.source,
            source_version=self.source_version,
            confidence=self.confidence,
        )


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} без часового пояса")
