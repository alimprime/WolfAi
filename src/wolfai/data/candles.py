"""Свечи OHLCV и проверка исторического ряда."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_candles(path: str | Path) -> list[Candle]:
    """Читает CSV `timestamp,open,high,low,close,volume` и проверяет ряд."""
    file_path = Path(path)
    with file_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{file_path}: файл не содержит свечей")
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [name for name in expected if name not in rows[0]]
    if missing:
        raise ValueError(f"{file_path}: нет колонок {', '.join(missing)}")
    candles = [_parse_row(index, row) for index, row in enumerate(rows, start=2)]
    validate_candles(candles)
    return candles


def validate_candles(candles: list[Candle]) -> None:
    """Проверяет цены, объём и строгий рост меток времени."""
    if not candles:
        raise ValueError("ряд свечей пуст")
    previous: datetime | None = None
    for index, candle in enumerate(candles):
        if candle.timestamp.tzinfo is None:
            raise ValueError(f"свеча {index}: timestamp без часового пояса")
        for name in ("open", "high", "low", "close"):
            value = getattr(candle, name)
            if value <= 0:
                raise ValueError(f"свеча {index}: {name} должна быть положительной")
        if candle.volume < 0:
            raise ValueError(f"свеча {index}: volume не может быть отрицательным")
        if candle.high < candle.low:
            raise ValueError(f"свеча {index}: high меньше low")
        if candle.high < max(candle.open, candle.close):
            raise ValueError(f"свеча {index}: high ниже open или close")
        if candle.low > min(candle.open, candle.close):
            raise ValueError(f"свеча {index}: low выше open или close")
        if previous is not None and candle.timestamp <= previous:
            raise ValueError(f"свеча {index}: timestamp не возрастает")
        previous = candle.timestamp


def _parse_row(line_number: int, row: dict[str, str]) -> Candle:
    try:
        timestamp = datetime.fromisoformat(row["timestamp"].strip())
        return Candle(
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"строка {line_number}: не удалось прочитать свечу") from error
