"""Дневной ряд этапа 2.3.2. Цены с 2022-01-01 не разбираются.

2022 не используется как тест и не входит в подбор. 2023–2024 и holdout
остаются закрытыми, потому что чтение останавливается раньше.
"""

from __future__ import annotations

import csv
from pathlib import Path

from wolfai.data.candles import Candle, parse_utc_timestamp
from wolfai.data.gateway import admit_daily_research
from wolfai.research.quality_contract import FORBIDDEN_FROM


def load_stage_candles(path: str | Path) -> list[Candle]:
    """Читает CSV, пока дата открытия строго раньше 2022-01-01.

    Строка с первой запрещённой датой не переводится в числа.
    """
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
            if timestamp.date() >= FORBIDDEN_FROM:
                break
            candles.append(Candle(
                timestamp=timestamp,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    if not candles:
        raise ValueError(f"{file_path}: в окне этапа нет свечей")
    if candles[-1].timestamp.date() >= FORBIDDEN_FROM:
        raise ValueError("в ряд этапа попала дата с 2022-01-01")
    admit_daily_research(candles)
    return candles
