"""Funding и open interest BTCUSDT USD-M из публичного архива Binance Vision.

Время печати funding и create_time снимка OI — это момент, когда значение уже
было на бирже. Так же проект читает close дневной свечи, а не час появления ZIP.
Архив может быть позже переиздан: source_version хранит SHA-256 конкретного ZIP.
Ликвидации из этого архива в проверенном виде не загружаются.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

from wolfai.intelligence.schema import USE_STRICT, Observation, SourceUnavailable

FUNDING_SERIES = "binance_vision.um.BTCUSDT.funding_rate"
OI_SERIES = "binance_vision.um.BTCUSDT.open_interest"
VISION = "https://data.binance.vision"
FUNDING_ROOT = f"{VISION}/data/futures/um/monthly/fundingRate/BTCUSDT"
OI_ROOT = f"{VISION}/data/futures/um/daily/metrics/BTCUSDT"


def parse_funding_csv(
    text: str,
    *,
    source: str,
    source_version: str,
    ingested_at: datetime,
) -> list[Observation]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "calc_time" not in reader.fieldnames or "last_funding_rate" not in reader.fieldnames:
        raise SourceUnavailable("в CSV funding нет calc_time и last_funding_rate")
    observations: list[Observation] = []
    for row in reader:
        event_at = datetime.fromtimestamp(int(row["calc_time"]) / 1000, tz=timezone.utc)
        interval = (row.get("funding_interval_hours") or "").strip()
        observations.append(Observation(
            series_id=FUNDING_SERIES,
            event_at=event_at,
            published_at=event_at,
            ingested_at=ingested_at,
            source=source,
            source_version=source_version,
            confidence=0.9,
            value=float(row["last_funding_rate"]),
            unit="rate_per_interval",
            category="derivatives",
            use_class=USE_STRICT,
            publication_time_exact=True,
            strict_backtest=True,
            notes=(
                "Ставка funding USD-M BTCUSDT в момент calc_time. "
                f"Интервал в файле: {interval or 'не указан'} ч. "
                "Это не on-chain перевод и не продажа. SHA-256 относится к ZIP архива."
            ),
            entity="binance_um_btcusdt",
        ))
    if not observations:
        raise SourceUnavailable("файл funding пуст")
    return observations


def parse_open_interest_csv(
    text: str,
    *,
    day: datetime,
    source: str,
    source_version: str,
    ingested_at: datetime,
) -> Observation | None:
    """Последний снимок внутри суток UTC. Пропуск файла не заменяется предыдущим значением."""
    if day.tzinfo is None or day.utcoffset() != timedelta(0) or day.hour or day.minute or day.second:
        raise ValueError("сутки OI должны начинаться в 00:00 UTC")
    reader = csv.DictReader(io.StringIO(text))
    required = {"create_time", "symbol", "sum_open_interest"}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise SourceUnavailable("в CSV open interest нет create_time, symbol, sum_open_interest")
    end = day + timedelta(days=1)
    chosen_time: datetime | None = None
    chosen_value: float | None = None
    for row in reader:
        if (row.get("symbol") or "").strip() != "BTCUSDT":
            continue
        stamp = datetime.strptime(row["create_time"].strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        if stamp < day or stamp >= end:
            continue
        value = float(row["sum_open_interest"])
        if value < 0:
            raise SourceUnavailable("open interest отрицателен")
        if chosen_time is None or stamp > chosen_time:
            chosen_time = stamp
            chosen_value = value
    if chosen_time is None or chosen_value is None:
        return None
    return Observation(
        series_id=OI_SERIES,
        event_at=chosen_time,
        published_at=chosen_time,
        ingested_at=ingested_at,
        source=source,
        source_version=source_version,
        confidence=0.85,
        value=chosen_value,
        unit="BTC",
        category="derivatives",
        use_class=USE_STRICT,
        publication_time_exact=True,
        strict_backtest=True,
        notes=(
            "Последний пятиминутный sum_open_interest внутри суток UTC. "
            "Пропущенные сутки не заполняются. Это не поток кошельков."
        ),
        entity="binance_um_btcusdt",
    )


def read_zip_text(path: str | Path) -> str:
    archive_path = Path(path)
    with ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise SourceUnavailable(f"{archive_path.name}: в ZIP ожидается один CSV")
        return archive.read(names[0]).decode("utf-8-sig")
