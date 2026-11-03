"""Потоки американских спотовых Bitcoin ETF.

Торги начались 11 января 2024 года. До этой даты признак недоступен.
Ноль туда не записывается: ноль был бы утверждением «потока не было».
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from wolfai.intelligence.schema import USE_STRICT, Observation, SourceUnavailable, parse_utc

US_SPOT_BTC_ETF_START = datetime(2024, 1, 11, tzinfo=timezone.utc)
ETF_SERIES = "etf.us_spot_btc.net_flow_usd_million"


def etf_value(as_of: datetime, flow: float | None) -> float | None:
    """Прячет любой поток до начала торгов, в том числе ошибочно переданный ноль."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of без часового пояса")
    if as_of < US_SPOT_BTC_ETF_START:
        return None
    return flow


def load_etf_csv(path: str | Path, *, ingested_at: datetime, source: str) -> list[Observation]:
    """Локальный файл с колонками date, net_flow_usd_million, published_at.

    published_at обязателен и должен быть с часовым поясом. Без него время
    выхода таблицы неизвестно, и строка не годится даже как описательная
    подстановка. Дата раньше 2024-01-11 отвергает файл целиком.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise SourceUnavailable(f"файл ETF не найден: {file_path}")
    with file_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"date", "net_flow_usd_million", "published_at"}
        if not required <= fields:
            raise SourceUnavailable(
                "в CSV ETF нет date, net_flow_usd_million и published_at; "
                "время публикации не угадывается"
            )
        observations: list[Observation] = []
        for line_number, row in enumerate(reader, start=2):
            event_at = parse_utc(f"{row['date'].strip()}T00:00:00+00:00")
            if event_at < US_SPOT_BTC_ETF_START:
                raise SourceUnavailable(
                    f"строка {line_number}: дата раньше запуска спотовых ETF США, ноль не записывается"
                )
            published_at = parse_utc(row["published_at"])
            value = float(row["net_flow_usd_million"])
            observations.append(Observation(
                series_id=ETF_SERIES,
                event_at=event_at,
                published_at=published_at,
                ingested_at=ingested_at,
                source=source,
                source_version=f"local-csv:{file_path.name}",
                confidence=0.6,
                value=value,
                unit="USD_million",
                category="etf",
                use_class=USE_STRICT,
                publication_time_exact=True,
                strict_backtest=True,
                notes=(
                    "Чистый поток спотовых Bitcoin ETF в млн USD по локальному файлу. "
                    "Это не on-chain покупка в момент публикации и не перевод конкретного кошелька."
                ),
                entity="us_spot_btc_etf",
            ))
    if not observations:
        raise SourceUnavailable("файл ETF пуст")
    return observations
