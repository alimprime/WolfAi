"""FRED graph CSV и EFFR Федерального резервного банка Нью-Йорка.

Оба ответа — последняя винтажная выгрузка без времени релиза в самой строке.
Такие ряды описывают рынок, но не входят в строгий backtest.
API FRED с ключом и винтажами ALFRED здесь не вызывается: ключа нет.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from wolfai.intelligence.schema import USE_DESCRIPTIVE, Observation, SourceUnavailable

FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
NYFED_EFFR = (
    "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
    "?startDate={start}&endDate={end}"
)
FRED_UNITS = {
    "DFF": "percent",
    "CPIAUCSL": "index",
    "VIXCLS": "index",
}


def parse_fred_csv(
    text: str,
    *,
    series_id: str,
    ingested_at: datetime,
) -> list[Observation]:
    if series_id not in FRED_UNITS:
        raise SourceUnavailable(f"серия FRED {series_id} не входит в согласованный набор")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "observation_date" not in reader.fieldnames or series_id not in reader.fieldnames:
        raise SourceUnavailable(f"CSV FRED {series_id} без observation_date")
    observations: list[Observation] = []
    for row in reader:
        raw = (row.get(series_id) or "").strip()
        if raw in {"", "."}:
            continue
        event_at = datetime.fromisoformat(row["observation_date"]).replace(tzinfo=timezone.utc)
        observations.append(Observation(
            series_id=f"fred.{series_id}",
            event_at=event_at,
            published_at=None,
            ingested_at=ingested_at,
            source=FRED_GRAPH.format(series_id=series_id),
            source_version="fred-graph-csv-latest-vintage",
            confidence=0.4,
            value=float(raw),
            unit=FRED_UNITS[series_id],
            category="macro",
            use_class=USE_DESCRIPTIVE,
            publication_time_exact=False,
            strict_backtest=False,
            notes=(
                "Последняя винтажность графика FRED. В файле нет vintage date и часа релиза. "
                "Пересмотры ряда могли выйти позже даты наблюдения. В строгий backtest не входит."
            ),
        ))
    if not observations:
        raise SourceUnavailable(f"FRED {series_id} не содержит чисел")
    return observations


def parse_nyfed_effr(payload: dict[str, object], *, ingested_at: datetime, source: str) -> list[Observation]:
    rates = payload.get("refRates")
    if not isinstance(rates, list):
        raise SourceUnavailable("NY Fed не вернул refRates")
    observations: list[Observation] = []
    for row in rates:
        if not isinstance(row, dict):
            raise SourceUnavailable("строка EFFR не объект")
        if row.get("type") not in (None, "EFFR"):
            continue
        percent = row.get("percentRate")
        effective = row.get("effectiveDate")
        if not isinstance(effective, str) or not isinstance(percent, (int, float)):
            continue
        revision = str(row.get("revisionIndicator") or "")
        event_at = datetime.fromisoformat(effective).replace(tzinfo=timezone.utc)
        observations.append(Observation(
            series_id="nyfed.effr",
            event_at=event_at,
            published_at=None,
            ingested_at=ingested_at,
            source=source,
            source_version="nyfed-effr-search-no-release-timestamp",
            confidence=0.35 if revision else 0.45,
            value=float(percent),
            unit="percent",
            category="macro",
            use_class=USE_DESCRIPTIVE,
            publication_time_exact=False,
            strict_backtest=False,
            notes=(
                "EFFR Нью-Йоркского ФРБ. В объекте есть effectiveDate и revisionIndicator, "
                "но нет часа публикации. Ставка за день обычно выходит на следующее утро. "
                f"revisionIndicator={revision or 'пусто'}. В строгий backtest не входит."
            ),
        ))
    if not observations:
        raise SourceUnavailable("в ответе NY Fed нет ставок EFFR")
    return observations


def load_nyfed_payload(text: str) -> dict[str, object]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SourceUnavailable("NY Fed вернул не объект")
    return payload
