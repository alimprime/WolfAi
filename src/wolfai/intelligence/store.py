"""Append-only SQLite для внешних наблюдений.

Повторная загрузка той же версии не переписывает строку. Новая source_version
добавляется рядом.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wolfai.intelligence.schema import Observation, parse_utc

_DDL = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY,
    series_id TEXT NOT NULL,
    event_at TEXT NOT NULL,
    published_at TEXT,
    ingested_at TEXT NOT NULL,
    source TEXT NOT NULL,
    source_version TEXT NOT NULL,
    confidence REAL NOT NULL,
    value REAL,
    unit TEXT NOT NULL,
    category TEXT NOT NULL,
    use_class TEXT NOT NULL,
    publication_time_exact INTEGER NOT NULL,
    strict_backtest INTEGER NOT NULL,
    entity TEXT,
    notes TEXT NOT NULL,
    UNIQUE (series_id, event_at, source, source_version)
)
"""


class ObservationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(_DDL)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def append(self, observations: list[Observation]) -> int:
        inserted = 0
        for observation in observations:
            cursor = self._connection.execute(
                """
                INSERT OR IGNORE INTO observations (
                    series_id, event_at, published_at, ingested_at, source, source_version,
                    confidence, value, unit, category, use_class, publication_time_exact,
                    strict_backtest, entity, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row(observation),
            )
            inserted += cursor.rowcount if cursor.rowcount > 0 else 0
        self._connection.commit()
        return inserted

    def load(self) -> list[Observation]:
        rows = self._connection.execute(
            "SELECT * FROM observations ORDER BY series_id, event_at, id"
        ).fetchall()
        return [_observation(row) for row in rows]

    def counts_by_series(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT series_id, COUNT(*) AS n FROM observations GROUP BY series_id ORDER BY series_id"
        ).fetchall()
        return {str(row["series_id"]): int(row["n"]) for row in rows}


def _row(observation: Observation) -> tuple[object, ...]:
    return (
        observation.series_id,
        observation.event_at.isoformat(),
        None if observation.published_at is None else observation.published_at.isoformat(),
        observation.ingested_at.isoformat(),
        observation.source,
        observation.source_version,
        observation.confidence,
        observation.value,
        observation.unit,
        observation.category,
        observation.use_class,
        int(observation.publication_time_exact),
        int(observation.strict_backtest),
        observation.entity,
        observation.notes,
    )


def _observation(row: sqlite3.Row) -> Observation:
    published = row["published_at"]
    return Observation(
        series_id=row["series_id"],
        event_at=parse_utc(row["event_at"]),
        published_at=None if published is None else parse_utc(published),
        ingested_at=parse_utc(row["ingested_at"]),
        source=row["source"],
        source_version=row["source_version"],
        confidence=float(row["confidence"]),
        value=None if row["value"] is None else float(row["value"]),
        unit=row["unit"],
        category=row["category"],
        use_class=row["use_class"],
        publication_time_exact=bool(row["publication_time_exact"]),
        strict_backtest=bool(row["strict_backtest"]),
        notes=row["notes"],
        entity=row["entity"],
    )


def coverage(observations: list[Observation]) -> list[dict[str, object]]:
    """Сводка по сериям: сколько строк, границы и сколько из них строгие."""
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.series_id, []).append(observation)
    report: list[dict[str, object]] = []
    for series_id in sorted(grouped):
        rows = grouped[series_id]
        events = [item.event_at for item in rows]
        published = [item.published_at for item in rows if item.published_at is not None]
        report.append({
            "series_id": series_id,
            "category": rows[0].category,
            "unit": rows[0].unit,
            "source": rows[0].source,
            "observations": len(rows),
            "strict_observations": sum(item.strict_backtest for item in rows),
            "missing_values": sum(item.value is None for item in rows),
            "event_start": min(events).isoformat(),
            "event_end": max(events).isoformat(),
            "published_start": min(published).isoformat() if published else None,
            "published_end": max(published).isoformat() if published else None,
            "publication_time_exact": all(item.publication_time_exact for item in rows),
            "confidence_min": min(item.confidence for item in rows),
            "confidence_max": max(item.confidence for item in rows),
            "use_class": sorted({item.use_class for item in rows}),
        })
    return report


def individual_whale_events(observations: list[Observation]) -> int:
    """Число отдельных переводов, а не дневных агрегатов биржи."""
    return sum(item.category == "whale_transfer" for item in observations)
