"""Хранилище проверенных дневных рядов: Parquet на диске, запросы через DuckDB.

Сырой снимок и проверенный ряд лежат в разных каталогах. Повторный импорт
тех же суток не добавляет вторую строку. Контрольная сумма считается по
каноническим полям ряда, поэтому то же содержимое даёт тот же результат.
Close с даты holdout в Parquet не записывается.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

from wolfai.data.availability import DataAvailability
from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import Candle, inspect_candles, parse_utc_timestamp
from wolfai.data.gateway import HOLDOUT_START, admit_daily_research, admit_observations

SCHEMA_VERSION = 1
DAY_US = 86_400_000_000
COLUMNS = (
    "event_time_us",
    "available_at_us",
    "source",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
COLUMN_TYPES = {
    "event_time_us": "BIGINT",
    "available_at_us": "BIGINT",
    "source": "VARCHAR",
    "open": "DOUBLE",
    "high": "DOUBLE",
    "low": "DOUBLE",
    "close": "DOUBLE",
    "volume": "DOUBLE",
}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    start: date
    end: date
    observations: int
    schema_version: int
    content_sha256: str
    created_at: datetime
    quality_passed: bool
    warning_codes: tuple[str, ...]
    holdout_excluded: int


class HistoricalStore:
    """Один источник — один Parquet и один манифест внутри корня хранилища."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.validated_dir = self.root / "validated"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.validated_dir.mkdir(parents=True, exist_ok=True)

    def import_csv(self, path: str | Path, *, source: str) -> DatasetManifest:
        """Читает CSV. Цену строки holdout не разбирает и в ряд не кладёт."""
        file_path = Path(path)
        kept, stamps, excluded = _parse_csv(file_path)
        if not stamps:
            raise ValueError(f"{file_path}: файл не содержит свечей")
        _require_daily_grid(stamps)
        if not kept:
            raise ValueError("после границы holdout не осталось наблюдений")
        return self._commit(kept, source, excluded)

    def import_candles(self, candles: list[Candle], *, source: str) -> DatasetManifest:
        """Принимает уже собранные свечи. Сетка суток проверяется до записи."""
        if not candles:
            raise ValueError("ряд свечей пуст")
        _require_daily_grid([candle.timestamp for candle in candles])
        kept: list[Candle] = []
        excluded = 0
        for candle in candles:
            if candle.timestamp.date() >= HOLDOUT_START:
                excluded += 1
                continue
            kept.append(candle)
        if not kept:
            raise ValueError("после границы holdout не осталось наблюдений")
        return self._commit(kept, source, excluded)

    def read_candles(
        self,
        source: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Candle]:
        """Возвращает проверенный ряд. Окно, задевающее holdout, не открывает цены."""
        _refuse_holdout_window(start, end)
        candles = [candle for candle, _record in self._load(source)]
        admit_daily_research(candles)
        return [
            candle
            for candle in candles
            if (start is None or candle.timestamp.date() >= start)
            and (end is None or candle.timestamp.date() <= end)
        ]

    def read_availability(self, source: str) -> list[DataAvailability]:
        records = [record for _candle, record in self._load(source)]
        as_of = records[-1].available_at
        if as_of is None:
            raise ValueError("available_at неизвестен; event_time не заменяет время публикации")
        return admit_observations(records, as_of)

    def manifest(self, source: str) -> DatasetManifest:
        loaded = self._read_manifest(source)
        if loaded is None:
            raise ValueError(f"нет манифеста для {source}")
        return loaded

    def query_summary(self, source: str) -> dict[str, object]:
        """Счётчик и границы через DuckDB. Колонка close в запрос не входит."""
        path = self._checked_parquet(source)
        quoted = _quote_path(path)
        with _connection() as con:
            row = con.execute(
                f"SELECT COUNT(*), MIN(event_time_us), MAX(event_time_us) FROM read_parquet({quoted})"
            ).fetchone()
        if row is None or row[1] is None or row[2] is None:
            raise ValueError(f"проверенный ряд {source} пуст")
        return {
            "observations": int(row[0]),
            "start": _us_to_datetime(int(row[1])).date(),
            "end": _us_to_datetime(int(row[2])).date(),
        }

    def column_types(self, source: str) -> dict[str, str]:
        return _describe(self._checked_parquet(source))

    def parquet_path(self, source: str) -> Path:
        return self.validated_dir / f"{_slug(source)}.parquet"

    def manifest_path(self, source: str) -> Path:
        return self.validated_dir / f"{_slug(source)}.manifest.json"

    def _commit(self, kept: list[Candle], source: str, excluded: int) -> DatasetManifest:
        report = inspect_candles(kept, daily=True)
        if report.errors:
            raise ValueError(report.errors[0].message)
        incoming = [_bar(candle, source) for candle in kept]
        merged = _merge(self._existing_rows(source), incoming)
        merged_candles = [_candle_from_row(row) for row in merged]
        merged_report = inspect_candles(merged_candles, daily=True)
        if merged_report.errors:
            raise ValueError(merged_report.errors[0].message)
        checksum = _content_checksum(merged)
        current = self._read_manifest(source)
        if (
            current is not None
            and current.content_sha256 == checksum
            and current.schema_version == SCHEMA_VERSION
        ):
            return current
        manifest = DatasetManifest(
            source=source,
            start=merged_candles[0].timestamp.date(),
            end=merged_candles[-1].timestamp.date(),
            observations=len(merged),
            schema_version=SCHEMA_VERSION,
            content_sha256=checksum,
            created_at=datetime.now(timezone.utc),
            quality_passed=True,
            warning_codes=tuple(sorted({issue.code for issue in merged_report.warnings})),
            holdout_excluded=excluded,
        )
        self._write(source, merged, manifest)
        return self.manifest(source)

    def _existing_rows(self, source: str) -> list[tuple[object, ...]]:
        path = self.parquet_path(source)
        if not path.exists():
            return []
        self._checked_parquet(source)
        quoted = _quote_path(path)
        with _connection() as con:
            fetched = con.execute(
                "SELECT event_time_us, available_at_us, source, open, high, low, close, volume "
                f"FROM read_parquet({quoted}) ORDER BY event_time_us"
            ).fetchall()
        return [tuple(row) for row in fetched]

    def _load(self, source: str) -> list[tuple[Candle, DataAvailability]]:
        path = self._checked_parquet(source)
        quoted = _quote_path(path)
        with _connection() as con:
            clocks = con.execute(
                f"SELECT event_time_us, available_at_us FROM read_parquet({quoted}) "
                "ORDER BY event_time_us"
            ).fetchall()
        _require_clocks(clocks)
        with _connection() as con:
            fetched = con.execute(
                "SELECT event_time_us, available_at_us, source, open, high, low, close, volume "
                f"FROM read_parquet({quoted}) ORDER BY event_time_us"
            ).fetchall()
        loaded: list[tuple[Candle, DataAvailability]] = []
        for row in fetched:
            candle = _candle_from_row(row)
            available = _us_to_datetime(int(row[1]))
            loaded.append((candle, DataAvailability(candle.timestamp, available, str(row[2]))))
        if not loaded:
            raise ValueError(f"проверенный ряд {source} пуст")
        return loaded

    def _checked_parquet(self, source: str) -> Path:
        path = self.parquet_path(source)
        if not path.exists():
            raise ValueError(f"нет проверенного ряда {source}")
        manifest = self._read_manifest(source)
        if manifest is not None and manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(f"схема датасета {manifest.schema_version} не поддерживается")
        actual = _describe(path)
        if actual != COLUMN_TYPES:
            raise ValueError(f"схема parquet не совпала: {actual}")
        quoted = _quote_path(path)
        holdout_us = _datetime_to_us(datetime(HOLDOUT_START.year, HOLDOUT_START.month, HOLDOUT_START.day, tzinfo=timezone.utc))
        with _connection() as con:
            found = con.execute(
                f"SELECT COUNT(*) FROM read_parquet({quoted}) WHERE event_time_us >= {holdout_us}"
            ).fetchone()
        if found and int(found[0]) > 0:
            raise ValueError("в проверенном ряду есть holdout")
        return path

    def _write(self, source: str, rows: list[tuple[object, ...]], manifest: DatasetManifest) -> None:
        path = self.parquet_path(source)
        temporary = path.with_name(path.name + ".tmp")
        with _connection() as con:
            con.execute(
                """
                CREATE TABLE candles (
                    event_time_us BIGINT,
                    available_at_us BIGINT,
                    source VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE
                )
                """
            )
            con.executemany("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            con.execute(
                "COPY (SELECT event_time_us, available_at_us, source, open, high, low, close, volume "
                f"FROM candles ORDER BY event_time_us) TO {_quote_path(temporary)} (FORMAT PARQUET)"
            )
        temporary.replace(path)
        payload = {
            "source": manifest.source,
            "start": manifest.start.isoformat(),
            "end": manifest.end.isoformat(),
            "observations": manifest.observations,
            "schema_version": manifest.schema_version,
            "content_sha256": manifest.content_sha256,
            "created_at": manifest.created_at.isoformat(),
            "quality_passed": manifest.quality_passed,
            "warning_codes": list(manifest.warning_codes),
            "holdout_excluded": manifest.holdout_excluded,
        }
        manifest_path = self.manifest_path(source)
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_tmp.replace(manifest_path)
        _write_raw_snapshot(
            self.raw_dir / _slug(source) / f"{_slug(source)}.csv",
            [_candle_from_row(row) for row in rows],
        )

    def _read_manifest(self, source: str) -> DatasetManifest | None:
        path = self.manifest_path(source)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        created_at = datetime.fromisoformat(payload["created_at"])
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at без часового пояса")
        return DatasetManifest(
            source=str(payload["source"]),
            start=date.fromisoformat(payload["start"]),
            end=date.fromisoformat(payload["end"]),
            observations=int(payload["observations"]),
            schema_version=int(payload["schema_version"]),
            content_sha256=str(payload["content_sha256"]),
            created_at=created_at,
            quality_passed=bool(payload["quality_passed"]),
            warning_codes=tuple(payload["warning_codes"]),
            holdout_excluded=int(payload["holdout_excluded"]),
        )


def _parse_csv(path: Path) -> tuple[list[Candle], list[datetime], int]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: файл не содержит свечей")
        missing = [
            name
            for name in ("timestamp", "open", "high", "low", "close", "volume")
            if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"{path}: нет колонок {', '.join(missing)}")
        kept: list[Candle] = []
        stamps: list[datetime] = []
        excluded = 0
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp, _shifted = parse_utc_timestamp(row.get("timestamp") or "")
            except (TypeError, ValueError) as error:
                raise ValueError(f"строка {line_number}: не удалось прочитать свечу") from error
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"строка {line_number}: timestamp без часового пояса")
            stamps.append(timestamp)
            if timestamp.date() >= HOLDOUT_START:
                excluded += 1
                continue
            try:
                kept.append(Candle(
                    timestamp=timestamp,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"строка {line_number}: не удалось прочитать свечу") from error
    return kept, stamps, excluded


def _require_daily_grid(stamps: list[datetime]) -> None:
    if not stamps:
        raise ValueError("ряд свечей пуст")
    previous: datetime | None = None
    for index, stamp in enumerate(stamps):
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError(f"свеча {index}: требуется начало суток UTC")
        if any((stamp.hour, stamp.minute, stamp.second, stamp.microsecond)):
            raise ValueError(f"свеча {index}: требуется начало суток UTC")
        if previous is not None and stamp == previous:
            raise ValueError(f"свеча {index}: timestamp дублируется и не возрастает")
        if previous is not None and stamp < previous:
            raise ValueError(f"свеча {index}: timestamp не возрастает")
        if previous is not None and stamp - previous != timedelta(days=1):
            raise ValueError(f"свеча {index}: пропуск или нарушение дневного шага")
        previous = stamp


def _bar(candle: Candle, source: str) -> tuple[object, ...]:
    available = daily_bar_available_at(candle)
    return (
        _datetime_to_us(candle.timestamp),
        _datetime_to_us(available),
        source,
        float(candle.open),
        float(candle.high),
        float(candle.low),
        float(candle.close),
        float(candle.volume),
    )


def _merge(existing: list[tuple[object, ...]], incoming: list[tuple[object, ...]]) -> list[tuple[object, ...]]:
    merged = {int(row[0]): tuple(row) for row in existing}
    for row in incoming:
        merged[int(row[0])] = tuple(row)
    return [merged[key] for key in sorted(merged)]


def _candle_from_row(row: tuple[object, ...]) -> Candle:
    event = _us_to_datetime(int(row[0]))
    return Candle(
        timestamp=event,
        open=float(row[3]),
        high=float(row[4]),
        low=float(row[5]),
        close=float(row[6]),
        volume=float(row[7]),
    )


def _content_checksum(rows: list[tuple[object, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: int(item[0])):
        digest.update(struct.pack("<qq", int(row[0]), int(row[1])))
        digest.update(str(row[2]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack(
            "<ddddd",
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
        ))
    return digest.hexdigest()


def _describe(path: Path) -> dict[str, str]:
    quoted = _quote_path(path)
    with _connection() as con:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet({quoted})").fetchall()
    return {str(row[0]): str(row[1]).upper() for row in rows}


def _require_clocks(clocks: list[tuple[object, ...]]) -> None:
    holdout_us = _datetime_to_us(
        datetime(HOLDOUT_START.year, HOLDOUT_START.month, HOLDOUT_START.day, tzinfo=timezone.utc)
    )
    for event_us, available_us in clocks:
        if event_us is None or available_us is None:
            raise ValueError("available_at неизвестен; event_time не заменяет время публикации")
        if int(event_us) >= holdout_us:
            raise ValueError("в проверенном ряду есть holdout")


def _refuse_holdout_window(start: date | None, end: date | None) -> None:
    if start is not None and start >= HOLDOUT_START:
        raise ValueError(
            f"holdout {HOLDOUT_START.isoformat()} закрыт: окно исследования его не открывает"
        )
    if end is not None and end >= HOLDOUT_START:
        raise ValueError(
            f"holdout {HOLDOUT_START.isoformat()} закрыт: окно исследования его не открывает"
        )


def _write_raw_snapshot(destination: Path, candles: list[Candle]) -> None:
    """Канонический CSV уже принятого ряда. Строки holdout сюда не попадают."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp.isoformat()},{candle.open},{candle.high},"
            f"{candle.low},{candle.close},{candle.volume}"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _datetime_to_us(value: datetime) -> int:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("метка без часового пояса")
    utc = value.astimezone(timezone.utc)
    delta = utc - _EPOCH
    return delta.days * DAY_US + delta.seconds * 1_000_000 + delta.microseconds


def _us_to_datetime(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=value)


def _slug(source: str) -> str:
    text = source.strip()
    if not text or any(char in text for char in "/\\"):
        raise ValueError("source пуст")
    slug = "".join(char if char.isalnum() or char in "-._" else "-" for char in text)
    if not slug or slug in {".", ".."}:
        raise ValueError("source пуст")
    return slug


def _quote_path(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


class _connection:
    def __enter__(self):
        self._con = duckdb.connect(database=":memory:")
        return self._con

    def __exit__(self, exc_type, exc, tb) -> None:
        self._con.close()
