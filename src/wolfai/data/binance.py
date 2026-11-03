"""Offline import of public Binance BTCUSDT Spot 1d ZIP archives.

Reads archives without extracting files or contacting the network. The caller
is responsible for obtaining archives and checking their published checksums.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from wolfai.data.candles import Candle, validate_candles

_ARCHIVE_RE = re.compile(r"BTCUSDT-1d-(\d{4})-(\d{2})(?:-(\d{2}))?\.zip\Z")
_MAX_CSV_BYTES = 2_000_000
_TIMESTAMP_MICROSECOND_FLOOR = 100_000_000_000_000


def validate_daily_candles(candles: list[Candle]) -> None:
    """Require contiguous, midnight-UTC daily bars (one bar = one calendar day)."""
    validate_candles(candles)
    previous = None
    for index, candle in enumerate(candles):
        if not _is_utc_midnight(candle.timestamp):
            raise ValueError(f"свеча {index}: требуется начало суток UTC")
        if previous is not None and candle.timestamp - previous != timedelta(days=1):
            raise ValueError(f"свеча {index}: пропуск или нарушение дневного шага")
        previous = candle.timestamp


def daily_bar_available_at(candle: Candle) -> datetime:
    """When a daily bar labeled by its open can be used in a historical test.

    Binance stamps the bar at 00:00 UTC. That close is known at the next
    00:00 UTC. Forecasts that join other series must use this instant,
    not the open label.
    """
    if not _is_utc_midnight(candle.timestamp):
        raise ValueError("дневная свеча должна начинаться в 00:00 UTC")
    return candle.timestamp + timedelta(days=1)


def load_binance_spot_daily_zip(path: str | Path) -> list[Candle]:
    """Read ONE BTCUSDT Spot 1d monthly/daily Binance public-data archive.

    Binance Spot archive open timestamps are milliseconds before 2025 and
    microseconds from 2025-01-01. The magnitude of the integer selects the
    unit, so either encoding of the same instant decodes to the same UTC
    time. No in-progress candle is inferred here. A monthly file that is
    internally contiguous but shorter than its month is rejected separately
    by assert_filename_period, which the command line importer calls.
    """
    path = Path(path)
    if not _ARCHIVE_RE.fullmatch(path.name):
        raise ValueError("ожидается архив BTCUSDT-1d-YYYY-MM[-DD].zip")
    try:
        archive = ZipFile(path)
    except BadZipFile as error:
        raise ValueError("архив ZIP повреждён или не является ZIP") from error
    with archive:
        expected_name = path.stem + ".csv"
        if archive.namelist() != [expected_name]:
            raise ValueError("в архиве ожидается ровно один BTCUSDT-1d CSV")
        info = archive.getinfo(expected_name)
        if info.file_size > _MAX_CSV_BYTES:
            raise ValueError(f"CSV в архиве превышает {_MAX_CSV_BYTES} байт")
        with archive.open(expected_name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            candles: list[Candle] = []
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 12:
                    raise ValueError(f"строка {line_number}: некорректный Binance kline (12 полей)")
                try:
                    candles.append(_candle_from_kline(row))
                except (OverflowError, OSError, ValueError) as error:
                    raise ValueError(
                        f"строка {line_number}: не удалось прочитать Binance kline"
                    ) from error
    validate_daily_candles(candles)
    return candles


def assert_filename_period(candles: list[Candle], archive_name: str) -> None:
    """Require every UTC day promised by BTCUSDT-1d-YYYY-MM[-DD].zip."""
    validate_daily_candles(candles)
    match = _ARCHIVE_RE.fullmatch(Path(archive_name).name)
    if match is None:
        raise ValueError("ожидается архив BTCUSDT-1d-YYYY-MM[-DD].zip")
    year = int(match.group(1))
    month = int(match.group(2))
    day_text = match.group(3)
    try:
        expected = _period_opens(year, month, int(day_text) if day_text is not None else None)
    except ValueError as error:
        raise ValueError(f"{archive_name}: некорректная дата в имени файла") from error
    actual = [candle.timestamp for candle in candles]
    if actual != expected:
        raise ValueError(
            f"{archive_name}: ожидалось {len(expected)} суток, в архиве {len(actual)}"
        )


def load_binance_spot_daily_archives(paths: Sequence[str | Path]) -> list[Candle]:
    """Склеивает месячные архивы в один непрерывный ряд.

    Пропущенный или повторный день между файлами останавливает сборку.
    Недостающие сутки синтетическими свечами не заполняются.
    """
    files = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if not files:
        raise ValueError("нет архивов для сборки ряда")
    merged: list[Candle] = []
    for path in files:
        month = load_binance_spot_daily_zip(path)
        assert_filename_period(month, path.name)
        if merged and month[0].timestamp - merged[-1].timestamp != timedelta(days=1):
            raise ValueError(
                "разрыв или перекрытие между "
                f"{merged[-1].timestamp.date().isoformat()} и "
                f"{month[0].timestamp.date().isoformat()}"
            )
        merged.extend(month)
    validate_daily_candles(merged)
    return merged


def read_sha256_checksum(path: str | Path) -> str:
    """Первое поле официального файла .CHECKSUM Binance."""
    token = Path(path).read_text(encoding="utf-8").strip().split()[0].lower()
    if len(token) != 64 or any(char not in "0123456789abcdef" for char in token):
        raise ValueError(f"{Path(path).name}: файл CHECKSUM не содержит SHA-256")
    return token


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"{Path(path).name}: SHA-256 не совпал с опубликованным checksum")


def save_normalized_candles(candles: list[Candle], path: str | Path) -> None:
    """Write normalized timestamp/open/high/low/close/volume CSV."""
    validate_daily_candles(candles)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "open", "high", "low", "close", "volume"))
        for candle in candles:
            writer.writerow((
                candle.timestamp.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ))


def _period_opens(year: int, month: int, day: int | None) -> list[datetime]:
    if day is not None:
        return [datetime(year, month, day, tzinfo=timezone.utc)]
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    opens: list[datetime] = []
    cursor = start
    while cursor < end:
        opens.append(cursor)
        cursor += timedelta(days=1)
    return opens


def _is_utc_midnight(timestamp: datetime) -> bool:
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        return False
    return not any((
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
    ))


def _candle_from_kline(row: list[str]) -> Candle:
    return Candle(
        timestamp=_open_time(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def _open_time(raw: int) -> datetime:
    """Decode a Binance spot open time without floating-point rounding.

    Values below 1e14 are milliseconds. Values at or above 1e14 are
    microseconds. Both encodings of one instant land on the same UTC time.
    """
    unit = 1_000_000 if raw >= _TIMESTAMP_MICROSECOND_FLOOR else 1000
    seconds, remainder = divmod(raw, unit)
    microseconds = remainder * (1_000_000 // unit)
    opened = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return opened + timedelta(microseconds=microseconds)
