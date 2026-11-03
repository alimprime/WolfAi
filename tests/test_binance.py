"""Synthetic archive tests; no external API or Internet connection needed."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import pytest

from wolfai.data import Candle, load_candles
from wolfai.data.binance import (
    assert_filename_period,
    daily_bar_available_at,
    load_binance_spot_daily_archives,
    load_binance_spot_daily_zip,
    read_sha256_checksum,
    save_normalized_candles,
    sha256_file,
    validate_daily_candles,
    verify_sha256,
)


def _archive(tmp_path: Path, dates: list[datetime], *, name: str = "BTCUSDT-1d-2024-12.zip") -> Path:
    path = tmp_path / name
    rows = []
    for date in dates:
        unit = 1_000_000 if date.year >= 2025 else 1000
        ts = int(date.timestamp() * unit)
        rows.append(f"{ts},100,110,90,105,12,0,0,0,0,0,0")
    with ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", "\n".join(rows) + "\n")
    return path


def test_import_millisecond_spot_archive_and_roundtrip(tmp_path: Path) -> None:
    dates = [datetime(2024, 12, 30, tzinfo=timezone.utc) + timedelta(days=i) for i in range(2)]
    archive = _archive(tmp_path, dates)
    candles = load_binance_spot_daily_zip(archive)
    output = tmp_path / "data" / "btc.csv"
    save_normalized_candles(candles, output)
    assert load_candles(output) == candles
    assert candles[0].timestamp == dates[0]


def test_import_microsecond_spot_archive(tmp_path: Path) -> None:
    dates = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(2)]
    candles = load_binance_spot_daily_zip(_archive(tmp_path, dates, name="BTCUSDT-1d-2025-01.zip"))
    assert [item.timestamp for item in candles] == dates


def test_daily_gap_is_rejected(tmp_path: Path) -> None:
    dates = [datetime(2024, 12, day, tzinfo=timezone.utc) for day in (1, 3)]
    with pytest.raises(ValueError, match="дневного шага"):
        load_binance_spot_daily_zip(_archive(tmp_path, dates))


def test_wrong_archive_pair_is_rejected(tmp_path: Path) -> None:
    dates = [datetime(2024, 12, 1, tzinfo=timezone.utc)]
    with pytest.raises(ValueError, match="BTCUSDT"):
        load_binance_spot_daily_zip(_archive(tmp_path, dates, name="ETHUSDT-1d-2024-12.zip"))


def test_non_utc_midnight_rejected(tmp_path: Path) -> None:
    dates = [datetime(2025, 1, 1, 1, tzinfo=timezone.utc)]
    with pytest.raises(ValueError, match="UTC"):
        load_binance_spot_daily_zip(_archive(tmp_path, dates, name="BTCUSDT-1d-2025-01.zip"))


def test_does_not_extract_zip_entries(tmp_path: Path) -> None:
    dates = [datetime(2024, 12, 1, tzinfo=timezone.utc)]
    archive = _archive(tmp_path, dates)
    with ZipFile(archive, "a") as handle:
        handle.writestr("../../untrusted.csv", "bad")
    with pytest.raises(ValueError, match="ровно один"):
        load_binance_spot_daily_zip(archive)
    assert not (tmp_path / "untrusted.csv").exists()


def test_non_finite_values_rejected(tmp_path: Path) -> None:
    from wolfai.data import validate_candles
    dt = datetime(2024, 12, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="конечной"):
        validate_candles([Candle(dt, 100, 110, 90, float("nan"), 2)])
    with pytest.raises(ValueError, match="volume"):
        validate_candles([Candle(dt, 100, 110, 90, 100, float("inf"))])


def test_duplicate_open_time_rejected(tmp_path: Path) -> None:
    moment = datetime(2024, 12, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="не возрастает"):
        load_binance_spot_daily_zip(_archive(tmp_path, [moment, moment]))


def test_subsecond_open_is_not_truncated_to_midnight(tmp_path: Path) -> None:
    moment = datetime(2024, 12, 1, tzinfo=timezone.utc)
    raw_ms = int(moment.timestamp()) * 1000 + 1
    path = tmp_path / "BTCUSDT-1d-2024-12.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", f"{raw_ms},100,110,90,105,12,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="UTC"):
        load_binance_spot_daily_zip(path)


def test_millisecond_and_microsecond_encodings_match(tmp_path: Path) -> None:
    moment = datetime(2025, 1, 1, tzinfo=timezone.utc)
    seconds = int(moment.timestamp())
    for unit in (1000, 1_000_000):
        path = tmp_path / "BTCUSDT-1d-2025-01.zip"
        row = f"{seconds * unit},100,110,90,105,12,0,0,0,0,0,0\n"
        with ZipFile(path, "w") as archive:
            archive.writestr(path.stem + ".csv", row)
        assert load_binance_spot_daily_zip(path)[0].timestamp == moment


def test_nan_price_inside_archive_rejected(tmp_path: Path) -> None:
    moment = datetime(2024, 12, 1, tzinfo=timezone.utc)
    raw_ms = int(moment.timestamp()) * 1000
    path = tmp_path / "BTCUSDT-1d-2024-12.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", f"{raw_ms},100,110,90,nan,12,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="конечной"):
        load_binance_spot_daily_zip(path)


def test_rejects_nested_member_and_sidecars(tmp_path: Path) -> None:
    nested = tmp_path / "BTCUSDT-1d-2024-12.zip"
    with ZipFile(nested, "w") as archive:
        archive.writestr("../BTCUSDT-1d-2024-12.csv", "1,1,1,1,1,1,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="ровно один"):
        load_binance_spot_daily_zip(nested)
    assert not (tmp_path.parent / "BTCUSDT-1d-2024-12.csv").exists()

    dates = [datetime(2024, 12, 1, tzinfo=timezone.utc)]
    archive_path = _archive(tmp_path, dates, name="BTCUSDT-1d-2024-12-01.zip")
    with ZipFile(archive_path, "a") as handle:
        handle.writestr("readme.txt", "no")
    with pytest.raises(ValueError, match="ровно один"):
        load_binance_spot_daily_zip(archive_path)
    assert not (tmp_path / "readme.txt").exists()


def test_corrupt_or_oversized_zip_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "BTCUSDT-1d-2024-12.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="ZIP"):
        load_binance_spot_daily_zip(corrupt)

    oversized = tmp_path / "BTCUSDT-1d-2024-12-01.zip"
    with ZipFile(oversized, "w") as archive:
        archive.writestr(oversized.stem + ".csv", "x" * 2_000_001)
    with pytest.raises(ValueError, match="2000000"):
        load_binance_spot_daily_zip(oversized)


def test_nonzero_offset_is_not_utc_midnight() -> None:
    stamp = datetime(2024, 12, 1, tzinfo=timezone(timedelta(hours=5)))
    candle = Candle(stamp, 100, 110, 90, 105, 1)
    with pytest.raises(ValueError, match="UTC"):
        validate_daily_candles([candle])
    with pytest.raises(ValueError, match="UTC"):
        daily_bar_available_at(Candle(
            datetime(2024, 12, 1, 0, 0, 0, 1, tzinfo=timezone.utc),
            100, 110, 90, 105, 1,
        ))


def test_daily_bar_becomes_available_at_the_next_midnight() -> None:
    candle = Candle(datetime(2024, 12, 31, tzinfo=timezone.utc), 100, 110, 90, 105, 1)
    assert daily_bar_available_at(candle) == datetime(2025, 1, 1, tzinfo=timezone.utc)


def _month(year: int, month: int) -> list[datetime]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    days: list[datetime] = []
    cursor = start
    while cursor < end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _candles(dates: list[datetime]) -> list[Candle]:
    return [Candle(date, 100, 110, 90, 105, 1) for date in dates]


def test_filename_period_covers_the_named_month_and_leap_day(tmp_path: Path) -> None:
    december = _month(2024, 12)
    candles = load_binance_spot_daily_zip(_archive(tmp_path, december))
    assert_filename_period(candles, "data/raw/BTCUSDT-1d-2024-12.zip")
    assert len(candles) == 31

    short = load_binance_spot_daily_zip(_archive(tmp_path, december[:-1]))
    with pytest.raises(ValueError, match="ожидалось 31 суток"):
        assert_filename_period(short, "BTCUSDT-1d-2024-12.zip")

    one_day = [datetime(2024, 12, 1, tzinfo=timezone.utc)]
    daily = load_binance_spot_daily_zip(
        _archive(tmp_path, one_day, name="BTCUSDT-1d-2024-12-01.zip")
    )
    assert_filename_period(daily, "BTCUSDT-1d-2024-12-01.zip")

    assert len(_month(2024, 2)) == 29
    assert len(_month(2025, 2)) == 28
    assert_filename_period(_candles(_month(2024, 2)), "BTCUSDT-1d-2024-02.zip")
    assert_filename_period(_candles(_month(2025, 2)), "BTCUSDT-1d-2025-02.zip")
    with pytest.raises(ValueError, match="некорректная дата"):
        assert_filename_period(daily, "BTCUSDT-1d-2024-13.zip")


def test_monthly_archives_join_millis_and_micros_without_a_gap(tmp_path: Path) -> None:
    december = _archive(tmp_path, _month(2024, 12))
    january = _archive(tmp_path, _month(2025, 1), name="BTCUSDT-1d-2025-01.zip")
    candles = load_binance_spot_daily_archives([january, december])
    assert len(candles) == 62
    assert candles[30].timestamp == datetime(2024, 12, 31, tzinfo=timezone.utc)
    assert candles[31].timestamp == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_gap_between_archives_is_not_filled(tmp_path: Path) -> None:
    december = _archive(tmp_path, _month(2024, 12))
    february = _archive(tmp_path, _month(2025, 2), name="BTCUSDT-1d-2025-02.zip")
    with pytest.raises(ValueError, match="разрыв или перекрытие"):
        load_binance_spot_daily_archives([december, february])
    with pytest.raises(ValueError, match="разрыв или перекрытие"):
        load_binance_spot_daily_archives([december, december])


def test_sha256_checksum_file(tmp_path: Path) -> None:
    payload = tmp_path / "BTCUSDT-1d-2024-12.zip"
    payload.write_bytes(b"archive-bytes")
    digest = sha256_file(payload)
    checksum = tmp_path / "BTCUSDT-1d-2024-12.zip.CHECKSUM"
    checksum.write_text(f"{digest}  BTCUSDT-1d-2024-12.zip\n", encoding="utf-8")
    assert read_sha256_checksum(checksum) == digest
    verify_sha256(payload, digest.upper())
    with pytest.raises(ValueError, match="SHA-256"):
        verify_sha256(payload, "0" * 64)
