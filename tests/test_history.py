"""DuckDB и Parquet: воспроизводимый ряд BTC без дублей и без holdout."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from wolfai.data import Candle
from wolfai.data.history import SCHEMA_VERSION, HistoricalStore
from wolfai.forecasting import HistoricalQuantileBaseline, walk_forward

UTC = timezone.utc
DAY_US = 86_400_000_000
SOURCE = "binance-vision"


def _span(start: date, days: int, *, volume: float = 2.0) -> list[Candle]:
    candles: list[Candle] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        close = 100.0 + offset
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=volume,
        ))
    return candles


def _csv(path: Path, candles: list[Candle], *, extra: str | None = None) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp.isoformat()},{candle.open},{candle.high},"
            f"{candle.low},{candle.close},{candle.volume}"
        )
    if extra is not None:
        lines.append(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_roundtrip_keeps_utc_instants_and_separates_directories(tmp_path: Path) -> None:
    candles = _span(date(2024, 1, 1), 40)
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_candles(candles, source=SOURCE)
    assert store.raw_dir.is_dir()
    assert store.validated_dir.is_dir()
    assert store.parquet_path(SOURCE).read_bytes()[:4] == b"PAR1"
    assert manifest.source == SOURCE
    assert manifest.start == date(2024, 1, 1)
    assert manifest.end == date(2024, 2, 9)
    assert manifest.observations == 40
    assert manifest.schema_version == SCHEMA_VERSION == 1
    assert len(manifest.content_sha256) == 64
    assert manifest.created_at.tzinfo is not None
    assert manifest.created_at.utcoffset() == timedelta(0)
    assert manifest.quality_passed is True
    assert manifest.holdout_excluded == 0
    restored = store.read_candles(SOURCE)
    assert restored == candles
    availability = store.read_availability(SOURCE)
    assert availability[0].event_time == candles[0].timestamp
    assert availability[0].available_at == datetime(2024, 1, 2, tzinfo=UTC)
    assert availability[0].available_at != availability[0].event_time
    assert availability[-1].available_at == candles[-1].timestamp + timedelta(days=1)
    summary = store.query_summary(SOURCE)
    assert summary["observations"] == 40
    assert summary["start"] == date(2024, 1, 1)
    assert summary["end"] == date(2024, 2, 9)


def test_microsecond_stamp_is_not_truncated_to_midnight(tmp_path: Path) -> None:
    candle = Candle(
        datetime(2024, 1, 1, 0, 0, 0, 1, tzinfo=UTC),
        100, 101, 99, 100, 1,
    )
    store = HistoricalStore(tmp_path / "btc")
    with pytest.raises(ValueError, match="UTC"):
        store.import_candles([candle], source=SOURCE)
    assert not store.parquet_path(SOURCE).exists()


def test_offset_csv_roundtrips_as_utc_midnight(tmp_path: Path) -> None:
    path = tmp_path / "offset.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T19:00:00-05:00,100,110,90,105,3\n"
        "2024-01-02T19:00:00-05:00,105,112,100,108,4\n",
        encoding="utf-8",
    )
    store = HistoricalStore(tmp_path / "btc")
    store.import_csv(path, source=SOURCE)
    restored = store.read_candles(SOURCE)
    assert [candle.timestamp for candle in restored] == [
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]
    availability = store.read_availability(SOURCE)
    assert availability[0].available_at == datetime(2024, 1, 3, tzinfo=UTC)
    assert availability[0].available_at - availability[0].event_time == timedelta(days=1)


def test_reimport_does_not_duplicate_and_keeps_the_checksum(tmp_path: Path) -> None:
    candles = _span(date(2024, 3, 1), 28)
    store = HistoricalStore(tmp_path / "btc")
    first = store.import_candles(candles, source=SOURCE)
    second = store.import_candles(candles, source=SOURCE)
    third = store.import_candles(candles[:10], source=SOURCE)
    assert second == first
    assert third == first
    assert store.query_summary(SOURCE)["observations"] == 28
    other = HistoricalStore(tmp_path / "other")
    parallel = other.import_candles(candles, source=SOURCE)
    assert parallel.content_sha256 == first.content_sha256
    assert parallel.observations == first.observations
    assert parallel.start == first.start
    assert parallel.end == first.end
    assert parallel.schema_version == first.schema_version
    reread = store.read_candles(SOURCE)
    again = store.import_candles(reread, source=SOURCE)
    assert again.content_sha256 == first.content_sha256
    assert again.created_at == first.created_at


def test_contiguous_batches_merge_without_a_duplicate_day(tmp_path: Path) -> None:
    candles = _span(date(2024, 4, 1), 40)
    store = HistoricalStore(tmp_path / "btc")
    store.import_candles(candles[:20], source=SOURCE)
    merged = store.import_candles(candles[20:], source=SOURCE)
    whole = HistoricalStore(tmp_path / "whole").import_candles(candles, source=SOURCE)
    assert merged.observations == 40
    assert merged.content_sha256 == whole.content_sha256
    assert len(store.read_candles(SOURCE)) == 40
    assert store.read_candles(SOURCE) == candles


def test_gap_and_duplicate_are_rejected_without_replacing_a_good_dataset(tmp_path: Path) -> None:
    candles = _span(date(2024, 5, 1), 20)
    store = HistoricalStore(tmp_path / "btc")
    first = store.import_candles(candles, source=SOURCE)
    with pytest.raises(ValueError, match="пропуск"):
        store.import_candles(candles[:10] + candles[11:], source=SOURCE)
    with pytest.raises(ValueError, match="пропуск"):
        store.import_candles(_span(date(2024, 5, 22), 5), source=SOURCE)
    duplicate = [candles[0], candles[0]]
    with pytest.raises(ValueError, match="дублир"):
        store.import_candles(duplicate, source=SOURCE)
    assert store.read_candles(SOURCE) == candles
    assert store.manifest(SOURCE) == first
    assert not store.parquet_path("missing-source").exists()


def test_holdout_prices_are_not_stored_or_returned(tmp_path: Path) -> None:
    candles = _span(date(2024, 12, 1), 31)
    path = tmp_path / "year.csv"
    _csv(
        path,
        candles,
        extra="2025-01-01T00:00:00+00:00,1,2,0.5,999999.25,1",
    )
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_csv(path, source=SOURCE)
    assert manifest.holdout_excluded == 1
    assert manifest.end == date(2024, 12, 31)
    assert manifest.observations == 31
    restored = store.read_candles(SOURCE)
    assert all(candle.timestamp.date() < date(2025, 1, 1) for candle in restored)
    assert all(candle.close != 999999.25 for candle in restored)
    with pytest.raises(ValueError, match="holdout") as caught:
        store.read_candles(SOURCE, start=date(2025, 1, 1), end=date(2025, 1, 31))
    assert "999999.25" not in str(caught.value)


def test_zero_volume_is_recorded_as_a_quality_warning(tmp_path: Path) -> None:
    candles = _span(date(2024, 6, 1), 5)
    candles[2] = Candle(
        candles[2].timestamp, candles[2].open, candles[2].high, candles[2].low, candles[2].close, 0,
    )
    manifest = HistoricalStore(tmp_path / "btc").import_candles(candles, source=SOURCE)
    assert manifest.quality_passed is True
    assert "zero_volume" in manifest.warning_codes


def test_schema_is_enforced(tmp_path: Path) -> None:
    import duckdb

    candles = _span(date(2024, 7, 1), 4)
    store = HistoricalStore(tmp_path / "btc")
    store.import_candles(candles, source=SOURCE)
    assert store.column_types(SOURCE) == {
        "event_time_us": "BIGINT",
        "available_at_us": "BIGINT",
        "source": "VARCHAR",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "volume": "DOUBLE",
    }
    path = store.parquet_path(SOURCE)
    broken = tmp_path / "broken.parquet"
    duckdb.connect().execute(
        f"COPY (SELECT 'bad' AS close) TO '{broken.as_posix()}' (FORMAT PARQUET)"
    )
    broken.replace(path)
    with pytest.raises(ValueError, match="схем"):
        store.read_candles(SOURCE)


def test_restored_series_matches_the_control_forecast(tmp_path: Path) -> None:
    candles = _span(date(2024, 1, 1), 45)
    store = HistoricalStore(tmp_path / "btc")
    store.import_candles(candles, source=SOURCE)
    restored = store.read_candles(SOURCE)
    assert HistoricalQuantileBaseline().forecast(restored, 1) == (
        HistoricalQuantileBaseline().forecast(candles, 1)
    )
    assert HistoricalQuantileBaseline().forecast(restored, 7) == (
        HistoricalQuantileBaseline().forecast(candles, 7)
    )
    assert walk_forward(restored, 1, min_history=30) == walk_forward(candles, 1, min_history=30)
    assert walk_forward(restored, 7, min_history=30) == walk_forward(candles, 7, min_history=30)


def test_same_instants_from_different_csv_text_share_one_checksum(tmp_path: Path) -> None:
    plus = tmp_path / "plus.csv"
    zed = tmp_path / "zed.csv"
    plus.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-08-01T00:00:00+00:00,100,101,99,100,2\n"
        "2024-08-02T00:00:00+00:00,101,102,100,101,2\n",
        encoding="utf-8",
    )
    zed.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-08-01T00:00:00Z,100,101,99,100,2\n"
        "2024-08-02T00:00:00Z,101,102,100,101,2\n",
        encoding="utf-8",
    )
    left = HistoricalStore(tmp_path / "left").import_csv(plus, source=SOURCE)
    right = HistoricalStore(tmp_path / "right").import_csv(zed, source=SOURCE)
    assert left.content_sha256 == right.content_sha256
    assert left.observations == right.observations == 2
    assert DAY_US == 86_400 * 1_000_000
