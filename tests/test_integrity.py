"""Целостность OHLCV и запрет look-ahead для горизонтов 1 и 7 дней."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from wolfai.data import Candle, load_candles, validate_candles, validate_daily_candles
from wolfai.data.availability import (
    DataAvailability,
    candle_availability,
    from_provenance,
    require_usable,
    usable_at,
)
from wolfai.data.binance import daily_bar_available_at
from wolfai.data.candles import inspect_candles, read_candles, to_utc
from wolfai.data.provenance import DataProvenance
from wolfai.forecasting import HistoricalQuantileBaseline
from wolfai.forecasting.baseline import _quantile, known_horizon_returns
from wolfai.research.features import load_research_candles

UTC = timezone.utc
PLUS_5 = timezone(timedelta(hours=5))
MINUS_5 = timezone(timedelta(hours=-5))


def _candle(day: int, close: float = 100.0, volume: float = 1.0) -> Candle:
    return Candle(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=volume,
    )


def _series(closes: list[float]) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        )
        for index, close in enumerate(closes)
    ]


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(
        "timestamp,open,high,low,close,volume\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_negative_offset_keeps_the_utc_instant_and_date(tmp_path: Path) -> None:
    # 19:00 в поясе -05:00 — это полночь следующего дня UTC, а не местная дата.
    _write_csv(
        tmp_path / "btc.csv",
        [
            "2024-01-01T19:00:00-05:00,100,110,90,105,3",
            "2024-01-02T19:00:00-05:00,105,112,100,108,4",
        ],
    )
    candles = load_candles(tmp_path / "btc.csv")
    assert candles[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert candles[0].timestamp.date() == date(2024, 1, 2)
    assert candles[0].timestamp.utcoffset() == timedelta(0)
    assert candles[1].timestamp == datetime(2024, 1, 3, tzinfo=UTC)
    validate_daily_candles(candles)


def test_positive_offset_midnight_is_the_previous_utc_evening(tmp_path: Path) -> None:
    _write_csv(tmp_path / "btc.csv", ["2024-01-02T00:00:00+05:00,100,110,90,105,3"])
    candle = load_candles(tmp_path / "btc.csv")[0]
    assert candle.timestamp == datetime(2024, 1, 1, 19, tzinfo=UTC)
    assert candle.timestamp.date() == date(2024, 1, 1)
    assert candle.timestamp.hour == 19


def test_daily_bars_written_at_plus_five_become_utc_midnights(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "btc.csv",
        [
            "2024-01-02T05:00:00+05:00,100,110,90,105,3",
            "2024-01-03T05:00:00+05:00,105,112,100,108,4",
        ],
    )
    loaded = read_candles(tmp_path / "btc.csv")
    assert loaded.candles[0].timestamp == datetime(2024, 1, 2, tzinfo=UTC)
    assert loaded.candles[1].timestamp == datetime(2024, 1, 3, tzinfo=UTC)
    assert [issue.code for issue in loaded.report.warnings] == [
        "normalized_timezone",
        "normalized_timezone",
    ]
    assert loaded.report.errors == ()
    validate_daily_candles(list(loaded.candles))


def test_dst_fold_offsets_stay_distinct_instants(tmp_path: Path) -> None:
    # Осенний перевод в Нью-Йорке: одна и та же местная метка — два разных UTC-момента.
    first = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0)
    second = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=1)
    assert to_utc(first) == datetime(2024, 11, 3, 5, 30, tzinfo=UTC)
    assert to_utc(second) == datetime(2024, 11, 3, 6, 30, tzinfo=UTC)
    _write_csv(
        tmp_path / "btc.csv",
        [
            "2024-11-03T01:30:00-04:00,100,110,90,105,3",
            "2024-11-03T01:30:00-05:00,101,111,91,106,4",
        ],
    )
    candles = load_candles(tmp_path / "btc.csv")
    assert candles[0].timestamp == datetime(2024, 11, 3, 5, 30, tzinfo=UTC)
    assert candles[1].timestamp == datetime(2024, 11, 3, 6, 30, tzinfo=UTC)
    assert candles[0].timestamp.microsecond == 0


def test_subsecond_offset_is_preserved_in_utc(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "btc.csv",
        ["2024-01-01T00:00:00.250000-05:00,100,110,90,105,3"],
    )
    candle = load_candles(tmp_path / "btc.csv")[0]
    assert candle.timestamp == datetime(2024, 1, 1, 5, 0, 0, 250000, tzinfo=UTC)


def test_z_suffix_and_explicit_utc_stay_on_the_utc_grid(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "btc.csv",
        [
            "2024-01-01T00:00:00Z,100,110,90,105,3",
            "2024-01-02T00:00:00+00:00,105,112,100,108,4",
        ],
    )
    candles = load_candles(tmp_path / "btc.csv")
    assert [candle.timestamp for candle in candles] == [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ]
    assert read_candles(tmp_path / "btc.csv").report.warnings == ()


def test_naive_timestamp_is_not_assumed_to_be_utc(tmp_path: Path) -> None:
    _write_csv(tmp_path / "btc.csv", ["2024-01-01T00:00:00,100,110,90,105,3"])
    with pytest.raises(ValueError, match="часового пояса"):
        load_candles(tmp_path / "btc.csv")
    naive = datetime(2024, 1, 1)
    assert to_utc(naive).tzinfo is None


def test_same_instant_in_two_offsets_is_a_duplicate(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "btc.csv",
        [
            "2024-01-01T00:00:00+00:00,100,110,90,105,3",
            "2024-01-01T05:00:00+05:00,105,112,100,108,4",
        ],
    )
    with pytest.raises(ValueError, match="дублир"):
        load_candles(tmp_path / "btc.csv")


def test_leap_day_gap_is_rejected_and_not_filled() -> None:
    candles = [
        Candle(datetime(2024, 2, 28, tzinfo=UTC), 100, 110, 90, 100, 1),
        Candle(datetime(2024, 3, 1, tzinfo=UTC), 100, 110, 90, 100, 1),
    ]
    snapshot = list(candles)
    with pytest.raises(ValueError, match="пропуск"):
        validate_candles(candles)
    assert candles == snapshot
    report = inspect_candles(candles)
    assert [issue.code for issue in report.errors] == ["missing_daily_bar"]
    assert report.warnings == ()


def test_non_leap_march_boundary_is_contiguous() -> None:
    validate_candles([
        Candle(datetime(2025, 2, 28, tzinfo=UTC), 100, 110, 90, 100, 1),
        Candle(datetime(2025, 3, 1, tzinfo=UTC), 100, 110, 90, 100, 1),
    ])


def test_calendar_gap_between_intraday_bars_is_a_warning() -> None:
    candles = [
        Candle(datetime(2024, 1, 1, 15, tzinfo=UTC), 100, 110, 90, 105, 1),
        Candle(datetime(2024, 1, 3, 15, tzinfo=UTC), 105, 112, 100, 108, 1),
    ]
    report = inspect_candles(candles)
    assert report.errors == ()
    assert [issue.code for issue in report.warnings] == ["missing_daily_bar"]
    validate_candles(candles)
    with pytest.raises(ValueError, match="UTC"):
        validate_daily_candles(candles)


@pytest.mark.parametrize(
    ("close", "volume", "code"),
    [
        (float("nan"), 1.0, "non_finite_price"),
        (float("inf"), 1.0, "non_finite_price"),
        (float("-inf"), 1.0, "non_finite_price"),
        (0.0, 1.0, "non_positive_price"),
        (-2.0, 1.0, "non_positive_price"),
        (100.0, float("nan"), "non_finite_volume"),
        (100.0, float("inf"), "non_finite_volume"),
        (100.0, float("-inf"), "non_finite_volume"),
        (100.0, -0.01, "negative_volume"),
    ],
)
def test_invalid_numbers_are_critical_errors(close: float, volume: float, code: str) -> None:
    candle = Candle(datetime(2024, 1, 1, tzinfo=UTC), 100, 110, 90, close, volume)
    report = inspect_candles([candle])
    assert report.errors[0].code == code
    assert report.warnings == ()
    with pytest.raises(ValueError):
        validate_candles([candle])


def test_nan_price_is_reported_before_a_negative_volume() -> None:
    candle = Candle(datetime(2024, 1, 1, tzinfo=UTC), 100, 110, 90, float("nan"), -1)
    report = inspect_candles([candle])
    assert [issue.code for issue in report.errors] == ["non_finite_price", "negative_volume"]
    with pytest.raises(ValueError, match="конечной"):
        validate_candles([candle])


def test_zero_volume_is_a_warning_and_stays_in_the_series() -> None:
    candles = [_candle(0, volume=0.0), _candle(1, volume=1.0)]
    report = inspect_candles(candles)
    assert report.errors == ()
    assert [issue.code for issue in report.warnings] == ["zero_volume"]
    validate_candles(candles)
    assert len(candles) == 2


def test_high_below_low_and_reversed_time_stay_critical() -> None:
    broken = Candle(datetime(2024, 1, 1, tzinfo=UTC), 100, 90, 110, 100, 1)
    report = inspect_candles([broken])
    assert report.errors[0].code == "high_below_low"
    with pytest.raises(ValueError, match="не возрастает"):
        validate_candles([_candle(1), _candle(0)])


def test_csv_nan_and_infinity_are_rejected(tmp_path: Path) -> None:
    _write_csv(tmp_path / "nan.csv", ["2024-01-01T00:00:00+00:00,100,110,90,nan,3"])
    _write_csv(tmp_path / "inf.csv", ["2024-01-01T00:00:00+00:00,100,110,90,105,inf"])
    with pytest.raises(ValueError, match="конечной"):
        load_candles(tmp_path / "nan.csv")
    with pytest.raises(ValueError, match="volume"):
        load_candles(tmp_path / "inf.csv")


def test_unknown_publication_time_is_not_replaced_by_event_time() -> None:
    event = datetime(2024, 6, 1, 12, tzinfo=UTC)
    record = DataAvailability.unpublished(event, "macro-release")
    assert record.available_at is None
    assert record.event_time == event
    assert record.source == "macro-release"
    month_later = event + timedelta(days=40)
    assert usable_at([record], month_later) == []
    with pytest.raises(ValueError, match="event_time"):
        require_usable([record], month_later)


def test_availability_uses_the_absolute_instant() -> None:
    event = datetime(2024, 12, 1, tzinfo=UTC)
    available = datetime(2024, 12, 1, 15, tzinfo=UTC)
    record = DataAvailability(event, available, "news")
    same_instant = datetime(2024, 12, 1, 20, tzinfo=PLUS_5)
    one_minute_later = datetime(2024, 12, 1, 20, 1, tzinfo=PLUS_5)
    assert usable_at([record], same_instant) == [record]
    assert usable_at([record], one_minute_later) == [record]
    assert usable_at([record], available - timedelta(microseconds=1)) == []
    assert require_usable([record], same_instant) == [record]


def test_provenance_publication_is_not_the_event_time() -> None:
    event = datetime(2024, 1, 1, tzinfo=UTC)
    published = datetime(2024, 1, 15, 8, 30, tzinfo=UTC)
    provenance = DataProvenance(
        event_time=event,
        published_at=published,
        ingested_at=datetime(2026, 10, 1, tzinfo=UTC),
        source="fred",
        source_version="v1",
        confidence=0.9,
    )
    record = from_provenance(provenance)
    assert record.available_at == published
    assert record.event_time == event
    assert record.source == "fred"
    assert usable_at([record], event + timedelta(days=10)) == []
    assert usable_at([record], published) == [record]


def test_daily_close_is_unavailable_at_its_open() -> None:
    candle = Candle(datetime(2024, 12, 31, tzinfo=UTC), 100, 110, 90, 105, 2)
    record = candle_availability(candle, source="binance-vision")
    assert record.event_time == candle.timestamp
    assert record.available_at == datetime(2025, 1, 1, tzinfo=UTC)
    assert record.available_at != record.event_time
    assert usable_at([record], record.event_time) == []
    assert usable_at([record], record.available_at) == [record]
    assert record.source == "binance-vision"


def test_forecast_rejects_a_calendar_gap_without_filling_it() -> None:
    candles = _series([100.0 + day for day in range(40)])
    gapped = candles[:20] + candles[21:]
    assert len(gapped) == 39
    with pytest.raises(ValueError, match="пропуск"):
        HistoricalQuantileBaseline().forecast(gapped, horizon_days=1)
    with pytest.raises(ValueError, match="пропуск"):
        HistoricalQuantileBaseline().forecast(gapped, horizon_days=7)
    assert len(gapped) == 39
    with pytest.raises(ValueError, match="пропуск"):
        known_horizon_returns(gapped, 1, daily_bar_available_at(gapped[-1]))


def test_horizon_returns_stop_at_closes_already_available() -> None:
    candles = _series([100.0 + day * 3 for day in range(30)])
    as_of = daily_bar_available_at(candles[15])
    for horizon in (1, 7):
        returns = known_horizon_returns(candles, horizon, as_of)
        assert len(returns) == 16 - horizon
        last = candles[15].close / candles[15 - horizon].close - 1
        assert returns[-1] == pytest.approx(last)
        mutated = list(candles)
        future = mutated[16]
        mutated[16] = Candle(future.timestamp, 1, 2, 0.5, 7, 1)
        assert known_horizon_returns(mutated, horizon, as_of) == returns
        changed = list(candles)
        current = changed[15]
        changed[15] = Candle(current.timestamp, 1, 2, 0.5, current.close * 2, 1)
        assert known_horizon_returns(changed, horizon, as_of) != returns


def test_forecast_median_matches_only_returns_known_at_as_of() -> None:
    candles = _series([100.0 + day * 3 for day in range(48)])
    model = HistoricalQuantileBaseline()
    for horizon in (1, 7):
        for end in range(horizon + 8, len(candles) - horizon + 1):
            history = candles[:end]
            forecast = model.forecast(history, horizon)
            as_of = forecast.as_of
            assert as_of == daily_bar_available_at(history[-1])
            returns = known_horizon_returns(history, horizon, as_of)
            assert returns
            assert all(
                daily_bar_available_at(history[index + horizon]) <= as_of
                for index in range(len(returns))
            )
            expected = history[-1].close * (1 + _quantile(returns, 0.5))
            assert forecast.median == pytest.approx(expected)
            target = candles[end - 1 + horizon]
            assert daily_bar_available_at(target) > as_of
            assert target.timestamp - history[-1].timestamp == timedelta(days=horizon)
            leaked = list(candles)
            leaked[end - 1 + horizon] = Candle(
                target.timestamp, 1, 2, 0.5, 1, target.volume,
            )
            assert model.forecast(leaked[:end], horizon) == forecast
            if end < len(candles):
                assert known_horizon_returns(candles[: end + 1], horizon, as_of) == returns


def test_research_loader_cuts_holdout_on_the_utc_date(tmp_path: Path) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    day = date(2024, 12, 1)
    while day <= date(2024, 12, 31):
        stamp = datetime(day.year, day.month, day.day, tzinfo=UTC).isoformat()
        lines.append(f"{stamp},100,110,90,105,2")
        day += timedelta(days=1)
    # Местная дата ещё 31 декабря, но это уже 2025-01-01 00:00 UTC.
    lines.append("2024-12-31T19:00:00-05:00,NOT_A_PRICE,1,1,1,1")
    path = tmp_path / "btc.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_research_candles(path)
    assert len(loaded) == 31
    assert loaded[-1].timestamp == datetime(2024, 12, 31, tzinfo=UTC)
    assert all(candle.close == 105 for candle in loaded)
