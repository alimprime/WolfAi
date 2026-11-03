"""Общий вход дневных признаков: дыры, holdout и чужое время доступности."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.data.availability import DataAvailability
from wolfai.data.gateway import admit_daily_research, admit_observations, require_calendar_days
from wolfai.research.features import (
    build_rows,
    direction_label,
    feature_vector,
    historical_up_frequency,
)

UTC = timezone.utc


def _span(start: date, days: int) -> list[Candle]:
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
            volume=2,
        ))
    return candles


def test_direct_feature_call_rejects_a_gap_inside_seven_days() -> None:
    candles = _span(date(2024, 1, 1), 50)
    gapped = candles[:36] + candles[37:]
    index = 40
    assert gapped[index].timestamp - gapped[index - 7].timestamp != timedelta(days=7)
    with pytest.raises(ValueError, match="пропуск|календар"):
        feature_vector(gapped, index)
    with pytest.raises(ValueError, match="пропуск|календар"):
        build_rows(gapped)
    with pytest.raises(ValueError, match="пропуск|календар"):
        historical_up_frequency(gapped, index)
    with pytest.raises(ValueError, match="пропуск|календар"):
        direction_label(gapped, index)
    with pytest.raises(ValueError, match="пропуск|календар"):
        admit_daily_research(gapped)


def test_seven_bars_on_a_continuous_series_are_seven_utc_days() -> None:
    candles = _span(date(2024, 1, 1), 50)
    index = 40
    require_calendar_days(candles, index - 7, index, 7)
    require_calendar_days(candles, index - 30, index, 30)
    features = feature_vector(candles, index)
    assert candles[index].timestamp - candles[index - 7].timestamp == timedelta(days=7)
    assert features["momentum_7"] == pytest.approx(
        candles[index].close / candles[index - 7].close - 1
    )
    assert features["log_return"] == pytest.approx(
        math.log(candles[index].close / candles[index - 1].close)
    )


def test_feature_vector_does_not_read_the_next_close() -> None:
    candles = _span(date(2024, 1, 1), 50)
    index = 40
    original = feature_vector(candles, index)
    future = candles[index + 1]
    leaked = list(candles)
    leaked[index + 1] = Candle(future.timestamp, 1, 2, 0.5, 1, 2)
    assert feature_vector(leaked, index) == original


def test_feature_vector_refuses_a_series_that_contains_holdout() -> None:
    candles = _span(date(2024, 12, 1), 31)
    candles.append(Candle(datetime(2025, 1, 1, tzinfo=UTC), 1, 2, 0.5, 999999.25, 1))
    with pytest.raises(ValueError, match="holdout") as caught:
        feature_vector(candles, 30)
    assert "999999.25" not in str(caught.value)
    with pytest.raises(ValueError, match="holdout"):
        build_rows(candles)


def test_unknown_available_at_is_not_treated_as_known() -> None:
    event = datetime(2024, 6, 1, tzinfo=UTC)
    hidden = DataAvailability.unpublished(event, "macro-release")
    visible = DataAvailability(event, event + timedelta(days=3), "macro-release")
    as_of = event + timedelta(days=10)
    assert admit_observations([visible], as_of) == [visible]
    with pytest.raises(ValueError, match="event_time"):
        admit_observations([visible, hidden], as_of)


def test_calendar_helper_rejects_bar_count_that_skips_a_day() -> None:
    candles = _span(date(2024, 1, 1), 10)
    gapped = candles[:4] + candles[5:]
    with pytest.raises(ValueError, match="календар"):
        require_calendar_days(gapped, 2, 6, 4)
