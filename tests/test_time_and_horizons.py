"""Время решения, границы меток и одинаковая выборка моделей."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data.candles import Candle
from wolfai.research.horizon_labels import build_aligned_rows, complete_rows, rows_by_role
from wolfai.research.horizon_study import evaluate_candles
from wolfai.research.ohlcv_features import causal_features
from wolfai.research.quality_contract import STATUS_SUPPORTED, dependence_block_length
from wolfai.research.reliability import assess_reliability
from wolfai.research.time_alignment import (
    TRAIN_LABEL_DEADLINE,
    decision_time,
    direction,
    target_known_time,
)

UTC = timezone.utc


def test_open_timestamp_is_not_the_decision_time() -> None:
    candle = Candle(
        timestamp=datetime(2019, 6, 1, tzinfo=UTC),
        open=10,
        high=11,
        low=9,
        close=10,
        volume=1,
    )
    assert candle.timestamp.utcoffset() == timedelta(0)
    assert decision_time(candle) == datetime(2019, 6, 2, tzinfo=UTC)
    assert target_known_time(candle) == decision_time(candle)
    assert direction(10, 10) == 0
    assert direction(10, 11) == 1
    with pytest.raises(ValueError):
        decision_time(Candle(
            timestamp=datetime(2019, 6, 1, 5, tzinfo=timezone(timedelta(hours=5))),
            open=10,
            high=11,
            low=9,
            close=10,
            volume=1,
        ))


def test_features_ignore_the_next_candle() -> None:
    candles = _path(date(2019, 1, 1), date(2019, 6, 1))
    index = 40
    original, reason = causal_features(candles, index)
    assert reason is None
    mutated = list(candles)
    mutated[index + 1] = Candle(
        timestamp=candles[index + 1].timestamp,
        open=1,
        high=50,
        low=0.5,
        close=40,
        volume=9_000,
    )
    changed, _reason = causal_features(mutated, index)
    assert changed == original


def test_train_labels_stop_before_2021_and_dev_stops_before_2022() -> None:
    candles = _path(date(2019, 1, 1), date(2021, 12, 31))
    for horizon in (1, 3, 7):
        rows = complete_rows(build_aligned_rows(candles, horizon))
        train = rows_by_role(rows, "train")
        dev = rows_by_role(rows, "dev")
        assert train
        assert dev
        assert max(row.target_time for row in train) <= TRAIN_LABEL_DEADLINE
        assert min(row.decision_time for row in dev) >= datetime(2021, 1, 1, tzinfo=UTC)
        assert max(row.target_time for row in dev) < datetime(2022, 1, 1, tzinfo=UTC)
        assert all(row.open_time.date() < date(2022, 1, 1) for row in rows)
    with pytest.raises(ValueError, match="2022"):
        build_aligned_rows(candles + [_candle(date(2022, 1, 1), 10)], 1)


def test_overlapping_week_uses_a_block_at_least_as_long_as_the_horizon() -> None:
    start = date(2021, 1, 1)
    daily = tuple(start + timedelta(days=offset) for offset in range(40))
    length, overlapping = dependence_block_length(daily, 7)
    assert overlapping is True
    assert length >= 7
    length_three, overlapping_three = dependence_block_length(daily, 3)
    assert overlapping_three is True
    assert length_three >= 3


def test_models_share_dev_dates_and_2021_cannot_be_supported() -> None:
    signal = evaluate_candles(_alternating(date(2019, 1, 1), date(2021, 12, 31)))
    horizon = signal["horizons"]["1"]
    counts = {model["count"] for model in horizon["models"]}
    assert counts == {horizon["dev_count"]}
    price = next(model for model in horizon["models"] if model["name"] == "logistic_price")
    assert price["brier"] < 0.05
    assert price["claim_status"] != STATUS_SUPPORTED
    assert price["evidence_role"] == "development_only"
    assert price["provisional"] is True
    noise = evaluate_candles(_random(date(2019, 1, 1), date(2021, 12, 31)))
    noisy = next(
        model for model in noise["horizons"]["1"]["models"] if model["name"] == "logistic_price"
    )
    assert noisy["claim_status"] != STATUS_SUPPORTED
    assert noisy["provisional"] is False


def test_research_role_can_see_the_synthetic_signal() -> None:
    candles = _alternating(date(2021, 1, 1), date(2021, 4, 30))
    dates = []
    probabilities = []
    labels = []
    day = date(2021, 1, 1)
    sign = 1.0
    while day <= date(2021, 4, 20):
        label = 0 if sign > 0 else 1
        probability = 0.05 if label == 0 else 0.95
        dates.append(day)
        probabilities.append(probability)
        labels.append(label)
        sign = -sign
        day += timedelta(days=1)
    research = assess_reliability(
        dates,
        probabilities,
        labels,
        [0.5] * len(labels),
        horizon_days=1,
        complexity=3,
        hypotheses_examined=1,
        selection_separated=True,
        evidence_role="research",
    )
    blocked = assess_reliability(
        dates,
        probabilities,
        labels,
        [0.5] * len(labels),
        horizon_days=1,
        complexity=3,
        hypotheses_examined=1,
        selection_separated=True,
        evidence_role="development_only",
    )
    assert research.status == STATUS_SUPPORTED
    assert blocked.status != STATUS_SUPPORTED
    assert blocked.provisional is True
    assert candles[0].timestamp.tzinfo == UTC


def _path(start: date, end: date) -> list[Candle]:
    candles = []
    day = start
    price = 100.0
    while day <= end:
        candles.append(_candle(day, price))
        price *= 1.001
        day += timedelta(days=1)
    return candles


def _alternating(start: date, end: date) -> list[Candle]:
    candles = []
    day = start
    price = 100.0
    sign = 1.0
    while day <= end:
        candles.append(_candle(day, price))
        price *= math.exp(sign)
        sign = -sign
        day += timedelta(days=1)
    return candles


def _random(start: date, end: date) -> list[Candle]:
    import random

    generator = random.Random(1729)
    candles = []
    day = start
    price = 100.0
    while day <= end:
        candles.append(_candle(day, price))
        price *= math.exp(0.01 if generator.random() < 0.5 else -0.01)
        day += timedelta(days=1)
    return candles


def _candle(day: date, close: float) -> Candle:
    return Candle(
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        open=close,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        volume=2 + (day.day % 5),
    )
