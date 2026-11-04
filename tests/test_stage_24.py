"""Границы этапа 2.4: время доступности, пропуски и запрет supported."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data.candles import Candle
from wolfai.research.external_join import derivative_features
from wolfai.research.group_compare import evaluate_groups
from wolfai.research.horizon_labels import build_aligned_rows, complete_rows
from wolfai.research.market_activity import volume_to_move
from wolfai.research.quality_contract import STATUS_SUPPORTED
from wolfai.research.research_memory import Stage24Memory
from wolfai.research.source_record import metric_record, usable_at
from wolfai.research.time_alignment import decision_time

UTC = timezone.utc
INGESTED = datetime(2026, 10, 26, tzinfo=UTC)


def test_closed_window_and_future_publication_are_blocked() -> None:
    candles = _path(date(2019, 1, 1), date(2021, 12, 31))
    with pytest.raises(ValueError, match="2022"):
        evaluate_groups(candles + [_candle(date(2022, 1, 1), 100)], [])
    future = metric_record(
        source="binance",
        metric_name="funding",
        event_time=datetime(2021, 6, 1, tzinfo=UTC),
        published_at=datetime(2021, 6, 2, tzinfo=UTC),
        available_at=datetime(2021, 6, 2, tzinfo=UTC),
        ingested_at=INGESTED,
        value=0.01,
        source_version="v1",
        availability_basis="публикация позже события и это указано явно",
    )
    assert usable_at([future], datetime(2021, 6, 1, 12, tzinfo=UTC)) == []
    with pytest.raises(ValueError, match="UTC"):
        metric_record(
            source="binance",
            metric_name="funding",
            event_time=datetime(2021, 6, 1),
            published_at=None,
            available_at=None,
            ingested_at=datetime(2021, 6, 2),
            value=0.01,
            source_version="v1",
        )


def test_event_time_does_not_become_available_at() -> None:
    blocked = metric_record(
        source="fred",
        metric_name="cpi",
        event_time=datetime(2021, 6, 1, tzinfo=UTC),
        published_at=None,
        available_at=None,
        ingested_at=INGESTED,
        value=100.0,
        source_version="latest-vintage",
    )
    assert blocked.quality_status == "forecast_blocked"
    assert blocked.available_at is None
    assert usable_at([blocked], datetime(2021, 6, 2, tzinfo=UTC)) == []


def test_volume_ratio_ignores_the_next_candle_and_gaps_stay_empty() -> None:
    candles = _path(date(2019, 1, 1), date(2019, 6, 1))
    rows = complete_rows(build_aligned_rows(candles, 1))
    target = next(row for row in rows if row.index == 40)
    mutated = list(candles)
    mutated[41] = Candle(
        timestamp=candles[41].timestamp,
        open=1,
        high=80,
        low=0.4,
        close=70,
        volume=50_000,
    )
    again = complete_rows(build_aligned_rows(mutated, 1))
    twin = next(row for row in again if row.index == 40)
    assert twin.features["log_return"] == target.features["log_return"]
    assert twin.features["volume_change"] == target.features["volume_change"]
    assert volume_to_move(float(target.features["volume_change"]), float(target.features["log_return"])) == (
        volume_to_move(float(twin.features["volume_change"]), float(twin.features["log_return"]))
    )
    as_of = decision_time(candles[40])
    later = metric_record(
        source="binance",
        metric_name="binance_vision.um.BTCUSDT.open_interest",
        event_time=as_of + timedelta(days=2),
        published_at=as_of + timedelta(days=2),
        available_at=as_of + timedelta(days=2),
        ingested_at=INGESTED,
        value=20.0,
        source_version="v",
        availability_basis="create_time снимка open interest — момент, когда снимок уже опубликован биржей",
    )
    past_only = _oi_pair(as_of - timedelta(days=1), as_of - timedelta(hours=1))
    fundings = [item for item in past_only if "funding" in item.metric_name]
    assert derivative_features(fundings + [later], as_of) is None
    assert usable_at([later], as_of) == []


def test_repeat_is_stable_and_memory_keeps_the_negative_result(tmp_path) -> None:
    candles = _alternating(date(2019, 1, 1), date(2021, 12, 31))
    first = evaluate_groups(candles, _funding_year())
    second = evaluate_groups(candles, _funding_year())
    assert first["horizons"][0]["models"][0]["brier"] == second["horizons"][0]["models"][0]["brier"]
    week = next(item for item in first["horizons"] if item["horizon_days"] == 7)
    activity = next(model for model in week["models"] if model["name"] == "logistic_market_activity")
    assert activity["block_length"] >= 7
    assert activity["claim_status"] != STATUS_SUPPORTED
    assert week["onchain"] == "not_evaluated"
    assert week["macro"] == "not_evaluated"
    assert week["all_groups"] == "not_evaluated"
    memory = Stage24Memory(tmp_path / "stage24.sqlite")
    payload = _memory_payload(activity["claim_status"])
    assert memory.remember("research", payload) == "stored"
    assert memory.remember("research", dict(payload)) == "duplicate"
    assert memory.count("research") == 1
    with pytest.raises(ValueError, match="supported"):
        memory.remember("candidate", _memory_payload(STATUS_SUPPORTED))
    with pytest.raises(ValueError, match="Validated"):
        memory.remember("validated", payload)
    memory.close()


def _memory_payload(status: str) -> dict[str, object]:
    return {
        "model_name": "logistic_market_activity",
        "config_hash": "l2",
        "dataset_fingerprint": "synthetic",
        "features": ["log_return"],
        "label_definition": "рост",
        "horizon_days": 1,
        "train_start": "2019",
        "train_end": "2020",
        "dev_start": "2021",
        "dev_end": "2021",
        "seed": 1729,
        "source_version": "v",
        "feature_group": "ohlcv",
        "evidence_role": "development_only",
        "claim_status": status,
        "brier": 0.25,
    }


def _funding_year() -> list:
    records = []
    day = datetime(2020, 9, 1, tzinfo=UTC)
    end = datetime(2021, 12, 31, tzinfo=UTC)
    oi = 100.0
    while day <= end:
        for hour in (0, 8, 16):
            stamp = day + timedelta(hours=hour)
            records.append(metric_record(
                source="binance-vision",
                metric_name="binance_vision.um.BTCUSDT.funding_rate",
                event_time=stamp,
                published_at=stamp,
                available_at=stamp,
                ingested_at=INGESTED,
                value=0.0001,
                source_version="zip",
                availability_basis="calc_time — момент, когда ставка уже стоит на бирже, а не время открытия свечи",
            ))
        records.append(metric_record(
            source="binance-vision",
            metric_name="binance_vision.um.BTCUSDT.open_interest",
            event_time=day + timedelta(hours=23, minutes=55),
            published_at=day + timedelta(hours=23, minutes=55),
            available_at=day + timedelta(hours=23, minutes=55),
            ingested_at=INGESTED,
            value=oi,
            source_version="zip",
            availability_basis="create_time снимка open interest — момент, когда снимок уже опубликован биржей",
        ))
        oi *= 1.001
        day += timedelta(days=1)
    return records


def _oi_pair(first: datetime, second: datetime) -> list:
    rows = []
    for index, stamp in enumerate((first, second)):
        rows.append(metric_record(
            source="binance-vision",
            metric_name="binance_vision.um.BTCUSDT.funding_rate",
            event_time=stamp,
            published_at=stamp,
            available_at=stamp,
            ingested_at=INGESTED,
            value=0.0001 * (index + 1),
            source_version="zip",
            availability_basis="calc_time — момент, когда ставка уже стоит на бирже, а не время открытия свечи",
        ))
        rows.append(metric_record(
            source="binance-vision",
            metric_name="binance_vision.um.BTCUSDT.open_interest",
            event_time=stamp,
            published_at=stamp,
            available_at=stamp,
            ingested_at=INGESTED,
            value=10.0 + index,
            source_version="zip",
            availability_basis="create_time снимка open interest — момент, когда снимок уже опубликован биржей",
        ))
    return rows


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


def _candle(day: date, close: float) -> Candle:
    return Candle(
        timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
        open=close,
        high=close * 1.02,
        low=close * 0.98,
        close=close,
        volume=2 + (day.toordinal() % 5),
    )
