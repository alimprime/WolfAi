"""Окно этапа обрывается до разбора цен 2022 и позже."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data.candles import Candle
from wolfai.research.comparison import known_pattern_status, split_stage, to_timed
from wolfai.research.feature_quality import audit_features
from wolfai.research.quality_contract import STATUS_SUPPORTED
from wolfai.research.regimes import regime_report
from wolfai.research.stage_data import load_stage_candles

UTC = timezone.utc


def test_loader_stops_before_parsing_2022_prices(tmp_path) -> None:
    path = tmp_path / "btc.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2019-01-01T00:00:00+00:00,10,11,9,10,1\n"
        "2019-01-02T00:00:00+00:00,10,11,9,10.5,1\n"
        "2022-01-01T00:00:00+00:00,1,1,1,FUTURE,1\n"
        "2023-06-01T00:00:00+00:00,1,1,1,ALSO_FUTURE,1\n",
        encoding="utf-8",
    )
    candles = load_stage_candles(path)
    assert [candle.timestamp.date() for candle in candles] == [date(2019, 1, 1), date(2019, 1, 2)]
    with pytest.raises(ValueError, match="2022"):
        split_stage(
            candles + [Candle(
                timestamp=datetime(2022, 1, 1, tzinfo=UTC),
                open=1,
                high=1,
                low=1,
                close=1,
                volume=1,
            )],
            1,
        )


def test_synthetic_signal_is_not_the_same_as_noise() -> None:
    signal = known_pattern_status(_alternating(date(2019, 1, 1), date(2021, 12, 31)))
    noise = known_pattern_status(_random_walk(date(2019, 1, 1), date(2021, 12, 31)))
    assert signal["signal"] == STATUS_SUPPORTED
    assert signal["noise"] != STATUS_SUPPORTED
    assert noise["signal"] != STATUS_SUPPORTED
    assert noise["noise"] != STATUS_SUPPORTED
    candles = _alternating(date(2019, 1, 1), date(2021, 12, 31))
    fit, check = split_stage(candles, 1)
    audit = audit_features(to_timed(fit + check), horizon_days=1)
    assert audit.dropped_features == ()
    assert all(item.kept and item.usefulness.predictive_claim is False for item in audit.features)
    log_return = next(item for item in audit.features if item.name == "log_return")
    assert log_return.usefulness.brier is not None
    assert log_return.usefulness.brier < log_return.usefulness.coin_brier
    regimes = regime_report(to_timed(fit + check), None, horizon_days=1)
    assert regimes.separate_model is False


def _alternating(start: date, end: date) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    price = 100.0
    sign = 1.0
    while day <= end:
        close = price
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=2,
        ))
        price = close * math.exp(sign)
        sign = -sign
        day += timedelta(days=1)
    return candles


def _random_walk(start: date, end: date) -> list[Candle]:
    import random

    generator = random.Random(1729)
    candles: list[Candle] = []
    day = start
    price = 100.0
    while day <= end:
        close = price
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=2,
        ))
        price = close * math.exp(0.01 if generator.random() < 0.5 else -0.01)
        day += timedelta(days=1)
    return candles
