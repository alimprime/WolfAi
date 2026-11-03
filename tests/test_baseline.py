from datetime import datetime, timedelta, timezone

import pytest

from wolfai.data import Candle, daily_bar_available_at
from wolfai.forecasting import HistoricalQuantileBaseline, walk_forward

UTC = timezone.utc


def _series(closes: list[float]) -> list[Candle]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Candle(
            timestamp=start + timedelta(days=index),
            open=close,
            high=close,
            low=close,
            close=close,
            volume=1,
        )
        for index, close in enumerate(closes)
    ]


def test_constant_daily_return_sets_median() -> None:
    closes = [100 * (1.01**day) for day in range(40)]
    forecast = HistoricalQuantileBaseline().forecast(_series(closes), horizon_days=1)
    assert forecast.median == pytest.approx(closes[-1] * 1.01)
    assert forecast.quantiles[0.1] == pytest.approx(forecast.quantiles[0.9])


def test_walk_forward_does_not_use_the_future() -> None:
    closes = [100 + day for day in range(50)]
    candles = _series(closes)
    model = HistoricalQuantileBaseline()
    errors: list[float] = []
    for end in range(10, len(candles) - 1 + 1):
        history = candles[:end]
        prediction = model.forecast(history, 1)
        target = candles[end - 1 + 1]
        assert prediction.as_of == daily_bar_available_at(history[-1])
        assert daily_bar_available_at(target) > prediction.as_of
        assert target.timestamp == prediction.as_of
        errors.append(abs(prediction.median - target.close))
    result = walk_forward(candles, horizon_days=1, min_history=10)
    assert result.observations == len(errors) == 40
    assert result.mean_absolute_error == pytest.approx(sum(errors) / len(errors))
    assert result.mean_absolute_percentage_error >= 0


def test_seven_day_horizon_uses_only_past_returns() -> None:
    closes = [100 + day for day in range(40)]
    candles = _series(closes)
    forecast = HistoricalQuantileBaseline().forecast(candles[:20], horizon_days=7)
    full = HistoricalQuantileBaseline().forecast(candles, horizon_days=7)
    assert forecast.as_of == daily_bar_available_at(candles[19])
    assert forecast.as_of == candles[19].timestamp + timedelta(days=1)
    assert forecast.last_close == candles[19].close
    assert forecast.median != full.median


def test_future_close_does_not_change_the_forecast() -> None:
    candles = _series([100 + day for day in range(40)])
    original = HistoricalQuantileBaseline().forecast(candles[:20], horizon_days=7)
    leaked = [
        candle if index < 20 else Candle(
            candle.timestamp, 10, 11, 9, 10.5, candle.volume,
        )
        for index, candle in enumerate(candles)
    ]
    assert HistoricalQuantileBaseline().forecast(leaked[:20], horizon_days=7) == original
    contaminated = HistoricalQuantileBaseline().forecast(leaked[:27], horizon_days=7)
    assert contaminated.as_of > original.as_of
    assert contaminated.median != original.median


def test_unordered_history_is_rejected() -> None:
    candles = _series([100 + day for day in range(12)])
    broken = candles[:10] + [candles[11], candles[10]]
    with pytest.raises(ValueError, match="ещё не известен"):
        HistoricalQuantileBaseline().forecast(broken, horizon_days=1)
