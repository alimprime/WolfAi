from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.data.fetch_public import contiguous_prefix, iter_months
from wolfai.forecasting import (
    EVALUATED_PERIODS,
    HistoricalQuantileBaseline,
    Period,
    RandomWalkBaseline,
    period_bounds,
    walk_forward,
)
from wolfai.forecasting.evaluate import (
    audit_period_leak,
    evaluate_split,
    render_markdown,
    score_controls,
)

UTC = timezone.utc


def _span(start: date, end: date) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    index = 0
    while day <= end:
        close = 100.0 + index
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=1,
        ))
        day += timedelta(days=1)
        index += 1
    return candles


def test_periods_are_fixed_before_any_metric() -> None:
    assert period_bounds(Period.TRAINING) == (date(2019, 1, 1), date(2022, 12, 31))
    assert period_bounds(Period.VALIDATION) == (date(2023, 1, 1), date(2024, 12, 31))
    assert period_bounds(Period.HOLDOUT) == (date(2025, 1, 1), date(2026, 9, 30))
    assert Period.HOLDOUT not in EVALUATED_PERIODS


def test_holdout_metrics_are_refused() -> None:
    candles = _span(date(2024, 12, 1), date(2025, 2, 1))
    with pytest.raises(ValueError, match="holdout"):
        evaluate_split(candles, 1, Period.HOLDOUT)


def test_random_walk_median_is_the_last_close_and_has_no_direction() -> None:
    closes = [100 * (1.01**day) for day in range(40)]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        Candle(start + timedelta(days=index), close, close, close, close, 1)
        for index, close in enumerate(closes)
    ]
    quantile = HistoricalQuantileBaseline().forecast(candles, 1)
    walk = RandomWalkBaseline().forecast(candles, 1)
    assert walk.median == candles[-1].close
    assert walk.as_of == quantile.as_of
    assert quantile.median == pytest.approx(candles[-1].close * 1.01)
    assert walk.median != pytest.approx(quantile.median)
    result = walk_forward(candles, horizon_days=1, min_history=10, model=RandomWalkBaseline())
    assert result.directional_calls == 0
    assert result.directional_accuracy is None
    assert result.mean_absolute_error > 0


def test_validation_forecast_never_reads_a_holdout_bar() -> None:
    candles = _span(date(2023, 1, 1), date(2025, 1, 20))
    seen: list[date] = []

    class _Recording:
        def forecast(self, history: list[Candle], horizon_days: int):
            seen.append(history[-1].timestamp.date())
            return HistoricalQuantileBaseline().forecast(history, horizon_days)

    evaluate_split(candles, 1, Period.VALIDATION, _Recording())
    assert seen
    assert max(seen) <= date(2024, 12, 30)
    assert all(day < date(2025, 1, 1) for day in seen)


def test_validation_metrics_ignore_closes_from_2025() -> None:
    candles = _span(date(2023, 1, 1), date(2025, 1, 20))
    checks = audit_period_leak(candles, Period.VALIDATION, date(2025, 1, 1), horizons=(1, 7))
    assert checks
    assert all(item["passed"] for item in checks)


def test_training_metrics_ignore_closes_from_2023() -> None:
    candles = _span(date(2022, 1, 1), date(2023, 2, 15))
    checks = audit_period_leak(candles, Period.TRAINING, date(2023, 1, 1), horizons=(1, 7))
    assert checks
    assert all(item["passed"] for item in checks)


def test_score_controls_requires_the_fixed_history_window() -> None:
    with pytest.raises(ValueError, match="2019-01-01"):
        score_controls(_span(date(2024, 1, 1), date(2024, 6, 1)))


def test_report_keeps_holdout_without_metrics() -> None:
    payload = {
        "rows": 3,
        "first_open": "2019-01-01T00:00:00+00:00",
        "last_open": "2026-09-30T00:00:00+00:00",
        "periods": {
            "training": {
                "start": "2019-01-01",
                "end": "2022-12-31",
                "models": {
                    "historical_quantile": {
                        "1": _metrics(),
                    },
                },
            },
            "validation": {
                "start": "2023-01-01",
                "end": "2024-12-31",
                "models": {
                    "random_walk": {
                        "7": _metrics(direction=None, calls=0),
                    },
                },
            },
        },
        "holdout": {
            "start": "2025-01-01",
            "end": "2026-09-30",
            "metrics": None,
            "reason": "закрытый период: метрики не считались",
        },
        "leakage_checks": [],
    }
    text = render_markdown(payload)
    assert "2025-01-01" in text
    assert "не считались" in text
    assert "holdout" in text
    payload["holdout"]["metrics"] = {"mean_absolute_error": 1}
    with pytest.raises(ValueError, match="holdout"):
        render_markdown(payload)


def test_month_prefix_stops_at_the_first_hole() -> None:
    expected = iter_months((2019, 1), (2026, 9))
    assert len(expected) == 93
    assert expected[0] == (2019, 1)
    assert expected[-1] == (2026, 9)
    present = set(expected) - {(2020, 6)}
    prefix = contiguous_prefix(expected, present)
    assert prefix[-1] == (2020, 5)
    assert (2020, 7) not in prefix


def _metrics(direction: float | None = 0.5, calls: int = 10) -> dict[str, object]:
    return {
        "observations": calls,
        "mean_absolute_error": 1.5,
        "mean_absolute_percentage_error": 0.01,
        "directional_accuracy": direction,
        "directional_calls": calls,
        "interval_coverage": 0.8,
        "mean_interval_width": 2.0,
        "mean_interval_score": 3.0,
    }
