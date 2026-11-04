"""Режимы только по прошлым признакам. Отдельная модель состояния не создаётся."""

from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.research.features import feature_vector
from wolfai.research.logistic import score_probabilities
from wolfai.research.periods import SEARCH_TRAIN
from wolfai.research.quality_contract import (
    STATUS_DESCRIPTIVE,
    STATUS_INSUFFICIENT,
    STATUS_SUPPORTED,
    TREND_STATES,
    VOLATILITY_STATES,
    RegimeThresholds,
    TimedRow,
    linear_percentile,
)
from wolfai.research.regimes import fit_regime_thresholds, label_regimes, regime_report

UTC = timezone.utc


def _features(**overrides: float) -> dict[str, float]:
    values = {
        "log_return": 0.0,
        "momentum_7": 0.0,
        "momentum_30": 0.0,
        "volatility_7": 1.0,
    }
    values.update(overrides)
    return values


def _row(
    decision: date,
    target: date,
    features: dict[str, float],
    y_up: int = 1,
    baseline: float = 0.5,
) -> TimedRow:
    return TimedRow(
        decision_date=decision,
        target_date=target,
        features=features,
        y_up=y_up,
        baseline_frequency=baseline,
    )


def _span(start: date, end: date) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    price = 100.0
    while day <= end:
        close = price
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=max(close - 1, 0.5),
            close=close,
            volume=2,
        ))
        price += 1.0
        day += timedelta(days=1)
    return candles


def _scale_closes_from(candles: list[Candle], start_index: int, factor: float) -> list[Candle]:
    scaled: list[Candle] = []
    for index, candle in enumerate(candles):
        if index < start_index:
            scaled.append(candle)
            continue
        close = candle.close * factor
        scaled.append(Candle(
            timestamp=candle.timestamp,
            open=candle.open,
            high=max(candle.high, candle.open, close),
            low=min(candle.low, candle.open, close),
            close=close,
            volume=candle.volume,
        ))
    return scaled


def _timed_rows(candles: list[Candle]) -> list[TimedRow]:
    rows: list[TimedRow] = []
    for index in range(30, len(candles) - 1):
        decision = candles[index].timestamp.date()
        target = candles[index + 1].timestamp.date()
        y_up = 1 if candles[index + 1].close > candles[index].close else 0
        rows.append(TimedRow(
            decision_date=decision,
            target_date=target,
            features=feature_vector(candles, index),
            y_up=y_up,
            baseline_frequency=0.5,
        ))
    return rows


def _slice(report, state: str, window: str):
    found = [item for item in report.slices if item.state == state and item.window == window]
    assert len(found) == 1
    return found[0]


def _expect_volatility(value: float, low: float, high: float) -> str:
    if low == high:
        if value > high:
            return "high_volatility"
        if value < low:
            return "low_volatility"
        return "mid_volatility"
    if value >= high:
        return "high_volatility"
    if value <= low:
        return "low_volatility"
    return "mid_volatility"


def _expect_trend(momentum: float, band: float) -> str:
    if momentum > band:
        return "uptrend"
    if momentum < -band:
        return "downtrend"
    return "sideways"


def _thresholds(low: float, high: float, band: float) -> RegimeThresholds:
    return RegimeThresholds(
        volatility_low=low,
        volatility_high=high,
        trend_band=band,
        fit_start=SEARCH_TRAIN[0],
        fit_end=SEARCH_TRAIN[1],
        fit_observations=60,
        rule="Терцили зафиксированы на 2019–2020 до проверки 2021.",
    )


def test_thresholds_fit_on_2019_2020_only() -> None:
    fit: list[TimedRow] = []
    for index in range(60):
        day = date(2019, 1, 1) + timedelta(days=index)
        fit.append(_row(
            day,
            day + timedelta(days=1),
            _features(volatility_7=float(index + 1), momentum_30=float((index % 9) - 4)),
            index % 2,
            0.4,
        ))
    check = [
        _row(
            date(2021, 2, 1) + timedelta(days=index),
            date(2021, 2, 2) + timedelta(days=index),
            _features(volatility_7=1_000_000.0, momentum_30=-1_000_000.0),
            0,
            0.6,
        )
        for index in range(30)
    ]
    boundary = _row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        _features(volatility_7=1_000_000_000.0, momentum_30=1_000_000_000.0),
        1,
        0.3,
    )
    rows = fit + check + [boundary]
    thresholds = fit_regime_thresholds(rows)
    volatility = [float(index + 1) for index in range(60)]
    magnitude = [abs(float((index % 9) - 4)) for index in range(60)]
    assert thresholds.volatility_low == linear_percentile(volatility, 1 / 3)
    assert thresholds.volatility_high == linear_percentile(volatility, 2 / 3)
    assert thresholds.trend_band == linear_percentile(magnitude, 1 / 3)
    assert thresholds.fit_observations == 60
    assert thresholds.fit_start == SEARCH_TRAIN[0]
    assert thresholds.fit_end == SEARCH_TRAIN[1]
    assert thresholds.fit_start == date(2019, 1, 1)
    assert thresholds.fit_end == date(2020, 12, 31)
    assert "Терцили зафиксированы на 2019–2020 до проверки 2021" in thresholds.rule
    flipped = [
        TimedRow(
            decision_date=row.decision_date,
            target_date=row.target_date,
            features=dict(row.features),
            y_up=1 - row.y_up,
            baseline_frequency=0.1,
        )
        for row in rows
    ]
    assert fit_regime_thresholds(flipped) == thresholds
    quiet_check = fit + [
        TimedRow(
            decision_date=row.decision_date,
            target_date=row.target_date,
            features=_features(volatility_7=0.0, momentum_30=0.0),
            y_up=row.y_up,
            baseline_frequency=row.baseline_frequency,
        )
        for row in check
    ] + [boundary]
    assert fit_regime_thresholds(quiet_check) == thresholds
    nan_day = date(2019, 7, 1)
    nan_features = _features(momentum_30=1.0)
    nan_features["volatility_7"] = float("nan")
    assert fit_regime_thresholds(rows + [_row(nan_day, nan_day + timedelta(days=1), nan_features)]) == thresholds
    changed = list(rows)
    changed[0] = _row(
        changed[0].decision_date,
        changed[0].target_date,
        _features(volatility_7=1e12, momentum_30=0.0),
        changed[0].y_up,
        changed[0].baseline_frequency,
    )
    assert fit_regime_thresholds(changed) != thresholds
    with pytest.raises(ValueError):
        fit_regime_thresholds(fit[:59] + check)
    with pytest.raises(ValueError):
        fit_regime_thresholds(check)
    assert regime_report(rows, horizon_days=1).thresholds == thresholds
    assert regime_report(rows, horizon_days=7).thresholds == thresholds


def test_labels_use_only_past_features() -> None:
    thresholds = _thresholds(1.0, 2.0, 0.3)
    first = _row(
        date(2019, 3, 1),
        date(2019, 3, 2),
        _features(log_return=0.1, momentum_7=0.2, volatility_7=2.5, momentum_30=0.5),
        1,
        0.2,
    )
    same_features = _row(
        date(2019, 3, 1),
        date(2019, 3, 2),
        _features(log_return=-4.0, momentum_7=9.0, volatility_7=2.5, momentum_30=0.5),
        0,
        0.9,
    )
    later = _row(
        date(2019, 3, 2),
        date(2019, 3, 3),
        _features(volatility_7=0.2, momentum_30=-0.5),
        1,
        0.4,
    )
    later_changed = _row(
        date(2019, 3, 2),
        date(2019, 3, 3),
        _features(volatility_7=5.0, momentum_30=1.0),
        0,
        0.1,
    )
    original = label_regimes([first, later], thresholds)
    altered = label_regimes([same_features, later_changed], thresholds)
    assert original[0] == altered[0]
    assert original[0].volatility_state == "high_volatility"
    assert original[0].trend_state == "uptrend"
    assert original[0].volatility_7 == 2.5
    assert original[0].momentum_30 == 0.5
    assert original[1].volatility_state == "low_volatility"
    assert original[1].trend_state == "downtrend"
    assert altered[1].volatility_state == "high_volatility"
    assert altered[1].trend_state == "uptrend"
    assert [point.decision_date for point in original] == [first.decision_date, later.decision_date]
    missing_volatility = _row(date(2019, 4, 1), date(2019, 4, 2), {"momentum_30": 0.1}, 1, 0.5)
    with pytest.raises(ValueError, match="volatility_7"):
        label_regimes([missing_volatility], thresholds)
    missing_momentum = _row(date(2019, 4, 1), date(2019, 4, 2), {"volatility_7": 1.5}, 1, 0.5)
    with pytest.raises(ValueError, match="momentum_30"):
        label_regimes([missing_momentum], thresholds)
    infinite = _features(volatility_7=float("inf"), momentum_30=0.2)
    with pytest.raises(ValueError, match="volatility_7"):
        label_regimes([_row(date(2019, 4, 3), date(2019, 4, 4), infinite, 0, 0.5)], thresholds)


def test_future_close_does_not_change_regime() -> None:
    candles = _span(date(2019, 1, 1), date(2019, 8, 1))
    rows = _timed_rows(candles)
    thresholds = fit_regime_thresholds(rows)
    original = label_regimes(rows, thresholds)
    mutated_candles = _scale_closes_from(candles, 90, 8.0)
    mutated_rows = _timed_rows(mutated_candles)
    relabeled = label_regimes(mutated_rows, thresholds)
    day = candles[89].timestamp.date()
    before = [point for point in original if point.decision_date == day]
    after = [point for point in relabeled if point.decision_date == day]
    assert before == after
    assert feature_vector(mutated_candles, 89) == feature_vector(candles, 89)
    assert feature_vector(mutated_candles, 90) != feature_vector(candles, 90)
    assert fit_regime_thresholds(mutated_rows) != thresholds


def test_small_slice_is_not_a_separate_model() -> None:
    rows: list[TimedRow] = []
    probabilities: list[float] = []
    for index in range(70):
        day = date(2019, 2, 1) + timedelta(days=index)
        rows.append(_row(
            day,
            day + timedelta(days=1),
            _features(volatility_7=1.0, momentum_30=0.0),
            1 if index % 2 == 0 else 0,
            0.25 if index % 2 == 0 else 0.75,
        ))
        probabilities.append(0.2 + (index % 5) * 0.1)
    rows.append(_row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        _features(volatility_7=1.0, momentum_30=0.0),
        1,
        0.5,
    ))
    probabilities.append(0.4)
    check_start = len(rows)
    for index in range(45):
        day = date(2021, 3, 1) + timedelta(days=index)
        rows.append(_row(
            day,
            day + timedelta(days=1),
            _features(volatility_7=1.0, momentum_30=0.0),
            1 if index % 2 == 0 else 0,
            0.25 if index % 2 == 0 else 0.75,
        ))
        probabilities.append(0.2 + (index % 5) * 0.1)
    report = regime_report(rows, probabilities, horizon_days=1)
    expected_order = [
        (state, window)
        for state in (*VOLATILITY_STATES, *TREND_STATES)
        for window in ("fit", "check")
    ]
    assert [(item.state, item.window) for item in report.slices] == expected_order
    assert [point.decision_date for point in report.points] == [row.decision_date for row in rows]
    assert {point.volatility_state for point in report.points} == {"mid_volatility"}
    assert {point.trend_state for point in report.points} == {"sideways"}
    assert report.thresholds.volatility_low == report.thresholds.volatility_high == 1.0
    assert report.thresholds.trend_band == 0.0
    assert report.thresholds.fit_observations == 70
    assert _slice(report, "mid_volatility", "fit").observations == 70
    assert _slice(report, "sideways", "fit").observations == 70
    assert _slice(report, "high_volatility", "fit").observations == 0
    assert _slice(report, "low_volatility", "fit").observations == 0
    assert _slice(report, "uptrend", "fit").observations == 0
    assert _slice(report, "downtrend", "fit").observations == 0
    check_rows = rows[check_start:]
    check_probabilities = probabilities[check_start:]
    scored = score_probabilities(
        check_probabilities,
        [row.y_up for row in check_rows],
        {
            "historical_frequency": [row.baseline_frequency for row in check_rows],
            "constant_half": [0.5] * len(check_rows),
        },
    )
    sideways = _slice(report, "sideways", "check")
    mid = _slice(report, "mid_volatility", "check")
    assert sideways.observations == mid.observations == 45
    assert sideways.brier == scored["brier"]
    assert sideways.frequency_brier == scored["skill"]["historical_frequency"]["reference_brier"]
    assert sideways.coin_brier == scored["skill"]["constant_half"]["reference_brier"]
    assert sideways.base_rate == sum(row.y_up for row in check_rows) / len(check_rows)
    assert sideways.base_rate == scored["base_rate"]
    assert sideways.frequency_brier != sideways.coin_brier
    assert sideways.status == STATUS_DESCRIPTIVE
    assert sideways.separate_model is False
    assert mid.status == STATUS_DESCRIPTIVE
    empty = _slice(report, "high_volatility", "check")
    assert empty.observations == 0
    assert empty.brier is None
    assert empty.frequency_brier is None
    assert empty.coin_brier is None
    assert empty.base_rate is None
    assert empty.status == STATUS_INSUFFICIENT
    assert empty.separate_model is False
    assert _slice(report, "uptrend", "check").observations == 0
    assert _slice(report, "downtrend", "check").observations == 0
    assert _slice(report, "low_volatility", "check").observations < 40
    assert report.separate_model is False
    assert report.status == STATUS_INSUFFICIENT
    assert report.status != STATUS_SUPPORTED
    assert all(item.separate_model is False for item in report.slices)
    assert all(item.status != STATUS_SUPPORTED for item in report.slices)
    assert "Отдельная модель режима не создана." in report.limitations
    quiet = regime_report(rows, horizon_days=1)
    assert all(item.brier is None and item.base_rate is None for item in quiet.slices)
    assert quiet.separate_model is False
    assert quiet.status == STATUS_INSUFFICIENT
    with pytest.raises(ValueError):
        regime_report(rows, [0.5], horizon_days=1)
    with pytest.raises(ValueError):
        regime_report(rows, [0.0] * len(rows), horizon_days=1)
    with pytest.raises(ValueError):
        regime_report(rows, [1.0] * len(rows), horizon_days=1)
    with pytest.raises(ValueError):
        regime_report(rows, [float("nan")] * len(rows), horizon_days=1)
    with pytest.raises(ValueError, match="горизонт"):
        regime_report(rows, horizon_days=2)


def test_high_low_and_trend_states_follow_frozen_fences() -> None:
    fit = [
        _row(
            date(2019, 1, 1) + timedelta(days=index),
            date(2019, 1, 2) + timedelta(days=index),
            _features(volatility_7=float(index), momentum_30=(index - 45) / 45),
            index % 2,
            0.5,
        )
        for index in range(90)
    ]
    boundary = _row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        _features(volatility_7=1_000_000.0, momentum_30=1_000_000.0),
    )
    check = _row(
        date(2021, 6, 1),
        date(2021, 6, 2),
        _features(volatility_7=1_000_000.0, momentum_30=-1_000_000.0),
        0,
    )
    rows = fit + [boundary, check]
    thresholds = fit_regime_thresholds(rows)
    volatility = [float(index) for index in range(90)]
    magnitude = [abs((index - 45) / 45) for index in range(90)]
    assert thresholds.volatility_low == linear_percentile(volatility, 1 / 3)
    assert thresholds.volatility_high == linear_percentile(volatility, 2 / 3)
    assert thresholds.trend_band == linear_percentile(magnitude, 1 / 3)
    assert thresholds.fit_observations == 90
    assert thresholds.volatility_low != thresholds.volatility_high
    points = label_regimes(rows, thresholds)
    for row, point in zip(rows, points, strict=True):
        assert point.volatility_7 == row.features["volatility_7"]
        assert point.momentum_30 == row.features["momentum_30"]
        assert point.volatility_state == _expect_volatility(
            point.volatility_7,
            thresholds.volatility_low,
            thresholds.volatility_high,
        )
        assert point.trend_state == _expect_trend(point.momentum_30, thresholds.trend_band)
    _assert_cases(_thresholds(1.0, 3.0, 0.5), (
        (3.0, 0.0, "high_volatility", "sideways"),
        (1.0, 0.5, "low_volatility", "sideways"),
        (1.0, -0.5, "low_volatility", "sideways"),
        (2.0, 0.5 + 1e-9, "mid_volatility", "uptrend"),
        (2.0, -0.5 - 1e-9, "mid_volatility", "downtrend"),
        (3.1, 0.4, "high_volatility", "sideways"),
        (0.9, -0.6, "low_volatility", "downtrend"),
    ))
    _assert_cases(_thresholds(2.0, 2.0, 0.2), (
        (2.0, 0.0, "mid_volatility", "sideways"),
        (2.0, 0.2, "mid_volatility", "sideways"),
        (2.0, -0.2, "mid_volatility", "sideways"),
        (2.0 + 1e-9, 0.2 + 1e-9, "high_volatility", "uptrend"),
        (2.0 - 1e-9, -0.2 - 1e-9, "low_volatility", "downtrend"),
    ))


def test_regime_rows_reject_2023() -> None:
    bad = _row(date(2021, 12, 30), date(2023, 1, 1), _features())
    later = _row(date(2023, 1, 1), date(2023, 1, 2), _features())
    year_2024 = _row(date(2021, 6, 1), date(2024, 1, 1), _features())
    good = _row(date(2019, 1, 1), date(2019, 1, 2), _features())
    dummy = _thresholds(1.0, 2.0, 0.1)
    with pytest.raises(ValueError, match="2023"):
        fit_regime_thresholds([bad])
    with pytest.raises(ValueError, match="2023"):
        label_regimes([bad], dummy)
    with pytest.raises(ValueError, match="2023"):
        regime_report([bad], horizon_days=1)
    with pytest.raises(ValueError, match="2023"):
        regime_report([later], horizon_days=1)
    with pytest.raises(ValueError, match="2023"):
        fit_regime_thresholds([year_2024])
    with pytest.raises(ValueError, match="2023"):
        regime_report([good, bad], horizon_days=1)
    with pytest.raises(ValueError, match="горизонт"):
        regime_report([bad], horizon_days=4)


def _assert_cases(
    thresholds: RegimeThresholds,
    cases: tuple[tuple[float, float, str, str], ...],
) -> None:
    rows = [
        _row(
            date(2019, 4, 1) + timedelta(days=index),
            date(2019, 4, 2) + timedelta(days=index),
            _features(volatility_7=volatility, momentum_30=momentum),
            index % 2,
            0.4,
        )
        for index, (volatility, momentum, _volatility_state, _trend_state) in enumerate(cases)
    ]
    points = label_regimes(rows, thresholds)
    for point, (_volatility, _momentum, volatility_state, trend_state) in zip(points, cases, strict=True):
        assert point.volatility_state == volatility_state
        assert point.trend_state == trend_state
