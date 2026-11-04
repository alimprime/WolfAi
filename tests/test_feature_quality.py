"""Аудит признаков: распределение, устойчивость, Brier и совпадение рангов."""

import math
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.research.feature_quality import audit_features
from wolfai.research.features import feature_vector
from wolfai.research.logistic import fit_l2_logistic, predict_probability, score_probabilities
from wolfai.research.quality_contract import (
    AUDITED_FEATURES,
    MIN_FEATURE_CHECK,
    MIN_FEATURE_FIT,
    REDUNDANCY_NOTE,
    STABILITY_RULE,
    USEFULNESS_MODEL,
    FeatureAudit,
    FeatureAuditReport,
    FeatureUsefulness,
    TimedRow,
    feature_values,
    linear_percentile,
    sample_stdev,
    stage_window,
)
from wolfai.research.regimes import fit_regime_thresholds, label_regimes

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


def _window(row: TimedRow) -> str | None:
    decision = stage_window(row.decision_date)
    target = stage_window(row.target_date)
    if decision == target:
        return decision
    return None


def _by_name(report: FeatureAuditReport, name: str) -> FeatureAudit:
    found = [item for item in report.features if item.name == name]
    assert len(found) == 1
    return found[0]


def _finite(rows: list[TimedRow], name: str) -> list[tuple[TimedRow, float]]:
    chosen: list[tuple[TimedRow, float]] = []
    for row in rows:
        value = feature_values(row, name)
        if value is not None and math.isfinite(value):
            chosen.append((row, value))
    return chosen


def _scale(value: float, mean: float, deviation: float) -> float:
    if deviation < 1e-8:
        return 0.0
    return (value - mean) / deviation


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


def _point_on(points, day: date):
    found = [point for point in points if point.decision_date == day]
    assert len(found) == 1
    return found[0]


def test_weak_check_does_not_drop_feature() -> None:
    fit: list[TimedRow] = []
    for index in range(60):
        day = date(2019, 6, 1) + timedelta(days=index)
        y_up = 1 if index % 2 == 0 else 0
        fit.append(_row(
            day,
            day + timedelta(days=1),
            _features(log_return=1.0 if y_up == 1 else -1.0, momentum_7=0.1, momentum_30=-0.2, volatility_7=0.3),
            y_up,
            0.5,
        ))
    check: list[TimedRow] = []
    for index in range(40):
        day = date(2021, 3, 1) + timedelta(days=index)
        y_up = 1 if index % 2 == 0 else 0
        check.append(_row(
            day,
            day + timedelta(days=1),
            _features(log_return=-1.0 if y_up == 1 else 1.0, momentum_7=0.1, momentum_30=-0.2, volatility_7=0.3),
            y_up,
            0.5,
        ))
    boundary = _row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        _features(log_return=100.0, momentum_7=0.1, momentum_30=-0.2, volatility_7=0.3),
        1,
        0.5,
    )
    rows = fit + [boundary] + check
    report = audit_features(rows, horizon_days=1)
    assert report.horizon_days == 1
    assert report.dropped_features == ()
    assert tuple(item.name for item in report.features) == AUDITED_FEATURES
    assert all(item.kept is True for item in report.features)
    assert all(item.usefulness.predictive_claim is False for item in report.features)
    assert all(item.usefulness.model == USEFULNESS_MODEL for item in report.features)
    audited = _by_name(report, "log_return")
    assert len(rows) == 101
    assert audited.distribution.observations >= MIN_FEATURE_FIT
    assert audited.usefulness.check_observations >= MIN_FEATURE_CHECK
    assert audited.distribution.observations == 60
    assert audited.usefulness.check_observations == 40
    assert audited.distribution.mean == 0.0
    improvement = audited.usefulness.brier_improvement_vs_frequency
    assert improvement is not None
    assert improvement < 0
    assert audited.kept is True
    assert audited.usefulness.predictive_claim is False


def test_usefulness_is_brier_not_correlation() -> None:
    assert "correlation" not in FeatureUsefulness.__dataclass_fields__
    fit: list[TimedRow] = []
    for index in range(40):
        day = date(2019, 4, 1) + timedelta(days=index)
        log_return = float(index) - 19.5
        y_up = 1 if log_return > 0 else 0
        baseline = 0.2 + (index % 5) * 0.15
        fit.append(_row(
            day,
            day + timedelta(days=1),
            _features(log_return=log_return, momentum_7=0.25, momentum_30=0.1 * ((index % 3) - 1), volatility_7=0.05),
            y_up,
            baseline,
        ))
    nan_day = date(2019, 8, 1)
    nan_features = _features(momentum_7=0.25, momentum_30=0.1, volatility_7=0.05)
    nan_features["log_return"] = float("nan")
    fit.append(_row(nan_day, nan_day + timedelta(days=1), nan_features, 1, 0.4))
    boundary = _row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        _features(log_return=500.0, momentum_7=500.0, momentum_30=500.0, volatility_7=500.0),
        0,
        0.9,
    )
    check: list[TimedRow] = []
    for index in range(30):
        day = date(2021, 4, 1) + timedelta(days=index)
        baseline = 0.2 + (index % 5) * 0.15
        check.append(_row(
            day,
            day + timedelta(days=1),
            _features(
                log_return=(index - 10) * 0.3,
                momentum_7=0.25,
                momentum_30=-0.4,
                volatility_7=0.08,
            ),
            1 if index % 2 == 0 else 0,
            baseline,
        ))
    rows = fit + [boundary] + check
    report = audit_features(rows, horizon_days=1)
    assert not hasattr(_by_name(report, "log_return").usefulness, "correlation")
    for name in ("log_return", "momentum_7"):
        audited = _by_name(report, name)
        assert audited.usefulness.brier is not None
        assert audited.usefulness.log_loss is not None
        assert isinstance(audited.usefulness.brier, float)
        assert isinstance(audited.usefulness.log_loss, float)
        assert audited.usefulness.predictive_claim is False
        assert audited.usefulness.model == USEFULNESS_MODEL
        assert "не доказательство прогноза" in audited.usefulness.limitation
        assert "модулю надёжности" in audited.usefulness.limitation
        expected = _expected_usefulness(rows, name)
        assert audited.usefulness.coefficients == expected["coefficients"]
        assert audited.usefulness.check_observations == expected["check_observations"]
        assert audited.usefulness.brier == expected["brier"]
        assert audited.usefulness.log_loss == expected["log_loss"]
        assert audited.usefulness.frequency_brier == expected["frequency_brier"]
        assert audited.usefulness.coin_brier == expected["coin_brier"]
        assert audited.usefulness.brier_improvement_vs_frequency == expected["frequency_improvement"]
        assert audited.usefulness.brier_improvement_vs_coin == expected["coin_improvement"]
        assert audited.usefulness.frequency_brier != audited.usefulness.coin_brier

    short_check = audit_features(_simple_rows(40, 29), horizon_days=1)
    short = _by_name(short_check, "log_return").usefulness
    assert short.coefficients is not None
    assert short.check_observations == 29
    assert short.check_observations < MIN_FEATURE_CHECK
    assert short.brier is None
    assert short.log_loss is None
    assert short.frequency_brier is None
    assert short.coin_brier is None
    assert short.brier_improvement_vs_frequency is None
    assert short.brier_improvement_vs_coin is None

    short_fit = audit_features(_simple_rows(39, 30), horizon_days=1)
    early = _by_name(short_fit, "log_return").usefulness
    assert early.check_observations == 30
    assert early.coefficients is None
    assert early.brier is None
    assert early.log_loss is None


def test_fit_coefficients_ignore_check_rows() -> None:
    fit: list[TimedRow] = []
    for index in range(48):
        day = date(2019, 3, 1) + timedelta(days=index)
        y_up = 1 if index % 2 == 0 else 0
        fit.append(_row(
            day,
            day + timedelta(days=1),
            _features(
                log_return=1.0 if y_up == 1 else -1.0,
                momentum_7=0.01 * index,
                momentum_30=-0.02 * ((index % 5) - 2),
                volatility_7=0.2 + 0.001 * index,
            ),
            y_up,
            0.45,
        ))
    check: list[TimedRow] = []
    for index in range(32):
        day = date(2021, 5, 1) + timedelta(days=index)
        check.append(_row(
            day,
            day + timedelta(days=1),
            _features(log_return=0.05 * index, momentum_7=-0.3, momentum_30=0.4, volatility_7=0.7),
            1 if index % 3 == 0 else 0,
            0.55,
        ))
    boundary = _row(
        date(2020, 12, 30),
        date(2021, 1, 2),
        _features(log_return=4.0, momentum_7=4.0, momentum_30=4.0, volatility_7=4.0),
        1,
        0.2,
    )
    original = fit + check + [boundary]
    report = audit_features(original, horizon_days=1)
    mutated_check: list[TimedRow] = []
    for row in original:
        if _window(row) != "check":
            mutated_check.append(row)
            continue
        mutated_check.append(TimedRow(
            decision_date=row.decision_date,
            target_date=row.target_date,
            features=_features(log_return=-80.0, momentum_7=70.0, momentum_30=-60.0, volatility_7=50.0),
            y_up=1 - row.y_up,
            baseline_frequency=0.15,
        ))
    changed = audit_features(mutated_check, horizon_days=1)
    for left, right in zip(report.features, changed.features, strict=True):
        assert left.usefulness.coefficients is not None
        assert left.usefulness.coefficients == right.usefulness.coefficients
        assert left.distribution.mean == right.distribution.mean
    assert _by_name(report, "log_return").usefulness.brier != _by_name(changed, "log_return").usefulness.brier
    moved_boundary = [
        TimedRow(
            decision_date=boundary.decision_date,
            target_date=boundary.target_date,
            features=_features(log_return=-999.0, momentum_7=-999.0, momentum_30=-999.0, volatility_7=-999.0),
            y_up=0,
            baseline_frequency=0.05,
        )
        if row is boundary else row
        for row in original
    ]
    assert audit_features(moved_boundary, horizon_days=1).features[0].usefulness.coefficients == (
        report.features[0].usefulness.coefficients
    )


def test_distribution_marks_train_fence_anomalies() -> None:
    fit: list[TimedRow] = []
    for index in range(20):
        day = date(2019, 2, 1) + timedelta(days=index)
        fit.append(_placed(day, log_return=0.0, y_up=0))
    for index in range(5):
        day = date(2019, 3, 1) + timedelta(days=index)
        if index == 0:
            features = {"momentum_30": float(day.toordinal()), "volatility_7": 3.0, "log_return": None}
        elif index == 1:
            features = {"momentum_30": float(day.toordinal()), "volatility_7": 3.0}
        elif index == 2:
            features = {"momentum_30": float(day.toordinal()), "volatility_7": 3.0, "log_return": float("nan")}
        elif index == 3:
            features = {"momentum_30": float(day.toordinal()), "volatility_7": 3.0, "log_return": float("inf")}
        else:
            features = {"momentum_30": float(day.toordinal()), "volatility_7": 3.0, "log_return": True}
        fit.append(_row(day, day + timedelta(days=1), features, 1, 0.5))
    for index in range(20):
        day = date(2019, 4, 1) + timedelta(days=index)
        fit.append(_placed(day, log_return=100.0, y_up=1))
    finite_logs = [value for _, value in _finite(fit, "log_return")]
    lower, upper = _fences(finite_logs)
    check = [
        _check_row(date(2021, 5, 1), lower),
        _check_row(date(2021, 5, 2), upper),
        _check_row(date(2021, 5, 3), lower - 1.0),
        _check_row(date(2021, 5, 4), upper + 1.0),
        _check_row(date(2021, 5, 5), 50.0),
    ]
    boundary = _row(
        date(2020, 12, 31),
        date(2021, 1, 1),
        {"log_return": 10_000.0, "momentum_30": 10_000.0, "volatility_7": 10_000.0},
        1,
        0.5,
    )
    rotated = fit[-1:] + fit[:-1]
    assert _half_shift([value for _, value in _finite(rotated, "momentum_30")]) != _half_shift(
        [value for _, value in sorted(_finite(fit, "momentum_30"), key=lambda item: item[0].decision_date)]
    )
    report = audit_features(rotated + [boundary] + check, horizon_days=1)
    assert tuple(item.name for item in report.features) == AUDITED_FEATURES
    assert report.dropped_features == ()
    log_audit = _by_name(report, "log_return")
    assert log_audit.distribution.window == "2019-2020"
    assert log_audit.distribution.observations == 40
    assert log_audit.distribution.missing == 3
    assert log_audit.distribution.non_finite == 2
    assert log_audit.distribution.mean == 50.0
    assert log_audit.distribution.stdev == pytest.approx(sample_stdev(finite_logs))
    assert log_audit.distribution.minimum == 0.0
    assert log_audit.distribution.maximum == 100.0
    assert log_audit.distribution.q10 == linear_percentile(finite_logs, 0.1)
    assert log_audit.distribution.q50 == linear_percentile(finite_logs, 0.5)
    assert log_audit.distribution.q90 == linear_percentile(finite_logs, 0.9)
    assert log_audit.distribution.anomalies == 0
    assert log_audit.check_anomalies == 2
    assert log_audit.usefulness.coefficients is not None
    assert log_audit.usefulness.brier is None
    assert log_audit.usefulness.check_observations == 5
    expected_shift = _half_shift([value for _, value in _finite(fit, "log_return")])
    assert expected_shift > 1
    assert log_audit.stability.standardized_shift == pytest.approx(expected_shift)
    assert log_audit.stability.early_mean == 0.0
    assert log_audit.stability.late_mean == 100.0
    assert log_audit.stability.limitation == STABILITY_RULE
    assert any("неустойчив" in item for item in log_audit.limitations)
    momentum = _by_name(report, "momentum_30")
    ordered_momentum = [value for _, value in sorted(_finite(fit, "momentum_30"), key=lambda item: item[0].decision_date)]
    assert momentum.stability.standardized_shift == pytest.approx(_half_shift(ordered_momentum))
    assert momentum.distribution.observations == 45
    empty = _by_name(report, "momentum_7")
    assert empty.distribution.observations == 0
    assert empty.distribution.missing == 45
    assert empty.distribution.non_finite == 0
    assert empty.distribution.mean is None
    assert empty.distribution.stdev is None
    assert empty.distribution.q10 is None
    assert empty.distribution.minimum is None
    assert empty.distribution.anomalies == 0
    assert empty.check_anomalies == 0
    assert empty.stability.early_mean is None
    assert empty.stability.late_mean is None
    assert empty.stability.standardized_shift is None
    assert empty.usefulness.coefficients is None
    assert empty.usefulness.check_observations == 0
    volatility = _by_name(report, "volatility_7")
    assert volatility.distribution.observations == 45
    assert volatility.distribution.stdev == 0.0
    assert volatility.distribution.anomalies == 0
    assert volatility.stability.standardized_shift == 0.0
    for item in report.features:
        assert item.kept is True
        assert item.usefulness.predictive_claim is False
        assert item.distribution.window == "2019-2020"
        assert "доказывает прогноз" not in " ".join(item.limitations)

    constant_fit = [
        _row(
            date(2019, 1, 1) + timedelta(days=index),
            date(2019, 1, 2) + timedelta(days=index),
            _features(log_return=3.0),
            index % 2,
        )
        for index in range(6)
    ]
    constant_check = [
        _row(date(2021, 1, 1), date(2021, 1, 2), _features(log_return=3.0), 1),
        _row(date(2021, 1, 2), date(2021, 1, 3), _features(log_return=9.0), 0),
    ]
    constant = _by_name(audit_features(constant_fit + constant_check, horizon_days=1), "log_return")
    assert constant.distribution.anomalies == 0
    assert constant.check_anomalies == 1
    assert constant.distribution.stdev == 0.0
    assert constant.stability.standardized_shift == 0.0
    assert constant.stability.early_mean == 3.0
    assert constant.stability.late_mean == 3.0
    assert all("неустойчив" not in item for item in constant.limitations)

    single = _by_name(
        audit_features(
            [
                _row(date(2019, 2, 1), date(2019, 2, 2), _features(log_return=1.0), 1),
                _row(date(2021, 2, 1), date(2021, 2, 2), _features(log_return=1000.0), 0),
            ],
            horizon_days=1,
        ),
        "log_return",
    )
    assert single.distribution.observations == 1
    assert single.distribution.anomalies == 0
    assert single.check_anomalies == 0
    assert single.distribution.mean == 1.0
    assert single.distribution.stdev is None
    assert single.distribution.q50 == 1.0
    assert single.stability.early_mean is None
    assert single.stability.late_mean is None
    assert single.stability.standardized_shift is None
    assert single.usefulness.coefficients is None
    assert single.usefulness.brier is None
    assert single.usefulness.check_observations == 1


def test_redundancy_is_descriptive_rank_agreement() -> None:
    rows = [
        _pair_row(date(2019, 5, 1), 1.0, 1.0, 5.0, 1.0),
        _pair_row(date(2019, 5, 2), 2.0, 3.0, 5.0, 1.0),
        _pair_row(date(2019, 5, 3), 2.0, 2.0, 5.0, None),
        _pair_row(date(2019, 5, 4), 3.0, 4.0, 5.0, None),
    ]
    report = audit_features(rows, horizon_days=1)
    names = AUDITED_FEATURES
    expected_pairs = [
        (left, right)
        for left_index, left in enumerate(names)
        for right in names[left_index + 1:]
    ]
    assert [(item.left, item.right) for item in report.redundancy] == expected_pairs
    assert all(item.note == REDUNDANCY_NOTE for item in report.redundancy)
    agreement = {(item.left, item.right): item.rank_agreement for item in report.redundancy}
    assert agreement[("log_return", "momentum_7")] == pytest.approx(0.3 * math.sqrt(10))
    assert agreement[("log_return", "momentum_30")] is None
    assert agreement[("log_return", "volatility_7")] is None
    assert agreement[("momentum_7", "momentum_30")] is None
    assert agreement[("momentum_7", "volatility_7")] is None
    assert agreement[("momentum_30", "volatility_7")] is None
    log_audit = _by_name(report, "log_return")
    assert log_audit.stability.standardized_shift == pytest.approx(math.sqrt(1.5))
    assert log_audit.stability.standardized_shift > 1
    assert "неустойчив" in log_audit.limitations[1]
    assert "momentum_7" in log_audit.limitations[2]
    assert log_audit.limitations[0] == log_audit.usefulness.limitation
    momentum = _by_name(report, "momentum_7")
    assert momentum.stability.standardized_shift == pytest.approx(math.sqrt(0.6))
    assert momentum.stability.standardized_shift < 1
    assert all("неустойчив" not in item for item in momentum.limitations)
    assert any("log_return" in item for item in momentum.limitations)
    assert _by_name(report, "momentum_30").limitations == (_by_name(report, "momentum_30").usefulness.limitation,)
    assert _by_name(report, "momentum_30").stability.standardized_shift == 0.0
    assert "доказывает прогноз" not in " ".join(log_audit.limitations)
    assert log_audit.usefulness.coefficients is None
    assert log_audit.usefulness.check_observations == 0

    reversed_rows = [
        _pair_row(date(2019, 6, 1) + timedelta(days=index), float(index + 1), float(5 - index), 0.0, 0.0)
        for index in range(5)
    ]
    reversed_report = audit_features(reversed_rows, horizon_days=1)
    reversed_agreement = {
        (item.left, item.right): item.rank_agreement for item in reversed_report.redundancy
    }
    assert reversed_agreement[("log_return", "momentum_7")] == pytest.approx(-1.0)
    assert reversed_agreement[("log_return", "momentum_7")] != pytest.approx(1.0)
    assert all(item.note == REDUNDANCY_NOTE for item in reversed_report.redundancy)


def test_features_do_not_read_future_candles() -> None:
    candles = _span(date(2019, 1, 1), date(2019, 8, 1))
    for index in (30, 89):
        mutated = _scale_closes_from(candles, index + 1, 8.0)
        assert feature_vector(mutated, index) == feature_vector(candles, index)
    changed_at_decision = _scale_closes_from(candles, 89, 8.0)
    assert feature_vector(changed_at_decision, 89) != feature_vector(candles, 89)
    thresholds = fit_regime_thresholds(_timed_rows(candles))
    day = candles[89].timestamp.date()
    original = label_regimes(_timed_rows(candles), thresholds)
    mutated = _scale_closes_from(candles, 90, 8.0)
    relabeled = label_regimes(_timed_rows(mutated), thresholds)
    assert _point_on(relabeled, day) == _point_on(original, day)
    assert feature_vector(mutated, 89) == feature_vector(candles, 89)


def test_stage_rejects_2022_target() -> None:
    bad = _row(date(2021, 12, 30), date(2022, 1, 1), _features())
    later = _row(date(2022, 6, 1), date(2022, 6, 2), _features())
    good = _row(date(2019, 1, 1), date(2019, 1, 8), _features())
    with pytest.raises(ValueError, match="2022"):
        audit_features([bad], horizon_days=1)
    with pytest.raises(ValueError, match="2022"):
        audit_features([later], horizon_days=1)
    with pytest.raises(ValueError, match="2022"):
        audit_features([good, bad], horizon_days=1)
    with pytest.raises(ValueError, match="горизонт"):
        audit_features([good], horizon_days=2)
    with pytest.raises(ValueError, match="горизонт"):
        audit_features([good], horizon_days=0)
    with pytest.raises(ValueError, match="горизонт"):
        audit_features([bad], horizon_days=3)
    report = audit_features([good], horizon_days=7)
    assert report.horizon_days == 7
    assert report.dropped_features == ()
    assert all(item.kept is True for item in report.features)
    with pytest.raises(ValueError):
        audit_features([], horizon_days=1)


def _simple_rows(fit_count: int, check_count: int) -> list[TimedRow]:
    rows: list[TimedRow] = []
    for index in range(fit_count):
        day = date(2019, 1, 1) + timedelta(days=index)
        rows.append(_row(day, day + timedelta(days=1), _features(log_return=float(index + 1)), index % 2, 0.4))
    for index in range(check_count):
        day = date(2021, 1, 1) + timedelta(days=index)
        rows.append(_row(day, day + timedelta(days=1), _features(log_return=float(index - 3)), index % 2, 0.6))
    return rows


def _expected_usefulness(rows: list[TimedRow], name: str) -> dict[str, object]:
    fit = _finite([row for row in rows if _window(row) == "fit"], name)
    check = _finite([row for row in rows if _window(row) == "check"], name)
    values = [value for _, value in fit]
    mean = sum(values) / len(values)
    deviation = sample_stdev(values)
    design = [[_scale(value, mean, deviation)] for value in values]
    labels = [row.y_up for row, _ in fit]
    coefficients = tuple(fit_l2_logistic(design, labels))
    probabilities = [
        predict_probability(list(coefficients), [_scale(value, mean, deviation)])
        for _, value in check
    ]
    scored = score_probabilities(
        probabilities,
        [row.y_up for row, _ in check],
        {
            "historical_frequency": [row.baseline_frequency for row, _ in check],
            "constant_half": [0.5] * len(check),
        },
    )
    skill = scored["skill"]
    return {
        "coefficients": coefficients,
        "check_observations": len(check),
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "frequency_brier": skill["historical_frequency"]["reference_brier"],
        "coin_brier": skill["constant_half"]["reference_brier"],
        "frequency_improvement": skill["historical_frequency"]["mean_brier_improvement"],
        "coin_improvement": skill["constant_half"]["mean_brier_improvement"],
    }


def _placed(day: date, log_return: float, y_up: int) -> TimedRow:
    return _row(
        day,
        day + timedelta(days=1),
        {
            "log_return": log_return,
            "momentum_30": float(day.toordinal()),
            "volatility_7": 3.0,
        },
        y_up,
        0.5,
    )


def _check_row(day: date, log_return: float) -> TimedRow:
    return _row(
        day,
        day + timedelta(days=1),
        {"log_return": log_return, "momentum_30": 1.0, "volatility_7": 3.0},
        1,
        0.5,
    )


def _fences(values: list[float]) -> tuple[float, float]:
    q1 = linear_percentile(values, 0.25)
    q3 = linear_percentile(values, 0.75)
    iqr = q3 - q1
    return q1 - 3.0 * iqr, q3 + 3.0 * iqr


def _half_shift(values: list[float]) -> float:
    ordered = list(values)
    middle = len(ordered) // 2
    early = ordered[:middle]
    late = ordered[middle:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    return abs(late_mean - early_mean) / sample_stdev(ordered)


def _pair_row(
    day: date,
    log_return: float,
    momentum_7: float,
    momentum_30: float,
    volatility_7: float | None,
) -> TimedRow:
    features: dict[str, float] = {
        "log_return": log_return,
        "momentum_7": momentum_7,
        "momentum_30": momentum_30,
    }
    if volatility_7 is not None:
        features["volatility_7"] = volatility_7
    return _row(day, day + timedelta(days=1), features, 1, 0.5)
