"""Устойчивость гипотезы: мало доказательств — это не успех."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import pytest

from wolfai.research.quality_contract import (
    BOOTSTRAP_REPLICATES,
    STATUS_INSUFFICIENT,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    ReliabilityReport,
    linear_percentile,
    moving_block_means,
    noise_champion,
    point_brier,
    sample_stdev,
)
from wolfai.research.reliability import assess_reliability


def _dates(count: int, *, start: date = date(2021, 1, 1), step: int = 1) -> list[date]:
    return [start + timedelta(days=step * offset) for offset in range(count)]


def _assess(
    dates: Sequence[date],
    probabilities: Sequence[float],
    labels: Sequence[int],
    baseline: Sequence[float],
    *,
    horizon_days: int = 1,
    complexity: int = 3,
    hypotheses_examined: int = 1,
    selection_separated: bool = True,
) -> ReliabilityReport:
    return assess_reliability(
        dates,
        probabilities,
        labels,
        baseline,
        horizon_days=horizon_days,
        complexity=complexity,
        hypotheses_examined=hypotheses_examined,
        selection_separated=selection_separated,
    )


def _strong(count: int, **overrides: object) -> ReliabilityReport:
    labels = [1 if index % 2 == 0 else 0 for index in range(count)]
    probabilities = [0.9 if label == 1 else 0.1 for label in labels]
    arguments = {
        "horizon_days": 1,
        "complexity": 3,
        "hypotheses_examined": 1,
        "selection_separated": True,
    }
    arguments.update(overrides)
    return _assess(
        _dates(count),
        probabilities,
        labels,
        [0.5] * count,
        horizon_days=int(arguments["horizon_days"]),
        complexity=int(arguments["complexity"]),
        hypotheses_examined=int(arguments["hypotheses_examined"]),
        selection_separated=bool(arguments["selection_separated"]),
    )


def _gaps(
    probabilities: Sequence[float],
    references: Sequence[float],
    labels: Sequence[int],
) -> list[float]:
    return [
        point_brier(reference, label) - point_brier(probability, label)
        for probability, reference, label in zip(probabilities, references, labels, strict=True)
    ]


def test_strong_separated_signal_can_be_supported() -> None:
    report = _strong(80)
    assert report.status == STATUS_SUPPORTED
    assert report.brier is not None and report.frequency_brier is not None
    assert report.brier < report.frequency_brier
    assert report.brier == pytest.approx(0.01)
    assert report.frequency_brier == pytest.approx(0.25)
    assert report.coin_brier == pytest.approx(0.25)
    assert report.log_loss is not None
    assert [item.bin_index for item in report.calibration] == [0, 1, 2, 3, 4]
    assert report.calibration[1].count == 0
    assert report.calibration[1].mean_probability is None
    assert report.calibration[1].empirical_frequency is None
    assert report.calibration[0].empirical_frequency == pytest.approx(0.0)
    assert report.calibration[4].empirical_frequency == pytest.approx(1.0)
    assert report.uncertainty is not None
    assert report.multiple_testing is not None
    assert report.uncertainty.method == "moving_block_bootstrap"
    assert report.uncertainty.replicates == BOOTSTRAP_REPLICATES
    assert report.uncertainty.overlapping is False
    assert report.uncertainty.block_length == 1
    assert report.uncertainty.low_vs_frequency > 0
    assert report.uncertainty.low_vs_coin > 0
    assert report.early_improvement_vs_frequency is not None
    assert report.late_improvement_vs_frequency is not None
    assert report.early_improvement_vs_frequency > 0
    assert report.late_improvement_vs_frequency > 0
    assert report.uncertainty.mean_improvement_vs_frequency > report.multiple_testing.noise_champion
    assert report.limitations == ()
    heavy = _strong(80, complexity=13)
    assert heavy.status == STATUS_SUPPORTED
    assert any("PySR" in item for item in heavy.limitations)


def test_constant_half_is_not_supported() -> None:
    labels = [1 if index % 2 == 0 else 0 for index in range(80)]
    report = _assess(_dates(80), [0.5] * 80, labels, [0.5] * 80)
    assert report.status == STATUS_INSUFFICIENT
    assert report.status != STATUS_SUPPORTED
    assert len(report.calibration) == 5
    assert report.calibration[2].count == 80
    assert report.calibration[2].mean_probability == pytest.approx(0.5)
    assert report.calibration[2].empirical_frequency == pytest.approx(0.5)
    assert report.uncertainty is not None
    assert report.uncertainty.low_vs_frequency == pytest.approx(0.0)
    assert report.uncertainty.high_vs_frequency == pytest.approx(0.0)


def test_consistently_worse_model_is_rejected() -> None:
    dates = _dates(80)
    probabilities = [0.9] * 80
    labels = [0] * 80
    baseline = [0.5] * 80
    report = _assess(dates, probabilities, labels, baseline)
    assert report.status == STATUS_REJECTED
    assert report.uncertainty is not None
    assert report.uncertainty.low_vs_frequency < 0
    assert report.uncertainty.high_vs_frequency < 0
    assert report.uncertainty.low_vs_coin < 0
    assert report.uncertainty.high_vs_coin < 0
    shared_window = _assess(
        dates,
        probabilities,
        labels,
        baseline,
        selection_separated=False,
    )
    assert shared_window.status == STATUS_REJECTED


def test_overlapping_week_uses_block_length() -> None:
    daily = _dates(40)
    daily_report = _assess(daily, [0.55] * 40, [1] * 40, [0.4] * 40, horizon_days=7)
    assert daily_report.uncertainty is not None
    assert daily_report.uncertainty.overlapping is True
    assert daily_report.uncertainty.block_length >= 7
    assert daily_report.uncertainty.method == "moving_block_bootstrap"
    overlap_note = " ".join(daily_report.limitations)
    assert "независим" in overlap_note
    assert "горизонт" in overlap_note

    weekly = _dates(40, step=7)
    last_target = weekly[-1] + timedelta(days=7)
    assert last_target <= date(2021, 12, 31)
    weekly_report = _assess(weekly, [0.55] * 40, [1] * 40, [0.4] * 40, horizon_days=7)
    assert weekly_report.uncertainty is not None
    assert weekly_report.uncertainty.overlapping is False
    assert weekly_report.uncertainty.block_length == 1


def test_multiple_testing_demotes_noisy_edge() -> None:
    count = 80
    probabilities = [
        1 - math.sqrt(0.05) if index % 2 == 0 else 1 - math.sqrt(0.29)
        for index in range(count)
    ]
    labels = [1] * count
    baseline = [0.5] * count
    dates = _dates(count)
    gaps = _gaps(probabilities, baseline, labels)
    means = moving_block_means(gaps, 1)
    error = sample_stdev(means)
    point_mean = sum(gaps) / len(gaps)

    one = _assess(dates, probabilities, labels, baseline, hypotheses_examined=1)
    many = _assess(dates, probabilities, labels, baseline, hypotheses_examined=10**9)

    assert one.uncertainty is not None and one.multiple_testing is not None
    assert many.uncertainty is not None and many.multiple_testing is not None
    assert one.uncertainty.mean_improvement_vs_frequency == point_mean
    assert many.uncertainty.mean_improvement_vs_frequency == point_mean
    assert one.uncertainty.low_vs_frequency == linear_percentile(means, 0.025)
    assert one.uncertainty.high_vs_frequency == linear_percentile(means, 0.975)
    assert one.uncertainty.bootstrap_se_vs_frequency == error
    assert one.uncertainty.low_vs_frequency > 0
    assert one.uncertainty.low_vs_coin > 0
    assert one.early_improvement_vs_frequency is not None
    assert one.late_improvement_vs_frequency is not None
    assert one.early_improvement_vs_frequency > 0
    assert one.late_improvement_vs_frequency > 0
    assert one.multiple_testing.noise_champion == 0.0
    assert one.status == STATUS_SUPPORTED
    assert many.hypotheses_examined == 10**9
    assert many.multiple_testing.hypotheses_examined == 10**9
    assert many.multiple_testing.noise_champion == noise_champion(error, 10**9)
    assert many.multiple_testing.noise_champion > many.uncertainty.mean_improvement_vs_frequency
    assert many.uncertainty.low_vs_frequency > 0
    assert many.status == STATUS_INSUFFICIENT
    assert "чемпион" in many.multiple_testing.justification
    assert "порог" in many.multiple_testing.justification
    assert any("шум" in item for item in many.limitations)


def test_selection_on_the_same_window_is_not_supported() -> None:
    report = _strong(80, selection_separated=False)
    assert report.brier is not None and report.frequency_brier is not None
    assert report.brier < report.frequency_brier
    assert report.status == STATUS_INSUFFICIENT
    assert report.status != STATUS_SUPPORTED
    assert any("формул" in item for item in report.limitations)


def test_too_few_observations_are_insufficient() -> None:
    report = _strong(10)
    assert report.brier is not None and report.frequency_brier is not None
    assert report.brier < report.frequency_brier
    assert report.status == STATUS_INSUFFICIENT
    assert report.uncertainty is not None
    assert report.multiple_testing is not None
    assert any("30" in item for item in report.limitations)

    boundary = _strong(30)
    assert boundary.status == STATUS_SUPPORTED
    assert _strong(29).status == STATUS_INSUFFICIENT


def test_dates_in_2022_are_rejected() -> None:
    with pytest.raises(ValueError):
        _assess([date(2022, 1, 1)], [0.6], [1], [0.5])
    with pytest.raises(ValueError):
        _assess([date(2022, 12, 25)], [0.6], [1], [0.5], horizon_days=7)
    with pytest.raises(ValueError):
        _assess([date(2021, 12, 25)], [0.6], [1], [0.5], horizon_days=7)
    edge = _assess([date(2021, 12, 24)], [0.4], [0], [0.2], horizon_days=7)
    assert edge.brier is not None
    assert edge.log_loss is not None
    assert edge.uncertainty is None
    assert edge.multiple_testing is None
    assert edge.status == STATUS_INSUFFICIENT


def test_frequency_reference_is_not_the_coin() -> None:
    count = 80
    probabilities = [0.8] * count
    labels = [1] * count
    baseline = [0.2] * count
    report = _assess(_dates(count), probabilities, labels, baseline)
    frequency_gaps = _gaps(probabilities, baseline, labels)
    coin_gaps = _gaps(probabilities, [0.5] * count, labels)
    assert report.uncertainty is not None
    assert report.uncertainty.mean_improvement_vs_frequency == sum(frequency_gaps) / count
    assert report.uncertainty.mean_improvement_vs_coin == sum(coin_gaps) / count
    assert report.frequency_brier == pytest.approx(point_brier(0.2, 1))
    assert report.coin_brier == pytest.approx(point_brier(0.5, 1))
    assert report.uncertainty.mean_improvement_vs_frequency != pytest.approx(
        report.uncertainty.mean_improvement_vs_coin
    )
    assert report.status == STATUS_SUPPORTED


def test_loss_against_only_one_reference_is_not_rejected() -> None:
    count = 80
    report = _assess(_dates(count), [0.4] * count, [1] * count, [0.1] * count)
    assert report.uncertainty is not None
    assert report.uncertainty.low_vs_frequency > 0
    assert report.uncertainty.high_vs_frequency > 0
    assert report.uncertainty.low_vs_coin < 0
    assert report.uncertainty.high_vs_coin < 0
    assert report.status == STATUS_INSUFFICIENT


def test_bootstrap_skipped_when_block_exceeds_series() -> None:
    short = _assess(_dates(4), [0.7] * 4, [1] * 4, [0.5] * 4, horizon_days=7)
    assert short.status == STATUS_INSUFFICIENT
    assert short.uncertainty is None
    assert short.multiple_testing is None
    assert short.brier is not None
    assert short.log_loss is not None
    assert len(short.calibration) == 5

    single = _assess(_dates(1), [0.7], [1], [0.5])
    assert single.uncertainty is None
    assert single.multiple_testing is None
    assert single.brier == pytest.approx(point_brier(0.7, 1))
    assert single.early_improvement_vs_frequency is None
    assert single.late_improvement_vs_frequency == pytest.approx(
        point_brier(0.5, 1) - point_brier(0.7, 1)
    )
    edges = (
        _assess(_dates(1), [0.2], [0], [0.0]),
        _assess(_dates(1), [0.2], [1], [1.0]),
    )
    assert edges[0].frequency_brier == pytest.approx(0.0)
    assert edges[1].frequency_brier == pytest.approx(0.0)


def test_invalid_rows_raise() -> None:
    valid_dates = _dates(2)
    with pytest.raises(ValueError):
        _assess([], [], [], [])
    with pytest.raises(ValueError):
        _assess(valid_dates, [0.2], [0, 1], [0.5, 0.5])
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.2], [2], [0.5])
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.0], [1], [0.5])
    with pytest.raises(ValueError):
        _assess(_dates(1), [1.0], [1], [0.5])
    with pytest.raises(ValueError):
        _assess(_dates(1), [float("nan")], [1], [0.5])
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.2], [1], [1.1])
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.2], [1], [-0.01])
    with pytest.raises(ValueError):
        _assess(valid_dates, [0.2, 0.3], [0, 1], [0.5, 0.5], horizon_days=2)
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.2], [1], [0.5], complexity=0)
    with pytest.raises(ValueError):
        _assess(_dates(1), [0.2], [1], [0.5], hypotheses_examined=0)
    with pytest.raises(ValueError):
        _assess(
            [date(2021, 1, 2), date(2021, 1, 2)],
            [0.2, 0.3],
            [0, 1],
            [0.5, 0.5],
        )
