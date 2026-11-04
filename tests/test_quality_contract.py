"""Границы этапа 2.3.2 зафиксированы до новых исследований."""

from datetime import date, timedelta

import pytest

from wolfai.research.periods import SEARCH_CHECK, SEARCH_TRAIN
from wolfai.research.pysr_engine import LIGHT_LIMITS
from wolfai.research.quality_contract import (
    AUDITED_FEATURES,
    FORBIDDEN_FROM,
    MIN_REGIME_OBSERVATIONS,
    MIN_RELIABILITY_OBSERVATIONS,
    PRIOR_231,
    STAGE_LAST_INCLUDED,
    SUCCESS_CRITERION,
    TimedRow,
    assert_stage_row,
    dependence_block_length,
    linear_percentile,
    moving_block_means,
    noise_champion,
    require_stage_rows,
    sample_stdev,
    stage_window,
)


def test_recorded_baseline_matches_light_budget() -> None:
    assert PRIOR_231["tests_passed"] == 164
    assert PRIOR_231["tests_skipped"] == 1
    assert PRIOR_231["pysr_budget"] == dict(LIGHT_LIMITS)
    assert PRIOR_231["pysr_budget"]["niterations"] == 4
    assert PRIOR_231["pysr_budget"]["procs"] == 1
    assert PRIOR_231["optuna_trials"] == 20
    assert "2021" in SUCCESS_CRITERION
    assert STAGE_LAST_INCLUDED == date(2021, 12, 31)
    assert FORBIDDEN_FROM == date(2022, 1, 1)
    assert SEARCH_TRAIN == (date(2019, 1, 1), date(2020, 12, 31))
    assert SEARCH_CHECK == (date(2021, 1, 1), date(2021, 12, 31))
    assert AUDITED_FEATURES == ("log_return", "momentum_7", "momentum_30", "volatility_7")
    assert MIN_RELIABILITY_OBSERVATIONS == 30
    assert MIN_REGIME_OBSERVATIONS == 40


def test_stage_rows_stop_before_2022() -> None:
    assert stage_window(date(2019, 6, 1)) == "fit"
    assert stage_window(date(2021, 6, 1)) == "check"
    with pytest.raises(ValueError, match="2022"):
        assert_stage_row(date(2022, 1, 1), date(2022, 1, 2))
    with pytest.raises(ValueError, match="2023-2024|2022"):
        assert_stage_row(date(2021, 12, 30), date(2023, 1, 2))
    with pytest.raises(ValueError, match="holdout|2022"):
        assert_stage_row(date(2024, 12, 31), date(2025, 1, 1))
    row = TimedRow(
        decision_date=date(2021, 12, 30),
        target_date=date(2021, 12, 31),
        features={"log_return": 0.01},
        y_up=1,
        baseline_frequency=0.5,
    )
    require_stage_rows([row])
    late = TimedRow(
        decision_date=date(2021, 12, 31),
        target_date=date(2021, 12, 31) + timedelta(days=1),
        features={},
        y_up=1,
        baseline_frequency=0.5,
    )
    with pytest.raises(ValueError):
        require_stage_rows([late])


def test_overlapping_horizon_is_not_an_independent_sample() -> None:
    start = date(2021, 1, 1)
    daily = tuple(start + timedelta(days=offset) for offset in range(20))
    weekly = tuple(start + timedelta(days=7 * offset) for offset in range(8))
    assert dependence_block_length(daily, 7) == (7, True)
    assert dependence_block_length(weekly, 7) == (1, False)
    assert dependence_block_length(daily, 1) == (1, False)


def test_noise_champion_grows_only_with_extra_hypotheses() -> None:
    assert noise_champion(0.02, 1) == 0.0
    assert noise_champion(0.02, 100) > noise_champion(0.02, 2)
    deltas = tuple(0.20 if index % 2 == 0 else -0.04 for index in range(80))
    means = moving_block_means(deltas, 1)
    error = sample_stdev(means)
    assert linear_percentile(means, 0.025) > 0.05
    assert noise_champion(error, 10**9) > sum(deltas) / len(deltas)
