"""Лаборатория гипотез: формулы, хронология, метрики и память эксперимента."""

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from wolfai.data import Candle
from wolfai.data.history import HistoricalStore
from wolfai.research.engine import (
    ENGINE_VERSION,
    ExperimentMemory,
    Hypothesis,
    build_experiment_rows,
    judge_hypothesis,
    run_experiment,
    score_direction,
)
from wolfai.research.formula import FormulaError

UTC = timezone.utc
SOURCE = "binance-vision"


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
            volume=2,
        ))
        day += timedelta(days=1)
        index += 1
    return candles


def _store(tmp_path: Path, candles: list[Candle]) -> tuple[HistoricalStore, str]:
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_candles(candles, source=SOURCE)
    return store, manifest.content_sha256


def _hypothesis(checksum: str, *, horizon: int = 1, expression: str = "sigmoid(0.5 * momentum_7)") -> Hypothesis:
    return Hypothesis(
        id="hyp-momentum-7",
        expression=expression,
        features=("momentum_7",),
        horizon_days=horizon,
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
        dataset_version=checksum,
        status="proposed",
    )


def test_direction_scores_separate_probability_from_price() -> None:
    scored = score_direction(
        [0.8, 0.2],
        [1, 0],
        frequency=[0.6, 0.6],
    )
    assert scored["brier"] == pytest.approx(0.04)
    assert scored["log_loss"] == pytest.approx(-math.log(0.8))
    assert len(scored["calibration"]) == 5
    assert "mean_absolute_error" not in scored
    assert scored["skill"]["historical_frequency"]["reference_brier"] == pytest.approx(0.26)
    assert scored["skill"]["constant_half"]["reference_brier"] == pytest.approx(0.25)


def test_train_only_improvement_does_not_accept() -> None:
    decision, reason = judge_hypothesis(
        train_brier=0.10,
        train_frequency_brier=0.20,
        validation_brier=0.30,
        validation_frequency_brier=0.22,
        validation_coin_brier=0.25,
    )
    assert decision == "rejected"
    assert "обучен" in reason
    accepted, accepted_reason = judge_hypothesis(
        train_brier=0.20,
        train_frequency_brier=0.20,
        validation_brier=0.15,
        validation_frequency_brier=0.22,
        validation_coin_brier=0.25,
    )
    assert accepted == "candidate"
    assert "не заменяется" in accepted_reason


def test_formula_cannot_execute_python(tmp_path: Path) -> None:
    candles = _span(date(2022, 1, 1), date(2024, 2, 1))
    store, checksum = _store(tmp_path, candles)
    marker = tmp_path / "pwned.txt"
    hypothesis = _hypothesis(
        checksum,
        expression=f"sigmoid(__import__('pathlib').Path({str(marker)!r}).write_text('x'))",
    )
    with pytest.raises(FormulaError):
        run_experiment(store, SOURCE, hypothesis)
    assert not marker.exists()
    with pytest.raises(FormulaError):
        run_experiment(store, SOURCE, _hypothesis(checksum, expression="eval('1+1')"))


def test_simple_hypothesis_is_scored_without_replacing_a_model(tmp_path: Path) -> None:
    candles = _span(date(2022, 1, 1), date(2024, 2, 1))
    store, checksum = _store(tmp_path, candles)
    memory = ExperimentMemory(tmp_path / "experiments.sqlite")
    hypothesis = _hypothesis(checksum)
    first = run_experiment(store, SOURCE, hypothesis, memory)
    second = run_experiment(store, SOURCE, hypothesis, memory)
    assert first.decision == "rejected"
    assert first.promoted is False
    assert first.revisions_confirmed is False
    assert first.availability_basis == "daily_close_known_next_utc_midnight"
    assert first.code_version == ENGINE_VERSION
    assert first.dataset_checksum == checksum
    assert first.horizon_days == 1
    assert first.train_start == date(2019, 1, 1)
    assert first.validation_end == date(2024, 12, 31)
    assert first.direction_metrics["validation"]["brier"] > (
        first.direction_metrics["validation"]["skill"]["historical_frequency"]["reference_brier"]
    )
    assert "brier" not in first.price_metrics["historical_quantile"]["validation"]
    assert "mean_absolute_error" in first.price_metrics["historical_quantile"]["validation"]
    assert "mean_absolute_error" not in first.direction_metrics["validation"]
    assert first.direction_metrics["validation"]["brier"] == second.direction_metrics["validation"]["brier"]
    assert first.reason == second.reason
    with pytest.raises(ValueError, match="не заменяется"):
        memory.promote(hypothesis.id)


def test_rows_do_not_use_the_future_close_as_a_feature(tmp_path: Path) -> None:
    candles = _span(date(2022, 6, 1), date(2022, 9, 1))
    rows = build_experiment_rows(candles, 7)
    row = rows[10]
    index = row.index
    assert row.features["log_return"] == pytest.approx(
        math.log(candles[index].close / candles[index - 1].close)
    )
    assert row.features["momentum_7"] == pytest.approx(
        candles[index].close / candles[index - 7].close - 1
    )
    assert row.label_available > row.decision_available
    assert row.target_date - row.decision_date == timedelta(days=7)
    leaked = list(candles)
    future = leaked[index + 7]
    leaked[index + 7] = Candle(future.timestamp, 1, 2, 0.5, 1, 2)
    changed = build_experiment_rows(leaked, 7)[10]
    assert changed.features == row.features
    assert changed.y_up != row.y_up


def test_seven_day_decision_uses_nonoverlapping_targets(tmp_path: Path) -> None:
    candles = _span(date(2022, 1, 1), date(2024, 2, 1))
    store, checksum = _store(tmp_path, candles)
    result = run_experiment(store, SOURCE, _hypothesis(checksum, horizon=7))
    overlapping = result.direction_metrics["validation_overlapping"]["observations"]
    effective = result.direction_metrics["validation"]["observations"]
    assert result.overlap_stride == 7
    assert effective < overlapping
    assert result.effective_validation_observations == effective
    assert result.promoted is False


def test_holdout_and_dataset_checksum_block_the_run(tmp_path: Path) -> None:
    candles = _span(date(2024, 12, 1), date(2024, 12, 31))
    path = tmp_path / "year.csv"
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp.isoformat()},{candle.open},{candle.high},"
            f"{candle.low},{candle.close},{candle.volume}"
        )
    lines.append("2025-01-01T00:00:00+00:00,1,2,0.5,999999.25,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_csv(path, source=SOURCE)
    assert manifest.holdout_excluded == 1
    hypothesis = _hypothesis(manifest.content_sha256)
    with pytest.raises(ValueError, match="мало") as caught:
        run_experiment(store, SOURCE, hypothesis)
    assert "999999.25" not in str(caught.value)
    loaded = store.read_candles(SOURCE)
    assert all(candle.close != 999999.25 for candle in loaded)
    with pytest.raises(ValueError, match="контрольной суммой"):
        run_experiment(store, SOURCE, _hypothesis("0" * 64))
    with pytest.raises(ValueError, match="первоначально известными"):
        run_experiment(
            store,
            SOURCE,
            _hypothesis(manifest.content_sha256),
            revisions_confirmed=True,
        )


def test_gap_never_reaches_the_experiment(tmp_path: Path) -> None:
    candles = _span(date(2024, 5, 1), date(2024, 5, 20))
    gapped = candles[:10] + candles[12:]
    store = HistoricalStore(tmp_path / "btc")
    with pytest.raises(ValueError, match="пропуск"):
        store.import_candles(gapped, source=SOURCE)
    assert not store.parquet_path(SOURCE).exists()
