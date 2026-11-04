"""Состояния рынка по уже известным признакам.

Терцили фиксируются на 2019–2020. Отдельная модель режима не создаётся.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from wolfai.research.logistic import score_probabilities
from wolfai.research.periods import SEARCH_TRAIN
from wolfai.research.quality_contract import (
    MIN_REGIME_OBSERVATIONS,
    MIN_THRESHOLD_OBSERVATIONS,
    NAMED_REGIMES,
    STATUS_DESCRIPTIVE,
    STATUS_INSUFFICIENT,
    TREND_STATES,
    VOLATILITY_STATES,
    RegimePoint,
    RegimeReport,
    RegimeSlice,
    RegimeThresholds,
    TimedRow,
    feature_values,
    linear_percentile,
    require_stage_rows,
    stage_window,
)

_THRESHOLD_RULE = (
    "Терцили зафиксированы на 2019–2020 до проверки 2021. "
    "Строки проверки на числа порогов не влияют."
)
_LIMITATIONS = (
    "Отдельная модель режима не создана.",
    "Срезы описывают уже известные состояния и не подбирают формулу режима.",
)


def fit_regime_thresholds(rows: Sequence[TimedRow]) -> RegimeThresholds:
    """Терцили volatility_7 и полоса |momentum_30| только по окну 2019–2020."""
    require_stage_rows(rows)
    chosen = _threshold_values(rows)
    if len(chosen) < MIN_THRESHOLD_OBSERVATIONS:
        raise ValueError("для порогов режима недостаточно строк 2019–2020")
    volatility = [item[0] for item in chosen]
    magnitude = [abs(item[1]) for item in chosen]
    return RegimeThresholds(
        volatility_low=linear_percentile(volatility, 1 / 3),
        volatility_high=linear_percentile(volatility, 2 / 3),
        trend_band=linear_percentile(magnitude, 1 / 3),
        fit_start=SEARCH_TRAIN[0],
        fit_end=SEARCH_TRAIN[1],
        fit_observations=len(chosen),
        rule=_THRESHOLD_RULE,
    )


def label_regimes(
    rows: Sequence[TimedRow],
    thresholds: RegimeThresholds,
) -> tuple[RegimePoint, ...]:
    """Метка строки только по её volatility_7 и momentum_30."""
    require_stage_rows(rows)
    points: list[RegimePoint] = []
    for row in rows:
        volatility = _finite_feature(row, "volatility_7")
        momentum = _finite_feature(row, "momentum_30")
        points.append(RegimePoint(
            decision_date=row.decision_date,
            volatility_state=_volatility_state(volatility, thresholds.volatility_low, thresholds.volatility_high),
            trend_state=_trend_state(momentum, thresholds.trend_band),
            volatility_7=volatility,
            momentum_30=momentum,
        ))
    return tuple(points)


def regime_report(
    rows: Sequence[TimedRow],
    probabilities: Sequence[float] | None = None,
    *,
    horizon_days: int,
) -> RegimeReport:
    """Пороги 2019–2020 и срезы состояний. Отдельная модель режима не создаётся."""
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    require_stage_rows(rows)
    if probabilities is not None:
        _validate_probabilities(probabilities, len(rows))
    thresholds = fit_regime_thresholds(rows)
    points = label_regimes(rows, thresholds)
    slices = _slices(rows, points, probabilities)
    return RegimeReport(
        thresholds=thresholds,
        points=points,
        slices=slices,
        separate_model=False,
        status=_overall_status(slices),
        limitations=_LIMITATIONS,
    )


def _threshold_values(rows: Sequence[TimedRow]) -> list[tuple[float, float]]:
    chosen: list[tuple[float, float]] = []
    for row in rows:
        if _placed_window(row) != "fit":
            continue
        volatility = feature_values(row, "volatility_7")
        momentum = feature_values(row, "momentum_30")
        if volatility is None or momentum is None:
            continue
        if not math.isfinite(volatility) or not math.isfinite(momentum):
            continue
        chosen.append((volatility, momentum))
    return chosen


def _finite_feature(row: TimedRow, name: str) -> float:
    value = feature_values(row, name)
    if value is None or not math.isfinite(value):
        raise ValueError(f"для состояния нужен конечный {name}")
    return value


def _volatility_state(value: float, low: float, high: float) -> str:
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


def _trend_state(momentum: float, band: float) -> str:
    if momentum > band:
        return "uptrend"
    if momentum < -band:
        return "downtrend"
    return "sideways"


def _placed_window(row: TimedRow) -> str | None:
    decision = stage_window(row.decision_date)
    target = stage_window(row.target_date)
    if decision == "fit" and target == "fit":
        return "fit"
    if decision == "check" and target == "check":
        return "check"
    return None


def _validate_probabilities(probabilities: Sequence[float], count: int) -> None:
    if len(probabilities) != count:
        raise ValueError("длина вероятностей не равна числу строк")
    for probability in probabilities:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("вероятность должна быть числом в (0, 1)")
        if not 0.0 < float(probability) < 1.0:
            raise ValueError("вероятность должна лежать в (0, 1)")


def _slices(
    rows: Sequence[TimedRow],
    points: Sequence[RegimePoint],
    probabilities: Sequence[float] | None,
) -> tuple[RegimeSlice, ...]:
    slices: list[RegimeSlice] = []
    for state in (*VOLATILITY_STATES, *TREND_STATES):
        for window in ("fit", "check"):
            indexes = [
                index
                for index, (row, point) in enumerate(zip(rows, points, strict=True))
                if _placed_window(row) == window and _matches(point, state)
            ]
            slices.append(_make_slice(state, window, rows, indexes, probabilities))
    return tuple(slices)


def _matches(point: RegimePoint, state: str) -> bool:
    if state in VOLATILITY_STATES:
        return point.volatility_state == state
    return point.trend_state == state


def _make_slice(
    state: str,
    window: str,
    rows: Sequence[TimedRow],
    indexes: Sequence[int],
    probabilities: Sequence[float] | None,
) -> RegimeSlice:
    observations = len(indexes)
    status = STATUS_INSUFFICIENT if observations < MIN_REGIME_OBSERVATIONS else STATUS_DESCRIPTIVE
    brier = None
    frequency_brier = None
    coin_brier = None
    base_rate = None
    if probabilities is not None and observations > 0:
        scored = score_probabilities(
            [float(probabilities[index]) for index in indexes],
            [rows[index].y_up for index in indexes],
            {
                "historical_frequency": [rows[index].baseline_frequency for index in indexes],
                "constant_half": [0.5] * observations,
            },
        )
        brier = _metric(scored, "brier")
        frequency_brier = _skill_metric(scored, "historical_frequency", "reference_brier")
        coin_brier = _skill_metric(scored, "constant_half", "reference_brier")
        base_rate = _metric(scored, "base_rate")
    return RegimeSlice(
        state=state,
        window=window,
        observations=observations,
        brier=brier,
        frequency_brier=frequency_brier,
        coin_brier=coin_brier,
        base_rate=base_rate,
        separate_model=False,
        status=status,
    )


def _overall_status(slices: Sequence[RegimeSlice]) -> str:
    counts = {item.state: item.observations for item in slices if item.window == "check"}
    for name in NAMED_REGIMES:
        if counts[name] < MIN_REGIME_OBSERVATIONS:
            return STATUS_INSUFFICIENT
    return STATUS_DESCRIPTIVE


def _metric(scored: dict[str, object], key: str) -> float:
    return _as_float(scored[key])


def _skill_metric(scored: dict[str, object], name: str, field: str) -> float:
    skill = scored["skill"]
    if not isinstance(skill, dict):
        raise ValueError("метрика вероятности не собрана")
    block = skill[name]
    if not isinstance(block, dict):
        raise ValueError("метрика вероятности не собрана")
    return _as_float(block[field])


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("метрика вероятности не собрана")
    return float(value)
