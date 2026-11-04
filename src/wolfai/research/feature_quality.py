"""Аудит признаков на 2019–2020 и проверка на 2021.

Слабый Brier не удаляет признак и не объявляется прогнозом.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
    FeatureDistribution,
    FeatureRedundancy,
    FeatureStability,
    FeatureUsefulness,
    TimedRow,
    feature_values,
    linear_percentile,
    require_stage_rows,
    sample_stdev,
    stage_window,
)

_SCALE_FLOOR = 1e-8
_SHIFT_FLOOR = 1e-12
_USEFULNESS_LIMITATION = (
    "Это не доказательство прогноза. Неопределённость отдана модулю надёжности."
)
_UNSTABLE_DISTRIBUTION = (
    "Распределение неустойчиво: стандартизованный сдвиг между ранней и поздней "
    "половинами 2019–2020 больше 1."
)


def audit_features(rows: Sequence[TimedRow], *, horizon_days: int) -> FeatureAuditReport:
    """Считает распределение, устойчивость, Brier и совпадение рангов.

    Окна не смешиваются: строка на границе 2020 и 2021 не входит ни в подбор,
    ни в проверку. Признак остаётся в отчёте даже при слабом Brier.
    """
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    require_stage_rows(rows)
    fit_rows, check_rows = _rows_by_window(rows)
    redundancy = _redundancy(fit_rows)
    features = tuple(
        _audit_feature(name, fit_rows, check_rows, redundancy) for name in AUDITED_FEATURES
    )
    return FeatureAuditReport(
        features=features,
        redundancy=redundancy,
        horizon_days=horizon_days,
        dropped_features=(),
    )


def _audit_feature(
    name: str,
    fit_rows: Sequence[TimedRow],
    check_rows: Sequence[TimedRow],
    redundancy: Sequence[FeatureRedundancy],
) -> FeatureAudit:
    distribution, fences = _distribution(name, fit_rows)
    stability = _stability(name, fit_rows)
    usefulness = _usefulness(name, fit_rows, check_rows)
    return FeatureAudit(
        name=name,
        distribution=distribution,
        stability=stability,
        usefulness=usefulness,
        kept=True,
        check_anomalies=_check_anomalies(name, check_rows, fences),
        limitations=_limitations(name, stability, redundancy),
    )


def _rows_by_window(rows: Sequence[TimedRow]) -> tuple[list[TimedRow], list[TimedRow]]:
    fit_rows: list[TimedRow] = []
    check_rows: list[TimedRow] = []
    for row in rows:
        window = _placed_window(row)
        if window == "fit":
            fit_rows.append(row)
        elif window == "check":
            check_rows.append(row)
    return fit_rows, check_rows


def _placed_window(row: TimedRow) -> str | None:
    decision = stage_window(row.decision_date)
    target = stage_window(row.target_date)
    if decision == "fit" and target == "fit":
        return "fit"
    if decision == "check" and target == "check":
        return "check"
    return None


def _value_kind(row: TimedRow, name: str) -> tuple[str, float | None]:
    value = feature_values(row, name)
    if value is None:
        return "missing", None
    if not math.isfinite(value):
        return "non_finite", None
    return "finite", value


def _finite_values(rows: Sequence[TimedRow], name: str) -> list[tuple[TimedRow, float]]:
    chosen: list[tuple[TimedRow, float]] = []
    for row in rows:
        kind, value = _value_kind(row, name)
        if kind == "finite" and value is not None:
            chosen.append((row, value))
    return chosen


def _distribution(
    name: str,
    rows: Sequence[TimedRow],
) -> tuple[FeatureDistribution, tuple[float, float] | None]:
    finite: list[float] = []
    missing = 0
    non_finite = 0
    for row in rows:
        kind, value = _value_kind(row, name)
        if kind == "missing":
            missing += 1
        elif kind == "non_finite":
            non_finite += 1
        elif value is not None:
            finite.append(value)
    count = len(finite)
    fences: tuple[float, float] | None = None
    anomalies = 0
    if count == 0:
        mean = None
        stdev = None
        minimum = None
        maximum = None
        q10 = None
        q50 = None
        q90 = None
    else:
        mean = sum(finite) / count
        stdev = sample_stdev(finite) if count >= 2 else None
        minimum = min(finite)
        maximum = max(finite)
        q10 = linear_percentile(finite, 0.1)
        q50 = linear_percentile(finite, 0.5)
        q90 = linear_percentile(finite, 0.9)
        if count >= 2:
            q1 = linear_percentile(finite, 0.25)
            q3 = linear_percentile(finite, 0.75)
            iqr = q3 - q1
            fences = (q1 - 3.0 * iqr, q3 + 3.0 * iqr)
            lower, upper = fences
            anomalies = sum(value < lower or value > upper for value in finite)
    return FeatureDistribution(
        name=name,
        window="2019-2020",
        observations=count,
        missing=missing,
        non_finite=non_finite,
        anomalies=anomalies,
        mean=mean,
        stdev=stdev,
        minimum=minimum,
        maximum=maximum,
        q10=q10,
        q50=q50,
        q90=q90,
    ), fences


def _check_anomalies(
    name: str,
    rows: Sequence[TimedRow],
    fences: tuple[float, float] | None,
) -> int:
    if fences is None:
        return 0
    lower, upper = fences
    count = 0
    for row in rows:
        kind, value = _value_kind(row, name)
        if kind == "finite" and value is not None and (value < lower or value > upper):
            count += 1
    return count


def _stability(name: str, rows: Sequence[TimedRow]) -> FeatureStability:
    finite = sorted(
        ((row.decision_date, value) for row, value in _finite_values(rows, name)),
        key=lambda item: item[0],
    )
    values = [value for _, value in finite]
    middle = len(values) // 2
    early = values[:middle]
    late = values[middle:]
    if not early or not late:
        return FeatureStability(
            name=name,
            early_mean=None,
            late_mean=None,
            standardized_shift=None,
            limitation=STABILITY_RULE,
        )
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    deviation = sample_stdev(values)
    if deviation < _SHIFT_FLOOR:
        shift = 0.0 if early_mean == late_mean else None
    else:
        shift = abs(late_mean - early_mean) / deviation
    return FeatureStability(
        name=name,
        early_mean=early_mean,
        late_mean=late_mean,
        standardized_shift=shift,
        limitation=STABILITY_RULE,
    )


def _usefulness(
    name: str,
    fit_rows: Sequence[TimedRow],
    check_rows: Sequence[TimedRow],
) -> FeatureUsefulness:
    fit = _finite_values(fit_rows, name)
    check = _finite_values(check_rows, name)
    coefficients: tuple[float, ...] | None = None
    brier = None
    log_loss = None
    frequency_brier = None
    coin_brier = None
    improvement_frequency = None
    improvement_coin = None
    if len(fit) >= MIN_FEATURE_FIT:
        values = [value for _, value in fit]
        mean = sum(values) / len(values)
        deviation = sample_stdev(values)
        design = [[_scale(value, mean, deviation)] for value in values]
        labels = [row.y_up for row, _ in fit]
        beta = fit_l2_logistic(design, labels)
        coefficients = tuple(beta)
        if len(check) >= MIN_FEATURE_CHECK:
            probabilities = [
                predict_probability(beta, [_scale(value, mean, deviation)])
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
            brier = _metric(scored, "brier")
            log_loss = _metric(scored, "log_loss")
            frequency_brier = _skill_metric(scored, "historical_frequency", "reference_brier")
            coin_brier = _skill_metric(scored, "constant_half", "reference_brier")
            improvement_frequency = _skill_metric(scored, "historical_frequency", "mean_brier_improvement")
            improvement_coin = _skill_metric(scored, "constant_half", "mean_brier_improvement")
    return FeatureUsefulness(
        name=name,
        model=USEFULNESS_MODEL,
        predictive_claim=False,
        coefficients=coefficients,
        check_observations=len(check),
        brier=brier,
        log_loss=log_loss,
        frequency_brier=frequency_brier,
        coin_brier=coin_brier,
        brier_improvement_vs_frequency=improvement_frequency,
        brier_improvement_vs_coin=improvement_coin,
        limitation=_USEFULNESS_LIMITATION,
    )


def _scale(value: float, mean: float, deviation: float) -> float:
    if deviation < _SCALE_FLOOR:
        return 0.0
    return (value - mean) / deviation


def _redundancy(rows: Sequence[TimedRow]) -> tuple[FeatureRedundancy, ...]:
    items: list[FeatureRedundancy] = []
    for left_index, left in enumerate(AUDITED_FEATURES):
        for right in AUDITED_FEATURES[left_index + 1:]:
            items.append(FeatureRedundancy(
                left=left,
                right=right,
                rank_agreement=_rank_agreement(rows, left, right),
                note=REDUNDANCY_NOTE,
            ))
    return tuple(items)


def _rank_agreement(rows: Sequence[TimedRow], left: str, right: str) -> float | None:
    paired: list[tuple[float, float]] = []
    for row in rows:
        left_kind, left_value = _value_kind(row, left)
        right_kind, right_value = _value_kind(row, right)
        if left_kind == "finite" and right_kind == "finite" and left_value is not None and right_value is not None:
            paired.append((left_value, right_value))
    if len(paired) < 3:
        return None
    return _pearson(
        _average_ranks([item[0] for item in paired]),
        _average_ranks([item[1] for item in paired]),
    )


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + 1 + end + 1) / 2.0
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left == 0.0 or var_right == 0.0:
        return None
    covariance = sum(
        (left_value - mean_left) * (right_value - mean_right)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return covariance / math.sqrt(var_left * var_right)


def _limitations(
    name: str,
    stability: FeatureStability,
    redundancy: Sequence[FeatureRedundancy],
) -> tuple[str, ...]:
    notes = [_USEFULNESS_LIMITATION]
    shift = stability.standardized_shift
    if shift is not None and shift > 1.0:
        notes.append(_UNSTABLE_DISTRIBUTION)
    for item in redundancy:
        agreement = item.rank_agreement
        if agreement is None or abs(agreement) < 0.8:
            continue
        if name == item.left:
            other = item.right
        elif name == item.right:
            other = item.left
        else:
            continue
        notes.append(
            f"Описательное совпадение рангов с {other}. Это не доказательство прогноза."
        )
    return tuple(notes)


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
