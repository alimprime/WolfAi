"""Отдельная оценка устойчивости вероятностной гипотезы.

Поддержка требует отделённого отбора, положительных блочных границ и
улучшения выше ожидаемого чемпиона шума. Порог p-value не назначается.
Перекрывающиеся прогнозы не считаются независимыми.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

from wolfai.research.logistic import score_probabilities
from wolfai.research.quality_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    MIN_RELIABILITY_OBSERVATIONS,
    RELIABILITY_RULE,
    STATUS_INSUFFICIENT,
    STATUS_REJECTED,
    STATUS_SUPPORTED,
    CalibrationBin,
    MultipleTestingNote,
    ReliabilityReport,
    UncertaintyEstimate,
    assert_stage_row,
    dependence_block_length,
    linear_percentile,
    moving_block_means,
    noise_champion,
    point_brier,
    sample_stdev,
)

_BOOTSTRAP_METHOD = "moving_block_bootstrap"
_LIGHT_PYSR_COMPLEXITY = 12
_OVERLAP_LIMIT = (
    "Перекрывающиеся прогнозы не считались независимыми: длина блока равна горизонту."
)
_SEPARATION_LIMIT = "Окно оценки не отделено от выбора формулы."
_COMPLEXITY_LIMIT = "Сложность выше лёгкого предела PySR."


def assess_reliability(
    decision_dates: Sequence[date],
    probabilities: Sequence[float],
    labels: Sequence[int],
    baseline_frequency: Sequence[float],
    *,
    horizon_days: int,
    complexity: int,
    hypotheses_examined: int,
    selection_separated: bool,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    evidence_role: str = "research",
) -> ReliabilityReport:
    """Оценить, хватает ли ряда, чтобы поддержать или отвергнуть гипотезу.

    Brier, log loss и пять корзин калибровки считаются всегда. Блочный
    интервал строится только когда в ряде хотя бы две точки и блок не длиннее
    ряда. Иначе статус insufficient_evidence, даже если точечный Brier мал.
    """
    dates, scored_probabilities, scored_labels, baseline = _validated_rows(
        decision_dates,
        probabilities,
        labels,
        baseline_frequency,
        horizon_days=horizon_days,
        complexity=complexity,
        hypotheses_examined=hypotheses_examined,
    )
    count = len(dates)
    for decision in dates:
        assert_stage_row(decision, decision + timedelta(days=horizon_days))
    block_length, overlapping = dependence_block_length(dates, horizon_days)
    scored = score_probabilities(
        scored_probabilities,
        scored_labels,
        {
            "historical_frequency": baseline,
            "constant_half": [0.5] * count,
        },
    )
    gaps_frequency = _improvements(scored_probabilities, baseline, scored_labels)
    gaps_coin = _improvements(scored_probabilities, [0.5] * count, scored_labels)
    middle = count // 2
    early = _mean_or_none(gaps_frequency[:middle])
    late = _mean_or_none(gaps_frequency[middle:])
    uncertainty, multiple_testing = _uncertainty(
        gaps_frequency,
        gaps_coin,
        block_length=block_length,
        overlapping=overlapping,
        replicates=replicates,
        seed=seed,
        hypotheses_examined=hypotheses_examined,
    )
    separated = selection_separated is True
    if evidence_role not in {"research", "development_only"}:
        raise ValueError("роль доказательства должна быть research или development_only")
    status = _status(
        count=count,
        selection_separated=separated,
        uncertainty=uncertainty,
        multiple_testing=multiple_testing,
        early=early,
        late=late,
    )
    provisional = False
    limitations = _limitations(
        status=status,
        overlapping=overlapping,
        selection_separated=separated,
        complexity=complexity,
        count=count,
        uncertainty=uncertainty,
        multiple_testing=multiple_testing,
        early=early,
        late=late,
    )
    if evidence_role == "development_only":
        if status == STATUS_SUPPORTED:
            provisional = True
            status = STATUS_INSUFFICIENT
        limitations = limitations + (
            "Окно 2021 уже использовано: роль development_only, статус supported запрещён.",
        )
    return ReliabilityReport(
        status=status,
        brier=_required_float(scored.get("brier"), "Brier не посчитан"),
        log_loss=_required_float(scored.get("log_loss"), "log loss не посчитан"),
        calibration=_calibration(scored.get("calibration")),
        frequency_brier=_reference_brier(scored, "historical_frequency"),
        coin_brier=_reference_brier(scored, "constant_half"),
        uncertainty=uncertainty,
        complexity=complexity,
        hypotheses_examined=hypotheses_examined,
        multiple_testing=multiple_testing,
        early_improvement_vs_frequency=early,
        late_improvement_vs_frequency=late,
        limitations=limitations,
        evidence_role=evidence_role,
        provisional=provisional,
    )


def _validated_rows(
    decision_dates: Sequence[date],
    probabilities: Sequence[float],
    labels: Sequence[int],
    baseline_frequency: Sequence[float],
    *,
    horizon_days: int,
    complexity: int,
    hypotheses_examined: int,
) -> tuple[list[date], list[float], list[int], list[float]]:
    dates = [_decision_date(value) for value in decision_dates]
    raw_probabilities = list(probabilities)
    raw_labels = list(labels)
    raw_baseline = list(baseline_frequency)
    count = len(dates)
    if (
        count == 0
        or len(raw_probabilities) != count
        or len(raw_labels) != count
        or len(raw_baseline) != count
    ):
        raise ValueError("ряды должны быть одной ненулевой длины")
    if isinstance(horizon_days, bool) or horizon_days not in (1, 3, 7):
        raise ValueError("горизонт должен быть 1, 3 или 7 дней")
    if isinstance(complexity, bool) or not isinstance(complexity, int) or complexity < 1:
        raise ValueError("сложность должна быть не меньше 1")
    if (
        isinstance(hypotheses_examined, bool)
        or not isinstance(hypotheses_examined, int)
        or hypotheses_examined < 1
    ):
        raise ValueError("число гипотез должно быть положительным")
    return (
        dates,
        [_probability(value) for value in raw_probabilities],
        [_label(value) for value in raw_labels],
        [_frequency(value) for value in raw_baseline],
    )


def _decision_date(value: object) -> date:
    if not isinstance(value, date):
        raise ValueError("дата решения должна быть датой")
    return value


def _label(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ValueError("метка направления должна быть 0 или 1")
    return value


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("вероятность должна быть конечной и лежать в (0, 1)")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("вероятность должна быть конечной и лежать в (0, 1)")
    return probability


def _frequency(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("частота прошлых ростов вне [0, 1]")
    frequency = float(value)
    if not math.isfinite(frequency) or not 0.0 <= frequency <= 1.0:
        raise ValueError("частота прошлых ростов вне [0, 1]")
    return frequency


def _improvements(
    probabilities: Sequence[float],
    references: Sequence[float],
    labels: Sequence[int],
) -> list[float]:
    return [
        point_brier(reference, label) - point_brier(probability, label)
        for probability, reference, label in zip(probabilities, references, labels, strict=True)
    ]


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _uncertainty(
    gaps_frequency: Sequence[float],
    gaps_coin: Sequence[float],
    *,
    block_length: int,
    overlapping: bool,
    replicates: int,
    seed: int,
    hypotheses_examined: int,
) -> tuple[UncertaintyEstimate | None, MultipleTestingNote | None]:
    count = len(gaps_frequency)
    if count < 2 or block_length > count:
        return None, None
    frequency_means = moving_block_means(gaps_frequency, block_length, replicates, seed)
    coin_means = moving_block_means(gaps_coin, block_length, replicates, seed)
    error = sample_stdev(frequency_means)
    frequency_mean = sum(gaps_frequency) / len(gaps_frequency)
    coin_mean = sum(gaps_coin) / len(gaps_coin)
    estimate = UncertaintyEstimate(
        method=_BOOTSTRAP_METHOD,
        block_length=block_length,
        replicates=replicates,
        overlapping=overlapping,
        mean_improvement_vs_frequency=frequency_mean,
        low_vs_frequency=linear_percentile(frequency_means, 0.025),
        high_vs_frequency=linear_percentile(frequency_means, 0.975),
        mean_improvement_vs_coin=coin_mean,
        low_vs_coin=linear_percentile(coin_means, 0.025),
        high_vs_coin=linear_percentile(coin_means, 0.975),
        bootstrap_se_vs_frequency=error,
    )
    note = MultipleTestingNote(
        hypotheses_examined=hypotheses_examined,
        noise_champion=noise_champion(error, hypotheses_examined),
        justification=_noise_justification(),
    )
    return estimate, note


def _noise_justification() -> str:
    text = noise_champion.__doc__
    if not text:
        return "Ожидаемый чемпион шума, не порог значимости."
    return " ".join(text.split())


def _contract_statuses() -> tuple[str, str, str]:
    """Коды статуса берутся из контракта, текст правила не копируется."""
    statuses = (STATUS_SUPPORTED, STATUS_REJECTED, STATUS_INSUFFICIENT)
    if any(code not in RELIABILITY_RULE for code in statuses):
        raise ValueError("правило надёжности не содержит коды статуса")
    return statuses


def _status(
    *,
    count: int,
    selection_separated: bool,
    uncertainty: UncertaintyEstimate | None,
    multiple_testing: MultipleTestingNote | None,
    early: float | None,
    late: float | None,
) -> str:
    supported, rejected, insufficient = _contract_statuses()
    if count < MIN_RELIABILITY_OBSERVATIONS or uncertainty is None or multiple_testing is None:
        return insufficient
    frequency_below = uncertainty.low_vs_frequency < 0.0 and uncertainty.high_vs_frequency < 0.0
    coin_below = uncertainty.low_vs_coin < 0.0 and uncertainty.high_vs_coin < 0.0
    if frequency_below and coin_below:
        return rejected
    halves_positive = (
        early is not None and late is not None and early > 0.0 and late > 0.0
    )
    if (
        selection_separated
        and uncertainty.low_vs_frequency > 0.0
        and uncertainty.low_vs_coin > 0.0
        and halves_positive
        and uncertainty.mean_improvement_vs_frequency > multiple_testing.noise_champion
    ):
        return supported
    return insufficient


def _limitations(
    *,
    status: str,
    overlapping: bool,
    selection_separated: bool,
    complexity: int,
    count: int,
    uncertainty: UncertaintyEstimate | None,
    multiple_testing: MultipleTestingNote | None,
    early: float | None,
    late: float | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if overlapping:
        notes.append(_OVERLAP_LIMIT)
    if not selection_separated:
        notes.append(_SEPARATION_LIMIT)
    if complexity > _LIGHT_PYSR_COMPLEXITY:
        notes.append(_COMPLEXITY_LIMIT)
    if status == STATUS_INSUFFICIENT:
        reason = _insufficient_reason(
            count=count,
            selection_separated=selection_separated,
            uncertainty=uncertainty,
            multiple_testing=multiple_testing,
            early=early,
            late=late,
        )
        if reason not in notes:
            notes.append(reason)
    return tuple(notes)


def _insufficient_reason(
    *,
    count: int,
    selection_separated: bool,
    uncertainty: UncertaintyEstimate | None,
    multiple_testing: MultipleTestingNote | None,
    early: float | None,
    late: float | None,
) -> str:
    if count < MIN_RELIABILITY_OBSERVATIONS:
        return f"Наблюдений меньше {MIN_RELIABILITY_OBSERVATIONS}."
    if uncertainty is None or multiple_testing is None:
        return "Бутстреп невозможен."
    if uncertainty.low_vs_frequency <= 0.0 or uncertainty.low_vs_coin <= 0.0:
        return "Нижняя граница улучшения не строго выше нуля."
    if early is None or late is None or early <= 0.0 or late <= 0.0:
        return "Хронологические половины не обе имеют положительное улучшение против частоты."
    if uncertainty.mean_improvement_vs_frequency <= multiple_testing.noise_champion:
        return "Среднее улучшение против частоты не больше ожидаемого чемпиона шума."
    if not selection_separated:
        return _SEPARATION_LIMIT
    return "Доказательств недостаточно."


def _calibration(raw: object) -> tuple[CalibrationBin, ...]:
    if not isinstance(raw, list):
        raise ValueError("калибровка не получена")
    bins: list[CalibrationBin] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("калибровка не получена")
        bins.append(
            CalibrationBin(
                bin_index=_required_int(item.get("bin"), "калибровка не получена"),
                count=_required_int(item.get("count"), "калибровка не получена"),
                mean_probability=_optional_float(item.get("mean_probability")),
                empirical_frequency=_optional_float(item.get("empirical_frequency")),
            )
        )
    bins.sort(key=lambda bin_item: bin_item.bin_index)
    if [item.bin_index for item in bins] != [0, 1, 2, 3, 4]:
        raise ValueError("калибровка должна содержать пять корзин")
    return tuple(bins)


def _reference_brier(scored: dict[str, object], name: str) -> float:
    skill = scored.get("skill")
    if not isinstance(skill, dict):
        raise ValueError("ориентир не посчитан")
    entry = skill.get(name)
    if not isinstance(entry, dict):
        raise ValueError("ориентир не посчитан")
    return _required_float(entry.get("reference_brier"), "ориентир не посчитан")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _required_float(value, "калибровка не получена")


def _required_float(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(message)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(message)
    return number


def _required_int(value: object, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)
    return value
