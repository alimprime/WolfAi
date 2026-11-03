"""Регуляризованная логистическая регрессия и метрики вероятности.

Коэффициенты подбираются детерминированным IRLS. Код модели не исполняется.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wolfai.research.formula import _sigmoid

LOGISTIC_L2 = 0.1
LOGISTIC_STEPS = 40
HAC_LAG = 5
PROBABILITY_EPSILON = 1e-12


class LogisticError(ValueError):
    """Коэффициенты не удалось оценить."""


@dataclass(frozen=True)
class Term:
    feature: str
    transform: str


def transform_value(transform: str, value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if transform == "identity":
        return value
    if transform == "abs":
        return abs(value)
    if transform == "tanh":
        return math.tanh(max(-20.0, min(20.0, value)))
    if transform == "sign":
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0
    raise LogisticError(f"неизвестное преобразование {transform}")


def fit_l2_logistic(design: list[list[float]], labels: list[int]) -> list[float]:
    """Веса [intercept, ...]. Штраф L2 не применяется к свободному члену."""
    if len(design) != len(labels) or len(design) < 2:
        raise LogisticError("для регрессии нужны хотя бы две строки")
    width = len(design[0]) + 1
    if any(len(row) + 1 != width for row in design):
        raise LogisticError("разная длина признаков")
    beta = [0.0] * width
    for _ in range(LOGISTIC_STEPS):
        hessian, gradient_rhs = _normal_equations(design, labels, beta)
        updated = _solve(hessian, gradient_rhs)
        updated = _clip_beta(updated)
        delta = max(abs(new - old) for new, old in zip(updated, beta, strict=True))
        beta = updated
        if delta < 1e-8:
            break
    if any(not math.isfinite(weight) for weight in beta):
        raise LogisticError("коэффициенты не конечны")
    return beta


def predict_probability(beta: list[float], row: list[float]) -> float:
    score = beta[0] + sum(weight * value for weight, value in zip(beta[1:], row, strict=True))
    probability = _sigmoid(score)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise LogisticError("вероятность вне (0, 1)")
    return probability


def score_probabilities(
    probabilities: list[float],
    labels: list[int],
    references: dict[str, list[float]],
) -> dict[str, object]:
    """Brier, log loss, калибровка и skill score. Интервал учитывает зависимость дней."""
    count = len(labels)
    if count == 0 or len(probabilities) != count:
        raise LogisticError("пустой набор вероятностей")
    brier = _mean(_point_brier(probability, label) for probability, label in zip(probabilities, labels, strict=True))
    log_loss = _mean(_point_log_loss(probability, label) for probability, label in zip(probabilities, labels, strict=True))
    result: dict[str, object] = {
        "observations": count,
        "brier": brier,
        "log_loss": log_loss,
        "mean_probability": _mean(probabilities),
        "probability_stdev": _stdev(probabilities),
        "base_rate": _mean(float(label) for label in labels),
        "calibration": _calibration(probabilities, labels),
        "skill": {},
    }
    skill: dict[str, object] = {}
    for name, reference in references.items():
        if len(reference) != count:
            raise LogisticError(f"ориентир {name} другой длины")
        reference_brier = _mean(
            _point_brier(probability, label) for probability, label in zip(reference, labels, strict=True)
        )
        gaps = [
            _point_brier(reference_probability, label) - _point_brier(probability, label)
            for probability, reference_probability, label in zip(probabilities, reference, labels, strict=True)
        ]
        mean_gap, gap_se = hac_mean_se(gaps, HAC_LAG)
        skill[name] = {
            "reference_brier": reference_brier,
            "brier_skill_score": None if reference_brier <= 0 else 1 - brier / reference_brier,
            "mean_brier_improvement": mean_gap,
            "improvement_se": gap_se,
            "improvement_low": mean_gap - 1.96 * gap_se,
            "improvement_high": mean_gap + 1.96 * gap_se,
        }
    result["skill"] = skill
    return result


def hac_mean_se(values: list[float], lag: int) -> tuple[float, float]:
    """Стандартная ошибка среднего с лагом Бартлетта."""
    count = len(values)
    if count == 0:
        raise LogisticError("нет наблюдений для ошибки среднего")
    mean = _mean(values)
    centered = [value - mean for value in values]
    variance = _autocovariance(centered, 0)
    limit = min(lag, count - 1)
    for step in range(1, limit + 1):
        weight = 1 - step / (lag + 1)
        variance += 2 * weight * _autocovariance(centered, step)
    variance = max(variance, 0.0) / count
    return mean, math.sqrt(variance)


def _normal_equations(
    design: list[list[float]],
    labels: list[int],
    beta: list[float],
) -> tuple[list[list[float]], list[float]]:
    width = len(beta)
    count = len(labels)
    hessian = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, label in zip(design, labels, strict=True):
        predictors = [1.0, *row]
        score = sum(weight * value for weight, value in zip(beta, predictors, strict=True))
        mu = _sigmoid(score)
        weight = max(mu * (1 - mu), 1e-6)
        working = score + (label - mu) / weight
        for left in range(width):
            rhs[left] += weight * predictors[left] * working
            for right in range(width):
                hessian[left][right] += weight * predictors[left] * predictors[right]
    scale = 1 / count
    for left in range(width):
        rhs[left] *= scale
        for right in range(width):
            hessian[left][right] *= scale
        hessian[left][left] += 1e-6
        if left > 0:
            hessian[left][left] += LOGISTIC_L2
    return hessian, rhs


def _clip_beta(beta: list[float]) -> list[float]:
    clipped = [max(-3.0, min(3.0, beta[0]))]
    clipped.extend(max(-5.0, min(5.0, weight)) for weight in beta[1:])
    return clipped


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise LogisticError("матрица вырождена")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for column_index in range(column, size + 1):
            augmented[column][column_index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for column_index in range(column, size + 1):
                augmented[row][column_index] -= factor * augmented[column][column_index]
    return [augmented[row][size] for row in range(size)]


def _point_brier(probability: float, label: int) -> float:
    return (probability - label) ** 2


def _point_log_loss(probability: float, label: int) -> float:
    clipped = min(1 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, probability))
    if label == 1:
        return -math.log(clipped)
    return -math.log(1 - clipped)


def _calibration(probabilities: list[float], labels: list[int]) -> list[dict[str, object]]:
    bins: list[dict[str, object]] = []
    for index in range(5):
        chosen = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if min(4, max(0, int(min(probability, 0.999999) * 5))) == index
        ]
        count = len(chosen)
        bins.append({
            "bin": index,
            "count": count,
            "mean_probability": None if count == 0 else _mean(item[0] for item in chosen),
            "empirical_frequency": None if count == 0 else _mean(float(item[1]) for item in chosen),
        })
    return bins


def _autocovariance(centered: list[float], lag: int) -> float:
    count = len(centered)
    return sum(centered[index] * centered[index - lag] for index in range(lag, count)) / count


def _mean(values) -> float:
    data = list(values)
    return sum(data) / len(data)


def _stdev(values: list[float]) -> float:
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
