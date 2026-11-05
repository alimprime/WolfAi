"""Внутренняя оценка гипотезы. Окна заданы до просмотра метрик.

2019 — обучение контрольной логистики. Июль–декабрь 2020 — общий тест.
2021 не входит в адаптацию. Даты с 2022 года отклоняют набор.
Формула не подбирается по результату. Улучшение на этом окне не является законом.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from wolfai.research.formula import FEATURES, FormulaError, evaluate, parse_formula
from wolfai.research.logistic import LogisticError, fit_l2_logistic, predict_probability, score_probabilities
from wolfai.research.qwen_brain import FEEDBACK_KEYS

PROTOCOL_VERSION = "3.1.0"
TRAIN_START = date(2019, 1, 1)
TRAIN_END = date(2019, 12, 31)
TEST_START = date(2020, 7, 1)
TEST_END = date(2020, 12, 31)
DEV_START = date(2021, 1, 1)
DEV_END = date(2021, 12, 31)
CLOSED_FROM = date(2022, 1, 1)
MIN_STABLE_OBSERVATIONS = 60
MIN_LOGISTIC_ROWS = 30
MIN_FREQUENCY_HISTORY = 5
HORIZONS = frozenset({1, 3, 7})


@dataclass(frozen=True)
class StudyPoint:
    horizon_days: int
    decision_date: date
    target_date: date
    decision_available: date
    label_available: date
    features: dict[str, float]
    label: int


@dataclass(frozen=True)
class Evaluation:
    accepted: bool
    code: str
    reason: str
    horizon_days: int
    expression: str
    train_count: int = 0
    sample_count: int = 0
    test_dates: tuple[date, ...] = ()
    gap_days: int = 0
    brier_score: float | None = None
    log_loss: float | None = None
    baseline_brier: float | None = None
    frequency_brier: float | None = None
    logistic_brier: float | None = None
    improvement: float | None = None
    improvement_low: float | None = None
    improvement_high: float | None = None
    uncertainty: str = ""
    dev_2021_skipped: int = 0
    fitted_on_2021: bool = False
    claim_status: str = "not_confirmed"
    validated_law: bool = False
    advantage_claim: bool = False
    experiment_id: str = ""
    data_hash: str = ""


def synthetic_study_points() -> tuple[StudyPoint, ...]:
    """Детерминированный ряд 2019–2020. Это не цены Bitcoin."""
    days = _days(TRAIN_START, TEST_END)
    points: list[StudyPoint] = []
    for horizon in (1, 3, 7):
        for index, day in enumerate(days):
            if index + horizon >= len(days):
                break
            target = days[index + horizon]
            features = _synthetic_features(index)
            label = 1 if math.sin(index / 5.0) > 0 else 0
            points.append(StudyPoint(
                horizon_days=horizon,
                decision_date=day,
                target_date=target,
                decision_available=day + timedelta(days=1),
                label_available=target + timedelta(days=1),
                features=features,
                label=label,
            ))
    return tuple(points)


def score_forecasts(
    probabilities: Sequence[float],
    labels: Sequence[int],
    frequency: Sequence[float],
) -> dict[str, object]:
    """Brier и log loss на одном и том же списке дат."""
    scored = score_probabilities(
        [float(item) for item in probabilities],
        [int(item) for item in labels],
        {
            "historical_frequency": [float(item) for item in frequency],
            "constant_half": [0.5] * len(labels),
        },
    )
    return scored


def evaluate_hypothesis(
    expression: str,
    feature_names: Sequence[str],
    horizon_days: int,
    points: Sequence[StudyPoint],
) -> Evaluation:
    """Считает формулу на заранее заданном тесте. Период модель не выбирает."""
    if horizon_days not in HORIZONS:
        return _rejected(expression, horizon_days, "bad_horizon", "горизонт должен быть 1, 3 или 7 дней")
    same_horizon = [point for point in points if point.horizon_days == horizon_days]
    problem = _audit(same_horizon, horizon_days)
    if problem is not None:
        code, reason = problem
        return _rejected(expression, horizon_days, code, reason, dev_2021_skipped=_dev_count(same_horizon))
    usable = [point for point in same_horizon if point.decision_date < DEV_START]
    dev_skipped = _dev_count(same_horizon)
    try:
        tree = parse_formula(expression)
    except FormulaError as error:
        return _rejected(expression, horizon_days, "unsafe_math", str(error), dev_2021_skipped=dev_skipped)
    train = _window(usable, TRAIN_START, TRAIN_END, horizon_days, TRAIN_END)
    test = _window(usable, TEST_START, TEST_END, horizon_days, TEST_END)
    if len(test) < 2:
        return _rejected(
            expression,
            horizon_days,
            "insufficient_sample",
            "во внутреннем тесте меньше двух точек",
            dev_2021_skipped=dev_skipped,
            train_count=len(train),
        )
    gap = (test[0].decision_date - (train[-1].label_available if train else TRAIN_END)).days
    if train and gap < horizon_days:
        return _rejected(
            expression,
            horizon_days,
            "time_boundary",
            "зазор между созревшей меткой обучения и тестом меньше горизонта",
            dev_2021_skipped=dev_skipped,
            train_count=len(train),
        )
    try:
        probabilities = [evaluate(tree, point.features) for point in test]
    except FormulaError as error:
        return _rejected(expression, horizon_days, "unsafe_math", str(error), dev_2021_skipped=dev_skipped)
    labels = [point.label for point in test]
    frequency = [_past_frequency(usable, point.decision_available) for point in test]
    scored = score_forecasts(probabilities, labels, frequency)
    logistic_brier = _logistic_brier(train, test, feature_names)
    skill = scored["skill"]
    coin = skill["constant_half"]
    freq = skill["historical_frequency"]
    if not isinstance(coin, dict) or not isinstance(freq, dict):
        return _rejected(expression, horizon_days, "unsafe_math", "контрольные вероятности не посчитаны")
    brier = round(float(scored["brier"]), 6)
    log_loss = round(float(scored["log_loss"]), 6)
    baseline = round(float(coin["reference_brier"]), 6)
    frequency_brier = round(float(freq["reference_brier"]), 6)
    improvement = round(baseline - brier, 6)
    low = float(coin["improvement_low"])
    high = float(coin["improvement_high"])
    data_hash = _points_hash(train + test)
    experiment_id = hashlib.sha256(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "expression": expression.strip(),
                "features": list(feature_names),
                "horizon_days": horizon_days,
                "data_hash": data_hash,
                "train_end": TRAIN_END.isoformat(),
                "test_start": TEST_START.isoformat(),
                "test_end": TEST_END.isoformat(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return Evaluation(
        accepted=True,
        code="ok",
        reason="внутренний счёт выполнен",
        horizon_days=horizon_days,
        expression=expression,
        train_count=len(train),
        sample_count=len(test),
        test_dates=tuple(point.decision_date for point in test),
        gap_days=gap,
        brier_score=brier,
        log_loss=log_loss,
        baseline_brier=baseline,
        frequency_brier=frequency_brier,
        logistic_brier=None if logistic_brier is None else round(logistic_brier, 6),
        improvement=improvement,
        improvement_low=round(low, 6),
        improvement_high=round(high, 6),
        uncertainty=_uncertainty(len(test), low, high),
        dev_2021_skipped=dev_skipped,
        fitted_on_2021=False,
        experiment_id=experiment_id,
        data_hash=data_hash,
    )


def feedback_payload(
    evaluation: Evaluation,
    *,
    hypothesis_id: str,
    failure_reason: str | None,
    limitations: str,
    revision_allowed: bool,
) -> dict[str, object]:
    """Короткий отчёт для следующей генерации. Ряд и SQL сюда не входят."""
    payload: dict[str, object] = {
        "hypothesis_id": hypothesis_id,
        "experiment_id": evaluation.experiment_id,
        "horizon_days": evaluation.horizon_days,
        "sample_count": evaluation.sample_count,
        "brier_score": evaluation.brier_score,
        "log_loss": evaluation.log_loss,
        "baseline_brier": evaluation.baseline_brier,
        "improvement": evaluation.improvement,
        "uncertainty": evaluation.uncertainty or failure_reason or "проверка не выполнена",
        "failure_reason": failure_reason,
        "limitations": limitations[:300],
        "revision_allowed": revision_allowed,
    }
    if set(payload) != FEEDBACK_KEYS:
        raise RuntimeError("обратная связь вышла за разрешённые поля")
    return payload


def _uncertainty(sample_count: int, low: float, high: float) -> str:
    if sample_count < MIN_STABLE_OBSERVATIONS:
        return "наблюдений меньше 60, улучшение статистически не устойчиво"
    if low > 0:
        return "на внутреннем окне интервал выше нуля; это не доказательство закономерности"
    if high < 0:
        return "внутреннее окно хуже константы 0.5"
    return "интервал улучшения содержит ноль, улучшение могло быть случайным"


def _logistic_brier(
    train: Sequence[StudyPoint],
    test: Sequence[StudyPoint],
    feature_names: Sequence[str],
) -> float | None:
    names = tuple(name for name in feature_names if name in FEATURES)
    if len(train) < MIN_LOGISTIC_ROWS or not names:
        return None
    design = [[point.features[name] for name in names] for point in train]
    labels = [point.label for point in train]
    try:
        beta = fit_l2_logistic(design, labels)
        probabilities = [
            predict_probability(beta, [point.features[name] for name in names])
            for point in test
        ]
    except LogisticError:
        return None
    scored = score_forecasts(probabilities, [point.label for point in test], [0.5] * len(test))
    return float(scored["brier"])


def _past_frequency(points: Sequence[StudyPoint], as_of: date) -> float:
    known = [point.label for point in points if point.label_available <= as_of]
    if len(known) < MIN_FREQUENCY_HISTORY:
        return 0.5
    return sum(known) / len(known)


def _window(
    points: Sequence[StudyPoint],
    start: date,
    end: date,
    horizon: int,
    label_deadline: date,
) -> list[StudyPoint]:
    chosen: list[StudyPoint] = []
    next_day: date | None = None
    for point in sorted(points, key=lambda item: item.decision_date):
        if point.decision_date < start or point.decision_date > end:
            continue
        if point.target_date > label_deadline or point.label_available > label_deadline:
            continue
        if point.label_available <= point.decision_available:
            continue
        if (point.label_available - point.decision_date).days < horizon + 1:
            continue
        if next_day is not None and point.decision_date < next_day:
            continue
        chosen.append(point)
        next_day = point.decision_date + timedelta(days=horizon)
    return chosen


def _audit(points: Sequence[StudyPoint], horizon: int) -> tuple[str, str] | None:
    for point in points:
        if point.horizon_days != horizon:
            continue
        if point.decision_date >= CLOSED_FROM or point.target_date >= CLOSED_FROM or point.label_available >= CLOSED_FROM:
            return "time_boundary", "точка использует дату после 2021 года"
        if point.label_available <= point.decision_available or point.target_date <= point.decision_date:
            return "label_leak", "метка доступна не позже решения"
        if point.label not in (0, 1):
            return "label_leak", "исход должен быть 0 или 1"
        if any(name not in point.features or not math.isfinite(point.features[name]) for name in FEATURES):
            return "missing_feature", "у точки нет полного конечного набора признаков"
    return None


def _dev_count(points: Sequence[StudyPoint]) -> int:
    return sum(1 for point in points if DEV_START <= point.decision_date <= DEV_END)


def _rejected(
    expression: str,
    horizon_days: int,
    code: str,
    reason: str,
    *,
    dev_2021_skipped: int = 0,
    train_count: int = 0,
) -> Evaluation:
    return Evaluation(
        accepted=False,
        code=code,
        reason=reason,
        horizon_days=horizon_days,
        expression=expression,
        train_count=train_count,
        uncertainty=reason,
        dev_2021_skipped=dev_2021_skipped,
    )


def _days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _synthetic_features(index: int) -> dict[str, float]:
    features = {
        name: math.sin((index + 1) * (offset + 1) / 17.0) * 0.3
        for offset, name in enumerate(FEATURES)
    }
    features["volatility_7"] = abs(features["volatility_7"]) + 0.05
    features["volatility_30"] = abs(features["volatility_30"]) + 0.05
    return features


def _points_hash(points: Sequence[StudyPoint]) -> str:
    payload = [
        {
            "decision": point.decision_date.isoformat(),
            "target": point.target_date.isoformat(),
            "label": point.label,
            "features": {name: round(point.features[name], 8) for name in sorted(point.features)},
        }
        for point in points
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()
