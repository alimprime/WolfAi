"""Замороженный контракт этапа 2.3.2.

Task A и Task B этот файл не меняют. Здесь нет поиска PySR, Optuna и Qwen.
Окна: 2019–2020 — подбор, 2021 — внутренняя проверка. С 2022-01-01 ряд
этапа закрыт: 2022 не независимый тест и не используется для подбора,
2023–2024 и holdout не открываются.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from wolfai.data.gateway import HOLDOUT_START
from wolfai.research.periods import FEEDBACK, FEEDBACK_NOTE, SEARCH_CHECK, SEARCH_TRAIN
from wolfai.research.pysr_engine import LIGHT_LIMITS

STAGE = "2.3.2"
PRIOR_STAGE = "2.3.1"
PRIOR_COMMIT = "bdf5840"
FORBIDDEN_FROM = date(2022, 1, 1)
STAGE_LAST_INCLUDED = date(2021, 12, 31)
AUDITED_FEATURES = ("log_return", "momentum_7", "momentum_30", "volatility_7")
VOLATILITY_STATES = ("high_volatility", "low_volatility", "mid_volatility")
TREND_STATES = ("uptrend", "downtrend", "sideways")
NAMED_REGIMES = ("high_volatility", "low_volatility", "uptrend", "downtrend", "sideways")
STATUS_INSUFFICIENT = "insufficient_evidence"
STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_DESCRIPTIVE = "descriptive_only"
MIN_THRESHOLD_OBSERVATIONS = 60
MIN_REGIME_OBSERVATIONS = 40
MIN_RELIABILITY_OBSERVATIONS = 30
MIN_FEATURE_FIT = 40
MIN_FEATURE_CHECK = 30
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_SEED = 1729
ANOMALY_RULE = (
    "Аномалия — значение ниже Q1 - 3 * IQR или выше Q3 + 3 * IQR. "
    "Квартили фиксируются на 2019–2020 и не пересчитываются на 2021."
)
STABILITY_RULE = (
    "Ранняя и поздняя половины 2019–2020. Сдвиг — модуль разности средних, "
    "делённый на стандартное отклонение всего окна подбора. Это устойчивость "
    "распределения, не доказательство прогноза."
)
REDUNDANCY_NOTE = "описательное совпадение рангов, не доказательство прогноза"
USEFULNESS_MODEL = "train_l2_logistic_identity"
RELIABILITY_RULE = (
    "supported только если одновременно: точек не меньше 30; отбор формулы "
    "отделён от оцениваемого окна; блочный интервал 2.5–97.5% улучшения и против "
    "частоты прошлых ростов, и против константы 0.5 лежит строго выше нуля; обе "
    "хронологические половины имеют положительное среднее улучшение против частоты; "
    "среднее улучшение против частоты больше se * sqrt(2 * ln K), где se — "
    "стандартное отклонение бутстреп-средних, K — число испытанных гипотез. "
    "При K = 1 поправка равна нулю, потому что чемпиона среди гипотез нет. "
    "rejected — только если оба интервала лежат строго ниже нуля. "
    "Иначе insufficient_evidence. Отдельный порог значимости не назначается. "
    "Перекрывающиеся прогнозы не считаются независимыми: длина блока в наблюдениях "
    "не меньше горизонта, если соседние даты решения ближе горизонта."
)
SUCCESS_CRITERION = (
    "Воспроизводимое улучшение относительно частоты прошлых ростов и константы 0.5 "
    "на окне 2021, без утечки будущих данных и с учётом числа испытанных гипотез. "
    "Один Brier одного запуска победу не даёт."
)

# Цифры отчёта 2.3.1. Они зафиксированы до изменений 2.3.2 и не пересчитываются здесь.
PRIOR_231 = {
    "tests_passed": 164,
    "tests_skipped": 1,
    "commit": PRIOR_COMMIT,
    "seed": 1729,
    "pysr_budget": dict(LIGHT_LIMITS),
    "optuna_trials": 20,
    "btc_fit": "2019-2020",
    "btc_check": "2021",
    "btc_pysr_internal_logit_logloss_about": 0.66,
    "btc_optuna_grid_coefficient": -0.846,
    "btc_optuna_grid_check_log_loss": 0.6922,
    "btc_optuna_grid_check_brier": 0.24953,
    "btc_optuna_tpe_coefficient": -0.915,
    "btc_optuna_tpe_check_log_loss": 0.6922,
    "btc_optuna_tpe_check_brier": 0.24954,
    "btc_frequency_brier_2021": 0.2515,
    "btc_coin_brier_2021": 0.250,
    "feedback_note": FEEDBACK_NOTE,
    "feedback_window": FEEDBACK,
}


@dataclass(frozen=True)
class TimedRow:
    """Точка решения. Признаки уже посчитаны только из свечей до этой даты."""

    decision_date: date
    target_date: date
    features: dict[str, float]
    y_up: int
    baseline_frequency: float


@dataclass(frozen=True)
class FeatureDistribution:
    name: str
    window: str
    observations: int
    missing: int
    non_finite: int
    anomalies: int
    mean: float | None
    stdev: float | None
    minimum: float | None
    maximum: float | None
    q10: float | None
    q50: float | None
    q90: float | None


@dataclass(frozen=True)
class FeatureStability:
    name: str
    early_mean: float | None
    late_mean: float | None
    standardized_shift: float | None
    limitation: str


@dataclass(frozen=True)
class FeatureUsefulness:
    """Счёт против контролей. Корреляции с будущим направлением здесь нет."""

    name: str
    model: str
    predictive_claim: bool
    coefficients: tuple[float, ...] | None
    check_observations: int
    brier: float | None
    log_loss: float | None
    frequency_brier: float | None
    coin_brier: float | None
    brier_improvement_vs_frequency: float | None
    brier_improvement_vs_coin: float | None
    limitation: str


@dataclass(frozen=True)
class FeatureRedundancy:
    left: str
    right: str
    rank_agreement: float | None
    note: str


@dataclass(frozen=True)
class FeatureAudit:
    name: str
    distribution: FeatureDistribution
    stability: FeatureStability
    usefulness: FeatureUsefulness
    kept: bool
    check_anomalies: int
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class FeatureAuditReport:
    features: tuple[FeatureAudit, ...]
    redundancy: tuple[FeatureRedundancy, ...]
    horizon_days: int
    dropped_features: tuple[str, ...]


@dataclass(frozen=True)
class RegimeThresholds:
    volatility_low: float
    volatility_high: float
    trend_band: float
    fit_start: date
    fit_end: date
    fit_observations: int
    rule: str


@dataclass(frozen=True)
class RegimePoint:
    decision_date: date
    volatility_state: str
    trend_state: str
    volatility_7: float
    momentum_30: float


@dataclass(frozen=True)
class RegimeSlice:
    state: str
    window: str
    observations: int
    brier: float | None
    frequency_brier: float | None
    coin_brier: float | None
    base_rate: float | None
    separate_model: bool
    status: str


@dataclass(frozen=True)
class RegimeReport:
    thresholds: RegimeThresholds
    points: tuple[RegimePoint, ...]
    slices: tuple[RegimeSlice, ...]
    separate_model: bool
    status: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationBin:
    bin_index: int
    count: int
    mean_probability: float | None
    empirical_frequency: float | None


@dataclass(frozen=True)
class UncertaintyEstimate:
    method: str
    block_length: int
    replicates: int
    overlapping: bool
    mean_improvement_vs_frequency: float
    low_vs_frequency: float
    high_vs_frequency: float
    mean_improvement_vs_coin: float
    low_vs_coin: float
    high_vs_coin: float
    bootstrap_se_vs_frequency: float


@dataclass(frozen=True)
class MultipleTestingNote:
    hypotheses_examined: int
    noise_champion: float
    justification: str


@dataclass(frozen=True)
class ReliabilityReport:
    status: str
    brier: float | None
    log_loss: float | None
    calibration: tuple[CalibrationBin, ...]
    frequency_brier: float | None
    coin_brier: float | None
    uncertainty: UncertaintyEstimate | None
    complexity: int
    hypotheses_examined: int
    multiple_testing: MultipleTestingNote | None
    early_improvement_vs_frequency: float | None
    late_improvement_vs_frequency: float | None
    limitations: tuple[str, ...]
    evidence_role: str = "research"
    provisional: bool = False


def assert_stage_row(decision_date: date, target_date: date) -> None:
    """Решение и цель должны лежать в 2019–2021. Цены 2022 и позже не входят."""
    if decision_date >= FORBIDDEN_FROM or target_date >= FORBIDDEN_FROM:
        raise ValueError(
            "этап 2.3.2 использует 2019–2021; 2022 не входит в подбор и не является тестом, "
            "2023–2024 и holdout закрыты"
        )
    if decision_date >= HOLDOUT_START or target_date >= HOLDOUT_START:
        raise ValueError("holdout закрыт")
    if target_date <= decision_date:
        raise ValueError("цель не позже решения")


def stage_window(day: date) -> str:
    """`fit` — 2019–2020, `check` — 2021. Другие даты отклоняются."""
    if day >= HOLDOUT_START:
        raise ValueError("holdout закрыт")
    if day >= FORBIDDEN_FROM:
        raise ValueError(
            "этап 2.3.2 использует 2019–2021; 2022 не входит в подбор и не является тестом, "
            "2023–2024 и holdout закрыты"
        )
    if SEARCH_TRAIN[0] <= day <= SEARCH_TRAIN[1]:
        return "fit"
    if SEARCH_CHECK[0] <= day <= SEARCH_CHECK[1]:
        return "check"
    raise ValueError("дата вне окон 2019–2020 и 2021")


def require_stage_rows(rows: Sequence[TimedRow]) -> None:
    if not rows:
        raise ValueError("нет строк этапа")
    for row in rows:
        assert_stage_row(row.decision_date, row.target_date)
        if row.y_up not in (0, 1):
            raise ValueError("метка направления должна быть 0 или 1")
        if not 0.0 <= row.baseline_frequency <= 1.0:
            raise ValueError("частота прошлых ростов вне [0, 1]")


def feature_values(row: TimedRow, name: str) -> float | None:
    if name not in AUDITED_FEATURES:
        raise ValueError(f"признак {name} не входит в аудит этапа")
    raw = row.features.get(name)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def mapping_copy(features: Mapping[str, float]) -> dict[str, float]:
    return {str(key): float(value) for key, value in features.items()}


def point_brier(probability: float, label: int) -> float:
    return (probability - label) ** 2


DIAGNOSTIC_HORIZONS = (1, 3, 7)


def dependence_block_length(decision_dates: Sequence[date], horizon_days: int) -> tuple[int, bool]:
    """Длина блока в наблюдениях. Соседние решения ближе горизонта — это перекрытие."""
    if horizon_days not in DIAGNOSTIC_HORIZONS:
        raise ValueError("горизонт диагностики должен быть 1, 3 или 7 дней")
    if len(decision_dates) < 2:
        return 1, False
    overlapping = False
    previous: date | None = None
    for day in decision_dates:
        if previous is not None:
            if day <= previous:
                raise ValueError("даты решения не возрастают")
            if (day - previous).days < horizon_days:
                overlapping = True
        previous = day
    if overlapping:
        return horizon_days, True
    return 1, False


def moving_block_means(
    values: Sequence[float],
    block_length: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, ...]:
    """Круговой блочный бутстреп среднего. Число повторностей и seed фиксированы."""
    if replicates < 100:
        raise ValueError("для интервала нужно хотя бы 100 повторностей")
    if block_length < 1:
        raise ValueError("длина блока меньше 1")
    data = [float(value) for value in values]
    count = len(data)
    if count < 2:
        raise ValueError("для бутстрепа нужны хотя бы две точки")
    if block_length > count:
        raise ValueError("блок длиннее ряда")
    generator = random.Random(seed)
    blocks = math.ceil(count / block_length)
    means: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        for _block in range(blocks):
            start = generator.randrange(count)
            for offset in range(block_length):
                if len(sample) >= count:
                    break
                sample.append(data[(start + offset) % count])
        means.append(sum(sample) / len(sample))
    return tuple(means)


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("квантиль вне диапазона")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sample_stdev(values: Sequence[float]) -> float:
    data = [float(value) for value in values]
    if len(data) < 2:
        raise ValueError("для отклонения нужны хотя бы две точки")
    mean = sum(data) / len(data)
    variance = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
    return math.sqrt(variance)


def noise_champion(standard_error: float, hypotheses_examined: int) -> float:
    """Ожидаемый чемпион чистого шума среди K оценок, не порог p-value.

    При независимых оценках N(0, se^2) максимум в среднем около se * sqrt(2 ln K).
    Зависимость гипотез делает эту планку осторожной: реальный шум обычно меньше.
    При одной гипотезе отбора чемпиона нет, поправка равна нулю.
    """
    if hypotheses_examined < 1:
        raise ValueError("число гипотез должно быть положительным")
    if not math.isfinite(standard_error) or standard_error < 0:
        raise ValueError("стандартная ошибка должна быть конечной и неотрицательной")
    if hypotheses_examined == 1:
        return 0.0
    return standard_error * math.sqrt(2.0 * math.log(hypotheses_examined))
