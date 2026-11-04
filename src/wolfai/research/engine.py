"""Проверка математических гипотез без вызова языковой модели.

Формула разбирается грамматикой, ряд читается только из HistoricalStore,
а решение принимается по валидации. Рабочая модель отсюда не назначается.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from wolfai.data.gateway import admit_daily_research, require_calendar_days
from wolfai.data.history import HistoricalStore
from wolfai.forecasting.baseline import HistoricalQuantileBaseline
from wolfai.forecasting.evaluate import evaluate_split
from wolfai.forecasting.random_walk import RandomWalkBaseline
from wolfai.forecasting.splits import Period, period_bounds
from wolfai.research.features import LOOKBACK, feature_vector
from wolfai.research.formula import (
    FormulaError,
    evaluate,
    features_used,
    node_count,
    parse_formula,
    tree_depth,
)
from wolfai.research.logistic import score_probabilities

ENGINE_VERSION = "2.1.0"
AVAILABILITY_BASIS = "daily_close_known_next_utc_midnight"
MIN_DECISION_ROWS = 20
DECISION_RULE = "validation_nonoverlapping_brier_beats_frequency_and_coin"


@dataclass(frozen=True)
class Hypothesis:
    """Гипотеза, которую проверяет Python. Статус active сюда не ставится."""

    id: str
    expression: str
    features: tuple[str, ...]
    horizon_days: int
    created_at: datetime
    dataset_version: str
    status: str = "proposed"


@dataclass(frozen=True)
class ExperimentRow:
    index: int
    decision_date: date
    target_date: date
    decision_available: date
    label_available: date
    features: dict[str, float]
    y_up: int
    baseline_frequency: float


@dataclass(frozen=True)
class ExperimentResult:
    hypothesis_id: str
    dataset_version: str
    dataset_checksum: str
    expression: str
    horizon_days: int
    features: tuple[str, ...]
    parameters: dict[str, object]
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    direction_metrics: dict[str, object]
    price_metrics: dict[str, object]
    decision: str
    reason: str
    promoted: bool
    revisions_confirmed: bool
    availability_basis: str
    point_in_time_note: str
    code_version: str
    settings: dict[str, object]
    overlap_stride: int
    effective_validation_observations: int


class ExperimentMemory:
    """Память экспериментов. Одинаковый ключ не получает второй, другой результат."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_key TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                dataset_checksum TEXT NOT NULL,
                code_version TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS discovery (
                search_key TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                dataset_checksum TEXT NOT NULL,
                code_version TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def find(self, experiment_key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiments WHERE experiment_key = ?",
            (experiment_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("память эксперимента повреждена")
        return payload

    def save(self, experiment_key: str, result: ExperimentResult) -> None:
        payload = _canonical(result_payload(result))
        existing = self.find(experiment_key)
        if existing is not None:
            if existing != payload:
                raise ValueError("результат не воспроизвёлся")
            return
        self._connection.execute(
            """
            INSERT INTO experiments (
                experiment_key, hypothesis_id, dataset_checksum, code_version, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                experiment_key,
                result.hypothesis_id,
                result.dataset_checksum,
                result.code_version,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._connection.commit()

    def promote(self, hypothesis_id: str) -> None:
        raise ValueError(
            f"рабочая модель не заменяется непроверенной гипотезой {hypothesis_id}"
        )

    def find_discovery(self, key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM discovery WHERE search_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("память поиска повреждена")
        return payload

    def save_discovery(
        self,
        key: str,
        payload: dict[str, object],
        *,
        origin: str,
        dataset_checksum: str,
    ) -> None:
        existing = self.find_discovery(key)
        if existing is not None:
            if _stable_discovery(existing) != _stable_discovery(payload):
                raise ValueError("результат не воспроизвёлся")
            return
        self._connection.execute(
            """
            INSERT INTO discovery (
                search_key, origin, dataset_checksum, code_version, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                key,
                origin,
                dataset_checksum,
                str(payload.get("code_version") or ""),
                json.dumps(_canonical(payload), ensure_ascii=False, sort_keys=True),
            ),
        )
        self._connection.commit()


def _stable_discovery(payload: dict[str, object]) -> dict[str, object]:
    cleaned = {
        key: value
        for key, value in payload.items()
        if key not in {"elapsed_seconds", "max_rss_bytes"}
    }
    return _canonical(cleaned)


def score_direction(
    probabilities: list[float],
    labels: list[int],
    frequency: list[float],
) -> dict[str, object]:
    """Brier, log loss и калибровка. Ошибки цены сюда не входят."""
    return score_probabilities(
        probabilities,
        labels,
        {
            "historical_frequency": frequency,
            "constant_half": [0.5] * len(labels),
        },
    )


def judge_hypothesis(
    *,
    train_brier: float,
    train_frequency_brier: float,
    validation_brier: float,
    validation_frequency_brier: float,
    validation_coin_brier: float,
) -> tuple[str, str]:
    """Улучшение только на обучении гипотезу не принимает."""
    beats_train = train_brier < train_frequency_brier
    beats_validation = (
        validation_brier < validation_frequency_brier and validation_brier < validation_coin_brier
    )
    if beats_train and not beats_validation:
        return "rejected", "улучшение видно только на обучении"
    if not beats_validation:
        return "rejected", "валидация не лучше контрольных вероятностей"
    return "candidate", "валидация лучше контрольных вероятностей; рабочая модель не заменяется"


def build_experiment_rows(candles: list, horizon_days: int) -> list[ExperimentRow]:
    """Признаки известны на as_of решения. Метка горизонта созревает позже."""
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    admit_daily_research(candles)
    rows: list[ExperimentRow] = []
    last = len(candles) - horizon_days
    for index in range(LOOKBACK, last):
        require_calendar_days(candles, index, index + horizon_days, horizon_days)
        decision = candles[index].timestamp.date()
        target = candles[index + horizon_days].timestamp.date()
        row = ExperimentRow(
            index=index,
            decision_date=decision,
            target_date=target,
            decision_available=decision + timedelta(days=1),
            label_available=target + timedelta(days=1),
            features=feature_vector(candles, index, _admitted=True),
            y_up=1 if candles[index + horizon_days].close > candles[index].close else 0,
            baseline_frequency=_past_frequency(candles, index, horizon_days),
        )
        if row.label_available <= row.decision_available:
            raise ValueError("метка созревает не позже признаков")
        rows.append(row)
    return rows


def run_experiment(
    store: HistoricalStore,
    source: str,
    hypothesis: Hypothesis,
    memory: ExperimentMemory | None = None,
    *,
    revisions_confirmed: bool = False,
) -> ExperimentResult:
    """Проверяет одну гипотезу на ряде из хранилища и записывает результат."""
    if revisions_confirmed:
        raise ValueError(
            "исправленные значения нельзя считать первоначально известными "
            "без отдельного времени публикации"
        )
    if hypothesis.status == "active":
        raise ValueError("рабочая модель не заменяется непроверенной гипотезой")
    if hypothesis.horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    if not hypothesis.id.strip():
        raise ValueError("id гипотезы пуст")
    manifest = store.manifest(source)
    if hypothesis.dataset_version != manifest.content_sha256:
        raise ValueError("гипотеза привязана к другой контрольной суммой датасета")
    candles = store.read_candles(source)
    tree = parse_formula(hypothesis.expression)
    if features_used(tree) != set(hypothesis.features):
        raise FormulaError("набор признаков в формуле не совпадает с полями features")
    rows = build_experiment_rows(candles, hypothesis.horizon_days)
    train_start, train_end = period_bounds(Period.TRAINING)
    validation_start, validation_end = period_bounds(Period.VALIDATION)
    train_rows = _between(rows, train_start, train_end)
    validation_rows = _between(rows, validation_start, validation_end)
    train_decision = _nonoverlapping(train_rows, hypothesis.horizon_days)
    validation_decision = _nonoverlapping(validation_rows, hypothesis.horizon_days)
    if len(train_decision) < MIN_DECISION_ROWS or len(validation_decision) < MIN_DECISION_ROWS:
        raise ValueError("в обучении или валидации слишком мало точек")
    train_metrics = _score_rows(train_decision, tree)
    validation_metrics = _score_rows(validation_decision, tree)
    validation_overlapping = _score_rows(validation_rows, tree)
    decision, reason = judge_hypothesis(
        train_brier=float(train_metrics["brier"]),
        train_frequency_brier=float(train_metrics["skill"]["historical_frequency"]["reference_brier"]),
        validation_brier=float(validation_metrics["brier"]),
        validation_frequency_brier=float(
            validation_metrics["skill"]["historical_frequency"]["reference_brier"]
        ),
        validation_coin_brier=float(validation_metrics["skill"]["constant_half"]["reference_brier"]),
    )
    result = ExperimentResult(
        hypothesis_id=hypothesis.id,
        dataset_version=hypothesis.dataset_version,
        dataset_checksum=manifest.content_sha256,
        expression=hypothesis.expression,
        horizon_days=hypothesis.horizon_days,
        features=tuple(hypothesis.features),
        parameters={
            "horizon_days": hypothesis.horizon_days,
            "features": list(hypothesis.features),
            "node_count": node_count(tree),
            "depth": tree_depth(tree),
        },
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        direction_metrics={
            "train": train_metrics,
            "validation": validation_metrics,
            "validation_overlapping": validation_overlapping,
        },
        price_metrics=_price_metrics(candles, hypothesis.horizon_days),
        decision=decision,
        reason=reason,
        promoted=False,
        revisions_confirmed=False,
        availability_basis=AVAILABILITY_BASIS,
        point_in_time_note=(
            "Close известен в следующую полночь UTC. Исправленная биржа история "
            "без отдельного времени публикации не считается первоначально известной."
        ),
        code_version=ENGINE_VERSION,
        settings={
            "engine_version": ENGINE_VERSION,
            "decision_rule": DECISION_RULE,
            "overlap_stride": hypothesis.horizon_days,
            "minimum_decision_rows": MIN_DECISION_ROWS,
        },
        overlap_stride=hypothesis.horizon_days,
        effective_validation_observations=int(validation_metrics["observations"]),
    )
    if memory is not None:
        memory.save(_experiment_key(hypothesis, manifest.content_sha256), result)
    return result


def _score_rows(rows: list[ExperimentRow], tree) -> dict[str, object]:
    probabilities: list[float] = []
    labels: list[int] = []
    frequency: list[float] = []
    for row in rows:
        probabilities.append(evaluate(tree, row.features))
        labels.append(row.y_up)
        frequency.append(row.baseline_frequency)
    return score_direction(probabilities, labels, frequency)


def _price_metrics(candles: list, horizon_days: int) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name, model in (
        ("historical_quantile", HistoricalQuantileBaseline()),
        ("random_walk", RandomWalkBaseline()),
    ):
        periods: dict[str, object] = {}
        for period in (Period.TRAINING, Period.VALIDATION):
            result = evaluate_split(candles, horizon_days, period, model)
            periods[period.value] = {
                "observations": result.observations,
                "mean_absolute_error": result.mean_absolute_error,
                "mean_absolute_percentage_error": result.mean_absolute_percentage_error,
                "interval_coverage": result.interval_coverage,
                "mean_interval_width": result.mean_interval_width,
                "mean_interval_score": result.mean_interval_score,
            }
        metrics[name] = periods
    return metrics


def _between(rows: list[ExperimentRow], start: date, end: date) -> list[ExperimentRow]:
    return [
        row for row in rows
        if start <= row.decision_date <= end and start <= row.target_date <= end
    ]


def _nonoverlapping(rows: list[ExperimentRow], horizon_days: int) -> list[ExperimentRow]:
    chosen: list[ExperimentRow] = []
    next_day: date | None = None
    for row in rows:
        if next_day is not None and row.decision_date < next_day:
            continue
        chosen.append(row)
        next_day = row.decision_date + timedelta(days=horizon_days)
    return chosen


def _past_frequency(candles: list, index: int, horizon_days: int) -> float:
    ends = range(horizon_days, index + 1)
    count = index - horizon_days + 1
    ups = sum(candles[end].close > candles[end - horizon_days].close for end in ends)
    return ups / count


def _experiment_key(hypothesis: Hypothesis, checksum: str) -> str:
    payload = {
        "id": hypothesis.id,
        "expression": hypothesis.expression,
        "features": list(hypothesis.features),
        "horizon_days": hypothesis.horizon_days,
        "dataset_checksum": checksum,
        "code_version": ENGINE_VERSION,
        "decision_rule": DECISION_RULE,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def result_payload(result: ExperimentResult) -> dict[str, object]:
    return {
        "hypothesis_id": result.hypothesis_id,
        "dataset_checksum": result.dataset_checksum,
        "expression": result.expression,
        "horizon_days": result.horizon_days,
        "features": list(result.features),
        "parameters": result.parameters,
        "train_start": result.train_start.isoformat(),
        "train_end": result.train_end.isoformat(),
        "validation_start": result.validation_start.isoformat(),
        "validation_end": result.validation_end.isoformat(),
        "direction_metrics": result.direction_metrics,
        "price_metrics": result.price_metrics,
        "decision": result.decision,
        "reason": result.reason,
        "promoted": result.promoted,
        "revisions_confirmed": result.revisions_confirmed,
        "code_version": result.code_version,
        "settings": result.settings,
        "overlap_stride": result.overlap_stride,
        "effective_validation_observations": result.effective_validation_observations,
    }


def _canonical(payload: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, sort_keys=True))
