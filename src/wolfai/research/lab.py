"""Контролируемый цикл: Qwen предлагает формулу, лаборатория 2.1 её проверяет.

Сводка для модели строится по окну 2019–2021. Обратная связь — только
проверочное окно 2022. Независимую оценку 2023–2024 выполняет `run_experiment`
и сохраняет её в `ExperimentMemory`; в следующий промпт эти числа не входят.
Holdout не читается. Движок этапа 2.1 не изменяется.

Бюджет одного запуска: не больше 5 гипотез, одна починка схемы на гипотезу,
контекст до 4096 токенов и ответ до 640. Повтор уже проверенной формулы не
запускает вторую итоговую оценку. Модель не скачивается.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from wolfai.data.candles import Candle
from wolfai.data.gateway import admit_daily_research
from wolfai.data.history import HistoricalStore
from wolfai.forecasting.splits import (
    COEFFICIENT_FIT_BOUNDS,
    INNER_CHECK_BOUNDS,
    Period,
    period_bounds,
)
from wolfai.research.engine import (
    MIN_DECISION_ROWS,
    ExperimentMemory,
    ExperimentResult,
    ExperimentRow,
    Hypothesis,
    build_experiment_rows,
    run_experiment,
    score_direction,
)
from wolfai.research.formula import FEATURES, FormulaError, evaluate, parse_formula
from wolfai.research.generator import (
    PROPOSAL_SCHEMA,
    BoundProposal,
    ResearchNote,
    build_lab_messages,
    note_payload,
    parse_proposal,
    quotable_expression,
)
from wolfai.research.hypothesis import HypothesisSchemaError
from wolfai.research.ollama import ModelRejectedOutput, OllamaUnavailable

MAX_LAB_HYPOTHESES = 5
SCHEMA_REPAIRS = 1
_STOP_MARKERS = (
    "воспроизв",
    "контрольной суммой",
    "рабочая модель",
    "первоначально известными",
)


class ResearchWindowError(ValueError):
    """В разрешённой истории нет раздельных окон нужной длины."""


@dataclass(frozen=True)
class ResearchFrame:
    summary: dict[str, object]
    rows_by_horizon: dict[int, tuple[ExperimentRow, ...]]


@dataclass(frozen=True)
class LabAttempt:
    hypothesis_id: str
    expression: str
    features: tuple[str, ...]
    horizon_days: int | None
    rationale: str
    works_when: str
    failure_modes: str
    research_note: ResearchNote
    final_evaluation: ExperimentResult | None
    operator_error: str | None
    rejected_errors: tuple[str, ...]


@dataclass(frozen=True)
class LabRun:
    stop_reason: str
    message: str
    attempts: tuple[LabAttempt, ...]
    model_name: str
    model_digest: str
    schema_mode: str
    data_checksum: str
    generation: dict[str, float | int]


def prepare_research_frame(candles: list[Candle]) -> ResearchFrame:
    """Ряд проходит общий вход. В сводку не попадают целевые значения по дням."""
    admit_daily_research(candles)
    rows = {
        1: tuple(build_experiment_rows(candles, 1)),
        7: tuple(build_experiment_rows(candles, 7)),
    }
    for horizon, bounds in (
        (1, COEFFICIENT_FIT_BOUNDS),
        (1, INNER_CHECK_BOUNDS),
        (7, INNER_CHECK_BOUNDS),
        (1, period_bounds(Period.TRAINING)),
        (7, period_bounds(Period.TRAINING)),
        (1, period_bounds(Period.VALIDATION)),
        (7, period_bounds(Period.VALIDATION)),
    ):
        _require_window(rows[horizon], horizon, bounds)
    return ResearchFrame(summary=summary_payload(rows[1]), rows_by_horizon=rows)


def summary_payload(rows: Sequence[ExperimentRow]) -> dict[str, object]:
    chosen = _decision_rows(rows, COEFFICIENT_FIT_BOUNDS[0], COEFFICIENT_FIT_BOUNDS[1], 1)
    if len(chosen) < MIN_DECISION_ROWS:
        raise ResearchWindowError("недостаточно истории для раздельных окон")
    means, stdevs = _moments(chosen)
    return {
        "designated_window": [
            COEFFICIENT_FIT_BOUNDS[0].isoformat(),
            COEFFICIENT_FIT_BOUNDS[1].isoformat(),
        ],
        "covered_start": chosen[0].decision_date.isoformat(),
        "covered_end": chosen[-1].decision_date.isoformat(),
        "observations": len(chosen),
        "feature_mean": means,
        "feature_stdev": stdevs,
        "targets_included": False,
    }


def score_check_window(
    rows: Sequence[ExperimentRow],
    tree,
    horizon: int,
) -> tuple[float, float, float, int]:
    """Brier проверочного окна 2022. Метрики 2023–2024 эта функция не считает."""
    chosen = _decision_rows(rows, INNER_CHECK_BOUNDS[0], INNER_CHECK_BOUNDS[1], horizon)
    if len(chosen) < MIN_DECISION_ROWS:
        raise ResearchWindowError("проверочное окно слишком короткое")
    probabilities = [evaluate(tree, row.features) for row in chosen]
    labels = [row.y_up for row in chosen]
    frequency = [row.baseline_frequency for row in chosen]
    scored = score_direction(probabilities, labels, frequency)
    skill = scored["skill"]
    if not isinstance(skill, dict):
        raise ValueError("сравнение с ориентиром не посчитано")
    frequency_score = skill["historical_frequency"]
    coin_score = skill["constant_half"]
    if not isinstance(frequency_score, dict) or not isinstance(coin_score, dict):
        raise ValueError("сравнение с ориентиром не посчитано")
    return (
        round(float(scored["brier"]), 6),
        round(float(frequency_score["reference_brier"]), 6),
        round(float(coin_score["reference_brier"]), 6),
        int(scored["observations"]),
    )


def run_qwen_lab(
    client,
    store: HistoricalStore,
    source: str,
    memory: ExperimentMemory,
    *,
    max_hypotheses: int = MAX_LAB_HYPOTHESES,
    share_research_history: bool = True,
) -> LabRun:
    """До пяти гипотез. Итоговая оценка сохраняется и модели не возвращается."""
    if type(max_hypotheses) is not int or not 1 <= max_hypotheses <= MAX_LAB_HYPOTHESES:
        raise ValueError("за один запуск допускается от 1 до 5 гипотез")
    if type(share_research_history) is not bool:
        raise ValueError("share_research_history должен быть логическим")
    checksum = ""
    try:
        manifest = store.manifest(source)
        checksum = manifest.content_sha256
        frame = prepare_research_frame(store.read_candles(source))
    except ResearchWindowError as error:
        return _empty(stop_reason="insufficient_history", message=str(error), data_checksum=checksum)
    except ValueError as error:
        return _empty(stop_reason="data_rejected", message=str(error), data_checksum=checksum)
    try:
        client.health()
    except OllamaUnavailable as error:
        return _empty(
            stop_reason="ollama_unavailable",
            message=str(error),
            data_checksum=checksum,
            client=client,
        )
    attempts: list[LabAttempt] = []
    stop_reason = "completed"
    message = "гипотезы проверены"
    for _ in range(max_hypotheses):
        notes = [attempt.research_note for attempt in attempts]
        try:
            bound, errors, raw = _request_bound(client, frame.summary, notes, share_research_history)
        except OllamaUnavailable as error:
            stop_reason = "ollama_unavailable"
            message = str(error)
            break
        if bound is None:
            expression = quotable_expression(raw)
            limitation = errors[-1] if errors else "формула отклонена"
            attempts.append(_invalid_attempt(expression, limitation, errors))
            continue
        prior = _prior_attempt(attempts, bound)
        if prior is not None:
            attempts.append(_duplicate_attempt(bound, prior))
            continue
        attempt, fatal = _evaluate_attempt(
            bound,
            errors,
            frame,
            store,
            source,
            checksum,
            memory,
        )
        attempts.append(attempt)
        if fatal:
            stop_reason = "not_reproducible"
            message = attempt.operator_error or "результат не воспроизвёлся"
            break
    return _empty(
        stop_reason=stop_reason,
        message=message,
        attempts=attempts,
        data_checksum=checksum,
        client=client,
    )


def lab_report(result: LabRun) -> dict[str, object]:
    """Отчёт оператора. Исследовательская обратная связь и итоговая оценка лежат рядом и не смешиваются."""
    attempts: list[dict[str, object]] = []
    for attempt in result.attempts:
        item: dict[str, object] = {
            "id": attempt.hypothesis_id,
            "proposal": {
                "expression": attempt.expression,
                "features": list(attempt.features),
                "horizon_days": attempt.horizon_days,
                "rationale": attempt.rationale,
                "works_when": attempt.works_when,
                "failure_modes": attempt.failure_modes,
            },
            "research_feedback": note_payload(attempt.research_note),
            "final_evaluation": _final_payload(attempt.final_evaluation),
        }
        if attempt.operator_error:
            item["operator_error"] = attempt.operator_error
        if attempt.rejected_errors:
            item["rejected_errors"] = list(attempt.rejected_errors)
        attempts.append(item)
    return {
        "stop_reason": result.stop_reason,
        "message": result.message,
        "model": result.model_name,
        "schema_mode": result.schema_mode,
        "dataset_checksum": result.data_checksum,
        "attempts": attempts,
    }


def _request_bound(
    client,
    summary: dict[str, object],
    notes: list[ResearchNote],
    share_research_history: bool,
) -> tuple[BoundProposal | None, tuple[str, ...], str]:
    errors: list[str] = []
    repair: str | None = None
    raw = ""
    for _ in range(1 + SCHEMA_REPAIRS):
        messages = build_lab_messages(
            summary=summary,
            notes=notes,
            repair_error=repair,
            share_research_history=share_research_history,
        )
        try:
            completion = client.complete(messages, PROPOSAL_SCHEMA)
        except ModelRejectedOutput as error:
            repair = str(error)[:200]
            errors.append(repair)
            raw = ""
            continue
        raw = completion.content
        try:
            return parse_proposal(raw), tuple(errors), raw
        except (HypothesisSchemaError, FormulaError) as error:
            repair = str(error)[:200]
            errors.append(repair)
    return None, tuple(errors), raw


def _evaluate_attempt(
    bound: BoundProposal,
    errors: tuple[str, ...],
    frame: ResearchFrame,
    store: HistoricalStore,
    source: str,
    checksum: str,
    memory: ExperimentMemory,
) -> tuple[LabAttempt, bool]:
    hypothesis = Hypothesis(
        id=f"qwen-{uuid.uuid4().hex[:12]}",
        expression=bound.expression,
        features=bound.features,
        horizon_days=bound.horizon_days,
        created_at=datetime.now(timezone.utc),
        dataset_version=checksum,
        status="proposed",
    )
    check_error: str | None = None
    metrics: tuple[float, float, float, int] | None = None
    try:
        metrics = score_check_window(
            frame.rows_by_horizon[bound.horizon_days],
            parse_formula(bound.expression),
            bound.horizon_days,
        )
    except (FormulaError, ValueError) as error:
        check_error = str(error)[:200]
    final: ExperimentResult | None = None
    operator_error: str | None = None
    fatal = False
    try:
        final = run_experiment(store, source, hypothesis, memory)
    except (FormulaError, ValueError) as error:
        operator_error = str(error)[:300]
        fatal = any(marker in operator_error for marker in _STOP_MARKERS)
    if metrics is None:
        limitation = check_error or "проверочное окно не посчитано"
        note = ResearchNote(
            expression=bound.expression,
            features=bound.features,
            horizon_days=bound.horizon_days,
            limitation=limitation,
        )
    else:
        brier, frequency, coin, observations = metrics
        note = ResearchNote(
            expression=bound.expression,
            features=bound.features,
            horizon_days=bound.horizon_days,
            limitation=_check_limitation(brier, frequency, coin),
            observations=observations,
            brier=brier,
            frequency_brier=frequency,
            coin_brier=coin,
        )
    attempt = LabAttempt(
        hypothesis_id=hypothesis.id,
        expression=bound.expression,
        features=bound.features,
        horizon_days=bound.horizon_days,
        rationale=bound.rationale,
        works_when=bound.works_when,
        failure_modes=bound.failure_modes,
        research_note=note,
        final_evaluation=final,
        operator_error=operator_error,
        rejected_errors=errors,
    )
    return attempt, fatal


def _invalid_attempt(expression: str, limitation: str, errors: tuple[str, ...]) -> LabAttempt:
    note = ResearchNote(
        expression=expression,
        features=(),
        horizon_days=None,
        limitation=limitation,
    )
    return LabAttempt(
        hypothesis_id=f"qwen-{uuid.uuid4().hex[:12]}",
        expression=expression,
        features=(),
        horizon_days=None,
        rationale="",
        works_when="",
        failure_modes="",
        research_note=note,
        final_evaluation=None,
        operator_error=None,
        rejected_errors=errors,
    )


def _duplicate_attempt(bound: BoundProposal, prior: LabAttempt) -> LabAttempt:
    previous = prior.research_note
    note = ResearchNote(
        expression=bound.expression,
        features=bound.features,
        horizon_days=bound.horizon_days,
        limitation="формула уже проверялась в этом запуске; повторная итоговая оценка не выполняется",
        observations=previous.observations,
        brier=previous.brier,
        frequency_brier=previous.frequency_brier,
        coin_brier=previous.coin_brier,
    )
    return LabAttempt(
        hypothesis_id=f"qwen-{uuid.uuid4().hex[:12]}",
        expression=bound.expression,
        features=bound.features,
        horizon_days=bound.horizon_days,
        rationale=bound.rationale,
        works_when=bound.works_when,
        failure_modes=bound.failure_modes,
        research_note=note,
        final_evaluation=None,
        operator_error=None,
        rejected_errors=(),
    )


def _prior_attempt(attempts: list[LabAttempt], bound: BoundProposal) -> LabAttempt | None:
    for attempt in attempts:
        if attempt.expression == bound.expression and attempt.horizon_days == bound.horizon_days:
            return attempt
    return None


def _check_limitation(brier: float, frequency: float, coin: float) -> str:
    if brier < frequency and brier < coin:
        return "проверочное окно 2022: Brier ниже обоих ориентиров"
    return "проверочное окно 2022: Brier не ниже обоих ориентиров"


def _final_payload(result: ExperimentResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    validation = result.direction_metrics["validation"]
    if not isinstance(validation, dict):
        raise ValueError("итоговая оценка повреждена")
    return {
        "decision": result.decision,
        "reason": result.reason,
        "promoted": result.promoted,
        "validation_brier": validation["brier"],
        "validation_start": result.validation_start.isoformat(),
        "validation_end": result.validation_end.isoformat(),
    }


def _decision_rows(
    rows: Sequence[ExperimentRow],
    start: date,
    end: date,
    horizon: int,
) -> list[ExperimentRow]:
    chosen: list[ExperimentRow] = []
    next_day: date | None = None
    for row in rows:
        if row.decision_date < start or row.decision_date > end:
            continue
        if row.target_date < start or row.target_date > end:
            continue
        if next_day is not None and row.decision_date < next_day:
            continue
        chosen.append(row)
        next_day = row.decision_date + timedelta(days=horizon)
    return chosen


def _require_window(rows: Sequence[ExperimentRow], horizon: int, bounds: tuple[date, date]) -> None:
    if len(_decision_rows(rows, bounds[0], bounds[1], horizon)) < MIN_DECISION_ROWS:
        raise ResearchWindowError("недостаточно истории для раздельных окон")


def _moments(rows: Sequence[ExperimentRow]) -> tuple[dict[str, float], dict[str, float]]:
    count = len(rows)
    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    for name in FEATURES:
        values = [row.features[name] for row in rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        means[name] = round(mean, 6)
        stdevs[name] = round(variance ** 0.5, 6)
    return means, stdevs


def _empty(
    *,
    stop_reason: str,
    message: str,
    attempts: list[LabAttempt] | None = None,
    data_checksum: str = "",
    client=None,
) -> LabRun:
    generation = getattr(client, "generation", {}) if client is not None else {}
    return LabRun(
        stop_reason=stop_reason,
        message=message,
        attempts=tuple(attempts or ()),
        model_name=str(getattr(client, "model_name", "") or ""),
        model_digest=str(getattr(client, "model_digest", "") or ""),
        schema_mode=str(getattr(client, "schema_mode", "") or ""),
        data_checksum=data_checksum,
        generation=dict(generation) if isinstance(generation, dict) else {},
    )
