"""Цикл: Qwen предлагает формулу, Python проверяет её и пишет журнал."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from wolfai.data.candles import Candle
from wolfai.forecasting.baseline import HistoricalQuantileBaseline, WalkForwardResult
from wolfai.forecasting.evaluate import evaluate_split
from wolfai.forecasting.random_walk import RandomWalkBaseline
from wolfai.forecasting.splits import Period
from wolfai.research.features import (
    FeatureRow,
    assert_research_series,
    build_rows,
    rows_for_period,
)
from wolfai.research.formula import (
    EvalState,
    Expr,
    FormulaError,
    evaluate,
    features_used,
    parse_formula,
)
from wolfai.research.hypothesis import HypothesisDraft, HypothesisSchemaError, parse_draft
from wolfai.research.journal import HypothesisRecord, Journal, utc_now
from wolfai.research.ollama import (
    GENERATION,
    ChatCompletion,
    OllamaUnavailable,
)

MAX_HYPOTHESES = 5
SCHEMA_REPAIRS = 1
MIN_OBSERVATIONS = 100
EFFECT_DELTA = 0.01
EFFECT_SCALE = 1e-4
CONSTANT_STDEV = 1e-4
HORIZON_DAYS = 1

SYSTEM_PROMPT = """Ты исследователь WolfAi. Предложи одну проверяемую формулу вероятности роста close BTCUSDT на следующий день.
Верни только JSON без пояснения вокруг него.
Корень формулы — sigmoid. Внутри разрешены признаки, числа от -10 до 10, скобки, + - * /, sigmoid, tanh, abs, sign.
Признаки: log_return, momentum_7, momentum_30, volatility_7, volatility_30, volume_change, range_position.
Деление на ноль Python заменяет нулём. Не используй код, функции Python и имена вне списка.
Поле effects задаёт знак влияния каждого заявленного признака: positive или negative.
status не указывай. Успешность формулы определяет только Python.
Не проси данные после 2024-12-31 и не ссылайся на них: их нет.
Пример: {"expression":"sigmoid(0.4 * momentum_7 - 0.2 * volatility_30)","features":["momentum_7","volatility_30"],"effects":[{"feature":"momentum_7","sign":"positive"},{"feature":"volatility_30","sign":"negative"}],"explanation":"рост импульса повышает вероятность продолжения, высокая волатильность снижает её","based_on":[]}
"""


@dataclass(frozen=True)
class RunResult:
    run_id: str
    stop_reason: str
    message: str
    hypothesis_ids: tuple[str, ...]
    data_version: str


class ResearchClient:
    """Минимальный контракт клиента. У тестовой подмены тот же набор методов."""

    model_name: str
    model_digest: str
    generation: dict[str, object]

    def health(self) -> object:
        raise NotImplementedError

    def complete(self, messages: list[dict[str, str]]) -> ChatCompletion:
        raise NotImplementedError


def series_version(candles: list[Candle]) -> str:
    import hashlib

    payload = "\n".join(
        f"{candle.timestamp.isoformat()},{candle.open},{candle.high},{candle.low},{candle.close},{candle.volume}"
        for candle in candles
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_research(
    client: ResearchClient,
    candles: list[Candle],
    journal: Journal,
    *,
    max_hypotheses: int = MAX_HYPOTHESES,
) -> RunResult:
    """До max_hypotheses попыток. Holdout в ряд и в промпт не входит."""
    if max_hypotheses < 1 or max_hypotheses > MAX_HYPOTHESES:
        raise ValueError("за один запуск допускается от 1 до 5 гипотез")
    assert_research_series(candles)
    data_version = series_version(candles)
    run_id = uuid.uuid4().hex
    journal.start_run(
        run_id,
        data_version=data_version,
        model_name=getattr(client, "model_name", None),
        generation=dict(getattr(client, "generation", GENERATION)),
        max_hypotheses=max_hypotheses,
    )
    try:
        info = client.health()
    except OllamaUnavailable as error:
        journal.finish_run(run_id, model_digest=None, baseline_metrics=None, stop_reason="ollama_unavailable")
        return RunResult(run_id, "ollama_unavailable", str(error), (), data_version)
    digest = str(getattr(info, "digest", "") or client.model_digest)
    rows = build_rows(candles)
    training_rows = rows_for_period(rows, Period.TRAINING)
    validation_rows = rows_for_period(rows, Period.VALIDATION)
    if len(training_rows) < MIN_OBSERVATIONS or not validation_rows:
        journal.finish_run(run_id, model_digest=digest, baseline_metrics=None, stop_reason="insufficient_history")
        return RunResult(
            run_id,
            "insufficient_history",
            "в обучении или валидации слишком мало точек для гипотезы",
            (),
            data_version,
        )
    baseline_metrics = _baseline_reference(candles)
    stats = _feature_stats(training_rows)
    created: list[str] = []
    feedback: list[str] = []
    stop_reason = "completed"
    message = "гипотезы проверены"
    for _attempt in range(max_hypotheses):
        messages = build_messages(
            stats=stats,
            previous=_recent_summaries(journal),
            feedback=feedback[-1] if feedback else None,
            data_version=data_version,
        )
        try:
            draft, completion, raw, schema_error = _request_draft(client, messages)
        except OllamaUnavailable as error:
            stop_reason = "ollama_unavailable"
            message = str(error)
            break
        if schema_error is not None or draft is None or completion is None:
            record = _failure_record(
                run_id=run_id,
                data_version=data_version,
                completion=completion,
                expression="",
                features=[],
                effects=[],
                explanation="",
                parent_ids=[],
                version=1,
                reason=schema_error or "JSON не прошёл схему",
                error_type="schema",
                raw_response=raw,
            )
            journal.add(record)
            created.append(record.id)
            feedback.append(schema_error or "JSON не прошёл схему")
            continue
        parent_ids = [item for item in draft.based_on if item in journal.known_ids()]
        version = 1
        if parent_ids:
            version = 1 + max(journal.get(item).version for item in parent_ids if journal.get(item))
        try:
            tree = parse_formula(draft.expression)
            used = features_used(tree)
            if used != set(draft.features):
                raise FormulaError("набор признаков в формуле не совпадает с полями features и effects")
            train_metrics = _score_direction(training_rows, tree)
            validation_metrics = _score_direction(validation_rows, tree)
            effect = _effect_report(training_rows, tree, draft)
        except FormulaError as error:
            record = _failure_record(
                run_id=run_id,
                data_version=data_version,
                completion=completion,
                expression=draft.expression,
                features=list(draft.features),
                effects=[effect.model_dump() for effect in draft.effects],
                explanation=draft.explanation,
                parent_ids=parent_ids,
                version=version,
                reason=str(error),
                error_type="formula",
                raw_response=raw,
            )
            journal.add(record)
            created.append(record.id)
            feedback.append(str(error))
            continue
        status = "experimental"
        reason = None
        error_type = effect["error_type"]
        if train_metrics["constant"]:
            status = "archived"
            reason = "вероятность почти не меняется"
            error_type = "constant_probability"
        elif effect["contradiction"]:
            status = "archived"
            reason = "знак влияния противоречит численной проверке на обучении"
        observations = int(train_metrics["observations"])
        record = HypothesisRecord(
            id=uuid.uuid4().hex,
            run_id=run_id,
            created_at=utc_now(),
            expression=draft.expression,
            features=list(draft.features),
            effects=[item.model_dump() for item in draft.effects],
            explanation=draft.explanation,
            version=version,
            status=status,
            parent_ids=parent_ids,
            data_version=data_version,
            model_name=completion.model_name,
            model_digest=completion.model_digest,
            generation=dict(completion.generation),
            rejection_reason=reason,
            error_type=error_type,
            error_scale=effect["error_scale"],
            observations=observations,
            sufficient_observations=bool(effect["sufficient"]),
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            raw_response=raw[:4000],
        )
        journal.add(record)
        created.append(record.id)
        feedback.append(_feedback_text(record))
    journal.finish_run(
        run_id,
        model_digest=digest,
        baseline_metrics=baseline_metrics,
        stop_reason=stop_reason,
    )
    return RunResult(run_id, stop_reason, message, tuple(created), data_version)


def build_messages(
    *,
    stats: dict[str, object],
    previous: list[dict[str, object]],
    feedback: str | None,
    data_version: str,
) -> list[dict[str, str]]:
    """Промпт из агрегатов и журнала. Цены свечей сюда не кладутся."""
    lines = [
        f"Версия исследовательского ряда (sha256): {data_version[:12]}.",
        "Горизонт: следующий close после последнего известного. Ряд заканчивается 2024-12-31.",
        "Сводка обучения, не цены:",
    ]
    for key, value in stats.items():
        lines.append(f"- {key}: {value}")
    if previous:
        lines.append("Последние записи журнала:")
        for item in previous:
            lines.append(
                f"- id={item['id']} status={item['status']} expression={item['expression']} "
                f"train_brier={item.get('train_brier')} validation_brier={item.get('validation_brier')} "
                f"error={item.get('error_type')} reason={item.get('rejection_reason')}"
            )
    else:
        lines.append("Журнал пока пуст.")
    if feedback:
        lines.append(f"Замечание Python к предыдущей попытке: {feedback}")
    lines.append("Предложи новую формулу, не копируй отвергнутое выражение дословно.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _request_draft(
    client: ResearchClient,
    messages: list[dict[str, str]],
) -> tuple[HypothesisDraft | None, ChatCompletion | None, str, str | None]:
    completion: ChatCompletion | None = None
    raw = ""
    error_text: str | None = None
    for _ in range(SCHEMA_REPAIRS + 1):
        completion = client.complete(messages)
        raw = completion.content
        try:
            return parse_draft(raw), completion, raw, None
        except HypothesisSchemaError as error:
            error_text = str(error)
            messages = [
                *messages,
                {"role": "assistant", "content": raw[:2000]},
                {
                    "role": "user",
                    "content": f"JSON не прошёл схему: {error}. Верни один объект JSON по примеру.",
                },
            ]
    return None, completion, raw, error_text


def _score_direction(rows: list[FeatureRow], tree: Expr) -> dict[str, object]:
    if not rows:
        raise FormulaError("нет наблюдений в периоде")
    probabilities: list[float] = []
    labels: list[int] = []
    baseline: list[float] = []
    protected = 0
    for row in rows:
        state = EvalState()
        probability = evaluate(tree, row.features, state)
        protected += state.protected_divisions
        probabilities.append(probability)
        labels.append(row.y_up)
        baseline.append(row.baseline_up_frequency)
    count = len(probabilities)
    mean = sum(probabilities) / count
    variance = sum((item - mean) ** 2 for item in probabilities) / count
    return {
        "horizon_days": HORIZON_DAYS,
        "observations": count,
        "brier": _brier(probabilities, labels),
        "brier_quantile_baseline": _brier(baseline, labels),
        "brier_random_walk": _brier([0.5] * count, labels),
        "base_rate": sum(labels) / count,
        "mean_probability": mean,
        "probability_stdev": math.sqrt(variance),
        "constant": math.sqrt(variance) < CONSTANT_STDEV,
        "protected_divisions": protected,
        "calibration": _calibration(probabilities, labels),
        "note": (
            "Brier сравнивается только с вероятностями направления. "
            "MAE ценового baseline с Brier не сопоставляется."
        ),
    }


def _effect_report(rows: list[FeatureRow], tree: Expr, draft: HypothesisDraft) -> dict[str, object]:
    sufficient = len(rows) >= MIN_OBSERVATIONS
    mismatches: list[dict[str, object]] = []
    largest = 0.0
    for effect in draft.effects:
        deltas: list[float] = []
        for row in rows:
            base = evaluate(tree, row.features, EvalState())
            bumped = dict(row.features)
            bumped[effect.feature] = bumped[effect.feature] + EFFECT_DELTA
            altered = evaluate(tree, bumped, EvalState())
            deltas.append(altered - base)
        mean_delta = sum(deltas) / len(deltas)
        expected_positive = effect.sign == "positive"
        opposite = abs(mean_delta) >= EFFECT_SCALE and (mean_delta > 0) != expected_positive
        largest = max(largest, abs(mean_delta))
        if opposite:
            mismatches.append({
                "feature": effect.feature,
                "declared_sign": effect.sign,
                "mean_delta": mean_delta,
            })
    contradiction = bool(sufficient and mismatches)
    if contradiction:
        error_type = "effect_sign_mismatch"
    elif mismatches:
        error_type = "effect_sign_inconclusive"
    else:
        error_type = None
    return {
        "error_type": error_type,
        "error_scale": largest if mismatches else None,
        "sufficient": sufficient,
        "contradiction": contradiction,
        "mismatches": mismatches,
    }


def _baseline_reference(candles: list[Candle]) -> dict[str, object]:
    """Ценовые метрики baseline. Это отдельный тип выхода, не рейтинг формулы."""
    reference: dict[str, object] = {
        "note": "MAE и интервалы baseline не сравниваются с Brier гипотезы.",
        "horizon_days": HORIZON_DAYS,
    }
    for name, model in (
        ("historical_quantile", HistoricalQuantileBaseline()),
        ("random_walk", RandomWalkBaseline()),
    ):
        periods: dict[str, object] = {}
        for period in (Period.TRAINING, Period.VALIDATION):
            result = evaluate_split(candles, HORIZON_DAYS, period, model)
            periods[period.value] = _price_metrics(result)
        reference[name] = periods
    return reference


def _price_metrics(result: WalkForwardResult) -> dict[str, object]:
    return {
        "observations": result.observations,
        "mean_absolute_error": result.mean_absolute_error,
        "mean_absolute_percentage_error": result.mean_absolute_percentage_error,
        "interval_coverage": result.interval_coverage,
        "mean_interval_width": result.mean_interval_width,
        "mean_interval_score": result.mean_interval_score,
    }


def _feature_stats(rows: list[FeatureRow]) -> dict[str, object]:
    names = list(rows[0].features)
    stats: dict[str, object] = {
        "observations": len(rows),
        "up_rate": round(sum(row.y_up for row in rows) / len(rows), 4),
    }
    for name in names:
        values = [row.features[name] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stats[name] = {"mean": round(mean, 6), "stdev": round(math.sqrt(variance), 6)}
    return stats


def _recent_summaries(journal: Journal) -> list[dict[str, object]]:
    summaries = []
    for record in journal.list_hypotheses()[-5:]:
        train = record.train_metrics or {}
        validation = record.validation_metrics or {}
        summaries.append({
            "id": record.id,
            "status": record.status,
            "expression": record.expression,
            "train_brier": train.get("brier"),
            "validation_brier": validation.get("brier"),
            "error_type": record.error_type,
            "rejection_reason": record.rejection_reason,
        })
    return summaries


def _feedback_text(record: HypothesisRecord) -> str:
    train = record.train_metrics or {}
    validation = record.validation_metrics or {}
    return (
        f"id={record.id} status={record.status} expression={record.expression} "
        f"train_brier={train.get('brier')} validation_brier={validation.get('brier')} "
        f"error={record.error_type} reason={record.rejection_reason}"
    )


def _failure_record(
    *,
    run_id: str,
    data_version: str,
    completion: ChatCompletion | None,
    expression: str,
    features: list[str],
    effects: list[dict[str, str]],
    explanation: str,
    parent_ids: list[str],
    version: int,
    reason: str,
    error_type: str,
    raw_response: str,
) -> HypothesisRecord:
    return HypothesisRecord(
        id=uuid.uuid4().hex,
        run_id=run_id,
        created_at=utc_now(),
        expression=expression,
        features=features,
        effects=effects,
        explanation=explanation,
        version=version,
        status="archived",
        parent_ids=parent_ids,
        data_version=data_version,
        model_name=None if completion is None else completion.model_name,
        model_digest=None if completion is None else completion.model_digest,
        generation={} if completion is None else dict(completion.generation),
        rejection_reason=reason,
        error_type=error_type,
        error_scale=None,
        observations=None,
        sufficient_observations=None,
        train_metrics=None,
        validation_metrics=None,
        raw_response=raw_response[:4000],
    )


def _brier(probabilities: list[float], labels: list[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(labels)


def _calibration(probabilities: list[float], labels: list[int]) -> list[dict[str, object]]:
    bins: list[dict[str, object]] = []
    for index in range(5):
        chosen = [
            (probability, label)
            for probability, label in zip(probabilities, labels, strict=True)
            if _bin_index(probability) == index
        ]
        count = len(chosen)
        bins.append({
            "bin": index,
            "lower": index / 5,
            "upper": (index + 1) / 5,
            "count": count,
            "mean_probability": None if count == 0 else sum(item[0] for item in chosen) / count,
            "empirical_frequency": None if count == 0 else sum(item[1] for item in chosen) / count,
        })
    return bins


def _bin_index(probability: float) -> int:
    if probability >= 1:
        return 4
    if probability < 0:
        return 0
    return min(4, int(probability * 5))
