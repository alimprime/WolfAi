"""Qwen задаёт структуру, Python обучает коэффициенты на 2019–2021.

Обратная связь модели — только внутренний 2022 год.
Сравнение на 2023–2024 выполняется один раз после фиксации всех кандидатов.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from wolfai.data.candles import Candle
from wolfai.forecasting.baseline import HistoricalQuantileBaseline
from wolfai.forecasting.evaluate import evaluate_split
from wolfai.forecasting.random_walk import RandomWalkBaseline
from wolfai.forecasting.splits import (
    COEFFICIENT_FIT_BOUNDS,
    INNER_CHECK_BOUNDS,
    Period,
    period_bounds,
)
from wolfai.research.diagnose import prompt_diagnosis
from wolfai.research.experiment import ResearchClient, series_version
from wolfai.research.features import (
    FEATURE_NAMES,
    FeatureRow,
    StandardizedRow,
    assert_research_series,
    build_rows,
    rows_between,
    standardize_rows,
)
from wolfai.research.formula import EvalState, evaluate, parse_formula
from wolfai.research.hypothesis import AdaptiveDraft, HypothesisSchemaError, parse_adaptive
from wolfai.research.journal import HypothesisRecord, Journal, utc_now
from wolfai.research.logistic import (
    Term,
    fit_l2_logistic,
    predict_probability,
    score_probabilities,
    transform_value,
)
from wolfai.research.ollama import GENERATION, ChatCompletion, OllamaUnavailable

MAX_HYPOTHESES = 5
SCHEMA_REPAIRS = 1
ADAPTIVE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "parent_id": {"type": "string"},
        "independent_reason": {"type": "string"},
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "feature": {"type": "string"},
                    "transform": {"type": "string", "enum": ["identity", "abs", "tanh", "sign"]},
                },
                "required": ["feature", "transform"],
            },
        },
        "change": {"type": "string"},
        "problem": {"type": "string"},
        "expected_improvement": {"type": "string"},
        "falsification": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reference": {
                    "type": "string",
                    "enum": ["constant_0.5", "historical_frequency", "logistic_all_features"],
                },
                "split": {"type": "string", "enum": ["inner_2022"]},
            },
            "required": ["reference", "split"],
        },
        "explanation": {"type": "string"},
    },
    "required": [
        "terms",
        "change",
        "problem",
        "expected_improvement",
        "falsification",
        "explanation",
    ],
}
ALL_TERMS = [Term(name, "identity") for name in FEATURE_NAMES]
SYSTEM_PROMPT = """Ты исследователь WolfAi. Предложи структуру логистической формулы вероятности роста следующего close.
Числовые коэффициенты и свободный член оценит Python на 2019–2021 после стандартизации признаков по прошлой истории.
Верни только JSON.
Преобразования членов: identity, abs, tanh, sign.
Признаки: log_return, momentum_7, momentum_30, volatility_7, volatility_30, volume_change, range_position.
Укажи parent_id существующей гипотезы либо independent_reason. Одно из двух.
Опиши change, problem, expected_improvement.
falsification.split обязан быть inner_2022. reference: constant_0.5, historical_frequency или logistic_all_features.
status не указывай. Не проси и не оценивай 2023–2024 и даты с 2025 года.
Пример: {"parent_id":"REPLACE","terms":[{"feature":"momentum_30","transform":"identity"},{"feature":"volatility_30","transform":"abs"}],"change":"убрал сырые коэффициенты","problem":"сырой momentum слишком мал и sigmoid стоит у 0.5","expected_improvement":"Brier на 2022 ниже константы 0.5","falsification":{"reference":"constant_0.5","split":"inner_2022"},"explanation":"длинный импульс и модуль волатильности"}
"""


@dataclass(frozen=True)
class AdaptiveRunResult:
    run_id: str
    stop_reason: str
    message: str
    hypothesis_ids: tuple[str, ...]
    data_version: str


def run_adaptive(
    client: ResearchClient,
    candles: list[Candle],
    journal: Journal,
    *,
    max_hypotheses: int = MAX_HYPOTHESES,
) -> AdaptiveRunResult:
    """До пяти структур. Метрики 2023–2024 считаются после фиксации всех кандидатов."""
    if max_hypotheses < 1 or max_hypotheses > MAX_HYPOTHESES:
        raise ValueError("за один запуск допускается от 1 до 5 гипотез")
    assert_research_series(candles)
    data_version = series_version(candles)
    run_id = uuid.uuid4().hex
    prior_ids = journal.known_ids()
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
        return AdaptiveRunResult(run_id, "ollama_unavailable", str(error), (), data_version)
    digest = str(getattr(info, "digest", "") or client.model_digest)
    raw_rows = build_rows(candles)
    standardized = standardize_rows(raw_rows)
    fit_rows = rows_between(standardized, *COEFFICIENT_FIT_BOUNDS)
    inner_rows = rows_between(standardized, *INNER_CHECK_BOUNDS)
    if len(fit_rows) < 100 or len(inner_rows) < 50:
        journal.finish_run(run_id, model_digest=digest, baseline_metrics=None, stop_reason="insufficient_history")
        return AdaptiveRunResult(run_id, "insufficient_history", "мало точек во внутреннем разбиении", (), data_version)
    logistic_beta = fit_terms(fit_rows, ALL_TERMS)
    created: list[str] = []
    feedback: list[str] = []
    stop_reason = "completed"
    message = "кандидаты зафиксированы"
    for _ in range(max_hypotheses):
        messages = build_adaptive_messages(
            diagnosis=prompt_diagnosis(raw_rows),
            history=_history_for_prompt(journal),
            feedback=feedback[-1] if feedback else None,
            known_ids=sorted(journal.known_ids()),
        )
        try:
            draft, completion, raw, schema_error = _request_adaptive(client, messages, journal.known_ids())
        except OllamaUnavailable as error:
            stop_reason = "ollama_unavailable"
            message = str(error)
            break
        if draft is None or completion is None:
            record = _archived_attempt(
                run_id=run_id,
                data_version=data_version,
                completion=completion,
                expression="",
                reason=schema_error or "JSON не прошёл схему",
                error_type="schema",
                raw=raw,
            )
            journal.add(record)
            created.append(record.id)
            feedback.append(record.rejection_reason or "схема")
            continue
        parent_ids = [draft.parent_id] if draft.parent_id else []
        version = 1
        if draft.parent_id:
            parent = journal.get(draft.parent_id)
            version = 1 + (parent.version if parent else 0)
        terms = [Term(term.feature, term.transform) for term in draft.terms]
        try:
            beta = fit_terms(fit_rows, terms)
            inner_metrics = _score_model(beta, terms, inner_rows, logistic_beta)
        except Exception as error:
            record = _archived_attempt(
                run_id=run_id,
                data_version=data_version,
                completion=completion,
                expression=_expression(terms, None),
                reason=str(error),
                error_type="fit",
                raw=raw,
                draft=draft,
                parent_ids=parent_ids,
                version=version,
            )
            journal.add(record)
            created.append(record.id)
            feedback.append(str(error))
            continue
        reference_brier = _reference_brier(inner_metrics, draft.falsification.reference)
        status = "experimental"
        reason = None
        error_type = None
        if reference_brier is not None and float(inner_metrics["brier"]) > reference_brier:
            status = "archived"
            reason = (
                "опровергнута на inner_2022: Brier хуже ориентира "
                f"{draft.falsification.reference}"
            )
            error_type = "falsified_inner_2022"
        record = HypothesisRecord(
            id=uuid.uuid4().hex,
            run_id=run_id,
            created_at=utc_now(),
            expression=_expression(terms, beta),
            features=sorted({term.feature for term in terms}),
            effects=[],
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
            error_scale=None if reference_brier is None else float(inner_metrics["brier"]) - reference_brier,
            observations=int(inner_metrics["observations"]),
            sufficient_observations=int(inner_metrics["observations"]) >= 100,
            train_metrics=_score_model(beta, terms, fit_rows, logistic_beta),
            validation_metrics=None,
            raw_response=raw[:4000],
            kind="optimized",
            coefficients=_coefficients(terms, beta),
            inner_metrics=inner_metrics,
            change_summary=draft.change,
            problem_addressed=draft.problem,
            expected_improvement=draft.expected_improvement,
            falsification=draft.falsification.model_dump(),
            independent_reason=draft.independent_reason,
        )
        journal.add(record)
        created.append(record.id)
        feedback.append(_inner_feedback(record))
    dev_rows = rows_between(standardized, *period_bounds(Period.VALIDATION))
    if dev_rows:
        for hypothesis_id in created:
            stored = journal.get(hypothesis_id)
            if stored is None or not stored.coefficients:
                continue
            beta, terms = _beta_from_record(stored)
            if beta is None:
                continue
            journal.record_development_validation(
                hypothesis_id,
                _score_model(beta, terms, dev_rows, logistic_beta),
            )
    comparison = _comparison_bundle(
        candles,
        raw_rows,
        standardized,
        fit_rows,
        inner_rows,
        dev_rows,
        logistic_beta,
        prior_ids,
        journal,
    )
    journal.finish_run(run_id, model_digest=digest, baseline_metrics=comparison, stop_reason=stop_reason)
    return AdaptiveRunResult(run_id, stop_reason, message, tuple(created), data_version)


def build_adaptive_messages(
    *,
    diagnosis: str,
    history: list[dict[str, object]],
    feedback: str | None,
    known_ids: list[str],
) -> list[dict[str, str]]:
    """Промпт без метрик 2023–2024 и без цен."""
    lines = [
        diagnosis,
        "Доступные id родителей: " + (", ".join(known_ids) if known_ids else "нет"),
        "Поиск идёт по 2019–2021 и проверке 2022. Более поздние метрики в это сообщение не входят.",
    ]
    if history:
        lines.append("Память журнала:")
        for item in history:
            lines.append(
                f"- id={item['id']} status={item['status']} expression={item['expression']} "
                f"inner_brier={item.get('inner_brier')} reason={item.get('rejection_reason')}"
            )
    else:
        lines.append("Новых адаптированных гипотез ещё нет.")
    if feedback:
        lines.append(f"Проверка Python на 2022: {feedback}")
    lines.append("Предложи одну новую структуру.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def fit_terms(rows: list[StandardizedRow], terms: list[Term]) -> list[float]:
    design = [_design(row, terms) for row in rows]
    return fit_l2_logistic(design, [row.y_up for row in rows])


def _score_model(
    beta: list[float],
    terms: list[Term],
    rows: list[StandardizedRow],
    logistic_beta: list[float],
) -> dict[str, object]:
    probabilities = [predict_probability(beta, _design(row, terms)) for row in rows]
    logistic_probabilities = [predict_probability(logistic_beta, _design(row, ALL_TERMS)) for row in rows]
    return score_probabilities(
        probabilities,
        [row.y_up for row in rows],
        {
            "constant_0.5": [0.5] * len(rows),
            "historical_frequency": [row.baseline_up_frequency for row in rows],
            "logistic_all_features": logistic_probabilities,
        },
    )


def _request_adaptive(
    client: ResearchClient,
    messages: list[dict[str, str]],
    known_ids: set[str],
) -> tuple[AdaptiveDraft | None, ChatCompletion | None, str, str | None]:
    completion: ChatCompletion | None = None
    raw = ""
    error_text: str | None = None
    for _ in range(SCHEMA_REPAIRS + 1):
        completion = client.complete(messages, ADAPTIVE_SCHEMA)
        raw = completion.content
        try:
            draft = parse_adaptive(raw)
        except HypothesisSchemaError as error:
            error_text = str(error)
        else:
            if draft.parent_id and draft.parent_id not in known_ids:
                error_text = "parent_id нет в журнале"
            else:
                return draft, completion, raw, None
        messages = [
            *messages,
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": f"JSON отклонён: {error_text}. Верни один объект по примеру."},
        ]
    return None, completion, raw, error_text


def _history_for_prompt(journal: Journal) -> list[dict[str, object]]:
    items = []
    for record in journal.list_hypotheses()[-8:]:
        inner = record.inner_metrics or {}
        items.append({
            "id": record.id,
            "status": record.status,
            "expression": record.expression,
            "inner_brier": inner.get("brier"),
            "rejection_reason": record.rejection_reason,
        })
    return items


def _inner_feedback(record: HypothesisRecord) -> str:
    inner = record.inner_metrics or {}
    coefficients = record.coefficients or {}
    return (
        f"id={record.id} status={record.status} expression={record.expression} "
        f"coefficients={coefficients} inner_brier={inner.get('brier')} "
        f"inner_log_loss={inner.get('log_loss')} reason={record.rejection_reason}"
    )


def _comparison_bundle(
    candles: list[Candle],
    raw_rows: list[FeatureRow],
    standardized: list[StandardizedRow],
    fit_rows: list[StandardizedRow],
    inner_rows: list[StandardizedRow],
    dev_rows: list[StandardizedRow],
    logistic_beta: list[float],
    prior_ids: set[str],
    journal: Journal,
) -> dict[str, object]:
    windows = {
        "fit_2019_2021": rows_between(standardized, *COEFFICIENT_FIT_BOUNDS),
        "inner_2022": inner_rows,
        "development_2023_2024": dev_rows,
    }
    logistic_scores = {
        name: _score_model(logistic_beta, ALL_TERMS, rows, logistic_beta)
        for name, rows in windows.items()
        if rows
    }
    fixed_scores = []
    for record in journal.list_hypotheses():
        if record.id not in prior_ids or not record.expression.startswith("sigmoid"):
            continue
        try:
            tree = parse_formula(record.expression)
        except ValueError:
            continue
        fixed_scores.append({
            "id": record.id,
            "expression": record.expression,
            "windows": {
                name: _score_fixed(tree, _raw_between(raw_rows, rows))
                for name, rows in (
                    ("fit_2019_2021", fit_rows),
                    ("inner_2022", inner_rows),
                    ("development_2023_2024", dev_rows),
                )
                if rows
            },
        })
    price = {}
    for period in (Period.TRAINING, Period.VALIDATION):
        try:
            price[period.value] = {
                "historical_quantile": _price(evaluate_split(candles, 1, period, HistoricalQuantileBaseline())),
                "random_walk": _price(evaluate_split(candles, 1, period, RandomWalkBaseline())),
            }
        except ValueError:
            continue
    return {
        "note": (
            "Brier — метрика вероятности. MAE ценового baseline хранится отдельно. "
            "2023–2024 посчитан после фиксации кандидатов и в промпт не возвращался. "
            "Лучший Brier среди кандидатов не делает гипотезу успешной."
        ),
        "logistic_all_features": logistic_scores,
        "fixed_controls": fixed_scores,
        "price_baseline": price,
        "logistic_coefficients": _coefficients(ALL_TERMS, logistic_beta),
    }


def _score_fixed(tree, rows: list[FeatureRow]) -> dict[str, object]:
    if not rows:
        return {"observations": 0}
    probabilities = [evaluate(tree, row.features, EvalState()) for row in rows]
    return score_probabilities(
        probabilities,
        [row.y_up for row in rows],
        {
            "constant_0.5": [0.5] * len(rows),
            "historical_frequency": [row.baseline_up_frequency for row in rows],
        },
    )


def _raw_between(raw_rows: list[FeatureRow], standardized: list[StandardizedRow]) -> list[FeatureRow]:
    dates = {row.decision_date for row in standardized}
    return [row for row in raw_rows if row.decision_date in dates]


def _price(result) -> dict[str, float | int | None]:
    return {
        "observations": result.observations,
        "mean_absolute_error": result.mean_absolute_error,
        "interval_coverage": result.interval_coverage,
    }


def _design(row: StandardizedRow, terms: list[Term]) -> list[float]:
    return [transform_value(term.transform, row.standardized[term.feature]) for term in terms]


def _expression(terms: list[Term], beta: list[float] | None) -> str:
    if beta is None:
        parts = [f"{term.transform}(z_{term.feature})" for term in terms]
        return "sigmoid(intercept + " + " + ".join(parts) + ")"
    parts = [f"{beta[0]:.6f}"]
    for weight, term in zip(beta[1:], terms, strict=True):
        parts.append(f"{weight:.6f} * {term.transform}(z_{term.feature})")
    return "sigmoid(" + " + ".join(parts) + ")"


def _coefficients(terms: list[Term], beta: list[float]) -> dict[str, object]:
    return {
        "intercept": beta[0],
        "l2": 0.1,
        "weights": [
            {"feature": term.feature, "transform": term.transform, "weight": weight}
            for term, weight in zip(terms, beta[1:], strict=True)
        ],
    }


def _beta_from_record(record: HypothesisRecord) -> tuple[list[float] | None, list[Term]]:
    payload = record.coefficients or {}
    weights = payload.get("weights")
    intercept = payload.get("intercept")
    if not isinstance(weights, list) or not isinstance(intercept, (int, float)):
        return None, []
    terms = []
    beta = [float(intercept)]
    for item in weights:
        if not isinstance(item, dict):
            return None, []
        terms.append(Term(str(item["feature"]), str(item["transform"])))
        beta.append(float(item["weight"]))
    return beta, terms


def _reference_brier(metrics: dict[str, object], reference: str) -> float | None:
    skill = metrics.get("skill")
    if not isinstance(skill, dict):
        return None
    block = skill.get(reference)
    if not isinstance(block, dict) or not isinstance(block.get("reference_brier"), (int, float)):
        return None
    return float(block["reference_brier"])


def _archived_attempt(
    *,
    run_id: str,
    data_version: str,
    completion: ChatCompletion | None,
    expression: str,
    reason: str,
    error_type: str,
    raw: str,
    draft: AdaptiveDraft | None = None,
    parent_ids: list[str] | None = None,
    version: int = 1,
) -> HypothesisRecord:
    return HypothesisRecord(
        id=uuid.uuid4().hex,
        run_id=run_id,
        created_at=utc_now(),
        expression=expression,
        features=[],
        effects=[],
        explanation="" if draft is None else draft.explanation,
        version=version,
        status="archived",
        parent_ids=parent_ids or [],
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
        raw_response=raw[:4000],
        kind="optimized",
        change_summary=None if draft is None else draft.change,
        problem_addressed=None if draft is None else draft.problem,
        expected_improvement=None if draft is None else draft.expected_improvement,
        falsification=None if draft is None else draft.falsification.model_dump(),
        independent_reason=None if draft is None else draft.independent_reason,
    )
