"""Один запуск Qwen 3.5 4B. Веса не меняются, инструментов нет.

Модель видит бриф на дату не позже 2022-12-31. Python после ответа один раз
считает 2023–2024 и помечает, что эти числа модели не показывались.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wolfai.data.candles import Candle
from wolfai.forecasting.splits import COEFFICIENT_FIT_BOUNDS, INNER_CHECK_BOUNDS, Period, period_bounds
from wolfai.intelligence.brief import build_qwen_brief
from wolfai.intelligence.frame import MarketRow, ScaledRow, between, build_market_rows, rows_with_features, standardize
from wolfai.intelligence.schema import Observation
from wolfai.research.logistic import (
    LogisticError,
    fit_l2_logistic,
    predict_probability,
    score_probabilities,
    transform_value,
)
from wolfai.research.ollama import OllamaUnavailable

BRIEF_LAST_DECISION = date(2022, 12, 31)
TRANSFORMS = frozenset({"identity", "abs", "tanh", "sign"})
SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "feature": {"type": "string"},
                    "transform": {"type": "string", "enum": ["identity", "abs", "tanh", "sign"]},
                    "interact_with": {"type": "string"},
                },
                "required": ["feature", "transform"],
            },
        },
        "analogs_used": {"type": "array", "items": {"type": "string"}},
        "change": {"type": "string"},
        "problem": {"type": "string"},
        "expected_improvement": {"type": "string"},
        "explanation": {"type": "string"},
        "falsification": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reference": {
                    "type": "string",
                    "enum": ["constant_0.5", "historical_frequency", "price_logistic"],
                },
                "split": {"type": "string", "enum": ["inner_2022"]},
            },
            "required": ["reference", "split"],
        },
    },
    "required": [
        "terms",
        "analogs_used",
        "change",
        "problem",
        "expected_improvement",
        "explanation",
        "falsification",
    ],
}
SYSTEM_PROMPT = """Ты исследователь WolfAi. Предложи одну структуру логистической вероятности роста следующего close BTCUSDT.
Коэффициенты оценит Python на 2019–2021. Верни только JSON.
Интернета, терминала, секретов и торгового счёта нет. Веса модели не меняются.
Признаки и преобразования бери только из брифа: allowed_features и identity, abs, tanh, sign.
interact_with необязателен и должен быть другим признаком из allowed_features.
analogs_used может содержать только decision_date из analogs.neighbors.
Если analogs.sufficient = false, analogs_used обязан быть пустым: не выдумывай исторические совпадения.
falsification.split обязан быть inner_2022.
Не проси данные 2023–2024 и не упоминай holdout. status не указывай.
Пример: {"terms":[{"feature":"momentum_30","transform":"identity"}],"analogs_used":[],"change":"один длинный импульс","problem":"сырой ряд шумный","expected_improvement":"Brier на 2022 ниже константы 0.5","explanation":"импульс 30 дней","falsification":{"reference":"constant_0.5","split":"inner_2022"}}
"""


class _Term(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature: str
    transform: str
    interact_with: str | None = None


class _Falsification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reference: str
    split: str


class _Draft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    terms: list[_Term] = Field(min_length=1, max_length=6)
    analogs_used: list[str] = Field(default_factory=list)
    change: str = Field(min_length=1, max_length=300)
    problem: str = Field(min_length=1, max_length=300)
    expected_improvement: str = Field(min_length=1, max_length=300)
    explanation: str = Field(min_length=1, max_length=400)
    falsification: _Falsification


@dataclass(frozen=True)
class _Column:
    feature: str
    transform: str
    interact_with: str | None

    @property
    def name(self) -> str:
        if self.interact_with:
            return f"{self.feature}*{self.interact_with}"
        return self.feature


class Phase3Client(Protocol):
    def health(self) -> object:
        """Проверка локальной модели."""

    def complete(self, messages: list[dict[str, str]], schema: dict[str, object] | None = None) -> object:
        """Один ответ без инструментов."""


def run_phase3(
    client: Phase3Client,
    candles: list[Candle],
    observations: list[Observation],
) -> dict[str, object]:
    """Один исследовательский ответ. При недоступной модели гипотеза не создаётся."""
    try:
        info = client.health()
    except OllamaUnavailable as error:
        return {"status": "ollama_unavailable", "message": str(error), "hypothesis": None}
    cut = [candle for candle in candles if candle.timestamp.date() <= BRIEF_LAST_DECISION]
    if len(cut) < 100:
        return {"status": "insufficient_history", "hypothesis": None}
    brief = build_qwen_brief(build_market_rows(cut, observations), observations)
    messages = build_messages(brief)
    try:
        completion = client.complete(messages, SCHEMA)
    except OllamaUnavailable as error:
        return {"status": "ollama_unavailable", "message": str(error), "hypothesis": None}
    raw = _content(completion)
    draft, error = _parse(raw, brief)
    if draft is None:
        repair = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"JSON отклонён: {error}. Верни только исправленный JSON."},
        ]
        try:
            completion = client.complete(repair, SCHEMA)
        except OllamaUnavailable as unavailable:
            return {"status": "ollama_unavailable", "message": str(unavailable), "hypothesis": None}
        raw = _content(completion)
        draft, error = _parse(raw, brief)
    if draft is None:
        return {
            "status": "schema_rejected",
            "message": error,
            "hypothesis": None,
            "raw_response": raw,
            "model_digest": str(getattr(info, "digest", "") or ""),
        }
    scored = _score(draft, build_market_rows(candles, observations))
    return {
        "status": "checked",
        "model_name": str(getattr(info, "name", "") or ""),
        "model_digest": str(getattr(info, "digest", "") or ""),
        "brief_decision_date": brief["decision_date"],
        "hypothesis": {
            "expression": _expression(draft),
            "terms": [
                {
                    "feature": column.feature,
                    "transform": column.transform,
                    "interact_with": column.interact_with,
                }
                for column in _columns(draft)
            ],
            "analogs_used": list(draft.analogs_used),
            "explanation": draft.explanation,
            "change": draft.change,
            "problem": draft.problem,
            "expected_improvement": draft.expected_improvement,
            "falsification": draft.falsification.model_dump(),
        },
        "metrics": scored,
        "shown_to_model": {"development_validation_2023_2024": False, "holdout": False},
    }


def build_messages(brief: dict[str, object]) -> list[dict[str, str]]:
    payload = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
    if "brier" in payload.lower():
        raise RuntimeError("в сообщение модели попал Brier")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload},
    ]


def _parse(raw: str, brief: dict[str, object]) -> tuple[_Draft | None, str | None]:
    try:
        payload = json.loads(raw)
        draft = _Draft.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as error:
        return None, str(error)
    allowed = set(brief["allowed_features"]) if isinstance(brief.get("allowed_features"), list) else set()
    analogs = brief.get("analogs")
    neighbor_dates: set[str] = set()
    sufficient = False
    if isinstance(analogs, dict):
        sufficient = bool(analogs.get("sufficient"))
        neighbors = analogs.get("neighbors")
        if isinstance(neighbors, list):
            neighbor_dates = {
                str(item["decision_date"])
                for item in neighbors
                if isinstance(item, dict) and "decision_date" in item
            }
    if not sufficient and draft.analogs_used:
        return None, "аналогов недостаточно, список analogs_used должен быть пустым"
    unknown_analogs = [item for item in draft.analogs_used if item not in neighbor_dates]
    if unknown_analogs:
        return None, "модель назвала аналог, которого нет в расчёте Python"
    if draft.falsification.split != "inner_2022":
        return None, "falsification.split должен быть inner_2022"
    if draft.falsification.reference not in {"constant_0.5", "historical_frequency", "price_logistic"}:
        return None, "неизвестный ориентир falsification"
    seen: set[tuple[str, str, str | None]] = set()
    for term in draft.terms:
        if term.feature not in allowed or term.transform not in TRANSFORMS:
            return None, "признак или преобразование вне брифа"
        if term.interact_with is not None and (
            term.interact_with not in allowed or term.interact_with == term.feature
        ):
            return None, "interact_with вне брифа"
        key = (term.feature, term.transform, term.interact_with)
        if key in seen:
            return None, "член формулы повторён"
        seen.add(key)
    return draft, None


def _score(draft: _Draft, rows: list[MarketRow]) -> dict[str, object]:
    columns = _columns(draft)
    base_names = tuple(dict.fromkeys(
        [column.feature for column in columns]
        + [column.interact_with for column in columns if column.interact_with]
    ))
    usable = rows_with_features(rows, base_names)
    scaled = standardize(usable, base_names)
    fit_rows = between(scaled, *COEFFICIENT_FIT_BOUNDS)
    if len(fit_rows) < 100:
        return {"status": "insufficient_history", "fit_observations": len(fit_rows)}
    try:
        beta = fit_l2_logistic(
            [_design(row, columns) for row in fit_rows],
            [row.y_up for row in fit_rows],
        )
    except LogisticError as error:
        return {"status": "fit_failed", "reason": str(error)}
    periods: dict[str, object] = {}
    windows = (
        ("inner_2022", INNER_CHECK_BOUNDS),
        ("development_validation_2023_2024", period_bounds(Period.VALIDATION)),
    )
    for label, bounds in windows:
        chosen = between(scaled, *bounds)
        if len(chosen) < 20:
            periods[label] = {
                "status": "insufficient_period",
                "observations": len(chosen),
                "shown_to_model": False,
            }
            continue
        probabilities = [predict_probability(beta, _design(row, columns)) for row in chosen]
        metrics = score_probabilities(
            probabilities,
            [row.y_up for row in chosen],
            {
                "constant_0.5": [0.5] * len(chosen),
                "historical_frequency": [row.baseline_up_frequency for row in chosen],
            },
        )
        metrics["status"] = "scored"
        metrics["shown_to_model"] = False
        periods[label] = metrics
    return {
        "status": "scored",
        "fit_observations": len(fit_rows),
        "coefficients": {
            "intercept": beta[0],
            "weights": [
                {"feature": column.name, "weight": weight}
                for column, weight in zip(columns, beta[1:], strict=True)
            ],
        },
        "periods": periods,
    }


def _columns(draft: _Draft) -> list[_Column]:
    return [
        _Column(term.feature, term.transform, term.interact_with)
        for term in draft.terms
    ]


def _design(row: ScaledRow, columns: list[_Column]) -> list[float]:
    values: list[float] = []
    for column in columns:
        left = transform_value(column.transform, row.standardized[column.feature])
        if column.interact_with:
            right = row.standardized[column.interact_with]
            values.append(left * right)
        else:
            values.append(left)
    return values


def _expression(draft: _Draft) -> str:
    parts = []
    for column in _columns(draft):
        if column.interact_with:
            parts.append(f"{column.transform}(z_{column.feature})*z_{column.interact_with}")
        else:
            parts.append(f"{column.transform}(z_{column.feature})")
    return "sigmoid(intercept + " + " + ".join(parts) + ")"


def _content(completion: object) -> str:
    content = getattr(completion, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise OllamaUnavailable("в ответе модели нет текста")
    return content
