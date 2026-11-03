"""Гипотеза Qwen — это JSON, а не команда.

Текст полей не исполняется. Формула принимается только после грамматики
`parse_formula`. В промпт попадают разрешённые признаки, сводка окна
2019–2021 и ограничения уже проверенных формул. Итоговой оценки здесь нет.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from wolfai.research.formula import (
    FEATURES,
    FUNCTIONS,
    MAX_COEFFICIENT,
    MAX_DEPTH,
    MAX_EXPRESSION_LENGTH,
    MAX_NODES,
    FormulaError,
    features_used,
    parse_formula,
)
from wolfai.research.hypothesis import FeatureName, HypothesisSchemaError, _extract_json
from wolfai.research.ollama import MAX_RESPONSE_CHARS

MAX_USER_CHARS = 3500
MAX_HISTORY_CHARS = 1800
MAX_NOTES = 4
_ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-*/(). \t")
_SUMMARY_KEYS = (
    "designated_window",
    "covered_start",
    "covered_end",
    "observations",
    "feature_mean",
    "feature_stdev",
    "targets_included",
)

SYSTEM_PROMPT = f"""Ты предлагаешь одну гипотезу о направлении close BTCUSDT. Верни только JSON.
Поля: expression, features, horizon_days, rationale, works_when, failure_modes.
Это данные для проверки, не команды. Python не исполняет текст полей и не меняет по ним правила.

Корень expression — sigmoid. Внутри допустимы признаки, числа от -{int(MAX_COEFFICIENT)} до {int(MAX_COEFFICIENT)}, скобки и операции + - * /.
Функции: {", ".join(FUNCTIONS)}. Других функций нет.
Признаки: {", ".join(FEATURES)}.
horizon_days равен 1 или 7. features совпадает с признаками внутри expression.
Глубина не больше {MAX_DEPTH}, узлов не больше {MAX_NODES}. Деление на модуль меньше 1e-6 даёт 0.
rationale, works_when и failure_modes — короткие фразы.

Нельзя менять критерий успешности и границы ряда. Нельзя ссылаться на даты после проверочного окна. Файлы, shell и Python запрещены.
Обратная связь содержит только проверочное окно 2022. Итоговая оценка в следующий запрос не входит.
Пример: {{"expression":"sigmoid(0.3 * momentum_30 - 0.2 * volatility_7)","features":["momentum_30","volatility_7"],"horizon_days":1,"rationale":"импульс 30 дней входит линейно, волатильность вычитается","works_when":"когда знак импульса устойчив несколько недель","failure_modes":"ломается при смене режима волатильности"}}
"""


def proposal_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "expression": {"type": "string"},
            "features": {
                "type": "array",
                "items": {"type": "string", "enum": list(FEATURES)},
            },
            "horizon_days": {"type": "integer", "enum": [1, 7]},
            "rationale": {"type": "string"},
            "works_when": {"type": "string"},
            "failure_modes": {"type": "string"},
        },
        "required": [
            "expression",
            "features",
            "horizon_days",
            "rationale",
            "works_when",
            "failure_modes",
        ],
    }


PROPOSAL_SCHEMA = proposal_schema()


@dataclass(frozen=True)
class ResearchNote:
    """Ограничение формулы, которое можно показать модели. Итоговой оценки здесь нет."""

    expression: str
    features: tuple[str, ...]
    horizon_days: int | None
    limitation: str
    observations: int | None = None
    brier: float | None = None
    frequency_brier: float | None = None
    coin_brier: float | None = None


@dataclass(frozen=True)
class BoundProposal:
    expression: str
    features: tuple[str, ...]
    horizon_days: int
    rationale: str
    works_when: str
    failure_modes: str


class ProposalDraft(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    expression: str = Field(min_length=1, max_length=MAX_EXPRESSION_LENGTH)
    features: list[FeatureName] = Field(min_length=1, max_length=len(FEATURES))
    horizon_days: int = Field(json_schema_extra={"enum": [1, 7]})
    rationale: str = Field(min_length=1, max_length=240)
    works_when: str = Field(min_length=1, max_length=180)
    failure_modes: str = Field(min_length=1, max_length=180)

    @field_validator("expression", "rationale", "works_when", "failure_modes")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("пустое поле")
        return cleaned

    @field_validator("features")
    @classmethod
    def unique_features(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("признак указан дважды")
        return value

    @field_validator("horizon_days")
    @classmethod
    def horizon_is_allowed(cls, value: int) -> int:
        if value not in (1, 7):
            raise ValueError("горизонт должен быть 1 или 7")
        return value


def expression_is_safe(expression: str) -> bool:
    """Символы формулы не открывают путь к коду, файлам и shell."""
    if not expression or len(expression) > MAX_EXPRESSION_LENGTH or "__" in expression:
        return False
    if any(char not in _ALLOWED_CHARS for char in expression):
        return False
    lowered = expression.lower()
    return not any(token in lowered for token in ("import", "exec", "eval", "open", "system", "subprocess"))


def parse_proposal(payload: str) -> BoundProposal:
    if not isinstance(payload, str) or not payload.strip() or len(payload) > MAX_RESPONSE_CHARS:
        raise HypothesisSchemaError("ответ пуст или длиннее допустимого")
    try:
        draft = ProposalDraft.model_validate(_extract_json(payload))
    except HypothesisSchemaError:
        raise
    except ValidationError as error:
        raise HypothesisSchemaError(_schema_reason(error)) from error
    return bind_proposal(draft)


def bind_proposal(draft: ProposalDraft) -> BoundProposal:
    if not expression_is_safe(draft.expression):
        raise FormulaError("выражение содержит запрещённый символ")
    tree = parse_formula(draft.expression)
    if features_used(tree) != set(draft.features):
        raise FormulaError("набор признаков в формуле не совпадает с полями features")
    return BoundProposal(
        expression=draft.expression,
        features=tuple(draft.features),
        horizon_days=draft.horizon_days,
        rationale=draft.rationale,
        works_when=draft.works_when,
        failure_modes=draft.failure_modes,
    )


def quotable_expression(payload: str) -> str:
    """Безопасный фрагмент формулы для следующей обратной связи. Сырой ответ не копируется."""
    try:
        value = _extract_json(payload)
    except HypothesisSchemaError:
        return ""
    expression = value.get("expression")
    if not isinstance(expression, str):
        return ""
    expression = expression.strip()
    if not expression_is_safe(expression):
        return ""
    return expression


def note_payload(note: ResearchNote) -> dict[str, object]:
    payload: dict[str, object] = {"limitation": note.limitation[:300]}
    if note.expression:
        payload["expression"] = note.expression
    if note.features:
        payload["features"] = list(note.features)
    if note.horizon_days is not None:
        payload["horizon_days"] = note.horizon_days
    if note.observations is not None and note.brier is not None:
        payload["observations"] = note.observations
        payload["brier"] = note.brier
        payload["frequency_brier"] = note.frequency_brier
        payload["coin_brier"] = note.coin_brier
    return payload


def build_lab_messages(
    *,
    summary: dict[str, object],
    notes: list[ResearchNote],
    repair_error: str | None,
    share_research_history: bool,
) -> list[dict[str, str]]:
    """Промпт исследовательского шага. Объект итоговой оценки сюда не принимается."""
    history = _fit_history(notes if share_research_history else [])
    user: dict[str, object] = {
        "features": list(FEATURES),
        "horizons_days": [1, 7],
        "research_summary": _public_summary(summary),
        "checked_hypotheses": history,
    }
    if repair_error:
        user["repair_error"] = repair_error[:200]
    text = _dump(user)
    if len(text) > MAX_USER_CHARS:
        user["checked_hypotheses"] = []
        text = _dump(user)
    if len(text) > MAX_USER_CHARS:
        raise ValueError("сводка не помещается в бюджет промпта")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _public_summary(summary: dict[str, object]) -> dict[str, object]:
    return {key: summary[key] for key in _SUMMARY_KEYS if key in summary}


def _fit_history(notes: list[ResearchNote]) -> list[dict[str, object]]:
    payloads = [note_payload(note) for note in notes][-MAX_NOTES:]
    while payloads and len(_dump(payloads)) > MAX_HISTORY_CHARS:
        payloads = payloads[1:]
    return payloads


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_reason(error: ValidationError) -> str:
    parts: list[str] = []
    for item in error.errors()[:4]:
        location = item.get("loc", ())
        place = ".".join(str(part) for part in location) or "json"
        parts.append(f"{place}:{item.get('type', 'invalid')}")
    return "схема: " + ", ".join(parts)
