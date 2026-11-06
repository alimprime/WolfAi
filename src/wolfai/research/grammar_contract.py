"""Контракт грамматики и бюджет этапа 3.2.

Числа взяты из безопасного разборщика формул. Схема описывает ответ.
Произвольный Python по этой схеме не исполняется.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from wolfai.research.formula import (
    DIVISION_EPSILON,
    FEATURES,
    FUNCTIONS,
    MAX_COEFFICIENT,
    MAX_DEPTH,
    MAX_EXPRESSION_LENGTH,
    MAX_NODES,
)
from wolfai.research.hypothesis_protocol import HORIZONS, REGIMES, proposal_schema

INTELLIGENCE_VERSION = "3.3.0"
GENERATOR_VERSION = "3.3.0"
FIXED_EVALUATION_METHOD = "fixed_internal_brier_log_loss"
STRATEGY_ID = "qwen-research-strategy"

# Потолок зафиксирован до просмотра метрик.
# Отдельный вызов плана не входит в пять обращений на генерацию и ремонт.
# Общий потолок вместе с планом равен шести.
MAX_PLAN_CALLS = 1
MAX_GENERATION_REPAIR_CALLS = 5
MAX_MODEL_CALLS = MAX_PLAN_CALLS + MAX_GENERATION_REPAIR_CALLS
MAX_SYNTAX_REPAIRS = 2
MAX_SCIENTIFIC_REVISIONS = 2
MAX_EVALUATED_FORMULAS = 3
MAX_INITIAL_HYPOTHESES = 1
OVERFIT_NODE_LIMIT = 12
GENERATION_SEED = 1729
GENERATION_TEMPERATURE = 0.0
MAX_USER_CHARS = 3500

PLAN_FIELDS = (
    "research_question",
    "target_horizon",
    "candidate_features",
    "proposed_relationship",
    "expected_market_conditions",
    "falsification_conditions",
    "evaluation_method",
    "limitations",
)
HYPOTHESIS_FIELDS = (
    "hypothesis_id",
    "description",
    "features",
    "horizon_days",
    "expression",
    "rationale",
    "expected_regime",
    "test_plan",
    "limitations",
    "parent_hypothesis_id",
)
REVISION_FIELDS = (
    "what_failed",
    "assumption_to_revise",
    "why_new_relationship",
    "difference_from_parent",
    "falsification_conditions",
)
SYNTAX_CODES = frozenset({
    "invalid_json",
    "unsafe_math",
    "too_complex",
    "unknown_feature",
    "feature_mismatch",
    "missing_feature",
})
_FUTURE_YEARS = ("2022", "2023", "2024", "2025", "2026", "2027", "2028")
_FUTURE_MARKERS = ("holdout", "y_up", "next_close", "future_return", "lead(", "shift(-")
_RULE_PHRASES = (
    "validated law",
    "validated_law",
    "назначь supported",
    "назначить supported",
    "статус supported",
    "status supported",
    "claim_status",
    "измени критерий",
    "изменить критерий",
    "смени порог",
    "изменить порог",
    "brier_threshold",
)
_OVERCLAIM = (
    "закономерность установлена",
    "закономерность подтверждена",
    "доказано, что",
    "доказано лучше",
    "превосходит baseline",
)
_CODE_TOKENS = ("ev" + "al(", "ex" + "ec(", "__import__", "subprocess", "os.system")
_FIT_2021 = re.compile(
    r"(подобрать|подбор коэффициент|калибров\w* по 2021|fit on 2021|train on 2021)",
    re.IGNORECASE,
)
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def grammar_rules() -> dict[str, object]:
    """Правила, которые модель получает как данные, а не как исполняемый код."""
    return {
        "root": "sigmoid",
        "functions": list(FUNCTIONS),
        "operations": ["+", "-", "*", "/"],
        "features": list(FEATURES),
        "horizons": sorted(HORIZONS),
        "max_depth": MAX_DEPTH,
        "max_nodes": MAX_NODES,
        "max_length": MAX_EXPRESSION_LENGTH,
        "coefficient_min": -MAX_COEFFICIENT,
        "coefficient_max": MAX_COEFFICIENT,
        "division_epsilon": DIVISION_EPSILON,
        "python_execution": False,
        "evaluation_method": FIXED_EVALUATION_METHOD,
    }


def allowed_format_text() -> str:
    names = ", ".join(FUNCTIONS)
    return (
        "Один JSON-объект заявленной схемы. "
        f"Корень expression — sigmoid. Функции: {names}. Операции: + - * /. "
        f"Коэффициенты от {-MAX_COEFFICIENT:g} до {MAX_COEFFICIENT:g}. "
        f"Глубина не больше {MAX_DEPTH}, узлов не больше {MAX_NODES}, "
        f"длина не больше {MAX_EXPRESSION_LENGTH}."
    )


def preserve_meaning_text() -> str:
    return (
        "Сохрани первоначальный исследовательский смысл. "
        "Не подменяй вопрос, горизонт и разрешённые признаки. "
        "Не объявляй формулу лучше baseline и не назначай статус."
    )


def plan_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "research_question": {"type": "string"},
            "target_horizon": {"type": "integer", "enum": [1, 3, 7]},
            "candidate_features": {
                "type": "array",
                "items": {"type": "string", "enum": list(FEATURES)},
            },
            "proposed_relationship": {"type": "string"},
            "expected_market_conditions": {"type": "string", "enum": sorted(REGIMES)},
            "falsification_conditions": {"type": "string"},
            "evaluation_method": {"type": "string", "enum": [FIXED_EVALUATION_METHOD]},
            "limitations": {"type": "string"},
        },
        "required": list(PLAN_FIELDS),
    }


def hypothesis_schema() -> dict[str, object]:
    schema = proposal_schema()
    properties = dict(schema["properties"])  # type: ignore[arg-type]
    properties["expression"] = {
        "type": "string",
        "description": "Корень sigmoid. Функции sigmoid, tanh, abs, sign. Операции + - * /.",
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(schema["required"]),  # type: ignore[arg-type]
    }


def revision_schema() -> dict[str, object]:
    schema = hypothesis_schema()
    properties = dict(schema["properties"])  # type: ignore[arg-type]
    properties["parent_hypothesis_id"] = {"type": "string"}
    for name in REVISION_FIELDS:
        properties[name] = {"type": "string"}
    required = list(schema["required"]) + list(REVISION_FIELDS)  # type: ignore[arg-type]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def scan_research_text(text: str) -> tuple[str, str] | None:
    """Ищет в тексте просьбу открыть закрытый период, сменить критерий или исполнить код."""
    if not isinstance(text, str):
        return "invalid_json", "текстовое поле имеет неверный тип"
    lowered = text.lower()
    if any(token in lowered for token in _CODE_TOKENS):
        return "unsafe_math", "в тексте есть вызов, который не исполняется"
    if any(phrase in lowered for phrase in _RULE_PHRASES):
        return "criteria_change", "текст пытается изменить критерий проверки"
    if any(phrase in lowered for phrase in _OVERCLAIM):
        return "premature_claim", "текст объявляет преимущество до отдельной проверки"
    if any(marker in lowered for marker in _FUTURE_MARKERS):
        return "time_boundary", "текст ссылается на закрытые или будущие данные"
    if any(year in text for year in _FUTURE_YEARS):
        return "time_boundary", "текст ссылается на закрытый календарный период"
    if _FIT_2021.search(text) is not None:
        return "time_boundary", "текст просит подбирать коэффициенты по 2021 году"
    return None


def forbidden_call(text: str) -> str | None:
    for name in _CALL.findall(text):
        if name not in FUNCTIONS:
            return name
    return None


def dump_user(payload: Mapping[str, object]) -> str:
    """Короткий JSON пользователя. Сначала уходит память, затем длинные пояснения."""
    body = dict(payload)
    text = _compact(body)
    if len(text) <= MAX_USER_CHARS:
        return text
    memory = body.get("untrusted_memory")
    if isinstance(memory, dict):
        shrunk = dict(memory)
        shrunk["records"] = []
        body["untrusted_memory"] = shrunk
        text = _compact(body)
    if len(text) <= MAX_USER_CHARS:
        return text
    if "grammar" in body:
        body["grammar"] = {"root": "sigmoid", "functions": list(FUNCTIONS)}
        text = _compact(body)
    if len(text) <= MAX_USER_CHARS:
        return text
    feedback = body.get("scientific_feedback")
    if isinstance(feedback, dict):
        short = dict(feedback)
        for key in ("limitations", "further_research", "no_proven_improvement_reason", "uncertainty", "applicability"):
            value = short.get(key)
            if isinstance(value, str):
                short[key] = value[:120]
        body["scientific_feedback"] = short
        text = _compact(body)
    if len(text) <= MAX_USER_CHARS:
        return text
    raise ValueError("контекст не помещается в бюджет промпта")


def _compact(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
