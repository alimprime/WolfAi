"""Подбор записей Research Memory. Текст памяти — данные, не инструкция."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wolfai.research.formula import FUNCTIONS, FormulaError, parse_expression
from wolfai.research.grammar_contract import scan_research_text

MAX_RECORDS = 4
_CONTROL_KEYS = frozenset({
    "instruction",
    "instructions",
    "system",
    "tool",
    "command",
    "prompt",
    "role",
})
_PUBLIC_KEYS = (
    "expression",
    "features",
    "horizon_days",
    "error",
    "rejection_reason",
    "limitations",
    "claim_status",
    "evidence_role",
    "hypothesis_id",
)


def memory_as_data(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "trust": "data",
        "not_instructions": True,
        "records": [dict(item) for item in records],
    }


def select_relevant_records(
    records: Sequence[Mapping[str, object]],
    *,
    features: Sequence[str] = (),
    horizon_days: int | None = None,
    expression: str | None = None,
    limit: int = MAX_RECORDS,
) -> tuple[dict[str, object], ...]:
    """Ограниченный набор прошлых попыток: признаки, горизонт, структура, ошибки, дубликаты."""
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("лимит памяти должен быть от 1 до 8")
    wanted = set(features)
    target_shape = _shape(expression) if expression else ""
    ranked: list[tuple[tuple[int, str], dict[str, object]]] = []
    for raw in records:
        public = _public_record(raw)
        if public is None:
            continue
        record_features = _names(public.get("features"))
        record_expression = public.get("expression")
        expression_text = record_expression if isinstance(record_expression, str) else ""
        shape = _shape(expression_text)
        duplicate = bool(expression) and _norm(expression_text) == _norm(expression or "")
        score = 0
        score += 3 * len(wanted & set(record_features))
        if horizon_days is not None and public.get("horizon_days") == horizon_days:
            score += 2
        if target_shape and shape and target_shape == shape:
            score += 2
        if public.get("error") or public.get("rejection_reason"):
            score += 1
        if duplicate:
            score += 4
        public["structure"] = shape
        public["duplicate_of_request"] = duplicate
        public["trust"] = "data"
        ranked.append(((-score, expression_text), public))
    ranked.sort(key=lambda item: item[0])
    return tuple(item[1] for item in ranked[:limit])


def _public_record(raw: Mapping[str, object]) -> dict[str, object] | None:
    if any(key in raw for key in _CONTROL_KEYS):
        # Ключ управления не становится командой: поле отбрасывается, остальные данные остаются.
        raw = {key: value for key, value in raw.items() if key not in _CONTROL_KEYS}
    public: dict[str, object] = {}
    for key in _PUBLIC_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, str):
            if scan_research_text(value) is not None:
                return None
            public[key] = value.strip()[:180]
        elif isinstance(value, bool) or value is None:
            continue
        elif isinstance(value, int) and key == "horizon_days":
            public[key] = value
        elif key == "features":
            names = _names(value)
            if names:
                public[key] = names
    if not public:
        return None
    return public


def _names(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            names.append(item.strip())
    return names


def _shape(expression: str | None) -> str:
    if not expression or not expression.strip():
        return ""
    try:
        tree = parse_expression(expression)
    except FormulaError:
        return ""
    names: list[str] = []

    def walk(node: object) -> None:
        op = getattr(node, "op", None)
        if isinstance(op, str) and (op in FUNCTIONS or op in {"add", "sub", "mul", "div", "neg"}):
            names.append(op)
        arg = getattr(node, "arg", None)
        if arg is not None:
            walk(arg)
        left = getattr(node, "left", None)
        right = getattr(node, "right", None)
        if left is not None:
            walk(left)
        if right is not None:
            walk(right)

    walk(tree)
    return "+".join(names)


def _norm(expression: str) -> str:
    return "".join(expression.split())
