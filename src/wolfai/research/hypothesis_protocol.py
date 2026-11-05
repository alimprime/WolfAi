"""Схема гипотезы Qwen. Синтаксически верный JSON ещё не является гипотезой.

Ответ не исполняется как код. Статус supported и правила проверки
из этого текста не назначаются.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from wolfai.research.formula import FEATURES, features_used, parse_formula
from wolfai.research.hypothesis import HypothesisSchemaError, _extract_json
from wolfai.research.math_screen import MathReport, screen_formula

HORIZONS = frozenset({1, 3, 7})
REGIMES = frozenset({
    "any",
    "low_volatility",
    "high_volatility",
    "mid_volatility",
    "uptrend",
    "downtrend",
    "sideways",
})
_REQUIRED = (
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
_RULE_KEYS = frozenset({
    "claim_status",
    "status",
    "validated_law",
    "validated",
    "success_criterion",
    "brier_threshold",
    "validation_rule",
    "assign_supported",
})
_IDENTITY_KEYS = frozenset({"author_id", "graded_by", "model_digest"})
_TEXT_FIELDS = (
    "hypothesis_id",
    "description",
    "expression",
    "rationale",
    "expected_regime",
    "test_plan",
    "limitations",
)
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
)
_CODE_TOKENS = ("ev" + "al(", "ex" + "ec(", "__import__", "subprocess", "os.system")
_FIT_2021 = re.compile(
    r"(подобрать|подбор коэффициент|калибров\w* по 2021|fit on 2021|train on 2021)",
    re.IGNORECASE,
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$")


def proposal_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "hypothesis_id": {"type": "string"},
            "description": {"type": "string"},
            "features": {
                "type": "array",
                "items": {"type": "string", "enum": list(FEATURES)},
            },
            "horizon_days": {"type": "integer", "enum": [1, 3, 7]},
            "expression": {"type": "string"},
            "rationale": {"type": "string"},
            "expected_regime": {"type": "string", "enum": sorted(REGIMES)},
            "test_plan": {"type": "string"},
            "limitations": {"type": "string"},
            "parent_hypothesis_id": {"type": ["string", "null"]},
        },
        "required": list(_REQUIRED),
    }


QWEN_HYPOTHESIS_SCHEMA = proposal_schema()


@dataclass(frozen=True)
class KnownFormula:
    expression: str
    features: tuple[str, ...]
    horizon_days: int
    status: str
    hypothesis_id: str


@dataclass(frozen=True)
class QwenDraft:
    suggested_id: str
    description: str
    features: tuple[str, ...]
    horizon_days: int
    expression: str
    rationale: str
    expected_regime: str
    test_plan: str
    limitations: str
    parent_hypothesis_id: str | None
    math_report: MathReport


@dataclass(frozen=True)
class ProtocolDecision:
    accepted: bool
    code: str
    reason: str
    draft: QwenDraft | None = None
    duplicate_of: str | None = None


def verify_model_output(
    payload: str,
    *,
    known: Sequence[KnownFormula] = (),
    available_features: Sequence[str] = FEATURES,
) -> ProtocolDecision:
    """Проверяет JSON, математику, горизонт, будущее и повтор формулы."""
    if not isinstance(payload, str) or not payload.strip():
        return _reject("invalid_json", "ответ пуст")
    if len(payload) > 4000:
        return _reject("invalid_json", "ответ длиннее допустимого")
    try:
        raw = _extract_json(payload)
    except HypothesisSchemaError as error:
        return _reject("invalid_json", str(error))
    extra = [key for key in raw if key not in _REQUIRED]
    identity = [key for key in extra if key in _IDENTITY_KEYS]
    if identity:
        return _reject("agent_isolation", f"модель передала чужое поле {identity[0]}")
    rules = [key for key in extra if key in _RULE_KEYS]
    if rules:
        return _reject("criteria_change", f"модель пытается задать поле {rules[0]}")
    if extra:
        return _reject("invalid_json", f"лишнее поле {extra[0]}")
    draft_fields, problem = _fields(raw)
    if problem is not None:
        return problem
    assert draft_fields is not None
    text_problem = _scan_text(draft_fields)
    if text_problem is not None:
        return text_problem
    math_report = screen_formula(draft_fields.expression, available_features)
    if not math_report.accepted:
        return _reject(math_report.code, math_report.reason)
    if set(draft_fields.features) != features_used(parse_formula(draft_fields.expression)):
        return _reject("feature_mismatch", "набор признаков в формуле не совпадает с полем features")
    draft = QwenDraft(
        suggested_id=draft_fields.suggested_id,
        description=draft_fields.description,
        features=draft_fields.features,
        horizon_days=draft_fields.horizon_days,
        expression=draft_fields.expression,
        rationale=draft_fields.rationale,
        expected_regime=draft_fields.expected_regime,
        test_plan=draft_fields.test_plan,
        limitations=draft_fields.limitations,
        parent_hypothesis_id=draft_fields.parent_hypothesis_id,
        math_report=math_report,
    )
    duplicate = _duplicate(draft_fields, known)
    if duplicate is not None:
        code_reason, hypothesis_id = duplicate
        return ProtocolDecision(False, "duplicate", code_reason, draft=draft, duplicate_of=hypothesis_id)
    return ProtocolDecision(True, "ok", "черновик прошёл независимую проверку", draft=draft)


@dataclass(frozen=True)
class _Fields:
    suggested_id: str
    description: str
    features: tuple[str, ...]
    horizon_days: int
    expression: str
    rationale: str
    expected_regime: str
    test_plan: str
    limitations: str
    parent_hypothesis_id: str | None


def _fields(raw: Mapping[str, object]) -> tuple[_Fields | None, ProtocolDecision | None]:
    suggested = raw.get("hypothesis_id")
    description = raw.get("description")
    features = raw.get("features")
    horizon = raw.get("horizon_days")
    expression = raw.get("expression")
    rationale = raw.get("rationale")
    regime = raw.get("expected_regime")
    plan = raw.get("test_plan")
    limits = raw.get("limitations")
    parent = raw.get("parent_hypothesis_id")
    if not isinstance(suggested, str) or _ID_PATTERN.fullmatch(suggested.strip()) is None:
        return None, _reject("invalid_json", "hypothesis_id пуст или содержит недопустимые символы")
    text_map = {
        "description": (description, 400),
        "expression": (expression, 300),
        "rationale": (rationale, 400),
        "test_plan": (plan, 300),
        "limitations": (limits, 300),
    }
    cleaned: dict[str, str] = {"hypothesis_id": suggested.strip()}
    for name, (value, limit) in text_map.items():
        if not isinstance(value, str) or not value.strip():
            return None, _reject("invalid_json", f"поле {name} пустое")
        if len(value.strip()) > limit:
            return None, _reject("invalid_json", f"поле {name} длиннее допустимого")
        cleaned[name] = value.strip()
    if not isinstance(regime, str) or regime.strip() not in REGIMES:
        return None, _reject("invalid_json", "expected_regime вне разрешённого списка")
    if type(horizon) is not int or horizon not in HORIZONS:
        return None, _reject("bad_horizon", "горизонт должен быть 1, 3 или 7 дней")
    if not isinstance(features, list) or not features or len(features) > len(FEATURES):
        return None, _reject("unknown_feature", "список признаков пуст или длиннее разрешённого")
    if any(type(item) is not str for item in features):
        return None, _reject("unknown_feature", "признак должен быть строкой")
    names = tuple(str(item) for item in features)
    if len(names) != len(set(names)):
        return None, _reject("unknown_feature", "признак указан дважды")
    if any(name not in FEATURES for name in names):
        missing = next(name for name in names if name not in FEATURES)
        return None, _reject("unknown_feature", f"неизвестный признак {missing}")
    if parent is not None and (not isinstance(parent, str) or _ID_PATTERN.fullmatch(parent.strip()) is None):
        return None, _reject("invalid_json", "parent_hypothesis_id имеет недопустимый вид")
    return (
        _Fields(
            suggested_id=cleaned["hypothesis_id"],
            description=cleaned["description"],
            features=names,
            horizon_days=horizon,
            expression=cleaned["expression"],
            rationale=cleaned["rationale"],
            expected_regime=regime.strip(),
            test_plan=cleaned["test_plan"],
            limitations=cleaned["limitations"],
            parent_hypothesis_id=None if parent is None else parent.strip(),
        ),
        None,
    )


def _scan_text(fields: _Fields) -> ProtocolDecision | None:
    blob = "\n".join(
        getattr(fields, name) if name != "hypothesis_id" else fields.suggested_id
        for name in _TEXT_FIELDS
    )
    lowered = blob.lower()
    if any(token in lowered for token in _CODE_TOKENS):
        return _reject("unsafe_math", "в тексте есть вызов, который не исполняется")
    if any(phrase in lowered for phrase in _RULE_PHRASES):
        return _reject("criteria_change", "текст пытается изменить критерий проверки")
    if any(marker in lowered for marker in _FUTURE_MARKERS) or any(year in blob for year in _FUTURE_YEARS):
        return _reject("time_boundary", "ответ ссылается на закрытые или будущие данные")
    if _FIT_2021.search(blob):
        return _reject("time_boundary", "2021 год остаётся DEV_ONLY и не используется для подбора")
    return None


def _duplicate(fields: _Fields, known: Sequence[KnownFormula]) -> tuple[str, str] | None:
    target = _norm(fields.expression)
    features = set(fields.features)
    for item in known:
        if _norm(item.expression) != target:
            continue
        if set(item.features) != features or item.horizon_days != fields.horizon_days:
            continue
        if item.status == "rejected":
            return "дубликат уже отвергнутой формулы", item.hypothesis_id
        return "дубликат уже проверенной формулы", item.hypothesis_id
    return None


def _norm(expression: str) -> str:
    return "".join(expression.split())


def _reject(code: str, reason: str) -> ProtocolDecision:
    return ProtocolDecision(False, code, reason)
