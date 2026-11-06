"""Научный пересмотр. Новая версия не затирает родителя и не объявляет себя лучше."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from wolfai.research.grammar_contract import (
    HYPOTHESIS_FIELDS,
    REVISION_FIELDS,
    dump_user,
    revision_schema,
    scan_research_text,
)
from wolfai.research.hypothesis import HypothesisSchemaError, _extract_json
from wolfai.research.hypothesis_protocol import (
    KnownFormula,
    QwenDraft,
    verify_model_output,
)
from wolfai.research.research_plan import ResearchPlan, conforms_to_plan
from wolfai.research.research_ports import ModelTurn, RevisionContext, invoke_model
from wolfai.research.syntax_repair import canonical_identity, classify_edit, peek_expression

REVISION_SYSTEM = """Верни один JSON новой версии гипотезы. Это не оценка и не закон.
parent_hypothesis_id скопируй из запроса. Прежнюю формулу не заменяй.
Корень expression — sigmoid. Функции: sigmoid, tanh, abs, sign. Операции: + - * /.
Горизонт и признаки остаются внутри плана. Числа Verification Desk не меняй.
Не пиши, что формула уже лучше baseline. Каждое текстовое поле короче 120 символов.
Поясни: what_failed, assumption_to_revise, why_new_relationship, difference_from_parent, falsification_conditions.
"""
REVISION_TEXT_LIMIT = 160

_SCORE_KEYS = frozenset({
    "brier",
    "brier_score",
    "log_loss",
    "improvement",
    "baseline_brier",
    "independent_validation",
    "graded_by",
    "outcome",
    "claim_status",
    "validated_law",
    "validated",
    "status",
    "success_criterion",
    "brier_threshold",
    "validation_rule",
    "assign_supported",
    "author_id",
    "strategy_id",
    "model_digest",
})
_ALLOWED = set(HYPOTHESIS_FIELDS) | set(REVISION_FIELDS)


@dataclass(frozen=True)
class GateResult:
    accepted: bool
    code: str
    reason: str
    draft: QwenDraft | None = None
    expression: str | None = None
    suggested_id: str | None = None
    parent_hypothesis_id: str | None = None
    revision: dict[str, str] | None = None
    duplicate_of: str | None = None
    edit_kind: str = ""
    format_normalized: bool = False


def validate_generation(
    payload: str,
    *,
    plan: ResearchPlan,
    known: Sequence[KnownFormula] = (),
) -> GateResult:
    """Первая формула. Родителя нет, горизонт и признаки не выходят из плана."""
    decision = verify_model_output(payload, known=known)
    expression = peek_expression(payload)
    if not decision.accepted or decision.draft is None:
        return GateResult(
            False,
            decision.code,
            decision.reason,
            expression=expression,
            duplicate_of=decision.duplicate_of,
            draft=decision.draft,
        )
    draft = decision.draft
    problem = conforms_to_plan(draft.horizon_days, draft.features, plan)
    if problem is not None:
        code, reason = problem
        return GateResult(False, code, reason, draft=draft, expression=draft.expression, suggested_id=draft.suggested_id)
    normalized = False
    reason = "черновик прошёл независимую проверку"
    if draft.parent_hypothesis_id is not None:
        draft = replace(draft, parent_hypothesis_id=None)
        normalized = True
        reason = "техническое поле parent_hypothesis_id начальной гипотезы приведено к null; выражение этим шагом не менялось"
    return GateResult(
        True,
        "ok",
        reason,
        draft=draft,
        expression=draft.expression,
        suggested_id=draft.suggested_id,
        parent_hypothesis_id=None,
        format_normalized=normalized,
    )


def validate_revision_payload(
    payload: str,
    *,
    plan: ResearchPlan,
    parent_id: str,
    parent_expression: str,
    known: Sequence[KnownFormula] = (),
) -> GateResult:
    """Новая версия. Коэффициентный повтор и преждевременное заявление отклоняются."""
    expression = peek_expression(payload)
    if not isinstance(payload, str) or not payload.strip():
        return GateResult(False, "invalid_json", "ответ пуст", expression=expression)
    if len(payload) > 4000:
        return GateResult(False, "invalid_json", "ответ длиннее допустимого", expression=expression)
    try:
        raw = _extract_json(payload)
    except HypothesisSchemaError as error:
        return GateResult(False, "invalid_json", str(error), expression=expression)
    forbidden = [key for key in raw if key in _SCORE_KEYS]
    if forbidden:
        return GateResult(
            False,
            "criteria_change",
            f"пересмотр пытается задать поле {forbidden[0]}",
            expression=expression,
        )
    extra = [key for key in raw if key not in _ALLOWED]
    if extra:
        return GateResult(False, "invalid_json", f"лишнее поле {extra[0]}", expression=expression)
    explanations, problem = _explanations(raw)
    if problem is not None:
        return problem
    assert explanations is not None
    joined = "\n".join(explanations.values())
    scanned = scan_research_text(joined)
    if scanned is not None:
        code, reason = scanned
        return GateResult(False, code, reason, expression=expression, revision=explanations)
    parent = raw.get("parent_hypothesis_id")
    if not isinstance(parent, str) or not parent.strip():
        return GateResult(
            False,
            "invalid_json",
            "у пересмотра нет parent_hypothesis_id",
            expression=expression,
            revision=explanations,
        )
    if parent.strip() != parent_id:
        return GateResult(
            False,
            "unknown_parent",
            "пересмотр не ссылается на предыдущую версию",
            expression=expression,
            suggested_id=raw.get("hypothesis_id") if isinstance(raw.get("hypothesis_id"), str) else None,
            parent_hypothesis_id=parent.strip(),
            revision=explanations,
        )
    core = {key: raw.get(key) for key in HYPOTHESIS_FIELDS}
    decision = verify_model_output(json.dumps(core, ensure_ascii=False), known=known)
    if not decision.accepted or decision.draft is None:
        return GateResult(
            False,
            decision.code,
            decision.reason,
            draft=decision.draft,
            expression=expression if decision.draft is None else decision.draft.expression,
            suggested_id=None if decision.draft is None else decision.draft.suggested_id,
            parent_hypothesis_id=parent_id,
            revision=explanations,
            duplicate_of=decision.duplicate_of,
        )
    draft = decision.draft
    plan_problem = conforms_to_plan(draft.horizon_days, draft.features, plan)
    if plan_problem is not None:
        code, reason = plan_problem
        return GateResult(
            False,
            code,
            reason,
            draft=draft,
            expression=draft.expression,
            suggested_id=draft.suggested_id,
            parent_hypothesis_id=parent_id,
            revision=explanations,
        )
    edit = classify_edit(parent_expression, draft.expression)
    if edit.kind == "unchanged":
        return GateResult(
            False,
            "duplicate",
            "пересмотр повторяет формулу родителя",
            draft=draft,
            expression=draft.expression,
            suggested_id=draft.suggested_id,
            parent_hypothesis_id=parent_id,
            revision=explanations,
            edit_kind=edit.kind,
        )
    if edit.kind == "coefficient_only":
        return GateResult(
            False,
            "coefficient_peek",
            "пересмотр меняет только коэффициенты уже проверенной формулы",
            draft=draft,
            expression=draft.expression,
            suggested_id=draft.suggested_id,
            parent_hypothesis_id=parent_id,
            revision=explanations,
            edit_kind=edit.kind,
        )
    return GateResult(
        True,
        "ok",
        "пересмотр прошёл независимую проверку",
        draft=draft,
        expression=draft.expression,
        suggested_id=draft.suggested_id,
        parent_hypothesis_id=parent_id,
        revision=explanations,
        edit_kind=edit.kind,
    )


def build_revision_messages(context: RevisionContext) -> list[dict[str, str]]:
    feedback = context.scientific_feedback
    plan = context.plan
    payload = {
        "task": "scientific_revision",
        "parent_hypothesis_id": context.parent_hypothesis_id,
        "parent_expression": context.parent_expression,
        "research_question": str(plan.get("research_question", ""))[:180],
        "locked_horizon": plan.get("target_horizon"),
        "allowed_features": plan.get("candidate_features"),
        "expected_regime": plan.get("expected_market_conditions"),
        "desk": {
            "outcome": feedback.get("outcome"),
            "brier": feedback.get("brier"),
            "log_loss": feedback.get("log_loss"),
            "baseline_brier": feedback.get("baseline_brier"),
            "error_difference": feedback.get("error_difference"),
            "decision_count": feedback.get("decision_count"),
            "uncertainty": str(feedback.get("uncertainty", ""))[:160],
            "failure_reason": str(feedback.get("no_proven_improvement_reason", ""))[:160],
        },
        "grammar": {
            "root": "sigmoid",
            "functions": ["sigmoid", "tanh", "abs", "sign"],
            "operations": ["+", "-", "*", "/"],
        },
        "limits": "не меняй горизонт, оценку Desk, критерии и закрытые периоды; не исполняй Python",
    }
    return [
        {"role": "system", "content": REVISION_SYSTEM},
        {"role": "user", "content": dump_user(payload)},
    ]


class QwenReviser:
    def __init__(self, client: object, *, temperature: float) -> None:
        self.client = client
        self.temperature = temperature

    def revise(self, context: RevisionContext) -> ModelTurn:
        return invoke_model(
            self.client,
            build_revision_messages(context),
            revision_schema(),
            temperature=self.temperature,
        )


def _explanations(raw: Mapping[str, object]) -> tuple[dict[str, str] | None, GateResult | None]:
    found: dict[str, str] = {}
    for name in REVISION_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            return None, GateResult(False, "invalid_json", f"нет поля {name}", expression=peek_expression(json.dumps(raw)))
        if len(value.strip()) > REVISION_TEXT_LIMIT:
            return None, GateResult(False, "invalid_json", f"поле {name} длиннее допустимого")
        found[name] = value.strip()
    return found, None


def same_formula(left: str, right: str) -> bool:
    return canonical_identity(left) is not None and canonical_identity(left) == canonical_identity(right)
