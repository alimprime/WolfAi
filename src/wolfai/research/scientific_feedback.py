"""Научный отзыв Verification Desk. Модель эти числа не назначает."""

from __future__ import annotations

from collections.abc import Mapping

from wolfai.research.evaluation_protocol import MIN_STABLE_OBSERVATIONS, Evaluation
from wolfai.research.grammar_contract import OVERFIT_NODE_LIMIT
from wolfai.research.math_screen import VERIFIER_ID

_LIMITATIONS = (
    "Внутренний счёт на заранее заданном окне не доказывает закономерность "
    "и не является независимым подтверждением. "
    "2021 год в адаптацию не входил. "
    "Сравнение идёт с константой 0.5, прошлой частотой и логистикой 2019 года."
)
_DIRECTIONS = {
    "formula_invalid": "Сначала нужна допустимая формула. Новый научный эксперимент не начат.",
    "not_better_than_baseline": "Имеет смысл пересмотреть вид зависимости, а не подгонять коэффициент по тому же окну.",
    "point_improvement_insufficient_evidence": "Точечная разница на внутреннем окне не подтверждена. Отдельная проверка здесь не открывается.",
    "overfitting_signs": "Имеет смысл упростить структуру. Число узлов не заменяет доказательства.",
    "insufficient_data": "Не стоит увеличивать число параметров, пока наблюдений меньше заранее заданного минимума.",
}


def classify_evaluation(evaluation: Evaluation, *, nodes: int) -> str:
    """Пять исходов заданы до просмотра конкретной формулы."""
    if not evaluation.accepted:
        if evaluation.code == "insufficient_sample":
            return "insufficient_data"
        return "formula_invalid"
    if evaluation.sample_count < MIN_STABLE_OBSERVATIONS:
        return "insufficient_data"
    improvement = evaluation.improvement
    if nodes > OVERFIT_NODE_LIMIT and improvement is not None and improvement > 0:
        return "overfitting_signs"
    if improvement is None or improvement <= 0:
        return "not_better_than_baseline"
    return "point_improvement_insufficient_evidence"


def build_scientific_feedback(
    evaluation: Evaluation,
    *,
    hypothesis_id: str,
    formula_version: int,
    nodes: int,
    applicability: str,
    expression: str,
) -> dict[str, object]:
    """Собирает отзыв только из полей Evaluation. Внешней подмены чисел нет."""
    if not evaluation.accepted or evaluation.brier_score is None or evaluation.log_loss is None:
        raise ValueError("научная обратная связь строится только после исполнения")
    if expression.strip() != evaluation.expression.strip():
        raise ValueError("формула отзыва не совпадает с проверенной")
    outcome = classify_evaluation(evaluation, nodes=nodes)
    improvement = evaluation.improvement
    development = (
        outcome == "point_improvement_insufficient_evidence"
        and improvement is not None
        and improvement > 0
    )
    limitations = _LIMITATIONS
    if formula_version > 1:
        limitations += (
            " То же внутреннее окно уже использовано для предыдущей версии "
            "и не становится независимым подтверждением."
        )
    payload: dict[str, object] = {
        "kind": "scientific",
        "research_id": evaluation.experiment_id,
        "hypothesis_id": hypothesis_id,
        "expression": evaluation.expression,
        "formula_version": formula_version,
        "horizon_days": evaluation.horizon_days,
        "decision_count": evaluation.sample_count,
        "brier": evaluation.brier_score,
        "log_loss": evaluation.log_loss,
        "baseline_brier": evaluation.baseline_brier,
        "frequency_brier": evaluation.frequency_brier,
        "logistic_brier": evaluation.logistic_brier,
        "error_difference": improvement,
        "uncertainty": evaluation.uncertainty,
        "applicability": applicability,
        "limitations": limitations,
        "no_proven_improvement_reason": _no_proof_reason(outcome),
        "further_research": _DIRECTIONS[outcome],
        "outcome": outcome,
        "syntactic_success": True,
        "execution_success": True,
        "development_improvement": development,
        "independent_validation": False,
        "graded_by": VERIFIER_ID,
        "claim_status": "not_confirmed",
        "validated_law": False,
        "fitted_on_2021": False,
    }
    assert_feedback_matches(payload, evaluation)
    return payload


def assert_feedback_matches(feedback: Mapping[str, object], evaluation: Evaluation) -> None:
    """Останавливает цикл, если отзыв расходится с Verification Desk."""
    if feedback.get("kind") != "scientific":
        raise RuntimeError("это не научная обратная связь")
    if feedback.get("brier") != evaluation.brier_score or feedback.get("log_loss") != evaluation.log_loss:
        raise RuntimeError("оценка Verification Desk изменена")
    if feedback.get("error_difference") != evaluation.improvement:
        raise RuntimeError("разница ошибок изменена")
    if feedback.get("independent_validation") is not False or feedback.get("validated_law") is not False:
        raise RuntimeError("неподтверждённый статус")
    if feedback.get("graded_by") != VERIFIER_ID:
        raise RuntimeError("оценку поставил не verification-desk")
    if feedback.get("claim_status") != "not_confirmed":
        raise RuntimeError("неподтверждённый статус")
    if feedback.get("fitted_on_2021") is not False:
        raise RuntimeError("отзыв отмечает подбор по 2021 году")


def _no_proof_reason(outcome: str) -> str:
    if outcome == "not_better_than_baseline":
        return "На внутреннем окне ошибка не ниже заранее заданной константы 0.5."
    if outcome == "point_improvement_insufficient_evidence":
        return (
            "Точечная разница на внутреннем окне не является независимым подтверждением. "
            "Повторное использование того же окна не доказывает преимущество."
        )
    if outcome == "overfitting_signs":
        return "Структура выражения слишком богата для этого окна. Снижение ошибки не считается подтверждением."
    if outcome == "insufficient_data":
        return f"Наблюдений меньше заранее заданного минимума {MIN_STABLE_OBSERVATIONS}."
    return "Формула не прошла проверку и не является успешным экспериментом."
