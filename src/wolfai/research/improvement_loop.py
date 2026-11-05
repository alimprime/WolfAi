"""Один ограниченный цикл: генерация, проверка, счёт, отзыв, пересмотр.

По умолчанию dry-run. Не больше одной начальной гипотезы и двух пересмотров.
Статистические пороги между попытками не меняются. Исходная формула не переписывается.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from wolfai.research.budget import SearchBusy, exclusive_search
from wolfai.research.evaluation_protocol import (
    CLOSED_FROM,
    PROTOCOL_VERSION,
    Evaluation,
    StudyPoint,
    evaluate_hypothesis,
    feedback_payload,
    synthetic_study_points,
)
from wolfai.research.hypothesis_engine import HypothesisStore, formula_key
from wolfai.research.hypothesis_protocol import verify_model_output
from wolfai.research.math_screen import VERIFIER_ID
from wolfai.research.ollama_adapter import OllamaAdapter
from wolfai.research.memory_context import MemoryExcerpt, load_research_excerpt
from wolfai.research.qwen_brain import GENERATOR_VERSION, BrainContext, QwenResearchBrain
from wolfai.research.research_cycle import reject_self_grade, reject_validated_law

MAX_INITIAL = 1
MAX_REVISIONS = 2
MAX_EXPRESSIONS = 3
STRATEGY_ID = "qwen-research-strategy"
_LIMITATION = (
    "Внутренний счёт не доказывает закономерность. "
    "Диагностический год в адаптацию не входил. "
    "Сравнение с константой 0.5, прошлой частотой и логистикой идёт на одних и тех же датах."
)


@dataclass(frozen=True)
class AttemptRecord:
    step: str
    status: str
    author_id: str
    strategy_id: str
    suggested_id: str | None
    hypothesis_id: str | None
    parent_hypothesis_id: str | None
    expression: str | None
    revision_reason: str | None
    feedback: dict[str, object]
    brier_score: float | None
    log_loss: float | None
    baseline_brier: float | None
    improvement: float | None
    uncertainty: str
    experiment_id: str
    model_name: str
    model_digest: str
    persisted: bool


@dataclass(frozen=True)
class LoopReport:
    status: str
    message: str
    dry_run: bool
    author_id: str
    strategy_id: str
    attempts: tuple[AttemptRecord, ...]
    unique_expressions: int
    revisions: int
    model_calls: int
    elapsed_seconds: float
    rss_bytes: int
    claim_status: str = "not_confirmed"
    validated_law: bool = False
    point_estimate_better: bool | None = None
    improvement_is_noise: bool | None = None
    code_version: str = PROTOCOL_VERSION


def run_improvement_loop(
    client: object | None,
    *,
    author_id: str = "qwen-research-brain",
    strategy_id: str = STRATEGY_ID,
    market_brief: str = "Синтетический ряд известен только до решения. Рыночный вывод не делается.",
    store: HypothesisStore | None = None,
    memory: object | None = None,
    points: Sequence[StudyPoint] | None = None,
    commit: bool = False,
    created_at: datetime | None = None,
    deadline_seconds: float = 180.0,
    now: Callable[[], float] | None = None,
    memory_guard: Callable[[], str | None] | None = None,
    data_hash: str = "synthetic-2019-2020",
) -> LoopReport:
    """Не больше трёх уникальных выражений. Бесконечного повтора нет."""
    clock = now or time.perf_counter
    guard = memory_guard or (lambda: None)
    started = clock()
    series = tuple(synthetic_study_points() if points is None else points)
    when = created_at or datetime.now(timezone.utc)

    def finish(status: str, message: str, attempts: list[AttemptRecord], revisions: int, calls: int) -> LoopReport:
        from wolfai.research.ollama_adapter import process_rss_bytes

        better, noise = _compare(attempts)
        return LoopReport(
            status=status,
            message=message,
            dry_run=not commit,
            author_id=author_id,
            strategy_id=strategy_id,
            attempts=tuple(attempts),
            unique_expressions=len({item.expression for item in attempts if item.expression and item.brier_score is not None}),
            revisions=revisions,
            model_calls=calls,
            elapsed_seconds=round(clock() - started, 6),
            rss_bytes=process_rss_bytes(),
            point_estimate_better=better,
            improvement_is_noise=noise,
        )

    if not author_id.strip() or not strategy_id.strip():
        return finish("aborted", "нужны author_id и strategy_id", [], 0, 0)
    if when.tzinfo is None or when.utcoffset() != timedelta(0):
        return finish("aborted", "время цикла должно быть в UTC", [], 0, 0)
    if commit and store is None:
        return finish("store_unavailable", "нет журнала гипотез для записи", [], 0, 0)
    if commit and memory is None:
        return finish("memory_unavailable", "нет research memory для записи", [], 0, 0)
    blocked = guard()
    if blocked:
        return finish("resource_limited", blocked, [], 0, 0)
    boundary = _series_boundary(series)
    if boundary is not None:
        return finish("time_boundary", boundary, [], 0, 0)
    try:
        excerpt, memory_error = load_research_excerpt(memory, store, author_id=author_id)
    except Exception as error:
        _rollback(store, memory)
        return finish("aborted", f"{error.__class__.__name__}: {error}", [], 0, 0)
    if memory_error is not None or excerpt is None:
        return finish("memory_unavailable", memory_error or "research memory недоступна", [], 0, 0)
    try:
        with exclusive_search("qwen"):
            return _run(
                finish,
                client=client,
                author_id=author_id,
                strategy_id=strategy_id,
                market_brief=market_brief,
                store=store,
                memory=memory,
                points=series,
                commit=commit,
                created_at=when,
                data_hash=data_hash,
                excerpt=excerpt,
                clock=clock,
                started=started,
                deadline_seconds=deadline_seconds,
                guard=guard,
            )
    except SearchBusy as error:
        return finish("search_busy", str(error), [], 0, 0)
    except Exception as error:
        _rollback(store, memory)
        return finish("aborted", f"{error.__class__.__name__}: {error}", [], 0, 0)


def _run(finish, **kwargs: object) -> LoopReport:
    author_id = str(kwargs["author_id"])
    strategy_id = str(kwargs["strategy_id"])
    store: HypothesisStore | None = kwargs["store"]  # type: ignore[assignment]
    memory = kwargs["memory"]
    commit = bool(kwargs["commit"])
    points: tuple[StudyPoint, ...] = kwargs["points"]  # type: ignore[assignment]
    excerpt: MemoryExcerpt = kwargs["excerpt"]  # type: ignore[assignment]
    created_at: datetime = kwargs["created_at"]  # type: ignore[assignment]
    clock: Callable[[], float] = kwargs["clock"]  # type: ignore[assignment]
    started = float(kwargs["started"])  # type: ignore[arg-type]
    deadline = float(kwargs["deadline_seconds"])  # type: ignore[arg-type]
    guard: Callable[[], str | None] = kwargs["guard"]  # type: ignore[assignment]
    brain = QwenResearchBrain(author_id, temperature=0.0, seed=1729)
    model = kwargs["client"]
    if isinstance(model, OllamaAdapter):
        model._release_wait_seconds = 20.0
    attempts: list[AttemptRecord] = []
    seen: set[tuple[str, tuple[str, ...], int]] = set()
    revisions = 0
    calls = 0
    feedback: dict[str, object] | None = None
    previous_id: str | None = None
    for slot in range(MAX_EXPRESSIONS):
        if clock() - started > deadline:
            return finish("time_limit", "исчерпан бюджет времени", attempts, revisions, calls)
        blocked = guard()
        if blocked:
            return finish("resource_limited", blocked, attempts, revisions, calls)
        is_revision = slot > 0
        if is_revision:
            if revisions >= MAX_REVISIONS or feedback is None or feedback.get("revision_allowed") is not True:
                break
            revisions += 1
        context = BrainContext(
            author_id=author_id,
            market_brief=str(kwargs["market_brief"]),
            excerpt=excerpt,
            feedback=feedback,
        )
        try:
            turn = brain.propose(context, kwargs["client"])
        except Exception as error:
            return finish("aborted", f"{error.__class__.__name__}: {error}", attempts, revisions, calls)
        calls += 1
        if turn.status != "ok" or not turn.content:
            attempts.append(_empty_attempt(
                "revision" if is_revision else "generate",
                turn.status,
                author_id,
                strategy_id,
                turn.message,
                turn.model_name,
                turn.model_digest,
                previous_id if is_revision else None,
            ))
            return finish(turn.status, turn.message, attempts, revisions, calls)
        decision = verify_model_output(turn.content, known=excerpt.known)
        step = "revision" if is_revision else "generate"
        if not decision.accepted or decision.draft is None:
            saved = _save_rejection(store, author_id, decision.code, decision.reason, created_at) if commit else False
            feedback = _failure_feedback(
                decision.code,
                decision.reason,
                revision_allowed=revisions < MAX_REVISIONS and slot + 1 < MAX_EXPRESSIONS and decision.code != "duplicate",
                hypothesis_id=decision.duplicate_of or previous_id or "none",
            )
            attempts.append(_empty_attempt(
                step,
                decision.code,
                author_id,
                strategy_id,
                decision.reason,
                turn.model_name,
                turn.model_digest,
                decision.duplicate_of,
                expression=None if decision.draft is None else decision.draft.expression,
                suggested_id=None if decision.draft is None else decision.draft.suggested_id,
                feedback=feedback,
                persisted=saved,
            ))
            if decision.code == "duplicate":
                return finish("duplicate", decision.reason, attempts, revisions, calls)
            continue
        draft = decision.draft
        if is_revision and draft.parent_hypothesis_id != previous_id:
            feedback = _failure_feedback(
                "unknown_parent",
                "пересмотр не ссылается на предыдущую версию",
                revision_allowed=revisions < MAX_REVISIONS and slot + 1 < MAX_EXPRESSIONS,
                hypothesis_id=previous_id or "none",
            )
            attempts.append(_empty_attempt(
                step,
                "unknown_parent",
                author_id,
                strategy_id,
                "пересмотр не ссылается на предыдущую версию",
                turn.model_name,
                turn.model_digest,
                previous_id,
                expression=draft.expression,
                suggested_id=draft.suggested_id,
                feedback=feedback,
            ))
            continue
        identity = ("".join(draft.expression.split()), tuple(sorted(draft.features)), draft.horizon_days)
        if identity in seen:
            feedback = _failure_feedback(
                "duplicate",
                "дубликат уже проверенной формулы",
                revision_allowed=False,
                hypothesis_id=previous_id or "none",
            )
            attempts.append(_empty_attempt(
                step,
                "duplicate",
                author_id,
                strategy_id,
                "дубликат уже проверенной формулы",
                turn.model_name,
                turn.model_digest,
                previous_id,
                expression=draft.expression,
                suggested_id=draft.suggested_id,
                feedback=feedback,
            ))
            return finish("duplicate", "дубликат уже проверенной формулы", attempts, revisions, calls)
        evaluation = evaluate_hypothesis(draft.expression, draft.features, draft.horizon_days, points)
        applicability = _applicability(strategy_id, draft.expected_regime)
        hypothesis_id = formula_key(draft.expression, draft.features, draft.horizon_days, applicability)
        if not evaluation.accepted:
            if commit:
                _save_rejection(store, author_id, evaluation.code, evaluation.reason, created_at)
            feedback = _failure_feedback(
                evaluation.code,
                evaluation.reason,
                revision_allowed=revisions < MAX_REVISIONS and slot + 1 < MAX_EXPRESSIONS,
                hypothesis_id=hypothesis_id,
            )
            attempts.append(AttemptRecord(
                step=step,
                status=evaluation.code,
                author_id=author_id,
                strategy_id=strategy_id,
                suggested_id=draft.suggested_id,
                hypothesis_id=hypothesis_id,
                parent_hypothesis_id=draft.parent_hypothesis_id,
                expression=draft.expression,
                revision_reason=draft.rationale if is_revision else None,
                feedback=feedback,
                brier_score=None,
                log_loss=None,
                baseline_brier=None,
                improvement=None,
                uncertainty=evaluation.uncertainty,
                experiment_id=evaluation.experiment_id,
                model_name=turn.model_name,
                model_digest=turn.model_digest,
                persisted=False,
            ))
            continue
        reject_validated_law("not_confirmed", False)
        reject_self_grade(author_id, VERIFIER_ID)
        persisted = False
        if commit:
            persisted = _persist(
                store=store,
                memory=memory,
                author_id=author_id,
                strategy_id=strategy_id,
                draft=draft,
                evaluation=evaluation,
                applicability=applicability,
                hypothesis_id=hypothesis_id,
                model_name=turn.model_name or "qwen3.5:4b",
                model_digest=turn.model_digest or "unspecified",
                created_at=created_at,
                data_hash=str(kwargs["data_hash"]),
            )
            if persisted == "duplicate":
                feedback = _success_feedback(evaluation, hypothesis_id, revision_allowed=False)
                attempts.append(_scored(
                    step, "duplicate", author_id, strategy_id, draft, hypothesis_id, evaluation,
                    feedback, turn.model_name, turn.model_digest, False, is_revision,
                ))
                return finish("duplicate", "дубликат уже проверенной формулы", attempts, revisions, calls)
        seen.add(identity)
        previous_id = hypothesis_id
        allow_revision = revisions < MAX_REVISIONS and len(seen) < MAX_EXPRESSIONS
        feedback = _success_feedback(evaluation, hypothesis_id, revision_allowed=allow_revision)
        attempts.append(_scored(
            step,
            "stored" if persisted is True else "dry_run",
            author_id,
            strategy_id,
            draft,
            hypothesis_id,
            evaluation,
            feedback,
            turn.model_name,
            turn.model_digest,
            persisted is True,
            is_revision,
        ))
    status = "stored" if commit and any(item.persisted for item in attempts) else "dry_run"
    return finish(status, "цикл завершён в заданном бюджете", attempts, revisions, calls)


def _success_feedback(evaluation: Evaluation, hypothesis_id: str, *, revision_allowed: bool) -> dict[str, object]:
    note = _LIMITATION
    if evaluation.logistic_brier is None:
        note += " Контрольная логистика на этом наборе недоступна."
    return feedback_payload(
        evaluation,
        hypothesis_id=hypothesis_id,
        failure_reason=None,
        limitations=note,
        revision_allowed=revision_allowed,
    )


def _failure_feedback(code: str, reason: str, *, revision_allowed: bool, hypothesis_id: str) -> dict[str, object]:
    blank = Evaluation(
        accepted=False,
        code=code,
        reason=reason,
        horizon_days=1,
        expression="",
        uncertainty=reason,
        experiment_id="none",
    )
    return feedback_payload(
        blank,
        hypothesis_id=hypothesis_id,
        failure_reason=f"{code}: {reason}"[:240],
        limitations=_LIMITATION,
        revision_allowed=revision_allowed,
    )


def _persist(
    *,
    store: HypothesisStore | None,
    memory: object,
    author_id: str,
    strategy_id: str,
    draft,
    evaluation: Evaluation,
    applicability: dict[str, object],
    hypothesis_id: str,
    model_name: str,
    model_digest: str,
    created_at: datetime,
    data_hash: str,
) -> bool | str:
    if store is None:
        raise RuntimeError("нет журнала гипотез для записи")
    reject_validated_law("not_confirmed", False)
    stored_id, state = store.create(
        author_id=author_id,
        name=draft.suggested_id,
        description=draft.description,
        creation_reason=draft.rationale,
        features=draft.features,
        horizon_days=draft.horizon_days,
        expression=draft.expression,
        applicability=applicability,
        created_at=created_at,
        data_hash=data_hash,
        parent_hypothesis_id=draft.parent_hypothesis_id,
    )
    if stored_id != hypothesis_id or state == "duplicate":
        return "duplicate"
    payload = {
        "claim_status": "not_confirmed",
        "validated_law": False,
        "advantage_claim": False,
        "graded_by": VERIFIER_ID,
        "author_id": author_id,
        "strategy_id": strategy_id,
        "generator_model": model_name,
        "generator_version": GENERATOR_VERSION,
        "code_version": PROTOCOL_VERSION,
        "expression": draft.expression,
        "parent_hypothesis_id": draft.parent_hypothesis_id,
        "revision_reason": draft.rationale,
        "experiment_id": evaluation.experiment_id,
        "brier_score": evaluation.brier_score,
        "log_loss": evaluation.log_loss,
        "baseline_brier": evaluation.baseline_brier,
        "frequency_brier": evaluation.frequency_brier,
        "logistic_brier": evaluation.logistic_brier,
        "improvement": evaluation.improvement,
        "uncertainty": evaluation.uncertainty,
        "limitations": _LIMITATION,
        "fitted_on_2021": False,
        "dev_2021_skipped": evaluation.dev_2021_skipped,
        "sample_count": evaluation.sample_count,
        "data_hash": evaluation.data_hash,
    }
    reject_validated_law(str(payload["claim_status"]), bool(payload["validated_law"]))
    store.add_experiment(stored_id, payload, when=created_at)
    record = getattr(memory, "record", None)
    if not callable(record):
        raise RuntimeError("research memory не принимает запись")
    record({
        "model_name": model_name,
        "config_hash": model_digest,
        "dataset_fingerprint": data_hash,
        "features": list(draft.features),
        "label_definition": "синтетическая метка внутреннего протокола 3.1; это не исход BTC",
        "horizon_days": draft.horizon_days,
        "train_start": "2019-01-01T00:00:00+00:00",
        "train_end": "2019-12-31T00:00:00+00:00",
        "dev_start": "2021-01-01T00:00:00+00:00",
        "dev_end": "2021-12-31T00:00:00+00:00",
        "seed": 1729,
        "evidence_role": "synthetic",
        "claim_status": "not_confirmed",
        "validated_law": False,
        "fitted": False,
        "fit_window": "none",
        "diagnostic_2021": "DEV_ONLY_NOT_USED",
        "expression": draft.expression,
        "brier": evaluation.brier_score,
        "log_loss": evaluation.log_loss,
        "improvement": evaluation.improvement,
        "uncertainty": evaluation.uncertainty,
        "experiment_id": evaluation.experiment_id,
        "author_id": author_id,
        "strategy_id": strategy_id,
        "hypothesis_id": stored_id,
        "parent_hypothesis_id": draft.parent_hypothesis_id,
        "code_version": PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
        "btc_claim": False,
    })
    stored = store.get(stored_id)
    if stored.expression != draft.expression or stored.claim_status != "not_confirmed" or store.confirms_law(stored_id):
        raise RuntimeError("запись изменила формулу или получила запрещённый статус")
    return True


def _applicability(strategy_id: str, regime: str) -> dict[str, object]:
    return {
        "market": "synthetic-2019-2020",
        "regimes": ["any"] if regime == "any" else [regime],
        "btc_claim": False,
        "strategy_id": strategy_id,
        "code_version": PROTOCOL_VERSION,
        "evidence_role": "not_confirmed",
    }


def _series_boundary(points: Sequence[StudyPoint]) -> str | None:
    for point in points:
        if point.decision_date >= CLOSED_FROM or point.target_date >= CLOSED_FROM:
            return "набор содержит дату после 2021 года"
        if point.label_available <= point.decision_available:
            return "в наборе метка доступна не позже решения"
    return None


def _compare(attempts: Sequence[AttemptRecord]) -> tuple[bool | None, bool | None]:
    scored = [item for item in attempts if item.brier_score is not None]
    if len(scored) < 2:
        return None, None
    better = scored[-1].brier_score < scored[0].brier_score  # type: ignore[operator]
    text = scored[-1].uncertainty
    noise = "случайн" in text or "не устойчив" in text or "содержит ноль" in text
    return better, noise


def _scored(
    step: str,
    status: str,
    author_id: str,
    strategy_id: str,
    draft,
    hypothesis_id: str,
    evaluation: Evaluation,
    feedback: dict[str, object],
    model_name: str,
    model_digest: str,
    persisted: bool,
    is_revision: bool,
) -> AttemptRecord:
    return AttemptRecord(
        step=step,
        status=status,
        author_id=author_id,
        strategy_id=strategy_id,
        suggested_id=draft.suggested_id,
        hypothesis_id=hypothesis_id,
        parent_hypothesis_id=draft.parent_hypothesis_id,
        expression=draft.expression,
        revision_reason=draft.rationale if is_revision else None,
        feedback=feedback,
        brier_score=evaluation.brier_score,
        log_loss=evaluation.log_loss,
        baseline_brier=evaluation.baseline_brier,
        improvement=evaluation.improvement,
        uncertainty=evaluation.uncertainty,
        experiment_id=evaluation.experiment_id,
        model_name=model_name,
        model_digest=model_digest,
        persisted=persisted,
    )


def _empty_attempt(
    step: str,
    status: str,
    author_id: str,
    strategy_id: str,
    reason: str,
    model_name: str,
    model_digest: str,
    parent_hypothesis_id: str | None,
    *,
    expression: str | None = None,
    suggested_id: str | None = None,
    feedback: dict[str, object] | None = None,
    persisted: bool = False,
) -> AttemptRecord:
    body = feedback or _failure_feedback(status, reason, revision_allowed=False, hypothesis_id=parent_hypothesis_id or "none")
    return AttemptRecord(
        step=step,
        status=status,
        author_id=author_id,
        strategy_id=strategy_id,
        suggested_id=suggested_id,
        hypothesis_id=None,
        parent_hypothesis_id=parent_hypothesis_id,
        expression=expression,
        revision_reason=None,
        feedback=body,
        brier_score=None,
        log_loss=None,
        baseline_brier=None,
        improvement=None,
        uncertainty=reason,
        experiment_id="",
        model_name=model_name,
        model_digest=model_digest,
        persisted=persisted,
    )


def _save_rejection(store: HypothesisStore | None, author_id: str, code: str, reason: str, created_at: datetime) -> bool:
    if store is None:
        return False
    store.note_attempt(
        kind="formula",
        author_id=author_id,
        succeeded=False,
        detail=f"{code}: {reason}"[:300],
        created_at=created_at,
    )
    return True


def _rollback(store: object | None, memory: object | None) -> None:
    for target in (store, memory):
        method = getattr(target, "rollback_open", None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            return
