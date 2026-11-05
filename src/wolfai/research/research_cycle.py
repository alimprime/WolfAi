"""Один исследовательский проход. По умолчанию dry-run.

Qwen вызывается не больше одного раза. Проверка формулы, дубликатов и
статуса выполняется отдельно от генерации. Validated Law не назначается.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from wolfai.data.gateway import HOLDOUT_START
from wolfai.research.budget import SearchBusy, exclusive_search
from wolfai.research.formula import FormulaError, evaluate, parse_formula
from wolfai.research.hypothesis_engine import HypothesisLifecycleError, HypothesisStore
from wolfai.research.hypothesis_generator import StructuredHypothesis
from wolfai.research.hypothesis_protocol import verify_model_output
from wolfai.research.math_screen import VERIFIER_ID, score_synthetic
from wolfai.research.memory_context import load_research_excerpt
from wolfai.research.ollama_adapter import OllamaAdapter
from wolfai.research.qwen_brain import (
    GENERATOR_VERSION,
    BrainContext,
    QwenResearchBrain,
)

MAX_CYCLE_HYPOTHESES = 1
STAGE_CODE = "3.0.0"
DEFAULT_BRIEF = (
    "Синтетическая ситуация: короткая волатильность и импульс известны только до решения. "
    "Рыночный вывод не делается."
)
DEFAULT_DATA_HASH = "synthetic-stage30"
_AUTHOR = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_FUTURE_YEARS = ("2022", "2023", "2024", "2025", "2026", "2027", "2028")
_MODEL_REJECTIONS = frozenset({
    "invalid_json",
    "unknown_feature",
    "unsafe_math",
    "too_complex",
    "bad_horizon",
    "feature_mismatch",
    "criteria_change",
    "time_boundary",
    "agent_isolation",
    "unknown_parent",
    "missing_feature",
    "calibration_refused",
})
_ALLOWED_HISTORY_START = date(2019, 1, 1)
_FORBIDDEN_FROM = date(2022, 1, 1)
_DEV_START = date(2021, 1, 1)
_DEV_END = date(2021, 12, 31)


@dataclass(frozen=True)
class HistoryPoint:
    decision_date: date
    features: dict[str, float]
    label: int


@dataclass(frozen=True)
class CycleReport:
    status: str
    message: str
    persisted: bool = False
    dry_run: bool = True
    author_id: str = ""
    model_name: str = ""
    model_digest: str = ""
    parameter_size: str = ""
    ollama_version: str = ""
    generator_version: str = GENERATOR_VERSION
    hypothesis_id: str | None = None
    suggested_id: str | None = None
    expression: str | None = None
    parent_hypothesis_id: str | None = None
    rejection_code: str | None = None
    rejection_reason: str | None = None
    rejection_saved: bool = False
    duplicate_of: str | None = None
    claim_status: str = "not_confirmed"
    validated_law: bool = False
    advantage_claim: bool = False
    graded_by: str = VERIFIER_ID
    attempts_used: int = 0
    attempt_count: int | None = None
    elapsed_seconds: float = 0.0
    rss_bytes: int = 0
    available_bytes: int | None = None
    synthetic: dict[str, object] | None = None
    internal_2019_2020: dict[str, object] | None = None
    dev_2021_skipped: int = 0
    fitted_on_2021: bool = False
    hypotheses_requested: int = 1
    model_calls: int = 0
    response_excerpt: str = ""


def reject_validated_law(claim_status: str, validated_law: bool) -> None:
    """Единственная точка, которая отклоняет подтверждённый закон."""
    normalized = claim_status.strip().lower()
    if validated_law or normalized == "supported" or "validated" in normalized:
        raise HypothesisLifecycleError("переход в Validated Law запрещён")


def reject_self_grade(author_id: str, graded_by: str) -> None:
    """Автор гипотезы не ставит оценку своему результату."""
    if not graded_by.strip() or graded_by == author_id or graded_by.startswith("qwen"):
        raise PermissionError("агент не оценивает собственный результат")


def run_research_cycle(
    client: object | None,
    *,
    author_id: str = "qwen-research-brain",
    market_brief: str = DEFAULT_BRIEF,
    store: HypothesisStore | None = None,
    memory: object | None = None,
    commit: bool = False,
    max_hypotheses: int = MAX_CYCLE_HYPOTHESES,
    temperature: float = 0.0,
    seed: int = 1729,
    created_at: datetime | None = None,
    data_hash: str = DEFAULT_DATA_HASH,
    period_start: date | None = None,
    period_end: date | None = None,
    fit_on_2021: bool = False,
    touches_holdout: bool = False,
    reads_after_2021: bool = False,
    prediction_ledger: object | None = None,
    history: Sequence[HistoryPoint] = (),
    available_features: Sequence[str] | None = None,
) -> CycleReport:
    """Один проход: контекст, одна гипотеза, проверка, при commit — запись."""

    def finish(status: str, message: str, **kwargs: object) -> CycleReport:
        payload: dict[str, object] = {
            "author_id": author_id,
            "dry_run": not commit,
            "attempt_count": store.attempt_count() if store is not None else None,
            "hypotheses_requested": max_hypotheses if type(max_hypotheses) is int else 0,
        }
        payload.update(kwargs)
        return CycleReport(status=status, message=message, **payload)  # type: ignore[arg-type]

    if type(max_hypotheses) is not int or max_hypotheses != MAX_CYCLE_HYPOTHESES:
        return finish("limit", "за один запуск допускается одна гипотеза")
    if prediction_ledger is not None:
        return finish("ledger_forbidden", "журнал прогнозов генератору не передаётся")
    if _AUTHOR.fullmatch(author_id) is None:
        return finish("aborted", "идентификатор автора имеет недопустимый вид")
    boundary = _request_boundary(
        market_brief=market_brief,
        period_start=period_start,
        period_end=period_end,
        fit_on_2021=fit_on_2021,
        touches_holdout=touches_holdout,
        reads_after_2021=reads_after_2021,
        history=history,
    )
    if boundary is not None:
        return finish("time_boundary", boundary)
    if commit and store is None:
        return finish("store_unavailable", "нет журнала гипотез для записи")
    if commit and memory is None:
        return finish("memory_unavailable", "нет research memory для записи")
    when = created_at or datetime.now(timezone.utc)
    if when.tzinfo is None or when.utcoffset() != timedelta(0):
        return finish("aborted", "время цикла должно быть в UTC")
    if commit and not data_hash.strip():
        return finish("aborted", "хэш исходных данных обязателен")
    try:
        excerpt, memory_error = load_research_excerpt(memory, store, author_id=author_id)
    except Exception as error:
        _rollback(store, memory)
        return finish("aborted", f"{error.__class__.__name__}: {error}")
    if memory_error is not None or excerpt is None:
        return finish("memory_unavailable", memory_error or "research memory недоступна")
    try:
        with exclusive_search("qwen"):
            return _one_pass(
                finish,
                client=client,
                author_id=author_id,
                market_brief=market_brief,
                store=store,
                memory=memory,
                commit=commit,
                temperature=temperature,
                seed=seed,
                created_at=when,
                data_hash=data_hash.strip(),
                fit_on_2021=fit_on_2021,
                touches_holdout=touches_holdout,
                reads_after_2021=reads_after_2021,
                prediction_ledger=prediction_ledger,
                history=history,
                excerpt=excerpt,
                available_features=available_features,
            )
    except SearchBusy as error:
        return finish("search_busy", str(error))
    except Exception as error:
        _rollback(store, memory)
        return finish("aborted", f"{error.__class__.__name__}: {error}")


def run_local_smoke(*, timeout: float = 120.0) -> CycleReport:
    """Один локальный вызов уже установленной модели. Скачивание не выполняется."""
    adapter = OllamaAdapter(model="qwen3.5:4b", timeout=timeout)
    return run_research_cycle(adapter, market_brief=DEFAULT_BRIEF, commit=False, temperature=0.0)


def _one_pass(finish, **kwargs: object) -> CycleReport:
    client = kwargs["client"]
    author_id = str(kwargs["author_id"])
    store: HypothesisStore | None = kwargs["store"]  # type: ignore[assignment]
    memory = kwargs["memory"]
    commit = bool(kwargs["commit"])
    excerpt = kwargs["excerpt"]
    history: Sequence[HistoryPoint] = kwargs["history"]  # type: ignore[assignment]
    created_at: datetime = kwargs["created_at"]  # type: ignore[assignment]
    brain = QwenResearchBrain(
        author_id,
        temperature=float(kwargs["temperature"]),  # type: ignore[arg-type]
        seed=int(kwargs["seed"]),  # type: ignore[arg-type]
    )
    context = BrainContext(
        author_id=author_id,
        market_brief=str(kwargs["market_brief"]),
        excerpt=excerpt,  # type: ignore[arg-type]
        touches_holdout=bool(kwargs["touches_holdout"]),
        reads_after_2021=bool(kwargs["reads_after_2021"]),
        fit_on_2021=bool(kwargs["fit_on_2021"]),
        prediction_ledger=kwargs["prediction_ledger"],
    )
    turn = brain.propose(context, client)
    common = {
        "model_name": turn.model_name,
        "model_digest": turn.model_digest,
        "parameter_size": turn.parameter_size,
        "ollama_version": turn.ollama_version,
        "generator_version": turn.generator_version,
        "elapsed_seconds": turn.elapsed_seconds,
        "rss_bytes": turn.rss_bytes,
        "available_bytes": turn.available_bytes,
        "model_calls": turn.model_calls,
        "attempts_used": turn.attempts_used,
        "response_excerpt": (turn.content or "")[:400],
    }
    if turn.status != "ok" or not turn.content:
        return finish(turn.status, turn.message, **common)
    available = kwargs["available_features"]
    decision = verify_model_output(
        turn.content,
        known=excerpt.known,  # type: ignore[attr-defined]
        available_features=available if available is not None else _default_features(),
    )
    if not decision.accepted or decision.draft is None:
        if decision.code == "duplicate":
            return finish(
                "duplicate",
                decision.reason,
                expression=None if decision.draft is None else decision.draft.expression,
                duplicate_of=decision.duplicate_of,
                rejection_code="duplicate",
                rejection_reason=decision.reason,
                **common,
            )
        saved = False
        if decision.code in _MODEL_REJECTIONS:
            saved = _save_rejection(store, author_id, decision.code, decision.reason, created_at)
        return finish(
            decision.code,
            decision.reason,
            rejection_code=decision.code,
            rejection_reason=decision.reason,
            rejection_saved=saved,
            **common,
        )
    draft = decision.draft
    if draft.parent_hypothesis_id is not None:
        if store is None or not _exists(store, draft.parent_hypothesis_id):
            saved = _save_rejection(
                store,
                author_id,
                "unknown_parent",
                "parent_hypothesis_id не найден в журнале гипотез",
                created_at,
            )
            return finish(
                "unknown_parent",
                "parent_hypothesis_id не найден в журнале гипотез",
                rejection_code="unknown_parent",
                rejection_reason="parent_hypothesis_id не найден в журнале гипотез",
                rejection_saved=saved,
                suggested_id=draft.suggested_id,
                expression=draft.expression,
                **common,
            )
    usable, skipped = _split_history(history)
    try:
        synthetic = score_synthetic(draft.expression)
        internal = _score_history(draft.expression, usable) if usable else None
    except FormulaError as error:
        saved = _save_rejection(store, author_id, "unsafe_math", str(error), created_at)
        return finish(
            "unsafe_math",
            str(error),
            rejection_code="unsafe_math",
            rejection_reason=str(error),
            rejection_saved=saved,
            expression=draft.expression,
            **common,
        )
    reject_validated_law("not_confirmed", False)
    reject_self_grade(author_id, VERIFIER_ID)
    if not commit:
        return finish(
            "dry_run",
            "проверка прошла, запись не выполнялась",
            persisted=False,
            dry_run=True,
            suggested_id=draft.suggested_id,
            expression=draft.expression,
            parent_hypothesis_id=draft.parent_hypothesis_id,
            synthetic=synthetic,
            internal_2019_2020=internal,
            dev_2021_skipped=skipped,
            fitted_on_2021=False,
            **common,
        )
    try:
        hypothesis_id, state = _register(
            store=store,
            memory=memory,
            author_id=author_id,
            draft=draft,
            synthetic=synthetic,
            internal=internal,
            model_name=turn.model_name,
            model_digest=turn.model_digest,
            created_at=created_at,
            data_hash=str(kwargs["data_hash"]),
            seed=int(kwargs["seed"]),  # type: ignore[arg-type]
            skipped_2021=skipped,
        )
    except Exception as error:
        _rollback(store, memory)
        return finish(
            "aborted",
            f"{error.__class__.__name__}: {error}",
            expression=draft.expression,
            synthetic=synthetic,
            **common,
        )
    if state == "duplicate":
        return finish(
            "duplicate",
            "дубликат уже проверенной формулы",
            duplicate_of=hypothesis_id,
            expression=draft.expression,
            synthetic=synthetic,
            **common,
        )
    return finish(
        "stored",
        "гипотеза записана со статусом not_confirmed",
        persisted=True,
        dry_run=False,
        hypothesis_id=hypothesis_id,
        suggested_id=draft.suggested_id,
        expression=draft.expression,
        parent_hypothesis_id=draft.parent_hypothesis_id,
        synthetic=synthetic,
        internal_2019_2020=internal,
        dev_2021_skipped=skipped,
        fitted_on_2021=False,
        **common,
    )


def _register(
    *,
    store: HypothesisStore | None,
    memory: object,
    author_id: str,
    draft,
    synthetic: dict[str, object],
    internal: dict[str, object] | None,
    model_name: str,
    model_digest: str,
    created_at: datetime,
    data_hash: str,
    seed: int,
    skipped_2021: int,
) -> tuple[str, str]:
    if store is None:
        raise HypothesisLifecycleError("нет журнала гипотез для записи")
    reject_validated_law("not_confirmed", False)
    applicability = {
        "market": "synthetic",
        "regimes": ["any"] if draft.expected_regime == "any" else [draft.expected_regime],
        "btc_claim": False,
        "generator_model": model_name or "unspecified",
        "generator_version": GENERATOR_VERSION,
        "generator_digest": model_digest or "unspecified",
        "evidence_role": "not_confirmed",
    }
    proposal = StructuredHypothesis(
        author_id=author_id,
        name=draft.suggested_id,
        description=draft.description,
        creation_reason=draft.rationale,
        features=draft.features,
        horizon_days=draft.horizon_days,
        expression=draft.expression,
        applicability=applicability,
        parent_hypothesis_id=draft.parent_hypothesis_id,
    )
    hypothesis_id, state = store.create(
        author_id=proposal.author_id,
        name=proposal.name,
        description=proposal.description,
        creation_reason=proposal.creation_reason,
        features=proposal.features,
        horizon_days=proposal.horizon_days,
        expression=proposal.expression,
        applicability=proposal.applicability,
        created_at=created_at,
        data_hash=data_hash,
        parent_hypothesis_id=proposal.parent_hypothesis_id,
    )
    if state == "duplicate":
        return hypothesis_id, state
    experiment = {
        "claim_status": "not_confirmed",
        "validated_law": False,
        "advantage_claim": False,
        "graded_by": VERIFIER_ID,
        "author_id": author_id,
        "generator_model": applicability["generator_model"],
        "generator_version": GENERATOR_VERSION,
        "expression": draft.expression,
        "suggested_hypothesis_id": draft.suggested_id,
        "synthetic": synthetic,
        "internal_2019_2020": internal,
        "dev_2021_skipped": skipped_2021,
        "fitted_on_2021": False,
        "limitations": draft.limitations,
        "test_plan": draft.test_plan,
    }
    reject_validated_law(str(experiment["claim_status"]), bool(experiment["validated_law"]))
    store.add_experiment(hypothesis_id, experiment, when=created_at)
    record = getattr(memory, "record", None)
    if not callable(record):
        raise HypothesisLifecycleError("research memory не принимает запись")
    record(_memory_payload(
        draft=draft,
        synthetic=synthetic,
        author_id=author_id,
        model_name=str(applicability["generator_model"]),
        model_digest=str(applicability["generator_digest"]),
        data_hash=data_hash,
        seed=seed,
        hypothesis_id=hypothesis_id,
        skipped_2021=skipped_2021,
    ))
    stored = store.get(hypothesis_id)
    if stored.claim_status != "not_confirmed" or store.confirms_law(hypothesis_id):
        raise HypothesisLifecycleError("запись получила запрещённый статус")
    return hypothesis_id, state


def _memory_payload(
    *,
    draft,
    synthetic: dict[str, object],
    author_id: str,
    model_name: str,
    model_digest: str,
    data_hash: str,
    seed: int,
    hypothesis_id: str,
    skipped_2021: int,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "config_hash": model_digest,
        "dataset_fingerprint": data_hash,
        "features": list(draft.features),
        "label_definition": "синтетическая метка: 1 если momentum_7 > 0; это не исход BTC",
        "horizon_days": draft.horizon_days,
        "train_start": "2019-01-01T00:00:00+00:00",
        "train_end": "2020-12-31T00:00:00+00:00",
        "dev_start": "2021-01-01T00:00:00+00:00",
        "dev_end": "2021-12-31T00:00:00+00:00",
        "seed": seed,
        "evidence_role": "synthetic",
        "claim_status": "not_confirmed",
        "validated_law": False,
        "advantage_claim": False,
        "fitted": False,
        "fit_window": "none",
        "diagnostic_2021": "DEV_ONLY_NOT_USED",
        "dev_2021_skipped": skipped_2021,
        "expression": draft.expression,
        "brier": synthetic.get("brier"),
        "observations": synthetic.get("observations"),
        "limitations": draft.limitations,
        "author_id": author_id,
        "hypothesis_id": hypothesis_id,
        "suggested_hypothesis_id": draft.suggested_id,
        "parent_hypothesis_id": draft.parent_hypothesis_id,
        "generator_version": GENERATOR_VERSION,
        "graded_by": VERIFIER_ID,
        "code_version": STAGE_CODE,
        "btc_claim": False,
    }


def _request_boundary(
    *,
    market_brief: str,
    period_start: date | None,
    period_end: date | None,
    fit_on_2021: bool,
    touches_holdout: bool,
    reads_after_2021: bool,
    history: Sequence[HistoryPoint],
) -> str | None:
    if touches_holdout:
        return "закрытый holdout не передаётся"
    if reads_after_2021:
        return "данные после 2021 года не используются для нового подбора"
    if fit_on_2021:
        return "2021 год остаётся DEV_ONLY и не используется для подбора"
    lowered = market_brief.lower()
    if "holdout" in lowered or any(year in market_brief for year in _FUTURE_YEARS):
        return "описание ситуации содержит закрытые или поздние данные"
    if period_start is not None and period_start >= _FORBIDDEN_FROM:
        return "период захватывает даты после 2021 года"
    if period_end is not None and period_end >= _FORBIDDEN_FROM:
        return "период захватывает даты после 2021 года"
    if period_start is not None and period_end is not None and period_end < period_start:
        return "период перепутан"
    for point in history:
        if point.label not in (0, 1):
            return "исход исторической точки должен быть 0 или 1"
        if point.decision_date >= _FORBIDDEN_FROM or point.decision_date >= HOLDOUT_START:
            return "историческая точка лежит после 2021 года"
        if point.decision_date < _ALLOWED_HISTORY_START:
            return "историческая точка раньше разрешённого окна 2019–2020"
    return None


def _split_history(history: Sequence[HistoryPoint]) -> tuple[list[HistoryPoint], int]:
    usable: list[HistoryPoint] = []
    skipped = 0
    for point in history:
        if _DEV_START <= point.decision_date <= _DEV_END:
            skipped += 1
            continue
        usable.append(point)
    return usable, skipped


def _score_history(expression: str, points: Sequence[HistoryPoint]) -> dict[str, object]:
    tree = parse_formula(expression)
    brier_sum = 0.0
    base_sum = 0.0
    for point in points:
        probability = evaluate(tree, point.features)
        brier_sum += (probability - point.label) ** 2
        base_sum += (0.5 - point.label) ** 2
    count = len(points)
    return {
        "period": "2019-2020",
        "observations": count,
        "brier": round(brier_sum / count, 6),
        "constant_half_brier": round(base_sum / count, 6),
        "claim_status": "not_confirmed",
        "validated_law": False,
        "advantage_claim": False,
        "fitted_on_2021": False,
        "graded_by": VERIFIER_ID,
    }


def _save_rejection(
    store: HypothesisStore | None,
    author_id: str,
    code: str,
    reason: str,
    created_at: datetime,
) -> bool:
    if store is None or code not in _MODEL_REJECTIONS:
        return False
    store.note_attempt(
        kind="formula",
        author_id=author_id,
        succeeded=False,
        detail=f"{code}: {reason}"[:300],
        created_at=created_at,
    )
    return True


def _exists(store: HypothesisStore, hypothesis_id: str) -> bool:
    try:
        store.get(hypothesis_id)
    except KeyError:
        return False
    return True


def _default_features() -> tuple[str, ...]:
    from wolfai.research.formula import FEATURES

    return FEATURES


def _rollback(store: object | None, memory: object | None) -> None:
    for target in (store, memory):
        method = getattr(target, "rollback_open", None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            return
