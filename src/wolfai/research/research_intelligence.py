"""Цикл исследования 3.2: план, формула, ремонт синтаксиса, счёт, отзыв, пересмотр.

По умолчанию dry-run. Один клиент, последовательные вызовы, заранее заданный потолок.
Синтаксическая ошибка не считается успешным научным экспериментом.
Снижение Brier на внутреннем окне не становится независимым подтверждением.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from wolfai.research.budget import SearchBusy, exclusive_search
from wolfai.research.evaluation_protocol import (
    CLOSED_FROM,
    PROTOCOL_VERSION,
    Evaluation,
    StudyPoint,
    evaluate_hypothesis,
    synthetic_study_points,
)
from wolfai.research.grammar_contract import (
    FIXED_EVALUATION_METHOD,
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    GENERATOR_VERSION,
    INTELLIGENCE_VERSION,
    MAX_EVALUATED_FORMULAS,
    MAX_GENERATION_REPAIR_CALLS,
    MAX_MODEL_CALLS,
    MAX_SCIENTIFIC_REVISIONS,
    MAX_SYNTAX_REPAIRS,
    STRATEGY_ID,
    scan_research_text,
)
from wolfai.research.hypothesis_engine import HypothesisStore, formula_key
from wolfai.research.hypothesis_protocol import KnownFormula
from wolfai.research.intelligence_log import IntelligenceLog
from wolfai.research.math_screen import VERIFIER_ID
from wolfai.research.memory_context import MemoryExcerpt, load_research_excerpt
from wolfai.research.memory_retrieval import memory_as_data, select_relevant_records
from wolfai.research.ollama_adapter import OllamaAdapter, default_memory_guard, process_rss_bytes
from wolfai.research.qwen_brain import GENERATOR_VERSION as QWEN_GENERATOR_VERSION
from wolfai.research.research_cycle import reject_self_grade, reject_validated_law
from wolfai.research.research_plan import QwenPlanner, ResearchPlan, validate_research_plan
from wolfai.research.research_ports import (
    GenerateContext,
    HypothesisGenerator,
    ModelTurn,
    PlanContext,
    RepairContext,
    ResearchPlanner,
    RevisionContext,
    ScientificReviser,
    SyntaxRepairAgent,
)
from wolfai.research.schema_generation import QwenGenerator
from wolfai.research.scientific_feedback import build_scientific_feedback, classify_evaluation
from wolfai.research.scientific_revision import GateResult, QwenReviser, validate_generation, validate_revision_payload
from wolfai.research.syntax_repair import (
    EditAssessment,
    QwenRepairer,
    build_repair_brief,
    build_syntax_feedback,
    canonical_identity,
    classify_edit,
    peek_expression,
)

MAX_BRIEF_CHARS = 500
_NO_LAW = "not_confirmed"


@dataclass(frozen=True)
class IntelligenceAttempt:
    step: str
    status: str
    author_id: str
    strategy_id: str
    generator_version: str
    suggested_id: str | None = None
    hypothesis_id: str | None = None
    parent_hypothesis_id: str | None = None
    expression: str | None = None
    plan: dict[str, object] | None = None
    repair: dict[str, object] | None = None
    scientific_feedback: dict[str, object] | None = None
    syntax_feedback: dict[str, object] | None = None
    semantic_change: bool = False
    equivalent_to_previous: bool | None = None
    separate_candidate: bool = False
    syntactic_success: bool = False
    execution_success: bool = False
    development_improvement: bool = False
    independent_validation: bool = False
    outcome: str = "formula_invalid"
    brier_score: float | None = None
    log_loss: float | None = None
    baseline_brier: float | None = None
    improvement: float | None = None
    uncertainty: str = ""
    experiment_id: str = ""
    test_dates: tuple[str, ...] = ()
    model_name: str = ""
    model_digest: str = ""
    persisted: bool = False
    revision_reason: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class IntelligenceReport:
    status: str
    message: str
    dry_run: bool
    author_id: str
    strategy_id: str
    generator_version: str
    attempts: tuple[IntelligenceAttempt, ...]
    model_calls: int
    plan_calls: int
    generation_repair_calls: int
    syntax_repairs: int
    revisions: int
    hypotheses: int
    unique_expressions: int
    evaluated_expressions: tuple[str, ...]
    elapsed_seconds: float
    rss_bytes: int
    claim_status: str = _NO_LAW
    validated_law: bool = False
    independent_validation: bool = False
    multiple_testing_note: str = ""
    comparison: tuple[dict[str, object], ...] = ()
    generation_config: dict[str, object] = field(default_factory=dict)
    point_estimate_better: bool | None = None
    improvement_is_internal_only: bool | None = None
    code_version: str = INTELLIGENCE_VERSION


def run_research_intelligence(
    client: object | None,
    *,
    author_id: str = "qwen-research-brain",
    strategy_id: str = STRATEGY_ID,
    market_brief: str = "Синтетический ряд известен только до решения. Рыночный вывод не делается.",
    store: HypothesisStore | None = None,
    memory: object | None = None,
    journal: IntelligenceLog | None = None,
    points: Sequence[StudyPoint] | None = None,
    commit: bool = False,
    created_at: datetime | None = None,
    deadline_seconds: float = 180.0,
    now: Callable[[], float] | None = None,
    memory_guard: Callable[[], str | None] | None = None,
    data_hash: str = "synthetic-2019-2020",
    planner: ResearchPlanner | None = None,
    generator: HypothesisGenerator | None = None,
    repairer: SyntaxRepairAgent | None = None,
    reviser: ScientificReviser | None = None,
    temperature: float = GENERATION_TEMPERATURE,
) -> IntelligenceReport:
    """Один проход. Потолок: 1 план, 5 генераций и ремонтов, 2 ремонта, 2 пересмотра, 3 формулы."""
    clock = now or time.perf_counter
    guard = default_memory_guard if memory_guard is None else memory_guard
    started = clock()
    series = tuple(synthetic_study_points() if points is None else points)
    when = created_at or datetime.now(timezone.utc)

    def finish(status: str, message: str, *, attempts: tuple[IntelligenceAttempt, ...] = (), calls: int = 0) -> IntelligenceReport:
        return IntelligenceReport(
            status=status,
            message=message,
            dry_run=not commit,
            author_id=author_id,
            strategy_id=strategy_id,
            generator_version=GENERATOR_VERSION,
            attempts=attempts,
            model_calls=calls,
            plan_calls=0,
            generation_repair_calls=0,
            syntax_repairs=0,
            revisions=0,
            hypotheses=0,
            unique_expressions=0,
            evaluated_expressions=(),
            elapsed_seconds=round(clock() - started, 6),
            rss_bytes=process_rss_bytes(),
            multiple_testing_note=_multiple_testing_note(0),
            generation_config=_config(client),
            independent_validation=False,
            validated_law=False,
            claim_status=_NO_LAW,
        )

    if not author_id.strip() or not strategy_id.strip():
        return finish("aborted", "нужны author_id и strategy_id")
    if when.tzinfo is None or when.utcoffset() != timedelta(0):
        return finish("aborted", "время цикла должно быть в UTC")
    if not market_brief.strip() or len(market_brief) > MAX_BRIEF_CHARS:
        return finish("context_rejected", "описание ситуации пустое или длиннее бюджета")
    brief_problem = scan_research_text(market_brief)
    if brief_problem is not None and brief_problem[0] == "time_boundary":
        return finish("time_boundary", brief_problem[1])
    if commit and store is None:
        return finish("store_unavailable", "нет журнала гипотез для записи")
    if commit and memory is None:
        return finish("memory_unavailable", "нет research memory для записи")
    if commit and journal is None:
        return finish("journal_unavailable", "нет журнала исследовательского цикла для записи")
    blocked = guard()
    if blocked:
        return finish("resource_limited", blocked)
    boundary = _series_boundary(series)
    if boundary is not None:
        return finish("time_boundary", boundary)
    try:
        excerpt, memory_error = load_research_excerpt(memory, store, author_id=author_id)
    except Exception as error:
        _rollback(store, memory, journal)
        return finish("aborted", f"{error.__class__.__name__}: {error}")
    if memory_error is not None or excerpt is None:
        return finish("memory_unavailable", memory_error or "research memory недоступна")
    if isinstance(client, OllamaAdapter):
        client._release_wait_seconds = 20.0
    agents = (
        planner or QwenPlanner(client, temperature=temperature),
        generator or QwenGenerator(client, temperature=temperature),
        repairer or QwenRepairer(client, temperature=temperature),
        reviser or QwenReviser(client, temperature=temperature),
    )
    try:
        with exclusive_search("qwen"):
            session = _Session(
                client=client,
                author_id=author_id,
                strategy_id=strategy_id,
                market_brief=market_brief,
                store=store,
                memory=memory,
                journal=journal,
                points=series,
                commit=commit,
                created_at=when,
                data_hash=data_hash,
                excerpt=excerpt,
                clock=clock,
                started=started,
                deadline=deadline_seconds,
                guard=guard,
                planner=agents[0],
                generator=agents[1],
                repairer=agents[2],
                reviser=agents[3],
                temperature=temperature,
            )
            return session.run()
    except SearchBusy as error:
        return finish("search_busy", str(error))
    except Exception as error:
        _rollback(store, memory, journal)
        return finish("aborted", f"{error.__class__.__name__}: {error}")


class _Session:
    def __init__(self, **kwargs: object) -> None:
        self.client = kwargs["client"]
        self.author_id = str(kwargs["author_id"])
        self.strategy_id = str(kwargs["strategy_id"])
        self.market_brief = str(kwargs["market_brief"])
        self.store: HypothesisStore | None = kwargs["store"]  # type: ignore[assignment]
        self.memory = kwargs["memory"]
        self.journal: IntelligenceLog | None = kwargs["journal"]  # type: ignore[assignment]
        self.points: tuple[StudyPoint, ...] = kwargs["points"]  # type: ignore[assignment]
        self.commit = bool(kwargs["commit"])
        self.created_at: datetime = kwargs["created_at"]  # type: ignore[assignment]
        self.data_hash = str(kwargs["data_hash"])
        self.excerpt: MemoryExcerpt = kwargs["excerpt"]  # type: ignore[assignment]
        self.clock: Callable[[], float] = kwargs["clock"]  # type: ignore[assignment]
        self.started = float(kwargs["started"])  # type: ignore[arg-type]
        self.deadline = float(kwargs["deadline"])  # type: ignore[arg-type]
        self.guard: Callable[[], str | None] = kwargs["guard"]  # type: ignore[assignment]
        self.planner: ResearchPlanner = kwargs["planner"]  # type: ignore[assignment]
        self.generator: HypothesisGenerator = kwargs["generator"]  # type: ignore[assignment]
        self.repairer: SyntaxRepairAgent = kwargs["repairer"]  # type: ignore[assignment]
        self.reviser: ScientificReviser = kwargs["reviser"]  # type: ignore[assignment]
        self.temperature = float(kwargs["temperature"])  # type: ignore[arg-type]
        self.attempts: list[IntelligenceAttempt] = []
        self.known: list[KnownFormula] = list(self.excerpt.known)
        self.calls = 0
        self.plan_calls = 0
        self.gen_calls = 0
        self.syntax_repairs = 0
        self.revisions = 0
        self.generate_calls = 0
        self.evaluated: list[str] = []
        self.seen: set[tuple[object, ...]] = set()
        self.plan: ResearchPlan | None = None
        self.stopped: tuple[str, str] | None = None
        self.final: tuple[str, str] | None = None
        self.syntax_failed = False
        self.model_name = ""
        self.model_digest = ""

    def run(self) -> IntelligenceReport:
        plan = self._plan()
        if plan is not None and self.stopped is None and self.final is None:
            current = self._generate(plan)
            if current is not None and current.status == "duplicate" and self.final is None:
                self.final = ("duplicate", current.detail or "дубликат уже проверенной формулы")
            self._revise(current)
        if self.final is None and self.stopped is not None:
            self.final = self.stopped
        if self.final is None:
            if self.syntax_failed and not any(item.execution_success for item in self.attempts):
                self.final = ("syntax_unrepaired", "формула не прошла синтаксическую проверку")
            elif self.commit and any(item.persisted for item in self.attempts):
                self.final = ("stored", "цикл завершён в заданном бюджете")
            else:
                self.final = ("dry_run", "цикл завершён в заданном бюджете")
        status, message = self.final
        return self._report(status, message)

    def _plan(self) -> ResearchPlan | None:
        turn = self._call("plan", lambda: self.planner.plan(PlanContext(
            market_brief=self.market_brief,
            untrusted_memory=self._memory_payload(),
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
        )))
        if turn.status != "ok" or not turn.content:
            self._log_turn("plan", turn, parent_id=None)
            return None
        decision = validate_research_plan(turn.content)
        self._log_plan(decision, turn)
        if decision.accepted and decision.plan is not None:
            self.plan = decision.plan
            return decision.plan
        if decision.code != "invalid_json" or self.stopped is not None:
            self.final = ("plan_rejected", decision.reason)
            return None
        brief = build_repair_brief(decision.code, decision.reason, None)
        if brief is None or self.syntax_repairs >= MAX_SYNTAX_REPAIRS or not self._room("repair"):
            self.final = ("plan_rejected", decision.reason)
            return None
        self.syntax_repairs += 1
        repair = self._call("repair", lambda: self.repairer.repair(RepairContext(
            brief=brief.as_dict(),
            mode="plan",
            approved_plan=None,
            parent_hypothesis_id=None,
            untrusted_excerpt=(turn.content or "")[:500],
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
        )))
        if repair.status != "ok" or not repair.content:
            self._log_turn("syntax_repair", repair, parent_id=None)
            return None
        repaired = validate_research_plan(repair.content)
        self._log_plan(repaired, repair, brief=brief.as_dict())
        if repaired.plan is None:
            self.final = ("plan_rejected", repaired.reason)
            return None
        self.plan = repaired.plan
        return repaired.plan

    def _generate(self, plan: ResearchPlan) -> IntelligenceAttempt | None:
        if self.stopped is not None or not self._room("generate"):
            if self.final is None:
                self.final = self.stopped or ("budget_exhausted", "исчерпан заранее заданный бюджет обращений к модели")
            return None
        before = self.calls
        turn = self._call("generate", lambda: self.generator.generate(GenerateContext(
            plan=plan.as_dict(),
            untrusted_memory=self._memory_payload(plan),
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
        )))
        if self.calls > before:
            self.generate_calls += 1
        if turn.status != "ok" or not turn.content:
            self._log_turn("generate", turn, parent_id=None)
            return None
        return self._resolve(turn.content, step="generate", mode="hypothesis", is_revision=False, parent_id=None, parent_expression=None)

    def _revise(self, current: IntelligenceAttempt | None) -> None:
        while self._can_revise(current):
            assert current is not None
            assert current.hypothesis_id is not None
            assert current.expression is not None
            assert current.scientific_feedback is not None
            assert self.plan is not None
            parent_id = current.hypothesis_id
            parent_expression = current.expression
            feedback = dict(current.scientific_feedback)
            before = self.calls
            turn = self._call("revision", lambda: self.reviser.revise(RevisionContext(
                plan=self.plan.as_dict() if self.plan is not None else {},
                parent_hypothesis_id=parent_id,
                parent_expression=parent_expression,
                scientific_feedback=feedback,
                untrusted_memory=self._memory_payload(self.plan, expression=parent_expression),
                author_id=self.author_id,
                strategy_id=self.strategy_id,
                generator_version=GENERATOR_VERSION,
            )))
            if self.calls == before:
                return
            self.revisions += 1
            if turn.status != "ok" or not turn.content:
                self._log_turn("scientific_revision", turn, parent_id=parent_id)
                return
            nxt = self._resolve(
                turn.content,
                step="scientific_revision",
                mode="revision",
                is_revision=True,
                parent_id=parent_id,
                parent_expression=parent_expression,
            )
            if self.stopped is not None or nxt is None:
                return
            if nxt.status == "duplicate":
                self.final = ("duplicate", nxt.detail or "дубликат уже проверенной формулы")
                return
            if nxt.status == "syntax_unrepaired":
                self.syntax_failed = True
                continue
            if nxt.execution_success:
                current = nxt

    def _can_revise(self, current: IntelligenceAttempt | None) -> bool:
        return (
            self.final is None
            and self.stopped is None
            and current is not None
            and current.execution_success
            and current.hypothesis_id is not None
            and current.expression is not None
            and current.scientific_feedback is not None
            and self.revisions < MAX_SCIENTIFIC_REVISIONS
            and len(self.evaluated) < MAX_EVALUATED_FORMULAS
            and self._room("revision")
        )

    def _resolve(
        self,
        content: str,
        *,
        step: str,
        mode: str,
        is_revision: bool,
        parent_id: str | None,
        parent_expression: str | None,
    ) -> IntelligenceAttempt:
        decision = self._gate(content, is_revision=is_revision, parent_id=parent_id, parent_expression=parent_expression)
        assessment: EditAssessment | None = None
        source = content
        while not decision.accepted:
            attempt = self._log_gate(step if assessment is None else "syntax_repair", decision, assessment)
            brief = build_repair_brief(decision.code, decision.reason, decision.expression)
            if brief is None or self.syntax_repairs >= MAX_SYNTAX_REPAIRS or not self._room("repair") or self.stopped is not None:
                if brief is not None and decision.code in {"invalid_json", "unsafe_math", "too_complex", "unknown_feature", "feature_mismatch", "missing_feature"}:
                    marked = replace(attempt, status="syntax_unrepaired", detail=f"{decision.code}: {decision.reason}"[:300])
                    self.attempts[-1] = marked
                    self._remember(marked)
                    self.syntax_failed = True
                    return marked
                return attempt
            self.syntax_repairs += 1
            repair = self._call("repair", lambda: self.repairer.repair(RepairContext(
                brief=brief.as_dict(),
                mode=mode,
                approved_plan=None if self.plan is None else self.plan.as_dict(),
                parent_hypothesis_id=parent_id,
                untrusted_excerpt=source[:500],
                author_id=self.author_id,
                strategy_id=self.strategy_id,
                generator_version=GENERATOR_VERSION,
            )))
            if repair.status != "ok" or not repair.content:
                self._log_turn("syntax_repair", repair, parent_id=parent_id)
                return attempt
            assessment = classify_edit(decision.expression or peek_expression(source) or "", peek_expression(repair.content) or "")
            source = repair.content
            decision = self._gate(source, is_revision=is_revision, parent_id=parent_id, parent_expression=parent_expression)
            step = "syntax_repair"
        return self._admit(decision, step=step, is_revision=is_revision, parent_expression=parent_expression, assessment=assessment)

    def _admit(
        self,
        decision: GateResult,
        *,
        step: str,
        is_revision: bool,
        parent_expression: str | None,
        assessment: EditAssessment | None,
    ) -> IntelligenceAttempt:
        draft = decision.draft
        if draft is None:
            return self._log_gate(step, decision, assessment)
        identity = _identity(draft.expression, draft.features, draft.horizon_days)
        if identity in self.seen:
            rejected = replace(
                decision,
                accepted=False,
                code="duplicate",
                reason="дубликат уже проверенной формулы",
            )
            return self._log_gate(step, rejected, assessment)
        if len(self.evaluated) >= MAX_EVALUATED_FORMULAS:
            self.final = ("budget_exhausted", "исчерпан бюджет формул, допущенных к оценке")
            rejected = replace(decision, accepted=False, code="budget_exhausted", reason=self.final[1])
            return self._log_gate(step, rejected, assessment)
        self.seen.add(identity)
        self.evaluated.append(draft.expression)
        self._remember_formula(draft, parent_expression)
        evaluation = evaluate_hypothesis(draft.expression, draft.features, draft.horizon_days, self.points)
        semantic_change = (assessment is not None and assessment.kind == "semantic_change") or decision.edit_kind == "semantic_change"
        equivalent = False if semantic_change else (None if assessment is None and not decision.edit_kind else assessment.equivalent if assessment else False)
        separate = bool(assessment is not None and assessment.equivalent is False and assessment.kind != "unparsable")
        hypothesis_id = formula_key(draft.expression, draft.features, draft.horizon_days, _applicability(self.strategy_id, draft.expected_regime))
        if not evaluation.accepted:
            return self._append(IntelligenceAttempt(
                step=step,
                status=evaluation.code,
                author_id=self.author_id,
                strategy_id=self.strategy_id,
                generator_version=GENERATOR_VERSION,
                suggested_id=draft.suggested_id,
                hypothesis_id=hypothesis_id,
                parent_hypothesis_id=draft.parent_hypothesis_id,
                expression=draft.expression,
                semantic_change=semantic_change,
                equivalent_to_previous=equivalent,
                separate_candidate=separate,
                syntactic_success=True,
                execution_success=False,
                outcome=classify_evaluation(evaluation, nodes=draft.math_report.nodes),
                uncertainty=evaluation.uncertainty,
                experiment_id=evaluation.experiment_id,
                model_name=self.model_name,
                model_digest=self.model_digest,
                revision_reason=_revision_reason(decision, is_revision),
                detail=evaluation.reason,
            ))
        version = sum(1 for item in self.attempts if item.execution_success) + 1
        feedback = build_scientific_feedback(
            evaluation,
            hypothesis_id=hypothesis_id,
            formula_version=version,
            nodes=draft.math_report.nodes,
            applicability=_applicability_text(self.strategy_id, draft.expected_regime),
            expression=draft.expression,
        )
        persisted = False
        status = "dry_run"
        if self.commit:
            stored = self._persist(draft, evaluation, hypothesis_id, decision, parent_expression)
            if stored == "duplicate":
                status = "duplicate"
                self.final = ("duplicate", "дубликат уже проверенной формулы")
            elif stored is True:
                status = "stored"
                persisted = True
        return self._append(IntelligenceAttempt(
            step=step,
            status=status,
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
            suggested_id=draft.suggested_id,
            hypothesis_id=hypothesis_id,
            parent_hypothesis_id=draft.parent_hypothesis_id,
            expression=draft.expression,
            scientific_feedback=feedback,
            semantic_change=semantic_change,
            equivalent_to_previous=equivalent,
            separate_candidate=separate,
            syntactic_success=True,
            execution_success=True,
            development_improvement=bool(feedback["development_improvement"]),
            independent_validation=False,
            outcome=str(feedback["outcome"]),
            brier_score=evaluation.brier_score,
            log_loss=evaluation.log_loss,
            baseline_brier=evaluation.baseline_brier,
            improvement=evaluation.improvement,
            uncertainty=evaluation.uncertainty,
            experiment_id=evaluation.experiment_id,
            test_dates=tuple(day.isoformat() for day in evaluation.test_dates),
            model_name=self.model_name,
            model_digest=self.model_digest,
            persisted=persisted,
            revision_reason=_revision_reason(decision, is_revision),
            detail=evaluation.reason,
        ))

    def _gate(self, content: str, *, is_revision: bool, parent_id: str | None, parent_expression: str | None) -> GateResult:
        if self.plan is None:
            return GateResult(False, "plan_rejected", "план не утверждён")
        if is_revision:
            return validate_revision_payload(
                content,
                plan=self.plan,
                parent_id=parent_id or "",
                parent_expression=parent_expression or "",
                known=self.known,
            )
        return validate_generation(content, plan=self.plan, known=self.known)

    def _call(self, bucket: str, factory: Callable[[], ModelTurn]) -> ModelTurn:
        if self.stopped is not None:
            return ModelTurn(self.stopped[0], self.stopped[1], None)
        if not callable(getattr(self.client, "generate", None)):
            self.stopped = ("ollama_unavailable", "клиент модели не подключён")
            return ModelTurn("ollama_unavailable", self.stopped[1], None)
        if not self._room(bucket):
            self.stopped = ("budget_exhausted", "исчерпан заранее заданный бюджет обращений к модели")
            return ModelTurn("budget_exhausted", self.stopped[1], None)
        if self.clock() - self.started > self.deadline:
            self.stopped = ("time_limit", "исчерпан бюджет времени")
            return ModelTurn("time_limit", self.stopped[1], None)
        blocked = self._wait_for_memory()
        if blocked:
            self.stopped = ("resource_limited", blocked)
            return ModelTurn("resource_limited", blocked, None)
        self.calls += 1
        if bucket == "plan":
            self.plan_calls += 1
        else:
            self.gen_calls += 1
        turn = factory()
        self._note_model(turn)
        if turn.status != "ok":
            self.stopped = (turn.status, turn.message or turn.status)
        return turn

    def _wait_for_memory(self) -> str | None:
        """До первого вызова нехватка памяти останавливает цикл сразу.

        Между вызовами одной и той же модели допускается короткое ожидание выгрузки.
        Второй экземпляр не запускается.
        """
        started = time.perf_counter()
        while True:
            blocked = self.guard()
            if not blocked:
                return None
            if self.calls == 0 or time.perf_counter() - started >= 20:
                return blocked
            time.sleep(1)

    def _room(self, bucket: str) -> bool:
        if self.calls >= MAX_MODEL_CALLS:
            return False
        if bucket == "plan":
            return self.plan_calls < 1
        return self.gen_calls < MAX_GENERATION_REPAIR_CALLS

    def _memory_payload(self, plan: ResearchPlan | None = None, expression: str | None = None) -> dict[str, object]:
        records: list[dict[str, object]] = [dict(item) for item in self.excerpt.hypotheses]
        records.extend(dict(item) for item in self.excerpt.experiments)
        for attempt in self.attempts:
            if attempt.execution_success or attempt.status in {"ok", "dry_run", "stored"} or not attempt.detail:
                continue
            records.append({
                "expression": attempt.expression or "",
                "horizon_days": plan.target_horizon if plan is not None else None,
                "error": attempt.detail[:180],
                "rejection_reason": attempt.status,
            })
        selected = select_relevant_records(
            records,
            features=() if plan is None else plan.candidate_features,
            horizon_days=None if plan is None else plan.target_horizon,
            expression=expression,
            limit=4,
        )
        return memory_as_data(selected)

    def _log_turn(self, step: str, turn: ModelTurn, *, parent_id: str | None) -> IntelligenceAttempt:
        return self._append(IntelligenceAttempt(
            step=step,
            status=turn.status,
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
            parent_hypothesis_id=parent_id,
            model_name=turn.model_name or self.model_name,
            model_digest=turn.model_digest or self.model_digest,
            detail=turn.message[:300],
            uncertainty=turn.message[:300],
        ))

    def _log_plan(self, decision, turn: ModelTurn, brief: dict[str, object] | None = None) -> IntelligenceAttempt:
        syntax = None
        if brief is not None:
            syntax = {
                "kind": "syntax",
                "scientific_experiment": False,
                "error_code": brief.get("error_code"),
                "invalid_expression": brief.get("invalid_expression"),
                "violated_rule": brief.get("violated_rule"),
                "allowed_format": brief.get("allowed_format"),
                "preserve_research_meaning": brief.get("preserve_research_meaning"),
                "repair_class": brief.get("repair_class"),
                "brier": None,
                "log_loss": None,
                "independent_validation": False,
                "validated_law": False,
                "claim_status": _NO_LAW,
            }
        elif not decision.accepted:
            repair = build_repair_brief(decision.code, decision.reason, None)
            syntax = None if repair is None else build_syntax_feedback(repair)
        return self._append(IntelligenceAttempt(
            step="syntax_repair" if brief is not None else "plan",
            status="ok" if decision.accepted else decision.code,
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
            plan=None if decision.plan is None else decision.plan.as_dict(),
            repair=brief,
            syntax_feedback=syntax,
            model_name=turn.model_name or self.model_name,
            model_digest=turn.model_digest or self.model_digest,
            detail=decision.reason[:300],
            uncertainty="" if decision.accepted else decision.reason[:300],
            outcome="plan" if decision.accepted else "formula_invalid",
        ))

    def _log_gate(self, step: str, decision: GateResult, assessment: EditAssessment | None) -> IntelligenceAttempt:
        brief = build_repair_brief(decision.code, decision.reason, decision.expression)
        semantic_change = bool(assessment and assessment.kind == "semantic_change") or decision.edit_kind == "semantic_change"
        return self._append(IntelligenceAttempt(
            step=step,
            status=decision.code,
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
            suggested_id=decision.suggested_id,
            parent_hypothesis_id=decision.parent_hypothesis_id,
            expression=decision.expression,
            syntax_feedback=None if brief is None else build_syntax_feedback(brief),
            semantic_change=semantic_change,
            equivalent_to_previous=None if assessment is None else assessment.equivalent,
            separate_candidate=bool(assessment and assessment.kind == "semantic_change"),
            model_name=self.model_name,
            model_digest=self.model_digest,
            revision_reason=None if decision.revision is None else decision.revision.get("difference_from_parent"),
            detail=decision.reason[:300],
            uncertainty=decision.reason[:300],
            outcome="formula_invalid",
        ))

    def _append(self, attempt: IntelligenceAttempt) -> IntelligenceAttempt:
        self.attempts.append(attempt)
        self._remember(attempt)
        return attempt

    def _remember(self, attempt: IntelligenceAttempt) -> None:
        if not self.commit:
            return
        if self.journal is not None:
            self.journal.append(_attempt_payload(attempt, self._config_now()))
        if self.store is None or attempt.status in {"ok", "dry_run", "stored"}:
            return
        self.store.note_attempt(
            kind="plan" if attempt.step == "plan" else "formula",
            author_id=self.author_id,
            succeeded=False,
            detail=f"{self.calls}:{attempt.step}:{attempt.status}:{attempt.expression or ''}:{attempt.detail}"[:300],
            created_at=self.created_at,
        )

    def _remember_formula(self, draft, parent_expression: str | None) -> None:
        self.known.append(KnownFormula(
            expression=draft.expression,
            features=tuple(draft.features),
            horizon_days=draft.horizon_days,
            status="checked",
            hypothesis_id=formula_key(
                draft.expression,
                draft.features,
                draft.horizon_days,
                _applicability(self.strategy_id, draft.expected_regime),
            ),
        ))
        del parent_expression

    def _note_model(self, turn: ModelTurn) -> None:
        if turn.model_name:
            self.model_name = turn.model_name
        if turn.model_digest:
            self.model_digest = turn.model_digest

    def _persist(self, draft, evaluation: Evaluation, hypothesis_id: str, decision: GateResult, parent_expression: str | None) -> bool | str:
        if self.store is None or self.memory is None:
            raise RuntimeError("нет хранилища для записи")
        reject_validated_law(_NO_LAW, False)
        reject_self_grade(self.author_id, VERIFIER_ID)
        applicability = _applicability(self.strategy_id, draft.expected_regime)
        stored_id, state = self.store.create(
            author_id=self.author_id,
            name=draft.suggested_id,
            description=draft.description,
            creation_reason=draft.rationale,
            features=draft.features,
            horizon_days=draft.horizon_days,
            expression=draft.expression,
            applicability=applicability,
            created_at=self.created_at,
            data_hash=self.data_hash,
            parent_hypothesis_id=draft.parent_hypothesis_id,
            experiment_id=evaluation.experiment_id,
        )
        if stored_id != hypothesis_id or state == "duplicate":
            return "duplicate"
        if draft.parent_hypothesis_id is not None:
            parent = self.store.get(draft.parent_hypothesis_id)
            if parent_expression is not None and parent.expression != parent_expression:
                raise RuntimeError("исходная формула была перезаписана")
        payload = {
            "claim_status": _NO_LAW,
            "validated_law": False,
            "advantage_claim": False,
            "graded_by": VERIFIER_ID,
            "author_id": self.author_id,
            "strategy_id": self.strategy_id,
            "generator_model": self.model_name or "qwen3.5:4b",
            "generator_version": GENERATOR_VERSION,
            "code_version": INTELLIGENCE_VERSION,
            "evaluation_protocol": PROTOCOL_VERSION,
            "expression": draft.expression,
            "parent_hypothesis_id": draft.parent_hypothesis_id,
            "revision_reason": _revision_reason(decision, draft.parent_hypothesis_id is not None),
            "experiment_id": evaluation.experiment_id,
            "brier_score": evaluation.brier_score,
            "log_loss": evaluation.log_loss,
            "baseline_brier": evaluation.baseline_brier,
            "frequency_brier": evaluation.frequency_brier,
            "logistic_brier": evaluation.logistic_brier,
            "improvement": evaluation.improvement,
            "uncertainty": evaluation.uncertainty,
            "fitted_on_2021": False,
            "dev_2021_skipped": evaluation.dev_2021_skipped,
            "sample_count": evaluation.sample_count,
            "data_hash": evaluation.data_hash,
            "independent_validation": False,
            "btc_claim": False,
        }
        reject_validated_law(str(payload["claim_status"]), bool(payload["validated_law"]))
        self.store.add_experiment(stored_id, payload, when=self.created_at)
        record = getattr(self.memory, "record", None)
        if not callable(record):
            raise RuntimeError("research memory не принимает запись")
        record({
            "model_name": self.model_name or "qwen3.5:4b",
            "config_hash": self.model_digest or "unspecified",
            "dataset_fingerprint": self.data_hash,
            "features": list(draft.features),
            "label_definition": f"синтетическая метка протокола 3.2; формула {draft.expression}; это не исход BTC",
            "horizon_days": draft.horizon_days,
            "train_start": "2019-01-01T00:00:00+00:00",
            "train_end": "2019-12-31T00:00:00+00:00",
            "dev_start": "2021-01-01T00:00:00+00:00",
            "dev_end": "2021-12-31T00:00:00+00:00",
            "seed": GENERATION_SEED,
            "evidence_role": "synthetic",
            "claim_status": _NO_LAW,
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
            "author_id": self.author_id,
            "strategy_id": self.strategy_id,
            "hypothesis_id": stored_id,
            "parent_hypothesis_id": draft.parent_hypothesis_id,
            "code_version": INTELLIGENCE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "evaluation_protocol": PROTOCOL_VERSION,
            "independent_validation": False,
            "btc_claim": False,
            "plan": None if self.plan is None else self.plan.as_dict(),
        })
        stored = self.store.get(stored_id)
        if stored.expression != draft.expression or stored.claim_status != _NO_LAW or self.store.confirms_law(stored_id):
            raise RuntimeError("запись изменила формулу или получила запрещённый статус")
        return True

    def _config_now(self) -> dict[str, object]:
        return _config(self.client, model_name=self.model_name, model_digest=self.model_digest)

    def _report(self, status: str, message: str) -> IntelligenceReport:
        scored = [item for item in self.attempts if item.execution_success]
        better = None
        internal: bool | None = None
        if scored:
            internal = any(item.development_improvement for item in scored)
        if len(scored) >= 2 and scored[0].brier_score is not None and scored[-1].brier_score is not None:
            better = scored[-1].brier_score < scored[0].brier_score
        return IntelligenceReport(
            status=status,
            message=message,
            dry_run=not self.commit,
            author_id=self.author_id,
            strategy_id=self.strategy_id,
            generator_version=GENERATOR_VERSION,
            attempts=tuple(self.attempts),
            model_calls=self.calls,
            plan_calls=self.plan_calls,
            generation_repair_calls=self.gen_calls,
            syntax_repairs=self.syntax_repairs,
            revisions=self.revisions,
            hypotheses=self.generate_calls + self.revisions,
            unique_expressions=len(self.evaluated),
            evaluated_expressions=tuple(self.evaluated),
            elapsed_seconds=round(self.clock() - self.started, 6),
            rss_bytes=process_rss_bytes(),
            multiple_testing_note=_multiple_testing_note(len(scored)),
            comparison=_comparison(scored),
            generation_config=self._config_now(),
            point_estimate_better=better,
            improvement_is_internal_only=internal,
            independent_validation=False,
            validated_law=False,
            claim_status=_NO_LAW,
        )


def public_report(report: IntelligenceReport) -> dict[str, object]:
    """Снимок для отчёта. Ряд точек и закрытые даты сюда не входят."""
    return {
        "status": report.status,
        "message": report.message,
        "dry_run": report.dry_run,
        "author_id": report.author_id,
        "strategy_id": report.strategy_id,
        "generator_version": report.generator_version,
        "model_calls": report.model_calls,
        "plan_calls": report.plan_calls,
        "generation_repair_calls": report.generation_repair_calls,
        "syntax_repairs": report.syntax_repairs,
        "revisions": report.revisions,
        "hypotheses": report.hypotheses,
        "unique_expressions": report.unique_expressions,
        "evaluated_expressions": list(report.evaluated_expressions),
        "elapsed_seconds": report.elapsed_seconds,
        "rss_bytes": report.rss_bytes,
        "claim_status": report.claim_status,
        "validated_law": report.validated_law,
        "independent_validation": report.independent_validation,
        "multiple_testing_note": report.multiple_testing_note,
        "point_estimate_better": report.point_estimate_better,
        "improvement_is_internal_only": report.improvement_is_internal_only,
        "comparison": [dict(item) for item in report.comparison],
        "generation_config": dict(report.generation_config),
        "attempts": [_attempt_payload(item, {}) for item in report.attempts],
    }


def _attempt_payload(attempt: IntelligenceAttempt, config: Mapping[str, object]) -> dict[str, object]:
    return {
        "step": attempt.step,
        "status": attempt.status,
        "author_id": attempt.author_id,
        "strategy_id": attempt.strategy_id,
        "generator_version": attempt.generator_version,
        "suggested_id": attempt.suggested_id,
        "hypothesis_id": attempt.hypothesis_id,
        "parent_hypothesis_id": attempt.parent_hypothesis_id,
        "expression": attempt.expression,
        "plan": attempt.plan,
        "repair": attempt.repair,
        "scientific_feedback": attempt.scientific_feedback,
        "syntax_feedback": attempt.syntax_feedback,
        "semantic_change": attempt.semantic_change,
        "equivalent_to_previous": attempt.equivalent_to_previous,
        "separate_candidate": attempt.separate_candidate,
        "syntactic_success": attempt.syntactic_success,
        "execution_success": attempt.execution_success,
        "development_improvement": attempt.development_improvement,
        "independent_validation": attempt.independent_validation,
        "outcome": attempt.outcome,
        "brier_score": attempt.brier_score,
        "log_loss": attempt.log_loss,
        "baseline_brier": attempt.baseline_brier,
        "improvement": attempt.improvement,
        "uncertainty": attempt.uncertainty,
        "experiment_id": attempt.experiment_id,
        "test_dates": list(attempt.test_dates),
        "model_name": attempt.model_name,
        "model_digest": attempt.model_digest,
        "persisted": attempt.persisted,
        "revision_reason": attempt.revision_reason,
        "detail": attempt.detail,
        "generation_config": dict(config),
        "qwen_generator_version": QWEN_GENERATOR_VERSION,
    }


def _comparison(scored: Sequence[IntelligenceAttempt]) -> tuple[dict[str, object], ...]:
    if len(scored) < 2:
        return ()
    dates = scored[0].test_dates
    rows: list[dict[str, object]] = []
    for item in scored:
        rows.append({
            "hypothesis_id": item.hypothesis_id,
            "expression": item.expression,
            "brier": item.brier_score,
            "log_loss": item.log_loss,
            "baseline_brier": item.baseline_brier,
            "error_difference": item.improvement,
            "uncertainty": item.uncertainty,
            "same_dates": item.test_dates == dates,
        })
    return tuple(rows)


def _multiple_testing_note(count: int) -> str:
    if count <= 1:
        return "Внутреннее окно 2020-07-01..2020-12-31 не является независимым подтверждением."
    return (
        f"Окно 2020-07-01..2020-12-31 использовано для {count} версий. "
        "Повторные попытки на том же окне не являются независимыми подтверждениями."
    )


def _identity(expression: str, features: Sequence[str], horizon: int) -> tuple[object, ...]:
    canonical = canonical_identity(expression)
    return (canonical if canonical is not None else "".join(expression.split()), tuple(sorted(features)), horizon)


def _applicability(strategy_id: str, regime: str) -> dict[str, object]:
    return {
        "market": "synthetic-2019-2020",
        "regimes": ["any"] if regime == "any" else [regime],
        "btc_claim": False,
        "strategy_id": strategy_id,
        "code_version": INTELLIGENCE_VERSION,
        "evidence_role": _NO_LAW,
    }


def _applicability_text(strategy_id: str, regime: str) -> str:
    return f"strategy={strategy_id}; regime={regime}; market=synthetic-2019-2020; btc_claim=false"


def _revision_reason(decision: GateResult, is_revision: bool) -> str | None:
    if not is_revision or decision.revision is None:
        return None
    return decision.revision.get("why_new_relationship")


def _series_boundary(points: Sequence[StudyPoint]) -> str | None:
    for point in points:
        if point.decision_date >= CLOSED_FROM or point.target_date >= CLOSED_FROM:
            return "набор содержит дату после 2021 года"
        if point.label_available <= point.decision_available:
            return "в наборе метка доступна не позже решения"
    return None


def _config(client: object, *, model_name: str = "", model_digest: str = "") -> dict[str, object]:
    return {
        "model_name": model_name or getattr(client, "model_name", "") or getattr(client, "model", ""),
        "model_digest": model_digest or getattr(client, "model_digest", ""),
        "generator_version": GENERATOR_VERSION,
        "protocol": INTELLIGENCE_VERSION,
        "temperature": GENERATION_TEMPERATURE,
        "seed": GENERATION_SEED,
        "num_ctx": getattr(client, "num_ctx", 2048),
        "num_predict": getattr(client, "num_predict", 512),
        "evaluation_method": FIXED_EVALUATION_METHOD,
        "max_model_calls": MAX_MODEL_CALLS,
        "max_generation_repair_calls": MAX_GENERATION_REPAIR_CALLS,
        "max_syntax_repairs": MAX_SYNTAX_REPAIRS,
        "max_revisions": MAX_SCIENTIFIC_REVISIONS,
        "max_evaluated_formulas": MAX_EVALUATED_FORMULAS,
        "ollama_version": getattr(client, "ollama_version", ""),
        "parameter_size": getattr(client, "parameter_size", ""),
    }


def _rollback(store: object | None, memory: object | None, journal: object | None) -> None:
    for target in (store, memory, journal):
        method = getattr(target, "rollback_open", None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            return

