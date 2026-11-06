"""Этап 3.3: научный пересмотр доходит до проверки и не выходит за бюджет."""

import json
from datetime import date, datetime, timezone
from io import BytesIO
from urllib.error import HTTPError

import pytest

from wolfai.research.evaluation_protocol import (
    TEST_END,
    TEST_START,
    StudyPoint,
    evaluate_hypothesis,
    synthetic_study_points,
)
from wolfai.research.formula import FEATURES
from wolfai.research.grammar_contract import (
    FIXED_EVALUATION_METHOD,
    MAX_MODEL_CALLS,
    revision_schema,
)
from wolfai.research.hypothesis_engine import HypothesisLifecycleError, HypothesisStore
from wolfai.research.intelligence_log import IntelligenceLog
from wolfai.research.ollama_adapter import STAGE30_NUM_PREDICT, AdapterResult, OllamaAdapter, build_adapter_body
from wolfai.research.research_intelligence import run_research_intelligence
from wolfai.research.research_memory import ResearchMemory
from wolfai.research.research_plan import validate_research_plan
from wolfai.research.research_ports import RevisionContext
from wolfai.research.scientific_revision import build_revision_messages, validate_revision_payload
from wolfai.research.syntax_repair import classify_edit

WHEN = datetime(2026, 11, 6, tzinfo=timezone.utc)
MOMENTUM = "sigmoid(0.2 * momentum_7)"
VOLATILITY = "sigmoid(0.2 * volatility_7)"
LOG_RETURN = "sigmoid(0.2 * log_return)"


def _plan(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "research_question": "Связан ли семидневный импульс с внутренним исходом на горизонте одного дня?",
        "target_horizon": 1,
        "candidate_features": ["momentum_7", "volatility_7", "log_return"],
        "proposed_relationship": "Импульс и волатильность входят в аргумент с фиксированными коэффициентами.",
        "expected_market_conditions": "any",
        "falsification_conditions": "Версия слабеет, если ошибка не ниже константы 0.5 на заранее заданном окне.",
        "evaluation_method": FIXED_EVALUATION_METHOD,
        "limitations": "Внутренний счёт не устанавливает закономерность.",
    }
    payload.update(extra)
    return payload


def _hypothesis(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "hypothesis_id": "hyp-momentum",
        "description": "Черновик про импульс на синтетическом ряде.",
        "features": ["momentum_7"],
        "horizon_days": 1,
        "expression": MOMENTUM,
        "rationale": "Короткий импульс отличается фиксированным коэффициентом.",
        "expected_regime": "any",
        "test_plan": "Сравнить Brier с константой 0.5 на заранее заданном внутреннем окне.",
        "limitations": "Внутреннее улучшение не является закономерностью.",
        "parent_hypothesis_id": None,
    }
    payload.update(extra)
    return payload


def _revision(user: dict[str, object], **extra: object) -> dict[str, object]:
    body = _hypothesis(
        hypothesis_id="hyp-child",
        parent_hypothesis_id=user["parent_hypothesis_id"],
        expression=VOLATILITY,
        features=["volatility_7"],
        rationale="Волатильность заменяет импульс как отдельное предположение.",
        what_failed="Прежняя формула не дала доказанного улучшения.",
        assumption_to_revise="Связь только с импульсом слишком узкая.",
        why_new_relationship="Волатильность описывает другой режим, не коэффициент той же формулы.",
        difference_from_parent="Меняется признак, а не только число.",
        falsification_conditions="Новая версия слабеет, если её ошибка не ниже константы 0.5.",
    )
    body.update(extra)
    return body


def _copy_parent(user: dict[str, object]) -> dict[str, object]:
    expression = str(user["parent_expression"])
    if "volatility_7" in expression:
        features = ["volatility_7"]
    elif "log_return" in expression:
        features = ["log_return"]
    else:
        features = ["momentum_7"]
    return _revision(user, hypothesis_id="hyp-copy", expression=expression, features=features)


class QueueClient:
    def __init__(self, items: list[object], *, generation_requests: int = 1) -> None:
        self.items = list(items)
        self.calls: list[str] = []
        self.generation_requests = generation_requests

    def generate(self, messages, schema=None, *, temperature: float = 0.0):
        user = json.loads(messages[-1]["content"])
        self.calls.append(str(user["task"]))
        if not self.items:
            raise AssertionError(f"лишний вызов {user['task']}")
        item = self.items.pop(0)
        if callable(item):
            item = item(user)
        content = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return AdapterResult(
            status="ok",
            message="scripted",
            content=content,
            model_name="qwen3.5:4b",
            model_digest="digest-test",
            generation_requests=self.generation_requests,
        )


def _run(client: object, **kwargs: object):
    kwargs.setdefault("created_at", WHEN)
    kwargs.setdefault("memory_guard", lambda: None)
    return run_research_intelligence(client, **kwargs)  # type: ignore[arg-type]


def _features(secret: float = 0.1) -> dict[str, float]:
    values = {name: 0.1 for name in FEATURES}
    values["momentum_7"] = secret
    values["volatility_7"] = 0.2
    values["volatility_30"] = 0.2
    return values


def _approved_plan():
    decision = validate_research_plan(json.dumps(_plan(), ensure_ascii=False))
    assert decision.plan is not None
    return decision.plan


def test_first_response_json_is_accepted_without_repair() -> None:
    client = QueueClient([_plan(), _hypothesis(), _copy_parent])
    report = _run(client)
    assert client.calls == ["plan", "generate", "scientific_revision"]
    assert report.syntax_repairs == 0
    assert report.first_pass_valid == 2
    assert report.response_count == 3
    assert report.attempts[1].expression == MOMENTUM
    assert report.attempts[1].execution_success is True
    schema = revision_schema()
    assert schema["additionalProperties"] is False
    assert "what_failed" in schema["required"]
    messages = build_revision_messages(RevisionContext(
        plan=_plan(),
        parent_hypothesis_id="parent-1",
        parent_expression=MOMENTUM,
        scientific_feedback={
            "outcome": "not_better_than_baseline",
            "brier": 0.31,
            "log_loss": 0.8,
            "baseline_brier": 0.25,
            "error_difference": -0.06,
            "decision_count": 182,
            "uncertainty": "интервал содержит ноль",
            "no_proven_improvement_reason": "Ошибка не ниже константы 0.5.",
        },
        untrusted_memory={"trust": "data", "records": []},
        author_id="qwen-research-brain",
        strategy_id="qwen-research-strategy",
        generator_version="3.3.0",
    ))
    user = json.loads(messages[1]["content"])
    assert user["parent_hypothesis_id"] == "parent-1"
    assert user["desk"]["brier"] == 0.31
    assert user["locked_horizon"] == 1
    assert len(messages[1]["content"]) < 2200
    assert STAGE30_NUM_PREDICT == 1024


def test_invalid_first_json_is_repaired() -> None:
    client = QueueClient([_plan(), "не json", _hypothesis(), _copy_parent])
    report = _run(client)
    assert client.calls[1] == "generate"
    assert client.calls[2] == "syntax_repair"
    assert report.attempts[1].first_pass_valid is False or report.attempts[1].execution_success is False
    assert report.attempts[1].brier_score is None
    repaired = next(item for item in report.attempts if item.execution_success)
    assert repaired.expression == MOMENTUM


def test_revision_without_required_field_is_rejected() -> None:
    plan = _approved_plan()
    body = _revision({"parent_hypothesis_id": "parent-1"})
    del body["what_failed"]
    decision = validate_revision_payload(
        json.dumps(body, ensure_ascii=False),
        plan=plan,
        parent_id="parent-1",
        parent_expression=MOMENTUM,
    )
    assert decision.accepted is False
    assert decision.code == "invalid_json"
    assert "what_failed" in decision.reason


def test_revision_with_wrong_parent_is_rejected() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        lambda user: _revision(user, parent_hypothesis_id="not-the-parent"),
        _revision,
    ])
    report = _run(client)
    rejected = next(item for item in report.attempts if item.status == "unknown_parent")
    assert rejected.brier_score is None
    assert rejected.parent_hypothesis_id == "not-the-parent"
    assert "syntax_repair" not in client.calls


def test_revision_can_be_repaired_after_a_bad_reply() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        "это не json",
        _revision,
        _copy_parent,
    ])
    report = _run(client)
    assert client.calls == [
        "plan",
        "generate",
        "scientific_revision",
        "syntax_repair",
        "scientific_revision",
    ]
    scored = [item for item in report.attempts if item.execution_success]
    assert [item.expression for item in scored] == [MOMENTUM, VOLATILITY]
    assert scored[1].parent_hypothesis_id == scored[0].hypothesis_id
    assert report.syntax_repairs == 1


def test_budget_stops_at_six_generation_calls() -> None:
    state = {"n": 0}

    def revision(user: dict[str, object]) -> object:
        state["n"] += 1
        if state["n"] == 1:
            return "это не json"
        return _revision(user, expression=LOG_RETURN, features=["log_return"], hypothesis_id="hyp-log")

    client = QueueClient([
        _plan(),
        _hypothesis(),
        revision,
        "всё ещё не json",
        "{",
        revision,
        _hypothesis(expression="sigmoid(0.4 * range_position)", features=["range_position"], hypothesis_id="hyp-extra"),
    ])
    report = _run(client)
    assert report.model_calls == MAX_MODEL_CALLS
    assert report.model_calls == 6
    assert len(client.items) == 1
    assert "sigmoid(0.4 * range_position)" not in report.evaluated_expressions
    assert report.ollama_generation_requests == 6


def test_hidden_extra_generation_is_counted_inside_the_cap() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        _revision,
        _revision,
    ], generation_requests=2)
    report = _run(client)
    assert client.calls == ["plan", "generate", "scientific_revision"]
    assert report.model_calls == 6
    assert report.ollama_generation_requests == 6
    assert report.revisions == 1


def test_sigmoid_wrap_is_recorded_as_a_semantic_change() -> None:
    edit = classify_edit("sign(momentum_7)", "sigmoid(sign(momentum_7))")
    assert edit.kind == "semantic_change"
    assert edit.adds_root_sigmoid is True
    assert edit.equivalent is False
    client = QueueClient([
        _plan(),
        _hypothesis(expression="sign(momentum_7)"),
        _hypothesis(expression="sigmoid(sign(momentum_7))"),
        _copy_parent,
    ])
    report = _run(client)
    scored = next(item for item in report.attempts if item.execution_success)
    assert scored.expression == "sigmoid(sign(momentum_7))"
    assert scored.semantic_change is True
    assert scored.equivalent_to_previous is False
    normalized = _run(QueueClient([
        _plan(),
        _hypothesis(parent_hypothesis_id="foreign-id"),
        _copy_parent,
    ]))
    first = next(item for item in normalized.attempts if item.execution_success)
    assert first.format_normalized is True
    assert first.expression == MOMENTUM
    assert first.parent_hypothesis_id is None


def test_exact_duplicate_is_not_scored_again() -> None:
    client = QueueClient([_plan(), _hypothesis(), _copy_parent])
    report = _run(client)
    assert report.status == "duplicate"
    assert report.unique_expressions == 1
    assert sum(1 for item in report.attempts if item.brier_score is not None) == 1


def test_coefficient_only_revision_is_not_a_new_idea() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        lambda user: _revision(
            user,
            expression="sigmoid(0.9 * momentum_7)",
            features=["momentum_7"],
            hypothesis_id="hyp-tweak",
            why_new_relationship="Меняется только масштаб прежней зависимости.",
            difference_from_parent="Другое число при том же признаке.",
        ),
        _copy_parent,
    ])
    report = _run(client)
    tweak = next(item for item in report.attempts if item.status == "coefficient_peek")
    assert tweak.brier_score is None
    assert tweak.execution_success is False
    assert report.unique_expressions == 1


def test_model_cannot_replace_the_desk_brier() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        lambda user: _revision(user, brier_score=0.01),
        _copy_parent,
    ])
    report = _run(client)
    original = next(item for item in report.attempts if item.expression == MOMENTUM and item.execution_success)
    assert original.scientific_feedback is not None
    assert original.brier_score != 0.01
    assert any(item.status == "criteria_change" for item in report.attempts)


def test_closed_periods_stay_closed() -> None:
    future = StudyPoint(
        horizon_days=1,
        decision_date=date(2022, 3, 1),
        target_date=date(2022, 3, 2),
        decision_available=date(2022, 3, 2),
        label_available=date(2022, 3, 3),
        features=_features(),
        label=1,
    )
    blocked = _run(QueueClient([_plan(), _hypothesis()]), points=(future,))
    assert blocked.status == "time_boundary"
    assert blocked.model_calls == 0
    leaked = _run(QueueClient([
        _plan(research_question="Что будет на закрытом окне 2024 года?"),
        _hypothesis(),
    ]))
    assert leaked.status == "plan_rejected"
    assert leaked.model_calls == 1


def test_unavailable_ollama_stops_cleanly() -> None:
    class Down:
        def generate(self, *args: object, **kwargs: object) -> AdapterResult:
            return AdapterResult(status="ollama_unavailable", message="Ollama недоступна")

    failed = _run(Down())
    assert failed.status == "ollama_unavailable"
    assert failed.model_calls == 1
    assert failed.validated_law is False

    class Missing:
        pass

    absent = _run(Missing())
    assert absent.status == "ollama_unavailable"
    assert absent.model_calls == 0


def test_dangerous_expression_is_not_executed(tmp_path) -> None:
    marker = tmp_path / "owned.txt"
    dangerous = "sigmoid(__import__('os'))"
    report = _run(QueueClient([
        _plan(),
        _hypothesis(expression=dangerous),
        _hypothesis(expression=dangerous, hypothesis_id="hyp-2"),
        _hypothesis(expression=dangerous, hypothesis_id="hyp-3"),
    ]))
    assert report.status == "syntax_unrepaired"
    assert not marker.exists()
    assert all(item.brier_score is None for item in report.attempts)


def test_two_revisions_are_compared_on_the_same_dates() -> None:
    client = QueueClient([
        _plan(),
        _hypothesis(),
        _revision,
        lambda user: _revision(user, expression=LOG_RETURN, features=["log_return"], hypothesis_id="hyp-log"),
    ])
    report = _run(client)
    scored = [item for item in report.attempts if item.execution_success]
    assert [item.expression for item in scored] == [MOMENTUM, VOLATILITY, LOG_RETURN]
    assert report.revisions == 2
    assert len(report.comparison) == 3
    assert all(row["same_dates"] is True for row in report.comparison)
    points = synthetic_study_points()
    parent = evaluate_hypothesis(MOMENTUM, ("momentum_7",), 1, points)
    child = evaluate_hypothesis(VOLATILITY, ("volatility_7",), 1, points)
    assert scored[1].parent_brier == parent.brier_score
    assert scored[1].parent_delta == pytest.approx(child.brier_score - parent.brier_score)
    assert scored[1].brier_score == child.brier_score
    assert scored[1].log_loss == child.log_loss
    assert all(TEST_START.isoformat() <= day <= TEST_END.isoformat() for day in scored[1].test_dates)
    assert report.independent_validation is False
    assert report.validated_law is False


def test_parent_formula_stays_unchanged(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    journal = IntelligenceLog(tmp_path / "journal.sqlite")
    report = _run(
        QueueClient([_plan(), _hypothesis(), _revision, _copy_parent]),
        store=store,
        memory=memory,
        journal=journal,
        commit=True,
    )
    parent = next(item for item in report.attempts if item.expression == MOMENTUM and item.execution_success)
    child = next(item for item in report.attempts if item.expression == VOLATILITY and item.execution_success)
    assert store.get(str(parent.hypothesis_id)).expression == MOMENTUM
    assert store.get(str(child.hypothesis_id)).parent_hypothesis_id == parent.hypothesis_id
    assert store.get(str(child.hypothesis_id)).expression == VOLATILITY


def test_low_memory_stops_before_the_model() -> None:
    report = _run(QueueClient([_plan()]), memory_guard=lambda: "мало свободной памяти")
    assert report.status == "resource_limited"
    assert report.model_calls == 0
    assert report.ollama_generation_requests == 0


def test_validated_law_is_not_created(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    journal = IntelligenceLog(tmp_path / "journal.sqlite")
    report = _run(
        QueueClient([_plan(), _hypothesis(), _copy_parent]),
        store=store,
        memory=memory,
        journal=journal,
        commit=True,
    )
    hypothesis_id = str(next(item.hypothesis_id for item in report.attempts if item.execution_success))
    assert report.validated_law is False
    assert report.claim_status == "not_confirmed"
    assert store.confirms_law(hypothesis_id) is False
    with pytest.raises(HypothesisLifecycleError):
        store.transition(hypothesis_id, "supported", "модель так сказала")


class _Body:
    def __init__(self, payload: dict[str, object]) -> None:
        self.raw = json.dumps(payload).encode()
        self.pos = 0

    def read(self, count: int) -> bytes:
        chunk = self.raw[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_schema_fallback_is_recorded_and_can_be_refused() -> None:
    class Transport:
        def __init__(self) -> None:
            self.formats: list[object] = []

        def __call__(self, request, timeout):
            body = json.loads(request.data.decode())
            self.formats.append(body["format"])
            if isinstance(body["format"], dict):
                raise HTTPError(request.full_url, 400, "bad schema", hdrs=None, fp=BytesIO(b""))
            return _Body({"message": {"content": '{"expression":"sigmoid(0.2 * momentum_7)"}'}, "done_reason": "stop"})

    transport = Transport()
    adapter = OllamaAdapter(timeout=5, http_attempts=1, transport=transport)
    adapter.model_name = "qwen3.5:4b"
    adapter.model_digest = "digest"
    adapter.schema_mode = "schema"
    adapter.generation_budget = 2
    accepted = adapter.generate([{"role": "user", "content": "ping"}], {"type": "object"}, temperature=0)
    assert accepted.format_fallback is True
    assert accepted.schema_mode == "json"
    assert accepted.generation_requests == 2
    assert accepted.fallback_reason.startswith("сервер вернул 400")
    assert transport.formats == [{"type": "object"}, "json"]

    transport.formats.clear()
    refused = OllamaAdapter(timeout=5, http_attempts=1, transport=transport)
    refused.model_name = "qwen3.5:4b"
    refused.model_digest = "digest"
    refused.schema_mode = "schema"
    refused.allow_format_fallback = False
    refused.generation_budget = 1
    blocked = refused.generate([{"role": "user", "content": "ping"}], {"type": "object"}, temperature=0)
    assert blocked.status == "schema_rejected"
    assert blocked.generation_requests == 1
    assert blocked.format_fallback is False
    assert transport.formats == [{"type": "object"}]

    body = build_adapter_body(
        "qwen3.5:4b",
        [{"role": "user", "content": "тест"}],
        {"type": "object"},
        schema_mode="schema",
        temperature=0,
        seed=1729,
        num_ctx=2048,
        num_predict=4096,
    )
    options = body["options"]
    assert isinstance(options, dict)
    assert options["num_predict"] == 1024
    assert options["num_ctx"] == 2048
    assert "tools" not in body
