"""Этап 3.1: год в идентификаторе не открывает ряд, пересмотр не затирает формулу."""

import json
import math
from datetime import date, datetime, timezone

import pytest

from wolfai.research.budget import exclusive_search
from wolfai.research.evaluation_protocol import (
    CLOSED_FROM,
    MIN_STABLE_OBSERVATIONS,
    TEST_END,
    TEST_START,
    StudyPoint,
    evaluate_hypothesis,
    score_forecasts,
    synthetic_study_points,
)
from wolfai.research.hypothesis_engine import HypothesisLifecycleError, HypothesisStore
from wolfai.research.hypothesis_protocol import verify_model_output
from wolfai.research.improvement_loop import run_improvement_loop
from wolfai.research.ollama_adapter import AdapterResult, build_adapter_body
from wolfai.research.qwen_brain import FEEDBACK_KEYS, QwenResearchBrain
from wolfai.research.research_memory import ResearchMemory

WHEN = datetime(2026, 11, 5, tzinfo=timezone.utc)
EXPRESSION = "sigmoid(0.2 * momentum_7)"


def _body(**extra: object) -> str:
    payload: dict[str, object] = {
        "hypothesis_id": "HYP-2024-SYNTH-001",
        "description": "Техническая метка 2024 не открывает цены и не является формулой sigmoid(exp(price)).",
        "features": ["momentum_7"],
        "horizon_days": 1,
        "expression": EXPRESSION,
        "rationale": "Короткий импульс отличается фиксированным коэффициентом.",
        "expected_regime": "any",
        "test_plan": "Сравнить Brier с константой 0.5 на заранее заданном внутреннем окне.",
        "limitations": "Внутреннее улучшение не является закономерностью.",
        "parent_hypothesis_id": None,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


class Scripted:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.calls = 0
        self.seen: list[str] = []

    def generate(self, messages, schema=None, *, temperature: float = 0.0):
        self.seen.append("\n".join(item["content"] for item in messages))
        user = json.loads(messages[-1]["content"])
        raw = json.loads(self.payloads[self.calls])
        if "feedback" in user:
            raw["parent_hypothesis_id"] = user["feedback"]["hypothesis_id"]
        self.calls += 1
        content = json.dumps(raw, ensure_ascii=False)
        return AdapterResult(
            status="ok",
            message="scripted",
            content=content,
            model_name="qwen3.5:4b",
            model_digest="digest-test",
            elapsed_seconds=0.01,
            rss_bytes=2048,
        )


class Chain:
    """Берёт parent_hypothesis_id из предыдущей обратной связи."""

    def __init__(self, specs: list[tuple[str, list[str], str]]) -> None:
        self.specs = specs
        self.calls = 0
        self.seen: list[str] = []

    def generate(self, messages, schema=None, *, temperature: float = 0.0):
        text = "\n".join(item["content"] for item in messages)
        self.seen.append(text)
        user = json.loads(messages[-1]["content"])
        parent = None
        if "feedback" in user:
            parent = user["feedback"]["hypothesis_id"]
        expression, features, name = self.specs[self.calls]
        self.calls += 1
        return AdapterResult(
            status="ok",
            message="scripted",
            content=_body(
                hypothesis_id=name,
                expression=expression,
                features=features,
                parent_hypothesis_id=parent,
                rationale="Пересмотр меняет состав признаков и не заменяет прежнюю строку.",
            ),
            model_name="qwen3.5:4b",
            model_digest="digest-test",
        )


class Boom:
    def generate(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("обрыв генерации")


def _point(**changes: object) -> StudyPoint:
    base = StudyPoint(
        horizon_days=1,
        decision_date=date(2019, 6, 1),
        target_date=date(2019, 6, 2),
        decision_available=date(2019, 6, 2),
        label_available=date(2019, 6, 3),
        features={name: 0.1 for name in (
            "log_return", "momentum_7", "momentum_30", "volatility_7",
            "volatility_30", "volume_change", "range_position",
        )},
        label=1,
    )
    data = {field: getattr(base, field) for field in base.__dataclass_fields__}
    data.update(changes)
    return StudyPoint(**data)


def test_year_inside_identifier_is_not_a_time_boundary() -> None:
    decision = verify_model_output(_body())
    assert decision.accepted is True
    assert decision.draft is not None
    assert decision.draft.suggested_id == "HYP-2024-SYNTH-001"
    assert decision.draft.expression == EXPRESSION
    assert "exp(price)" not in decision.draft.expression
    report = run_improvement_loop(
        Scripted([_body(), _body()]),
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert report.attempts[0].status == "dry_run"
    assert report.attempts[0].brier_score is not None
    assert report.attempts[0].status != "time_boundary"
    assert report.status == "duplicate"
    assert report.validated_law is False
    assert report.claim_status == "not_confirmed"


def test_real_future_reference_is_rejected(tmp_path) -> None:
    leaked = verify_model_output(_body(
        hypothesis_id="clean-id",
        description="Обычное описание без календарной метки.",
        expression="sigmoid(0.2 * future_return)",
        features=["momentum_7"],
    ))
    assert leaked.accepted is False
    assert leaked.code == "time_boundary"
    future = run_improvement_loop(
        Scripted([_body()]),
        points=(_point(decision_date=date(2022, 3, 1), target_date=date(2022, 3, 2), label_available=date(2022, 3, 3)),),
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert future.status == "time_boundary"
    assert future.model_calls == 0
    assert future.attempts == ()


def test_allowed_formula_is_scored_on_the_fixed_window() -> None:
    evaluation = evaluate_hypothesis(EXPRESSION, ("momentum_7",), 1, synthetic_study_points())
    assert evaluation.accepted is True
    assert evaluation.sample_count >= 2
    assert evaluation.brier_score is not None
    assert 0.0 <= evaluation.brier_score <= 1.0
    assert evaluation.fitted_on_2021 is False
    assert evaluation.advantage_claim is False
    assert evaluation.claim_status == "not_confirmed"
    assert all(TEST_START <= day <= TEST_END for day in evaluation.test_dates)
    assert all(day < CLOSED_FROM for day in evaluation.test_dates)


def test_generated_code_is_not_executed(tmp_path) -> None:
    marker = tmp_path / "owned.txt"
    decision = verify_model_output(_body(expression="sigmoid(__import__('os'))"))
    assert decision.accepted is False
    assert decision.code == "unsafe_math"
    assert not marker.exists()
    bad = _body(expression="sigmoid(__import__('os'))")
    report = run_improvement_loop(
        Scripted([bad, bad, bad]),
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert report.attempts[0].status == "unsafe_math"
    assert report.attempts[0].brier_score is None
    assert not marker.exists()


def test_brier_and_log_loss_match_the_definition() -> None:
    probabilities = [0.5, 0.8]
    labels = [1, 1]
    frequency = [0.4, 0.4]
    scored = score_forecasts(probabilities, labels, frequency)
    expected_brier = ((0.5 - 1) ** 2 + (0.8 - 1) ** 2) / 2
    expected_log_loss = (-math.log(0.5) + -math.log(0.8)) / 2
    assert scored["brier"] == pytest.approx(expected_brier)
    assert scored["log_loss"] == pytest.approx(expected_log_loss)
    skill = scored["skill"]
    assert isinstance(skill, dict)
    assert skill["constant_half"]["reference_brier"] == pytest.approx(0.25)


def test_dates_share_one_horizon_and_keep_the_gap() -> None:
    evaluation = evaluate_hypothesis(EXPRESSION, ("momentum_7",), 1, synthetic_study_points())
    assert evaluation.accepted is True
    assert evaluation.gap_days >= 1
    assert evaluation.sample_count == len(evaluation.test_dates)
    assert evaluation.test_dates == tuple(sorted(evaluation.test_dates))
    assert evaluation.logistic_brier is not None
    assert evaluation.frequency_brier is not None
    assert evaluation.baseline_brier is not None
    week = evaluate_hypothesis(EXPRESSION, ("momentum_7",), 7, synthetic_study_points())
    assert week.accepted is True
    assert week.gap_days >= 7
    assert week.sample_count < MIN_STABLE_OBSERVATIONS
    assert "не устойчив" in week.uncertainty


def test_future_label_is_not_scored() -> None:
    leaked = _point(label_available=date(2019, 6, 2), decision_available=date(2019, 6, 2))
    evaluation = evaluate_hypothesis(EXPRESSION, ("momentum_7",), 1, (leaked,))
    assert evaluation.accepted is False
    assert evaluation.code == "label_leak"
    assert evaluation.brier_score is None
    clean = evaluate_hypothesis(EXPRESSION, ("momentum_7",), 1, synthetic_study_points())
    assert clean.accepted is True
    assert all(day <= TEST_END for day in clean.test_dates)


def test_feedback_reaches_the_next_generation_without_closed_data() -> None:
    client = Chain([
        (EXPRESSION, ["momentum_7"], "first-id"),
        ("sigmoid(0.3 * volatility_7)", ["volatility_7"], "second-id"),
        ("sigmoid(0.1 * momentum_30)", ["momentum_30"], "third-id"),
    ])
    report = run_improvement_loop(client, created_at=WHEN, memory_guard=lambda: None)
    assert client.calls == 3
    second = json.loads(client.seen[1].split("\n")[-1])
    feedback = second["feedback"]
    assert set(feedback) == FEEDBACK_KEYS
    assert feedback["revision_allowed"] is True
    assert isinstance(feedback["brier_score"], float)
    assert "holdout" not in client.seen[1].lower()
    assert "select " not in client.seen[1].lower()
    assert report.attempts[1].parent_hypothesis_id == report.attempts[0].hypothesis_id
    assert report.attempts[1].feedback["hypothesis_id"] == report.attempts[1].hypothesis_id


def test_child_hypothesis_keeps_its_own_id(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    client = Chain([
        (EXPRESSION, ["momentum_7"], "first-id"),
        ("sigmoid(0.3 * volatility_7)", ["volatility_7"], "second-id"),
        ("sigmoid(0.1 * momentum_30)", ["momentum_30"], "third-id"),
    ])
    report = run_improvement_loop(
        client,
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert report.status == "stored"
    parent = store.get(str(report.attempts[0].hypothesis_id))
    child = store.get(str(report.attempts[1].hypothesis_id))
    assert child.hypothesis_id != parent.hypothesis_id
    assert child.parent_hypothesis_id == parent.hypothesis_id
    assert child.expression != parent.expression
    assert store.derived_ids(parent.hypothesis_id) == [child.hypothesis_id]
    assert child.author_id == report.author_id
    assert report.strategy_id == "qwen-research-strategy"


def test_parent_formula_is_not_overwritten(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    report = run_improvement_loop(
        Chain([
            (EXPRESSION, ["momentum_7"], "first-id"),
            ("sigmoid(0.3 * volatility_7)", ["volatility_7"], "second-id"),
            ("sigmoid(0.1 * momentum_30)", ["momentum_30"], "third-id"),
        ]),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    parent = store.get(str(report.attempts[0].hypothesis_id))
    assert parent.expression == EXPRESSION
    assert parent.features == ("momentum_7",)
    assert memory.count() == 3
    assert all(item["claim_status"] == "not_confirmed" for item in memory.public_records(limit=8))


def test_same_experiment_is_not_stored_twice(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    same = _body(hypothesis_id="same-id", description="Повтор той же формулы.")
    client = Scripted([same, same])
    first = run_improvement_loop(
        client,
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert first.attempts[0].persisted is True
    assert first.status == "duplicate"
    assert store.count() == 1
    hypothesis_id = str(first.attempts[0].hypothesis_id)
    experiments = store.experiment_count(hypothesis_id)
    second = run_improvement_loop(
        Scripted([same]),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert second.status == "duplicate"
    assert store.count() == 1
    assert store.experiment_count(hypothesis_id) == experiments
    assert memory.count() == 1


def test_loop_stops_after_two_revisions() -> None:
    specs = [
        (EXPRESSION, ["momentum_7"], "one"),
        ("sigmoid(0.3 * volatility_7)", ["volatility_7"], "two"),
        ("sigmoid(0.1 * momentum_30)", ["momentum_30"], "three"),
        ("sigmoid(0.4 * log_return)", ["log_return"], "four"),
    ]
    client = Chain(specs)
    report = run_improvement_loop(client, created_at=WHEN, memory_guard=lambda: None)
    assert client.calls == 3
    assert report.revisions == 2
    assert report.unique_expressions == 3
    assert report.model_calls == 3
    assert [item.expression for item in report.attempts] == [item[0] for item in specs[:3]]


def test_error_stops_without_touching_the_journal(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    crashed = run_improvement_loop(
        Boom(),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    assert crashed.status == "aborted"
    assert "RuntimeError" in crashed.message
    assert store.count() == 0
    assert memory.count() == 0
    limited = run_improvement_loop(
        Scripted([_body()]),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: "мало свободной памяти",
    )
    assert limited.status == "resource_limited"
    assert limited.model_calls == 0
    assert store.count() == 0
    with exclusive_search("pysr"):
        busy = run_improvement_loop(
            Scripted([_body()]),
            store=store,
            memory=memory,
            commit=True,
            created_at=WHEN,
            memory_guard=lambda: None,
        )
    assert busy.status == "search_busy"
    assert store.count() == 0


def test_supported_status_cannot_be_assigned(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    memory = ResearchMemory(tmp_path / "memory.sqlite")
    report = run_improvement_loop(
        Scripted([_body(), _body()]),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        memory_guard=lambda: None,
    )
    hypothesis_id = str(report.attempts[0].hypothesis_id)
    assert store.confirms_law(hypothesis_id) is False
    assert store.get(hypothesis_id).claim_status == "not_confirmed"
    with pytest.raises(HypothesisLifecycleError):
        store.transition(hypothesis_id, "supported", "модель так сказала")
    forced = verify_model_output(_body(validated_law=True, hypothesis_id="other-id"))
    assert forced.code == "criteria_change"
    assert "supported" not in json.dumps(report.attempts[0].feedback)


def test_synthetic_dataset_is_not_a_bitcoin_claim() -> None:
    points = synthetic_study_points()
    assert points
    assert all(point.decision_date < date(2021, 1, 1) or point.decision_date <= date(2020, 12, 31) for point in points)
    assert all(point.target_date < CLOSED_FROM for point in points)
    assert all(point.label_available > point.decision_available for point in points)
    evaluation = evaluate_hypothesis("sigmoid(0.2 * momentum_7)", ("momentum_7",), 1, points)
    assert evaluation.accepted is True
    assert evaluation.advantage_claim is False
    assert evaluation.validated_law is False
    assert evaluation.fitted_on_2021 is False
    body = build_adapter_body(
        "qwen3.5:4b",
        [{"role": "user", "content": "тест"}],
        None,
        schema_mode="json",
        temperature=0,
        seed=1729,
        num_ctx=2048,
        num_predict=128,
        keep_alive=30,
    )
    assert body["keep_alive"] == 30
    assert not hasattr(QwenResearchBrain, "score")
    assert not hasattr(QwenResearchBrain, "declare_winner")
