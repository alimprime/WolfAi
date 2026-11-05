"""Этап 3.0: Qwen предлагает одну гипотезу, Python проверяет её без Ollama в сети."""

import inspect
import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import URLError

import pytest

from wolfai.research.budget import exclusive_search
from wolfai.research.formula import FEATURES
from wolfai.research.hypothesis_engine import HypothesisLifecycleError, HypothesisStore
from wolfai.research.math_screen import screen_formula, screen_score, score_synthetic
from wolfai.research.memory_context import load_research_excerpt
from wolfai.research.ollama_adapter import (
    STAGE30_NUM_CTX,
    STAGE30_NUM_PREDICT,
    OllamaAdapter,
    build_adapter_body,
)
from wolfai.research.prediction_ledger import PredictionLedger
from wolfai.research.qwen_brain import QwenResearchBrain
from wolfai.research.research_cycle import (
    HistoryPoint,
    reject_self_grade,
    reject_validated_law,
    run_research_cycle,
)
from wolfai.research.research_memory import ResearchMemory
import wolfai.research.hypothesis_protocol as hypothesis_protocol
import wolfai.research.math_screen as math_screen
import wolfai.research.memory_context as memory_context
import wolfai.research.ollama_adapter as ollama_adapter
import wolfai.research.qwen_brain as qwen_brain
import wolfai.research.research_cycle as research_cycle

WHEN = datetime(2026, 11, 5, tzinfo=timezone.utc)
EXPRESSION = "sigmoid(0.2 * momentum_7)"


def _hypothesis(**extra: object) -> str:
    body: dict[str, object] = {
        "hypothesis_id": "synth-momentum-7",
        "description": "Короткий импульс как синтетическая гипотеза.",
        "features": ["momentum_7"],
        "horizon_days": 1,
        "expression": EXPRESSION,
        "rationale": "Формула отличается от уже проверенных фиксированным коэффициентом.",
        "expected_regime": "any",
        "test_plan": "Посчитать Brier на синтетических точках и сравнить с константой 0.5.",
        "limitations": "Неопределённость высокая: синтетика не является рыночным преимуществом.",
        "parent_hypothesis_id": None,
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def _registry(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": "prior",
        "config_hash": "prior-hash",
        "dataset_fingerprint": "prior-data",
        "features": ["log_return"],
        "label_definition": "y = 1 только если close вырос",
        "horizon_days": 1,
        "train_start": "2019-02-01T00:00:00+00:00",
        "train_end": "2020-12-31T00:00:00+00:00",
        "dev_start": "2021-01-01T00:00:00+00:00",
        "dev_end": "2021-12-31T00:00:00+00:00",
        "seed": 1729,
        "evidence_role": "development_only",
        "claim_status": "insufficient_evidence",
        "rejection_reason": "MARKER-REJECT",
        "leak_suspicion": "MARKER-LEAK",
        "limitations": "окно подбора заканчивается в 2020",
        "holdout_brier": 0.123456,
    }
    payload.update(extra)
    return payload


class JsonBody:
    def __init__(self, payload: dict[str, object]) -> None:
        self.raw = json.dumps(payload).encode("utf-8")
        self.position = 0

    def read(self, limit: int = -1) -> bytes:
        if limit is None or limit < 0:
            chunk = self.raw[self.position:]
            self.position = len(self.raw)
            return chunk
        chunk = self.raw[self.position:self.position + limit]
        self.position += len(chunk)
        return chunk

    def __enter__(self) -> "JsonBody":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class Scripted:
    def __init__(self, content: str | None, *, status: str = "ok") -> None:
        self.content = content
        self.status = status
        self.calls = 0
        self.seen: list[str] = []

    def generate(self, messages, schema=None, *, temperature: float = 0.0):
        self.calls += 1
        self.seen.append("\n".join(item["content"] for item in messages))
        self.temperature = temperature
        self.schema = schema
        from wolfai.research.ollama_adapter import AdapterResult

        return AdapterResult(
            status=self.status,
            message="scripted",
            content=self.content,
            model_name="qwen3.5:4b",
            model_digest="digest-test",
            elapsed_seconds=0.01,
            rss_bytes=2048,
            attempts_used=1,
        )


class Boom:
    def generate(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("обрыв генерации")


def _store(tmp_path: Path, name: str = "hypotheses.sqlite") -> HypothesisStore:
    return HypothesisStore(tmp_path / name)


def _memory(tmp_path: Path, name: str = "memory.sqlite") -> ResearchMemory:
    return ResearchMemory(tmp_path / name)


def test_unavailable_ollama_returns_status(tmp_path: Path) -> None:
    def transport(request, timeout):
        raise URLError("down")

    adapter = OllamaAdapter(transport=transport, memory_guard=lambda: None, timeout=1)
    result = adapter.generate(
        [{"role": "user", "content": "ping"}],
        hypothesis_protocol.QWEN_HYPOTHESIS_SCHEMA,
    )
    assert result.status == "ollama_unavailable"
    assert result.attempts_used == 3
    store = _store(tmp_path)
    memory = _memory(tmp_path)
    report = run_research_cycle(adapter, store=store, memory=memory, commit=True, created_at=WHEN)
    assert report.status == "ollama_unavailable"
    assert store.count() == 0
    assert store.attempt_count() == 0
    assert memory.count() == 0


def test_missing_model_is_not_replaced(tmp_path: Path) -> None:
    seen: list[str] = []

    def transport(request, timeout):
        seen.append(request.full_url)
        if request.full_url.endswith("/api/version"):
            return JsonBody({"version": "0.40.9"})
        if request.full_url.endswith("/api/tags"):
            return JsonBody({"models": [
                {"name": "qwen3:8b", "digest": "eight"},
                {"name": "qwen3.6:latest", "digest": "large"},
            ]})
        raise AssertionError(request.full_url)

    adapter = OllamaAdapter(transport=transport, memory_guard=lambda: None, model="qwen3.5:4b")
    result = adapter.generate([{"role": "user", "content": "ping"}], hypothesis_protocol.QWEN_HYPOTHESIS_SCHEMA)
    assert result.status == "model_missing"
    assert "qwen3.5:4b" in result.message
    assert "скачивание не выполняется" in result.message
    assert result.model_name != "qwen3:8b"
    assert all("/api/pull" not in url for url in seen)
    assert all(not url.endswith("/api/chat") for url in seen)
    report = run_research_cycle(adapter, store=_store(tmp_path), memory=_memory(tmp_path), commit=True, created_at=WHEN)
    assert report.status == "model_missing"
    assert report.model_calls == 1
    assert report.persisted is False


def test_invalid_json_is_saved_as_rejection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = run_research_cycle(Scripted("{это не json"), store=store, created_at=WHEN)
    assert report.status == "invalid_json"
    assert report.rejection_saved is True
    assert report.persisted is False
    assert store.count() == 0
    assert any("JSON" in item for item in store.failure_details())
    horizon = run_research_cycle(Scripted(_hypothesis(horizon_days=2)), store=_store(tmp_path, "other.sqlite"), created_at=WHEN)
    assert horizon.status == "bad_horizon"
    assert horizon.persisted is False


def test_unknown_feature_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = run_research_cycle(
        Scripted(_hypothesis(expression="sigmoid(0.2 * funding_rate)", features=["funding_rate"])),
        store=store,
        created_at=WHEN,
    )
    assert report.status == "unknown_feature"
    assert report.rejection_saved is True
    assert store.count() == 0
    assert "funding_rate" in (report.rejection_reason or "")


def test_dangerous_math_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "owned.txt"
    store = _store(tmp_path)
    dangerous = run_research_cycle(
        Scripted(_hypothesis(expression="sigmoid(__import__('os'))")),
        store=store,
        created_at=WHEN,
    )
    assert dangerous.status == "unsafe_math"
    assert not marker.exists()
    unknown = run_research_cycle(
        Scripted(_hypothesis(expression="sigmoid(exp(momentum_7))")),
        store=store,
        created_at=WHEN,
    )
    assert unknown.status == "unsafe_math"
    assert store.count() == 0
    for module in (
        hypothesis_protocol,
        math_screen,
        memory_context,
        qwen_brain,
        research_cycle,
        ollama_adapter,
    ):
        source = inspect.getsource(module)
        assert "eval(" not in source
        assert "exec(" not in source


def test_too_complex_formula_is_rejected(tmp_path: Path) -> None:
    expression = "sigmoid(tanh(tanh(tanh(tanh(tanh(momentum_7))))))"
    store = _store(tmp_path)
    report = run_research_cycle(
        Scripted(_hypothesis(expression=expression)),
        store=store,
        created_at=WHEN,
    )
    assert report.status == "too_complex"
    assert report.rejection_saved is True
    assert store.count() == 0


def test_duplicate_of_rejected_formula_is_not_new_research(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hypothesis_id, state = store.create(
        author_id="qwen-research-brain",
        name="old-momentum",
        description="Прежняя формула короткого импульса.",
        creation_reason="синтетическая регистрация",
        features=("momentum_7",),
        horizon_days=1,
        expression=EXPRESSION,
        applicability={"market": "synthetic", "regimes": ["any"], "btc_claim": False},
        created_at=WHEN,
        data_hash="synthetic-stage30",
    )
    assert state == "stored"
    store.transition(hypothesis_id, "testing", "появились наблюдения", when=WHEN)
    store.transition(hypothesis_id, "rejected", "systematic_deterioration", when=WHEN)
    before = store.attempt_count()
    report = run_research_cycle(
        Scripted(_hypothesis(expression="sigmoid( 0.2 * momentum_7 )")),
        store=store,
        created_at=WHEN,
    )
    assert report.status == "duplicate"
    assert report.duplicate_of == hypothesis_id
    assert "отвергнут" in report.message
    assert report.expression is not None
    assert store.count() == 1
    assert store.get(hypothesis_id).status == "rejected"
    assert store.get(hypothesis_id).expression == EXPRESSION
    assert store.attempt_count() == before
    assert store.failure_details() == []


def test_unavailable_research_memory_stops_before_model(tmp_path: Path) -> None:
    class Broken:
        def public_records(self, limit: int = 8) -> list[dict[str, object]]:
            raise sqlite3.DatabaseError("файл повреждён")

    client = Scripted(_hypothesis())
    store = _store(tmp_path)
    report = run_research_cycle(client, store=store, memory=Broken(), created_at=WHEN)
    assert report.status == "memory_unavailable"
    assert client.calls == 0
    assert store.count() == 0
    assert store.attempt_count() == 0


def test_time_boundaries_block_new_fitting(tmp_path: Path) -> None:
    client = Scripted(_hypothesis())
    store = _store(tmp_path)
    cases = (
        {"touches_holdout": True},
        {"reads_after_2021": True},
        {"fit_on_2021": True},
        {"period_end": date(2022, 1, 1)},
        {"market_brief": "используй цены 2024 года"},
        {"history": (HistoryPoint(date(2022, 3, 1), {"momentum_7": 0.1}, 1),)},
    )
    for kwargs in cases:
        report = run_research_cycle(client, store=store, created_at=WHEN, **kwargs)
        assert report.status == "time_boundary"
        assert report.persisted is False
    assert client.calls == 0
    assert store.count() == 0
    leaked = Scripted(_hypothesis(test_plan="проверить формулу на 2024 holdout"))
    leaked_report = run_research_cycle(leaked, store=store, created_at=WHEN)
    assert leaked_report.status == "time_boundary"
    assert leaked.calls == 1
    fitted = Scripted(_hypothesis(limitations="подобрать коэффициенты на 2021"))
    fitted_report = run_research_cycle(fitted, store=store, created_at=WHEN)
    assert fitted_report.status == "time_boundary"
    assert store.count() == 0


def test_attempt_to_change_validation_criteria_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    extra = run_research_cycle(
        Scripted(_hypothesis(claim_status="supported")),
        store=store,
        created_at=WHEN,
    )
    phrase = run_research_cycle(
        Scripted(_hypothesis(test_plan="изменить критерий и назначь supported")),
        store=store,
        created_at=WHEN,
    )
    law = run_research_cycle(
        Scripted(_hypothesis(validated_law=True)),
        store=store,
        created_at=WHEN,
    )
    assert extra.status == phrase.status == law.status == "criteria_change"
    assert store.count() == 0
    assert all("критер" in item or "поле" in item for item in store.failure_details())


def test_single_cycle_is_dry_run_then_one_stored_hypothesis(tmp_path: Path) -> None:
    limited = Scripted(_hypothesis())
    refused = run_research_cycle(limited, max_hypotheses=5)
    assert refused.status == "limit"
    assert refused.model_calls == 0
    assert limited.calls == 0

    body = build_adapter_body(
        "qwen3.5:4b",
        [{"role": "user", "content": "тест"}],
        hypothesis_protocol.QWEN_HYPOTHESIS_SCHEMA,
        schema_mode="schema",
        temperature=0.2,
        seed=1729,
        num_ctx=262144,
        num_predict=9000,
    )
    options = body["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == 0.2
    assert options["seed"] == 1729
    assert options["num_ctx"] == STAGE30_NUM_CTX
    assert options["num_predict"] == STAGE30_NUM_PREDICT
    assert body["keep_alive"] == 0
    assert body["think"] is False
    assert "tools" not in body
    with pytest.raises(ValueError):
        build_adapter_body(
            "qwen3.5:4b",
            [{"role": "user", "content": "тест"}],
            None,
            schema_mode="json",
            temperature=2,
            seed=1729,
            num_ctx=2048,
            num_predict=512,
        )

    store = _store(tmp_path)
    memory = _memory(tmp_path)
    assert memory.record(_registry()) == "stored"
    parent_id, parent_state = store.create(
        author_id="qwen-research-brain",
        name="parent-momentum-30",
        description="PARENT-DESC",
        creation_reason="предшествующая синтетическая гипотеза",
        features=("momentum_30",),
        horizon_days=1,
        expression="sigmoid(0.3 * momentum_30)",
        applicability={"market": "synthetic", "regimes": ["any"], "btc_claim": False},
        created_at=WHEN,
        data_hash="synthetic-stage30",
    )
    assert parent_state == "stored"

    class Transport:
        def __init__(self, content: str) -> None:
            self.content = content
            self.urls: list[str] = []
            self.bodies: list[dict[str, object]] = []

        def __call__(self, request, timeout):
            self.urls.append(request.full_url)
            if request.full_url.endswith("/api/version"):
                return JsonBody({"version": "0.40.9"})
            if request.full_url.endswith("/api/tags"):
                return JsonBody({"models": [
                    {"name": "qwen3:8b", "digest": "eight"},
                    {
                        "name": "qwen3.5:4b",
                        "digest": "digest-stage30",
                        "details": {"parameter_size": "4.2B"},
                    },
                ]})
            if request.full_url.endswith("/api/chat"):
                self.bodies.append(json.loads(request.data.decode()))
                return JsonBody({"message": {"content": self.content}})
            raise AssertionError(request.full_url)

    transport = Transport(_hypothesis(parent_hypothesis_id=parent_id))
    adapter = OllamaAdapter(transport=transport, memory_guard=lambda: None, model="qwen3.5:4b")
    dry = run_research_cycle(
        adapter,
        store=store,
        memory=memory,
        created_at=WHEN,
        data_hash="synthetic-stage30",
    )
    assert dry.status == "dry_run"
    assert dry.persisted is False
    assert dry.model_calls == 1
    assert dry.model_name == "qwen3.5:4b"
    assert dry.claim_status == "not_confirmed"
    assert dry.validated_law is False
    assert dry.advantage_claim is False
    assert dry.fitted_on_2021 is False
    assert dry.synthetic is not None
    assert dry.synthetic["btc_claim"] is False
    assert dry.graded_by == "verification-desk"
    assert store.count() == 1
    prompt = str(transport.bodies[0]["messages"])
    assert "MARKER-REJECT" in prompt
    assert "MARKER-LEAK" in prompt
    assert "PARENT-DESC" in prompt
    assert "0.123456" not in prompt
    assert "holdout" not in prompt.lower()
    chat = transport.bodies[0]
    assert chat["model"] == "qwen3.5:4b"
    assert "tools" not in chat
    assert chat["options"]["temperature"] == 0
    assert chat["options"]["seed"] == 1729
    assert all("/api/pull" not in url for url in transport.urls)

    stored = run_research_cycle(
        adapter,
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
        data_hash="synthetic-stage30",
    )
    assert stored.status == "stored"
    assert stored.persisted is True
    assert stored.dry_run is False
    assert stored.model_calls == 1
    assert stored.hypothesis_id is not None
    assert stored.hypothesis_id != "synth-momentum-7"
    assert len(stored.hypothesis_id) == 64
    assert store.count() == 2
    assert store.derived_ids(parent_id) == [stored.hypothesis_id]
    view = store.get(stored.hypothesis_id)
    assert view.author_id == "qwen-research-brain"
    assert view.claim_status == "not_confirmed"
    assert view.status == "testing"
    assert view.version == 2
    assert view.parent_hypothesis_id == parent_id
    assert view.expression == EXPRESSION
    assert store.confirms_law(stored.hypothesis_id) is False
    assert memory.count() == 2
    recorded = [item for item in memory.public_records(limit=8) if item.get("hypothesis_id") == stored.hypothesis_id]
    assert recorded[0]["claim_status"] == "not_confirmed"
    assert recorded[0]["validated_law"] is False
    assert recorded[0]["fitted"] is False
    assert recorded[0]["diagnostic_2021"] == "DEV_ONLY_NOT_USED"
    assert recorded[0]["generator_version"] == "3.0.0"


def test_validated_law_cannot_be_created(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory = _memory(tmp_path)
    report = run_research_cycle(
        Scripted(_hypothesis()),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
    )
    assert report.status == "stored"
    assert report.validated_law is False
    assert report.claim_status == "not_confirmed"
    assert report.advantage_claim is False
    assert store.confirms_law(str(report.hypothesis_id)) is False
    with pytest.raises(HypothesisLifecycleError):
        store.transition(str(report.hypothesis_id), "supported", "модель так сказала")
    with pytest.raises(HypothesisLifecycleError):
        reject_validated_law("supported", False)
    with pytest.raises(HypothesisLifecycleError):
        reject_validated_law("validated_law", False)
    assert store.get(str(report.hypothesis_id)).claim_status == "not_confirmed"
    forced = run_research_cycle(
        Scripted(_hypothesis(hypothesis_id="other-id", expression="sigmoid(0.4 * log_return)", features=["log_return"], validated_law=True)),
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
    )
    assert forced.status == "criteria_change"
    assert store.count() == 1


def test_future_agents_stay_isolated(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory = _memory(tmp_path)
    agent_a, _state = store.create(
        author_id="agent-a",
        name="secret-a",
        description="SECRET-A",
        creation_reason="частная гипотеза агента A",
        features=("log_return",),
        horizon_days=1,
        expression="sigmoid(0.4 * log_return)",
        applicability={"market": "synthetic", "regimes": ["any"], "btc_claim": False},
        created_at=WHEN,
        data_hash="synthetic-stage30",
    )
    excerpt, error = load_research_excerpt(memory, store, author_id="agent-b")
    assert error is None and excerpt is not None
    visible = json.dumps(excerpt.as_dict(), ensure_ascii=False)
    assert "SECRET-A" not in visible
    assert "sigmoid(0.4 * log_return)" in visible
    assert excerpt.hypotheses[0]["author"] == "other"
    brain_a = QwenResearchBrain("agent-a")
    brain_b = QwenResearchBrain("agent-b")
    assert brain_a.author_id != brain_b.author_id
    assert not hasattr(brain_b, "transition")
    assert not hasattr(brain_b, "prediction_ledger")
    assert "PredictionLedger" not in inspect.getsource(qwen_brain)
    ledger = PredictionLedger(tmp_path / "predictions.sqlite")
    blocked = run_research_cycle(
        Scripted(_hypothesis()),
        author_id="agent-b",
        store=store,
        memory=memory,
        prediction_ledger=ledger,
        created_at=WHEN,
    )
    assert blocked.status == "ledger_forbidden"
    connection = sqlite3.connect(tmp_path / "predictions.sqlite")
    assert connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    connection.close()
    impersonation = run_research_cycle(
        Scripted(_hypothesis(author_id="agent-a")),
        author_id="agent-b",
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
    )
    assert impersonation.status == "agent_isolation"
    assert store.count() == 1
    assert store.get(agent_a).status == "proposed"
    client = Scripted(_hypothesis(
        hypothesis_id="agent-b-vol",
        expression="sigmoid(0.5 * volatility_7)",
        features=["volatility_7"],
        description="Гипотеза агента B.",
    ))
    stored = run_research_cycle(
        client,
        author_id="agent-b",
        store=store,
        memory=memory,
        commit=True,
        created_at=WHEN,
    )
    assert stored.status == "stored"
    assert store.get(str(stored.hypothesis_id)).author_id == "agent-b"
    assert store.get(agent_a).author_id == "agent-a"
    assert store.get(agent_a).description == "SECRET-A"
    assert "SECRET-A" not in client.seen[0]
    assert stored.graded_by == "verification-desk"
    assert stored.graded_by != "agent-b"
    with pytest.raises(PermissionError):
        reject_self_grade("agent-b", "agent-b")
    experiments = store.experiments_for("agent-a")
    assert experiments == []
    assert store.experiments_for("agent-b")[0]["author_id"] == "agent-b"


def test_failure_leaves_hypothesis_journal_intact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory = _memory(tmp_path)
    crashed = run_research_cycle(Boom(), store=store, memory=memory, commit=True, created_at=WHEN)
    assert crashed.status == "aborted"
    assert "RuntimeError" in crashed.message
    assert store.count() == 0
    assert store.attempt_count() == 0
    assert memory.count() == 0

    calls: list[str] = []

    def transport(request, timeout):
        calls.append(request.full_url)
        raise AssertionError("модель не должна вызываться")

    guarded = OllamaAdapter(transport=transport, memory_guard=lambda: "мало свободной памяти", model="qwen3.5:4b")
    limited = run_research_cycle(guarded, store=store, memory=memory, commit=True, created_at=WHEN)
    assert limited.status == "resource_limited"
    assert calls == []
    assert store.count() == 0

    with exclusive_search("pysr"):
        busy = run_research_cycle(
            Scripted(_hypothesis()),
            store=store,
            memory=memory,
            commit=True,
            created_at=WHEN,
        )
    assert busy.status == "search_busy"
    assert store.count() == 0
    assert memory.count() == 0


def test_synthetic_evaluation_makes_no_market_claim(tmp_path: Path) -> None:
    synthetic = score_synthetic(EXPRESSION)
    assert synthetic["observations"] == 40
    assert 0.0 <= float(synthetic["brier"]) <= 1.0
    assert math_isfinite(float(synthetic["brier"]))
    assert synthetic["claim_status"] == "not_confirmed"
    assert synthetic["validated_law"] is False
    assert synthetic["advantage_claim"] is False
    assert synthetic["btc_claim"] is False
    assert synthetic["fitted_on_2021"] is False
    assert synthetic["graded_by"] == "verification-desk"
    refused = screen_score("0.2 * momentum_7", "fit_2021", FEATURES)
    assert refused.accepted is False
    assert refused.code == "calibration_refused"
    wrapped = screen_score("0.2 * momentum_7", "explicit_sigmoid", FEATURES)
    assert wrapped.accepted is True
    assert wrapped.probabilities
    assert all(0.0 < value < 1.0 and math_isfinite(value) for value in wrapped.probabilities)
    division = screen_formula("sigmoid(momentum_7 / 0.0000001)", FEATURES)
    assert division.accepted is False
    assert division.code == "unsafe_math"
    missing = screen_formula(EXPRESSION, ["log_return"])
    assert missing.code == "missing_feature"
    features = {name: 0.1 for name in FEATURES}
    report = run_research_cycle(
        Scripted(_hypothesis()),
        history=(
            HistoryPoint(date(2019, 6, 1), features, 1),
            HistoryPoint(date(2021, 6, 1), features, 0),
        ),
        created_at=WHEN,
    )
    assert report.status == "dry_run"
    assert report.dev_2021_skipped == 1
    assert report.fitted_on_2021 is False
    assert report.internal_2019_2020 is not None
    assert report.internal_2019_2020["period"] == "2019-2020"
    assert report.internal_2019_2020["observations"] == 1
    assert report.internal_2019_2020["claim_status"] == "not_confirmed"
    assert report.internal_2019_2020["validated_law"] is False
    assert report.synthetic is not None
    assert report.synthetic["period"] == "synthetic"


def math_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
