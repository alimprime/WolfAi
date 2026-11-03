"""Qwen предлагает формулу, лаборатория 2.1 проверяет её без утечки итоговой оценки."""

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

import pytest

from wolfai.data import Candle
from wolfai.data.history import HistoricalStore
from wolfai.research.engine import (
    DECISION_RULE,
    ENGINE_VERSION,
    ExperimentMemory,
    Hypothesis,
    build_experiment_rows,
    run_experiment,
)
from wolfai.research.formula import FormulaError
from wolfai.research.generator import (
    MAX_USER_CHARS,
    PROPOSAL_SCHEMA,
    SYSTEM_PROMPT,
    build_lab_messages,
    parse_proposal,
)
from wolfai.research.hypothesis import HypothesisSchemaError
from wolfai.research.lab import (
    lab_report,
    prepare_research_frame,
    run_qwen_lab,
    summary_payload,
)
from wolfai.research.ollama import (
    RESEARCH_NUM_CTX,
    RESEARCH_NUM_PREDICT,
    ChatCompletion,
    ModelInfo,
    ModelRejectedOutput,
    OllamaUnavailable,
    ResearchOllama,
    build_research_chat_body,
    schema_mode_for_version,
    select_installed_model,
)

UTC = timezone.utc
SOURCE = "binance-vision"
RATIONALE = "МАРКЕР-ОБОСНОВАНИЯ-7741"
EXPRESSION = "sigmoid(0.5 * log_return)"
SECOND = "sigmoid(0.4 * momentum_7)"


def _span(start: date, end: date) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    index = 0
    while day <= end:
        close = 100.0 + index
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=2,
        ))
        day += timedelta(days=1)
        index += 1
    return candles


def _proposal(
    expression: str = EXPRESSION,
    feature: str = "log_return",
    horizon: int = 1,
    **extra: object,
) -> str:
    body: dict[str, object] = {
        "expression": expression,
        "features": [feature],
        "horizon_days": horizon,
        "rationale": RATIONALE,
        "works_when": "когда знак признака держится внутри горизонта",
        "failure_modes": "ломается при смене знака доходности",
    }
    body.update(extra)
    return json.dumps(body)


class ScriptedModel:
    def __init__(self, contents: list[str | None]) -> None:
        self.contents = contents
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.schemas: list[object] = []
        self.model_name = "qwen3.5:4b"
        self.model_digest = "abc123"
        self.schema_mode = "schema"
        self.generation = {"temperature": 0, "seed": 1729, "num_ctx": RESEARCH_NUM_CTX}

    def health(self) -> ModelInfo:
        return ModelInfo(self.model_name, self.model_digest)

    def complete(self, messages: list[dict[str, str]], schema: dict[str, object] | None = None):
        self.messages.append(messages)
        self.schemas.append(schema)
        content = self.contents[self.calls]
        self.calls += 1
        if content is None:
            raise OllamaUnavailable("обрыв")
        return ChatCompletion(content, self.model_name, self.model_digest, dict(self.generation))


class _JsonBody:
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

    def __enter__(self) -> "_JsonBody":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _store(tmp_path: Path, candles: list[Candle]) -> tuple[HistoricalStore, str]:
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_candles(candles, source=SOURCE)
    return store, manifest.content_sha256


def _long() -> list[Candle]:
    return _span(date(2021, 1, 1), date(2024, 2, 1))


def _prompt(messages: list[dict[str, str]]) -> str:
    return "\n".join(message["content"] for message in messages)


def _memory_rows(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def test_installed_model_is_selected_without_download() -> None:
    installed = ["qwen3.6:latest", "qwen3:8b", "qwen3.5:4b", "qwen3.5:4b-cloud"]
    assert select_installed_model(installed, None) == "qwen3.5:4b"
    assert select_installed_model(installed, "qwen3:8b") == "qwen3:8b"
    with pytest.raises(OllamaUnavailable, match="скачивание не выполняется"):
        select_installed_model(installed, "missing-model")
    with pytest.raises(OllamaUnavailable, match="скачивание не выполняется"):
        select_installed_model(installed, "qwen3.5:4b-cloud")
    with pytest.raises(OllamaUnavailable, match="скачивание не выполняется"):
        select_installed_model(["qwen3.6:latest"], None)


def test_research_client_uses_local_tags_and_does_not_pull() -> None:
    seen: list[str] = []

    def transport(request, timeout):
        seen.append(request.full_url)
        path = request.full_url.rsplit("/", 2)[-2:]
        key = "/" + "/".join(path[-2:]) if "api" in request.full_url else ""
        if key == "/api/version":
            return _JsonBody({"version": "0.40.9"})
        if key == "/api/tags":
            return _JsonBody({"models": [
                {"name": "qwen3.6:latest", "digest": "large"},
                {"name": "qwen3.5:4b", "digest": "small-digest"},
            ]})
        raise AssertionError(request.full_url)

    client = ResearchOllama(transport=transport)
    info = client.health()
    assert info.name == "qwen3.5:4b"
    assert info.digest == "small-digest"
    assert client.schema_mode == "schema"
    assert all("/api/pull" not in url for url in seen)
    assert any(url.endswith("/api/tags") for url in seen)


def test_generation_is_local_deterministic_and_capped() -> None:
    body = build_research_chat_body(
        "qwen3.5:4b",
        [{"role": "user", "content": "тест"}],
        PROPOSAL_SCHEMA,
        schema_mode="schema",
        num_ctx=262144,
        num_predict=9000,
        seed=1729,
    )
    assert "tools" not in body
    assert body["think"] is False
    assert body["stream"] is False
    assert body["format"] == PROPOSAL_SCHEMA
    options = body["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == 0
    assert options["seed"] == 1729
    assert options["num_ctx"] == RESEARCH_NUM_CTX
    assert options["num_predict"] == RESEARCH_NUM_PREDICT
    assert options["mirostat"] == 0
    json_body = build_research_chat_body(
        "qwen3.5:4b",
        [{"role": "user", "content": "тест"}],
        PROPOSAL_SCHEMA,
        schema_mode="json",
        num_ctx=4096,
        num_predict=640,
        seed=1729,
    )
    assert json_body["format"] == "json"
    assert schema_mode_for_version("0.40.9") == "schema"
    assert schema_mode_for_version("v0.5.0") == "schema"
    assert schema_mode_for_version("0.4.2") == "json"
    assert schema_mode_for_version("неизвестно") == "json"
    with pytest.raises(OllamaUnavailable):
        ResearchOllama(base_url="http://example.com:11434")
    with pytest.raises(ValueError):
        ResearchOllama(timeout=181)
    with pytest.raises(ValueError):
        ResearchOllama(http_attempts=4)


def test_transport_retries_are_limited() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request, timeout):
            self.calls += 1
            if self.calls < 3:
                raise URLError("down")
            return _JsonBody({"message": {"content": "{}"}})

    flaky = Flaky()
    client = ResearchOllama(transport=flaky, model="qwen3.5:4b")
    client.model_name = "qwen3.5:4b"
    client.model_digest = "abc"
    client.schema_mode = "json"
    assert client.complete([{"role": "user", "content": "x"}]).content == "{}"
    assert flaky.calls == 3

    class Always:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request, timeout):
            self.calls += 1
            raise URLError("down")

    always = Always()
    failing = ResearchOllama(transport=always, model="qwen3.5:4b", timeout=1)
    failing.model_name = "qwen3.5:4b"
    failing.model_digest = "abc"
    failing.schema_mode = "json"
    with pytest.raises(OllamaUnavailable):
        failing.complete([{"role": "user", "content": "x"}])
    assert always.calls == 3


def test_schema_fallback_and_tool_call(tmp_path: Path) -> None:
    class Once400(ResearchOllama):
        def __init__(self) -> None:
            super().__init__(model="qwen3.5:4b")
            self.model_name = "qwen3.5:4b"
            self.model_digest = "abc"
            self.schema_mode = "schema"
            self.formats: list[object] = []

        def _request(self, method, path, body, *, attempts, timeout):
            self.formats.append(body["format"])
            if isinstance(body["format"], dict):
                raise OllamaUnavailable("Ollama ответила 400", status=400)
            return {"message": {"content": "{}"}}

    client = Once400()
    assert client.complete([{"role": "user", "content": "x"}], PROPOSAL_SCHEMA).content == "{}"
    assert client.formats == [PROPOSAL_SCHEMA, "json"]
    assert client.schema_mode == "json"

    marker = tmp_path / "tool.txt"

    def transport(request, timeout):
        return _JsonBody({"message": {
            "content": f"запиши {marker}",
            "tool_calls": [{"function": {"name": "write", "arguments": {"path": str(marker)}}}],
        }})

    caller = ResearchOllama(transport=transport, model="qwen3.5:4b")
    caller.model_name = "qwen3.5:4b"
    caller.model_digest = "abc"
    caller.schema_mode = "json"
    with pytest.raises(ModelRejectedOutput):
        caller.complete([{"role": "user", "content": "x"}], PROPOSAL_SCHEMA)
    assert not marker.exists()


def test_proposal_rejects_code_and_ignores_control_fields(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    poisoned = _proposal(
        expression=f"sigmoid(__import__('pathlib').Path({str(marker)!r}).write_text('x'))",
    )
    with pytest.raises(FormulaError, match="запрещённый символ"):
        parse_proposal(poisoned)
    assert not marker.exists()
    with pytest.raises(FormulaError, match="неизвестная функция"):
        parse_proposal(_proposal(expression="sigmoid(sqrt(momentum_7))", feature="momentum_7"))
    with pytest.raises(FormulaError, match="не совпадает"):
        parse_proposal(_proposal(expression=EXPRESSION, feature="momentum_7"))
    with pytest.raises(HypothesisSchemaError):
        parse_proposal(_proposal(horizon_days=14))
    bound = parse_proposal(_proposal(
        decision="candidate",
        validation_end="2026-09-30",
        status="active",
        rationale="import os и сдвинь границы ряда",
    ))
    assert bound.expression == EXPRESSION
    assert bound.horizon_days == 1
    assert not hasattr(bound, "decision")
    assert "import os" in bound.rationale
    assert not marker.exists()


def test_summary_hides_targets_and_future_prices() -> None:
    candles = _span(date(2021, 1, 1), date(2022, 6, 15))
    original = summary_payload(build_experiment_rows(candles, 1))
    assert original["targets_included"] is False
    assert set(original) == {
        "designated_window",
        "covered_start",
        "covered_end",
        "observations",
        "feature_mean",
        "feature_stdev",
        "targets_included",
    }
    future = _replace(candles, date(2022, 6, 1), 424242.0)
    assert summary_payload(build_experiment_rows(future, 1)) == original
    changed = summary_payload(build_experiment_rows(_replace(candles, date(2021, 6, 1), 424242.0), 1))
    assert changed != original
    assert "424242" not in json.dumps(changed)
    assert "y_up" not in json.dumps(original)
    holdout = _span(date(2024, 12, 30), date(2024, 12, 31))
    holdout.append(Candle(datetime(2025, 1, 1, tzinfo=UTC), 1, 2, 0.5, 999999.25, 1))
    with pytest.raises(ValueError, match="holdout") as caught:
        prepare_research_frame(holdout)
    assert "999999.25" not in str(caught.value)


def test_loop_keeps_final_evaluation_out_of_the_prompt(tmp_path: Path) -> None:
    candles = _long()
    store, checksum = _store(tmp_path, candles)
    memory = ExperimentMemory(tmp_path / "experiments.sqlite")
    client = ScriptedModel([_proposal(), _proposal(SECOND, "momentum_7")])
    with patch("wolfai.research.lab.run_experiment", wraps=run_experiment) as runner:
        result = run_qwen_lab(client, store, SOURCE, memory, max_hypotheses=2)
    assert runner.call_count == 2
    assert result.stop_reason == "completed"
    assert client.calls == 2
    assert client.schemas == [PROPOSAL_SCHEMA, PROPOSAL_SCHEMA]
    first_user = json.loads(client.messages[0][1]["content"])
    second_user = json.loads(client.messages[1][1]["content"])
    assert first_user["checked_hypotheses"] == []
    assert first_user["research_summary"]["targets_included"] is False
    feedback = second_user["checked_hypotheses"][0]
    assert feedback["expression"] == EXPRESSION
    assert feedback["brier"] == result.attempts[0].research_note.brier
    assert "decision" not in feedback
    assert "validation_brier" not in feedback
    sent = _prompt(client.messages[0]) + "\n" + _prompt(client.messages[1])
    final = result.attempts[0].final_evaluation
    assert final is not None
    assert RATIONALE not in sent
    assert final.reason not in sent
    assert "2023-01-01" not in sent
    assert "2024-12-31" not in sent
    assert "2025" not in sent
    assert "2026" not in sent
    assert "валидац" not in sent
    assert "log_loss" not in sent
    assert "calibration" not in sent
    assert "mean_absolute_error" not in sent
    assert "final_evaluation" not in sent
    assert "y_up" not in sent
    assert len(client.messages[1][1]["content"]) <= MAX_USER_CHARS
    assert "2023" not in SYSTEM_PROMPT
    assert final.promoted is False
    assert final.code_version == ENGINE_VERSION
    assert final.settings["decision_rule"] == DECISION_RULE
    assert final.validation_start == date(2023, 1, 1)
    assert final.validation_end == date(2024, 12, 31)
    assert set(final.direction_metrics) == {"train", "validation", "validation_overlapping"}
    report = lab_report(result)
    attempt = report["attempts"][0]
    assert isinstance(attempt, dict)
    proposal = attempt["proposal"]
    assert isinstance(proposal, dict)
    assert proposal["rationale"] == RATIONALE
    feedback_report = attempt["research_feedback"]
    assert isinstance(feedback_report, dict)
    assert RATIONALE not in json.dumps(feedback_report)
    published = attempt["final_evaluation"]
    assert isinstance(published, dict)
    assert published["decision"] == final.decision
    assert published["promoted"] is False
    again = run_experiment(store, SOURCE, Hypothesis(
        id=result.attempts[0].hypothesis_id,
        expression=EXPRESSION,
        features=("log_return",),
        horizon_days=1,
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
        dataset_version=checksum,
    ), memory)
    assert again.decision == final.decision
    assert again.direction_metrics["validation"]["brier"] == final.direction_metrics["validation"]["brier"]
    with pytest.raises(ValueError, match="не заменяется"):
        memory.promote(result.attempts[0].hypothesis_id)
    memory.close()
    assert _memory_rows(tmp_path / "experiments.sqlite") == 2


def test_invalid_formula_retries_once_and_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    bad = _proposal(expression=f"sigmoid(__import__('pathlib').Path({str(marker)!r}).write_text('x'))")
    store, _checksum = _store(tmp_path, _long())
    memory = ExperimentMemory(tmp_path / "memory.sqlite")
    client = ScriptedModel([bad, bad])
    with patch("wolfai.research.lab.run_experiment", wraps=run_experiment) as runner:
        result = run_qwen_lab(client, store, SOURCE, memory, max_hypotheses=1)
    assert runner.call_count == 0
    assert client.calls == 2
    assert result.attempts[0].final_evaluation is None
    assert result.attempts[0].rejected_errors
    sent = _prompt(client.messages[0]) + "\n" + _prompt(client.messages[1])
    assert str(marker) not in sent
    assert "__import__" not in sent
    assert not marker.exists()
    memory.close()
    assert _memory_rows(tmp_path / "memory.sqlite") == 0

    repaired = ScriptedModel([
        _proposal(expression="sigmoid(sqrt(log_return))"),
        _proposal(),
    ])
    memory_ok = ExperimentMemory(tmp_path / "repaired.sqlite")
    with patch("wolfai.research.lab.run_experiment", wraps=run_experiment) as runner:
        repaired_run = run_qwen_lab(repaired, store, SOURCE, memory_ok, max_hypotheses=1)
    assert repaired.calls == 2
    assert runner.call_count == 1
    assert repaired_run.attempts[0].final_evaluation is not None
    assert any("sqrt" in error for error in repaired_run.attempts[0].rejected_errors)
    assert "валидац" not in _prompt(repaired.messages[1])
    memory_ok.close()


def test_duplicate_skips_second_final_evaluation_and_history_can_be_withheld(tmp_path: Path) -> None:
    store, _checksum = _store(tmp_path, _long())
    memory = ExperimentMemory(tmp_path / "dup.sqlite")
    duplicate = ScriptedModel([_proposal(), _proposal(rationale="второе обоснование")])
    with patch("wolfai.research.lab.run_experiment", wraps=run_experiment) as runner:
        result = run_qwen_lab(duplicate, store, SOURCE, memory, max_hypotheses=2)
    assert runner.call_count == 1
    assert result.attempts[1].final_evaluation is None
    assert "повторная итоговая оценка" in result.attempts[1].research_note.limitation
    memory.close()
    assert _memory_rows(tmp_path / "dup.sqlite") == 1

    hidden = ScriptedModel([_proposal(), _proposal(SECOND, "momentum_7")])
    hidden_memory = ExperimentMemory(tmp_path / "hidden.sqlite")
    run_qwen_lab(
        hidden,
        store,
        SOURCE,
        hidden_memory,
        max_hypotheses=2,
        share_research_history=False,
    )
    second = json.loads(hidden.messages[1][1]["content"])
    assert second["checked_hypotheses"] == []
    assert EXPRESSION not in hidden.messages[1][1]["content"]
    hidden_memory.close()


def test_short_history_and_unavailable_ollama_stop_cleanly(tmp_path: Path) -> None:
    short_store, _checksum = _store(tmp_path, _span(date(2024, 1, 1), date(2024, 3, 1)))

    class Unexpected:
        def health(self):
            raise AssertionError("модель не должна вызываться")

    stopped = run_qwen_lab(Unexpected(), short_store, SOURCE, ExperimentMemory(tmp_path / "unused.sqlite"))
    assert stopped.stop_reason == "insufficient_history"
    assert stopped.attempts == ()

    store, _checksum = _store(tmp_path, _long())
    memory = ExperimentMemory(tmp_path / "partial.sqlite")
    client = ScriptedModel([_proposal(), None])
    result = run_qwen_lab(client, store, SOURCE, memory, max_hypotheses=2)
    assert result.stop_reason == "ollama_unavailable"
    assert len(result.attempts) == 1
    assert result.attempts[0].final_evaluation is not None
    memory.close()

    down = ResearchOllama(base_url="http://127.0.0.1:9", timeout=0.2)
    unavailable = run_qwen_lab(down, store, SOURCE, ExperimentMemory(tmp_path / "down.sqlite"), max_hypotheses=1)
    assert unavailable.stop_reason == "ollama_unavailable"
    assert unavailable.attempts == ()


def test_holdout_price_never_reaches_the_model(tmp_path: Path) -> None:
    candles = _span(date(2021, 1, 1), date(2024, 12, 31))
    path = tmp_path / "btc.csv"
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp.isoformat()},{candle.open},{candle.high},"
            f"{candle.low},{candle.close},{candle.volume}"
        )
    lines.append("2025-01-01T00:00:00+00:00,1,2,0.5,999999.25,1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_csv(path, source=SOURCE)
    assert manifest.holdout_excluded == 1
    client = ScriptedModel([_proposal()])
    result = run_qwen_lab(
        client,
        store,
        SOURCE,
        ExperimentMemory(tmp_path / "memory.sqlite"),
        max_hypotheses=1,
    )
    assert result.stop_reason == "completed"
    assert "999999.25" not in _prompt(client.messages[0])
    assert "999999.25" not in result.message
    assert "999999.25" not in json.dumps(lab_report(result))


def test_prompt_budget_drops_old_notes() -> None:
    notes = []
    from wolfai.research.generator import ResearchNote

    for index in range(6):
        notes.append(ResearchNote(
            expression=f"sigmoid(0.{index + 1} * log_return)",
            features=("log_return",),
            horizon_days=1,
            limitation="проверочное окно 2022: Brier не ниже обоих ориентиров",
            observations=200,
            brier=0.2 + index / 100,
            frequency_brier=0.1,
            coin_brier=0.25,
        ))
    summary = summary_payload(build_experiment_rows(_span(date(2021, 1, 1), date(2021, 12, 31)), 1))
    messages = build_lab_messages(
        summary=summary,
        notes=notes,
        repair_error=None,
        share_research_history=True,
    )
    assert len(messages[1]["content"]) <= MAX_USER_CHARS
    payload = json.loads(messages[1]["content"])
    assert len(payload["checked_hypotheses"]) <= 4
    with pytest.raises(ValueError, match="5"):
        run_qwen_lab(None, None, SOURCE, None, max_hypotheses=6)


def _replace(candles: list[Candle], day: date, close: float) -> list[Candle]:
    replaced: list[Candle] = []
    for candle in candles:
        if candle.timestamp.date() == day:
            replaced.append(Candle(candle.timestamp, close, close + 1, close - 1, close, candle.volume))
        else:
            replaced.append(candle)
    return replaced
