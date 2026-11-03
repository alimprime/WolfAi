import json
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.research.experiment import build_messages, run_research
from wolfai.research.features import (
    HOLDOUT_START,
    build_rows,
    direction_label,
    feature_vector,
    load_research_candles,
)
from wolfai.research.formula import parse_formula
from wolfai.research.experiment import _score_direction
from wolfai.research.journal import HypothesisRecord, Journal, utc_now
from wolfai.research.ollama import (
    NUM_CTX,
    ChatCompletion,
    ModelInfo,
    OllamaClient,
    OllamaUnavailable,
    build_chat_body,
)
from wolfai.forecasting.splits import Period
from wolfai.research.features import rows_for_period

UTC = timezone.utc


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
            low=max(close - 1, 0.5),
            close=close,
            volume=2,
        ))
        day += timedelta(days=1)
        index += 1
    return candles


def _payload(expression: str, feature: str, sign: str, based_on: list[str] | None = None) -> str:
    return json.dumps({
        "expression": expression,
        "features": [feature],
        "effects": [{"feature": feature, "sign": sign}],
        "explanation": "проверка знака на прошлых доходностях",
        "based_on": based_on or [],
    })


class ScriptedClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.model_name = "qwen3.5:4b"
        self.model_digest = "abc123"
        self.generation = {"seed": 1729, "num_ctx": NUM_CTX}

    def health(self) -> ModelInfo:
        return ModelInfo(self.model_name, self.model_digest)

    def complete(self, messages: list[dict[str, str]]) -> ChatCompletion:
        self.messages.append(messages)
        content = self.contents[self.calls]
        self.calls += 1
        return ChatCompletion(content, self.model_name, self.model_digest, dict(self.generation))


def test_chat_body_has_no_tools_and_caps_context() -> None:
    body = build_chat_body("qwen3.5:4b", [{"role": "user", "content": "тест"}])
    assert "tools" not in body
    assert body["think"] is False
    assert body["stream"] is False
    options = body["options"]
    assert isinstance(options, dict)
    assert options["num_ctx"] == NUM_CTX
    assert options["seed"] == 1729
    assert options["num_predict"] <= 900


def test_ollama_unavailable() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:9", timeout=0.2)
    with pytest.raises(OllamaUnavailable):
        client.health()


def test_future_close_does_not_change_features() -> None:
    candles = _span(date(2024, 1, 1), date(2024, 4, 1))
    index = 40
    original = feature_vector(candles, index)
    altered = list(candles)
    current = altered[index]
    altered[index + 1] = Candle(altered[index + 1].timestamp, 1, 2, 0.5, 1, 2)
    assert feature_vector(altered, index) == original
    assert direction_label(candles, index) == 1
    assert direction_label(altered, index) == 0
    assert current.close == candles[index].close


def test_loader_stops_before_holdout_prices(tmp_path) -> None:
    candles = _span(date(2024, 12, 1), date(2024, 12, 31))
    path = tmp_path / "btc.csv"
    lines = ["timestamp,open,high,low,close,volume"]
    for candle in candles:
        lines.append(
            f"{candle.timestamp.isoformat()},{candle.open},{candle.high},{candle.low},{candle.close},{candle.volume}"
        )
    lines.append("2025-01-01T00:00:00+00:00,-1,-1,-1,-1,-1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    loaded = load_research_candles(path)
    assert loaded[-1].timestamp.date() == date(2024, 12, 31)
    assert all(candle.timestamp.date() < HOLDOUT_START for candle in loaded)
    assert all(candle.close > 0 for candle in loaded)
    with pytest.raises(ValueError, match="holdout"):
        build_rows(loaded + [Candle(datetime(2025, 1, 1, tzinfo=UTC), 1, 2, 0.5, 999999.25, 1)])


def test_journal_roundtrip_keeps_failures_and_rejects_active(tmp_path) -> None:
    journal = Journal(tmp_path / "research.sqlite")
    journal.start_run("run", data_version="abc", model_name="qwen3.5:4b", generation={"seed": 1}, max_hypotheses=5)
    record = HypothesisRecord(
        id="h1",
        run_id="run",
        created_at=utc_now(),
        expression="sigmoid(0.1 * log_return)",
        features=["log_return"],
        effects=[{"feature": "log_return", "sign": "positive"}],
        explanation="тест",
        version=1,
        status="archived",
        parent_ids=[],
        data_version="abc",
        model_name="qwen3.5:4b",
        model_digest="abc123",
        generation={"seed": 1729},
        rejection_reason="проверка",
        error_type="formula",
        error_scale=None,
        observations=None,
        sufficient_observations=None,
        train_metrics={"brier": 0.25},
        validation_metrics=None,
        raw_response="{}",
    )
    journal.add(record)
    again = Journal(tmp_path / "research.sqlite")
    loaded = again.get("h1")
    assert loaded is not None
    assert loaded.status == "archived"
    assert loaded.train_metrics is not None
    assert loaded.train_metrics["brier"] == pytest.approx(0.25)
    assert len(again.list_hypotheses()) == 1
    rejected = HypothesisRecord(
        id="h2", run_id="run", created_at=utc_now(), expression="", features=[], effects=[],
        explanation="", version=1, status="active", parent_ids=[], data_version="abc",
        model_name=None, model_digest=None, generation={}, rejection_reason=None,
        error_type=None, error_scale=None, observations=None, sufficient_observations=None,
        train_metrics=None, validation_metrics=None, raw_response=None,
    )
    with pytest.raises(ValueError, match="active"):
        again.add(rejected)
    again.close()
    journal.close()


def test_direction_score_is_repeatable_and_run_rejects_holdout(tmp_path) -> None:
    candles = _span(date(2022, 1, 1), date(2023, 4, 1))
    rows = rows_for_period(build_rows(candles), Period.TRAINING)
    tree = parse_formula("sigmoid(0.5 * momentum_7)")
    first = _score_direction(rows, tree)
    second = _score_direction(rows, tree)
    assert first["brier"] == second["brier"]
    assert first["calibration"] == second["calibration"]
    journal = Journal(tmp_path / "journal.sqlite")
    client = ScriptedClient([_payload("sigmoid(0.5 * momentum_7)", "momentum_7", "positive")])
    with pytest.raises(ValueError, match="holdout"):
        run_research(
            client,
            candles + [Candle(datetime(2025, 1, 1, tzinfo=UTC), 1, 2, 0.5, 999999.25, 1)],
            journal,
            max_hypotheses=1,
        )
    assert client.calls == 0
    journal.close()


def test_mock_qwen_is_scored_and_remembered_without_promotion(tmp_path) -> None:
    candles = _span(date(2022, 1, 1), date(2023, 4, 1))
    journal = Journal(tmp_path / "journal.sqlite")
    client = ScriptedClient([
        "это не json",
        _payload("sigmoid(0.5 * momentum_7)", "momentum_7", "positive"),
        _payload("sigmoid(8 * momentum_7)", "momentum_7", "positive"),
    ])
    result = run_research(client, candles, journal, max_hypotheses=2)
    assert result.stop_reason == "completed"
    rows = journal.list_hypotheses()
    assert len(rows) == 2
    assert rows[0].status == "experimental"
    assert rows[0].expression == "sigmoid(0.5 * momentum_7)"
    assert rows[0].train_metrics is not None
    assert rows[0].validation_metrics is not None
    assert "brier_quantile_baseline" in rows[0].train_metrics
    assert "brier_random_walk" in rows[0].validation_metrics
    assert rows[1].status == "experimental"
    assert all(row.status != "active" for row in rows)
    assert "JSON" in client.messages[1][-1]["content"]
    assert "sigmoid(0.5 * momentum_7)" in client.messages[2][1]["content"]
    assert "999999.25" not in client.messages[2][1]["content"]
    runs = journal.list_runs()
    assert runs[0].baseline_metrics is not None
    assert "historical_quantile" in runs[0].baseline_metrics
    assert runs[0].model_digest == "abc123"
    assert runs[0].generation["seed"] == 1729
    journal.close()


def test_sign_contradiction_is_archived(tmp_path) -> None:
    candles = _span(date(2022, 1, 1), date(2023, 4, 1))
    journal = Journal(tmp_path / "journal.sqlite")
    client = ScriptedClient([
        _payload("sigmoid(-2 * momentum_7)", "momentum_7", "positive"),
    ])
    run_research(client, candles, journal, max_hypotheses=1)
    row = journal.list_hypotheses()[0]
    assert row.status == "archived"
    assert row.error_type == "effect_sign_mismatch"
    assert row.sufficient_observations is True
    assert row.error_scale is not None and row.error_scale > 0
    journal.close()


def test_unavailable_client_stops_before_formulas(tmp_path) -> None:
    candles = _span(date(2022, 1, 1), date(2023, 4, 1))

    class DownClient:
        model_name = "qwen3.5:4b"
        model_digest = ""
        generation = {"seed": 1729}

        def health(self) -> ModelInfo:
            raise OllamaUnavailable("нет связи")

        def complete(self, messages: list[dict[str, str]]) -> ChatCompletion:
            raise AssertionError("запрос не должен уйти")

    journal = Journal(tmp_path / "journal.sqlite")
    result = run_research(DownClient(), candles, journal, max_hypotheses=5)
    assert result.stop_reason == "ollama_unavailable"
    assert journal.list_hypotheses() == []
    assert journal.list_runs()[0].stop_reason == "ollama_unavailable"
    journal.close()


def test_prompt_builder_uses_journal_not_prices() -> None:
    text = build_messages(
        stats={"up_rate": 0.51, "momentum_7": {"mean": 0.01, "stdev": 0.08}},
        previous=[{
            "id": "abc",
            "status": "archived",
            "expression": "sigmoid(0.1 * log_return)",
            "train_brier": 0.24,
            "validation_brier": 0.26,
            "error_type": "effect_sign_mismatch",
            "rejection_reason": "знак",
        }],
        feedback="коэффициент вне диапазона",
        data_version="deadbeef",
    )
    content = text[1]["content"]
    assert "sigmoid(0.1 * log_return)" in content
    assert "999999.25" not in content
    assert "2024-12-31" in content
