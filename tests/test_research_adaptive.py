import json
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.forecasting.splits import (
    COEFFICIENT_FIT_BOUNDS,
    INNER_CHECK_BOUNDS,
    period_bounds,
    Period,
)
from wolfai.research.adaptive import build_adaptive_messages, fit_terms, run_adaptive
from wolfai.research.features import (
    FEATURE_NAMES,
    FeatureRow,
    build_rows,
    standardize_rows,
)
from wolfai.research.formula import extract_linear_terms, parse_formula
from wolfai.research.hypothesis import HypothesisSchemaError, parse_adaptive
from wolfai.research.journal import HypothesisRecord, Journal, utc_now
from wolfai.research.logistic import Term, fit_l2_logistic
from wolfai.research.ollama import ChatCompletion, ModelInfo

UTC = timezone.utc


def _candles(start: date, days: int) -> list[Candle]:
    candles = []
    for index in range(days):
        day = start + timedelta(days=index)
        close = 100.0 + index + (4.0 if index % 3 == 0 else 0.0)
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=max(close - 1, 0.5),
            close=close,
            volume=2 + (index % 5),
        ))
    return candles


def _feature_rows(count: int, spike: int | None = None) -> list[FeatureRow]:
    rows = []
    day = date(2024, 1, 1)
    for index in range(count):
        features = {name: 0.01 * ((index + offset) % 5) for offset, name in enumerate(FEATURE_NAMES)}
        if spike is not None and index == spike:
            features["momentum_7"] = 1000.0
        rows.append(FeatureRow(
            index=index + 30,
            decision_date=day + timedelta(days=index),
            target_date=day + timedelta(days=index + 1),
            features=features,
            y_up=index % 2,
            baseline_up_frequency=0.5,
        ))
    return rows


class ScriptedClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []
        self.model_name = "qwen3.5:4b"
        self.model_digest = "digest"
        self.generation = {"seed": 1729}

    def health(self) -> ModelInfo:
        return ModelInfo(self.model_name, self.model_digest)

    def complete(self, messages: list[dict[str, str]], schema: dict | None = None) -> ChatCompletion:
        self.messages.append(messages)
        content = self.contents[self.calls]
        self.calls += 1
        return ChatCompletion(content, self.model_name, self.model_digest, {"seed": 1729})


def test_inner_splits_stay_inside_training_and_before_development() -> None:
    training = period_bounds(Period.TRAINING)
    validation = period_bounds(Period.VALIDATION)
    holdout = period_bounds(Period.HOLDOUT)
    assert COEFFICIENT_FIT_BOUNDS[0] == training[0]
    assert COEFFICIENT_FIT_BOUNDS[1] < INNER_CHECK_BOUNDS[0]
    assert INNER_CHECK_BOUNDS[1] == training[1]
    assert INNER_CHECK_BOUNDS[1] < validation[0]
    assert validation[1] < holdout[0]


def test_standardization_uses_only_the_past_and_handles_edge_values() -> None:
    base = _feature_rows(80)
    spiked = _feature_rows(80, spike=70)
    base_z = standardize_rows(base)
    spiked_z = standardize_rows(spiked)
    assert base_z[0].standardized == spiked_z[0].standardized
    assert base_z[8].standardized == spiked_z[8].standardized
    assert spiked_z[10].standardized["momentum_7"] == pytest.approx(8.0)
    assert base_z[0].label_available > base_z[0].decision_available
    flat = _feature_rows(80)
    for row in flat:
        for name in FEATURE_NAMES:
            row.features[name] = 1.0
    flat_z = standardize_rows(flat)
    assert all(value == 0.0 for value in flat_z[0].standardized.values())
    broken = _feature_rows(80)
    broken[70].features["log_return"] = float("nan")
    scaled = standardize_rows(broken)
    assert scaled[10].standardized["log_return"] == 0.0
    assert all(value == value and abs(value) != float("inf") for value in scaled[10].standardized.values())


def test_logistic_fit_is_repeatable_and_ignores_later_labels() -> None:
    design = [[index / 10, (index % 2) - 0.5] for index in range(40)]
    labels = [1 if row[0] + row[1] > 1.5 else 0 for row in design]
    assert fit_l2_logistic(design, labels) == fit_l2_logistic(design, labels)
    candles = _candles(date(2019, 1, 1), 1200)
    mutated = list(candles)
    mutated[-1] = Candle(mutated[-1].timestamp, 1, 2, 0.5, 1, 2)
    original = standardize_rows(build_rows(candles))
    changed = standardize_rows(build_rows(mutated))
    early = original[0].decision_date
    assert next(row.standardized for row in original if row.decision_date == early) == next(
        row.standardized for row in changed if row.decision_date == early
    )
    fit_end = COEFFICIENT_FIT_BOUNDS[1]
    first = fit_terms([row for row in original if row.target_date <= fit_end], [Term("momentum_7", "identity")])
    second = fit_terms([row for row in changed if row.target_date <= fit_end], [Term("momentum_7", "identity")])
    assert first == second


def test_linear_terms_and_adaptive_schema() -> None:
    terms = extract_linear_terms(parse_formula("sigmoid(0.3 * momentum_7 - 0.2 * volatility_30)"))
    assert terms == [(0.3, "momentum_7"), (-0.2, "volatility_30")]
    draft = parse_adaptive(json.dumps({
        "parent_id": "abc",
        "terms": [{"feature": "momentum_30", "transform": "identity"}],
        "change": "убрал лишний признак",
        "problem": "сырой масштаб",
        "expected_improvement": "ниже brier на 2022",
        "falsification": {"reference": "constant_0.5", "split": "inner_2022"},
        "explanation": "длинный импульс",
        "status": "active",
    }))
    assert not hasattr(draft, "status")
    with pytest.raises(HypothesisSchemaError):
        parse_adaptive(json.dumps({
            "terms": [{"feature": "momentum_30", "transform": "identity"}],
            "change": "x",
            "problem": "y",
            "expected_improvement": "z",
            "falsification": {"reference": "constant_0.5", "split": "development_2023"},
            "explanation": "нет родителя",
        }))


def test_adaptive_run_keeps_parent_and_hides_development_metrics(tmp_path) -> None:
    candles = _candles(date(2019, 1, 1), 1250)
    journal = Journal(tmp_path / "journal.sqlite")
    parent = HypothesisRecord(
        id="parent1",
        run_id="old",
        created_at=utc_now(),
        expression="sigmoid(0.3 * momentum_7)",
        features=["momentum_7"],
        effects=[{"feature": "momentum_7", "sign": "positive"}],
        explanation="старая",
        version=1,
        status="experimental",
        parent_ids=[],
        data_version="old",
        model_name="qwen3.5:4b",
        model_digest="d",
        generation={"seed": 1},
        rejection_reason=None,
        error_type=None,
        error_scale=None,
        observations=10,
        sufficient_observations=True,
        train_metrics={"brier": 0.2},
        validation_metrics={"brier": 0.424242},
        raw_response="{}",
    )
    journal.start_run("old", data_version="old", model_name="qwen3.5:4b", generation={"seed": 1}, max_hypotheses=1)
    journal.add(parent)
    payload = {
        "parent_id": "parent1",
        "terms": [
            {"feature": "momentum_30", "transform": "identity"},
            {"feature": "volatility_7", "transform": "abs"},
        ],
        "change": "заменил сырые веса структурой из двух членов",
        "problem": "вероятности стояли у 0.5",
        "expected_improvement": "внутренний brier ниже константы",
        "falsification": {"reference": "constant_0.5", "split": "inner_2022"},
        "explanation": "длинный импульс и модуль короткой волатильности",
    }
    client = ScriptedClient([json.dumps(payload)])
    result = run_adaptive(client, candles, journal, max_hypotheses=1)
    assert result.stop_reason == "completed"
    rows = [row for row in journal.list_hypotheses() if row.id != "parent1"]
    assert len(rows) == 1
    assert rows[0].parent_ids == ["parent1"]
    assert rows[0].version == 2
    assert rows[0].status in {"experimental", "archived"}
    assert rows[0].status != "active"
    assert rows[0].coefficients is not None
    assert "intercept" in rows[0].coefficients
    assert rows[0].inner_metrics is not None
    user_prompt = client.messages[0][1]["content"]
    assert "parent1" in user_prompt
    assert "0.424242" not in user_prompt
    assert "sigmoid(0.3 * momentum_7)" in user_prompt
    journal.close()
