"""Журнал прогнозов, созревание горизонта и запрет ложного supported."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from wolfai.research.forward_eval import evaluate_settled
from wolfai.research.forward_registry import ForwardRegistry
from wolfai.research.prediction_ledger import PredictionImmutable, PredictionLedger
from wolfai.research.provenance_grade import admits_strict_history, grade_timing
from wolfai.research.quality_contract import STATUS_SUPPORTED
from wolfai.research.shadow import main, run_shadow_once
from wolfai.research.strategy import ConstantHalfStrategy, FeatureView

UTC = timezone.utc
CODE = "test"


def test_probability_is_immutable_and_future_features_are_rejected(tmp_path) -> None:
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    decision = datetime(2026, 11, 4, tzinfo=UTC)
    created = decision + timedelta(hours=1)
    features = {"log_return": 0.01}
    available = {"log_return": decision}
    prediction_id, state = ledger.append(
        model_id="constant-half",
        model_version="1",
        created_at=created,
        decision_time=decision,
        horizon_days=1,
        probability_up=0.5,
        features=features,
        feature_available_at=available,
        dataset_version="fixture",
        code_version=CODE,
        channel="forward",
    )
    assert state == "stored"
    again, duplicate = ledger.append(
        model_id="constant-half",
        model_version="1",
        created_at=created,
        decision_time=decision,
        horizon_days=1,
        probability_up=0.5,
        features=features,
        feature_available_at=available,
        dataset_version="fixture",
        code_version=CODE,
        channel="forward",
    )
    assert duplicate == "duplicate"
    assert again == prediction_id
    with pytest.raises(PredictionImmutable):
        ledger.append(
            model_id="constant-half",
            model_version="1",
            created_at=created,
            decision_time=decision,
            horizon_days=1,
            probability_up=0.9,
            features=features,
            feature_available_at=available,
            dataset_version="fixture",
            code_version=CODE,
            channel="forward",
        )
    corrected, stored = ledger.append(
        model_id="constant-half",
        model_version="1",
        created_at=created + timedelta(minutes=5),
        decision_time=decision,
        horizon_days=1,
        probability_up=0.4,
        features=features,
        feature_available_at=available,
        dataset_version="fixture",
        code_version=CODE,
        channel="forward",
        replaces_prediction_id=prediction_id,
    )
    assert stored == "stored"
    assert ledger.get(prediction_id).probability_up == 0.5
    assert ledger.get(corrected).replaces_prediction_id == prediction_id
    with pytest.raises(PredictionImmutable, match="чужой"):
        ledger.append(
            model_id="other-strategy",
            model_version="1",
            created_at=created,
            decision_time=decision,
            horizon_days=1,
            probability_up=0.2,
            features=features,
            feature_available_at=available,
            dataset_version="fixture",
            code_version=CODE,
            channel="forward",
            replaces_prediction_id=prediction_id,
        )
    late = dict(available)
    late["log_return"] = decision + timedelta(days=1)
    with pytest.raises(ValueError, match="decision_time"):
        ledger.append(
            model_id="constant-half",
            model_version="1",
            created_at=created,
            decision_time=decision,
            horizon_days=1,
            probability_up=0.5,
            features=features,
            feature_available_at=late,
            dataset_version="fixture",
            code_version=CODE,
            channel="forward",
        )
    ledger.close()


def test_channels_horizons_and_outcome_cannot_be_rewritten(tmp_path) -> None:
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    decision = datetime(2019, 6, 1, tzinfo=UTC)
    features = {"log_return": -0.2}
    available = {"log_return": decision}
    synthetic, _ = ledger.append(
        model_id="sign",
        model_version="1",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
        decision_time=decision,
        horizon_days=7,
        probability_up=0.1,
        features=features,
        feature_available_at=available,
        dataset_version="synthetic",
        code_version=CODE,
        channel="synthetic",
    )
    target = decision + timedelta(days=7)
    with pytest.raises(ValueError, match="не созрел"):
        ledger.record_outcome(synthetic, 0, target - timedelta(seconds=1))
    assert ledger.record_outcome(synthetic, 0, target) == "stored"
    assert ledger.record_outcome(synthetic, 0, target) == "duplicate"
    with pytest.raises(PredictionImmutable):
        ledger.record_outcome(synthetic, 1, target)
    with pytest.raises(ValueError, match="2022"):
        ledger.append(
            model_id="sign",
            model_version="1",
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
            decision_time=datetime(2022, 1, 1, tzinfo=UTC),
            horizon_days=1,
            probability_up=0.5,
            features=features,
            feature_available_at={"log_return": datetime(2022, 1, 1, tzinfo=UTC)},
            dataset_version="reconstruction",
            code_version=CODE,
            channel="reconstruction",
        )
    with pytest.raises(ValueError, match="forward"):
        ledger.append(
            model_id="sign",
            model_version="1",
            created_at=target + timedelta(days=1),
            decision_time=decision,
            horizon_days=1,
            probability_up=0.5,
            features=features,
            feature_available_at=available,
            dataset_version="live",
            code_version=CODE,
            channel="forward",
        )
    ledger.close()


def test_known_synthetic_outcome_is_scored_but_not_supported(tmp_path) -> None:
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    start = datetime(2021, 1, 1, tzinfo=UTC)
    for offset in range(40):
        decision = start + timedelta(days=offset)
        label = offset % 2
        prediction_id, _state = ledger.append(
            model_id="sign",
            model_version="1",
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
            decision_time=decision,
            horizon_days=1,
            probability_up=0.9 if label == 1 else 0.1,
            features={"signal": 1.0 if label == 1 else -1.0},
            feature_available_at={"signal": decision},
            dataset_version="synthetic",
            code_version=CODE,
            channel="synthetic",
        )
        ledger.record_outcome(prediction_id, label, decision + timedelta(days=1))
    report = evaluate_settled(ledger, "synthetic")
    assert report["role"] == "synthetic_check"
    assert report["observations"] == 40
    assert float(report["brier"]) < 0.02
    assert report["claim_status"] != STATUS_SUPPORTED
    assert report["validated_law"] is False
    week = PredictionLedger(tmp_path / "week.sqlite")
    for offset in range(0, 40, 1):
        decision = start + timedelta(days=offset)
        prediction_id, _state = week.append(
            model_id="constant-half",
            model_version="1",
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
            decision_time=decision,
            horizon_days=7,
            probability_up=0.5,
            features={"log_return": 0.0},
            feature_available_at={"log_return": decision},
            dataset_version="synthetic",
            code_version=CODE,
            channel="synthetic",
        )
        week.record_outcome(prediction_id, 0, decision + timedelta(days=7))
    weekly = evaluate_settled(week, "synthetic")
    assert weekly["uncertainty"]["block_length"] >= 7
    ledger.close()
    week.close()


def test_shadow_cli_is_idempotent_and_refuses_a_fake_live_source(tmp_path) -> None:
    ledger_path = tmp_path / "shadow.sqlite"
    registry_path = tmp_path / "registry.sqlite"
    unavailable = main(["--ledger", str(ledger_path), "--channel", "forward"])
    assert unavailable["status"] == "source_unavailable"
    missing = main(["--ledger", str(ledger_path), "--channel", "synthetic"])
    assert missing["status"] == "market_data_missing"
    fixture = tmp_path / "fixture.json"
    decision = datetime(2026, 11, 4, tzinfo=UTC)
    fixture.write_text(json.dumps({
        "now": "2026-11-04T01:00:00+00:00",
        "decision_time": decision.isoformat(),
        "horizon_days": 1,
        "features": {"log_return": 0.01},
        "feature_available_at": {"log_return": decision.isoformat()},
        "outcome": 1,
        "observed_at": "2026-11-05T00:00:00+00:00",
        "dataset_version": "fixture",
    }), encoding="utf-8")
    first = main([
        "--ledger", str(ledger_path),
        "--registry", str(registry_path),
        "--channel", "synthetic",
        "--fixture", str(fixture),
    ])
    second = main([
        "--ledger", str(ledger_path),
        "--registry", str(registry_path),
        "--channel", "synthetic",
        "--fixture", str(fixture),
    ])
    assert first["write"] == "stored"
    assert second["write"] == "duplicate"
    assert second["outcomes_stored"] == 0
    registry = ForwardRegistry(registry_path)
    assert registry.count("evidence") == 1
    with pytest.raises(ValueError, match="Validated"):
        registry.remember("validated_law", {
            "name": "x",
            "model_id": "x",
            "model_version": "1",
            "dataset_version": "fixture",
            "code_version": "2.5.0",
            "horizon_days": 1,
            "decision_time": decision.isoformat(),
            "claim_status": "insufficient_evidence",
        })
    registry.close()
    view = FeatureView(decision, {"log_return": 0.01}, {"log_return": decision})
    assert ConstantHalfStrategy().predict(view) == 0.5
    live = run_shadow_once(
        ledger=PredictionLedger(tmp_path / "unused.sqlite"),
        registry=None,
        channel="forward",
        now=decision,
        strategy=ConstantHalfStrategy(),
        features={"log_return": 0.01},
        feature_available_at={"log_return": decision},
        decision_time=decision,
        horizon_days=1,
        outcome=None,
        observed_at=None,
        live_source_verified=False,
        dataset_version="none",
    )
    assert live["status"] == "source_unavailable"


def test_exchange_clock_is_not_verified_history() -> None:
    event = datetime(2020, 1, 1, tzinfo=UTC)
    declared = grade_timing(
        event_time=event,
        published_at=event,
        available_at=event,
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
        basis="exchange_calc_time",
    )
    assert declared == "source_declared"
    assert admits_strict_history(declared) is False
    verified = grade_timing(
        event_time=event,
        published_at=event + timedelta(days=1),
        available_at=event + timedelta(days=1),
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
        basis="daily_bar_next_midnight",
    )
    assert verified == "verified"
    assert admits_strict_history(verified) is True
    unknown = grade_timing(
        event_time=event,
        published_at=None,
        available_at=None,
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
        basis="",
    )
    assert unknown == "unknown"
    assert event != event + timedelta(days=1)
