"""Один проход наблюдения. Торговых счетов и фонового процесса нет.

Настоящий forward-прогноз требует явно проверенный живой источник.
Исторический CSV вместо него не подставляется: без источника статус
`source_unavailable`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from wolfai.research.forward_eval import evaluate_settled
from wolfai.research.forward_registry import ForwardRegistry
from wolfai.research.prediction_ledger import PredictionLedger
from wolfai.research.strategy import ConstantHalfStrategy, FeatureView, ResearchStrategy

CODE_VERSION = "2.5.0"


def run_shadow_once(
    *,
    ledger: PredictionLedger,
    registry: ForwardRegistry | None,
    channel: str,
    now: datetime,
    strategy: ResearchStrategy,
    features: dict[str, float] | None,
    feature_available_at: dict[str, datetime] | None,
    decision_time: datetime | None,
    horizon_days: int,
    outcome: int | None,
    observed_at: datetime | None,
    live_source_verified: bool,
    dataset_version: str,
) -> dict[str, object]:
    if channel == "forward" and not live_source_verified:
        return {"status": "source_unavailable", "predictions_stored": 0, "outcomes_stored": 0}
    if features is None or feature_available_at is None or decision_time is None:
        return {"status": "market_data_missing", "predictions_stored": 0, "outcomes_stored": 0}
    view = FeatureView(
        decision_time=decision_time,
        features=dict(features),
        feature_available_at=dict(feature_available_at),
    )
    probability = strategy.predict(view)
    prediction_id, state = ledger.append(
        model_id=strategy.strategy_id,
        model_version=strategy.version,
        created_at=now,
        decision_time=decision_time,
        horizon_days=horizon_days,
        probability_up=probability,
        features=view.features,
        feature_available_at=view.feature_available_at,
        dataset_version=dataset_version,
        code_version=CODE_VERSION,
        channel=channel,
        baseline_probability=0.5,
    )
    outcomes = 0
    outcome_state = "not_mature"
    if outcome is not None and observed_at is not None:
        target = decision_time + timedelta(days=horizon_days)
        if observed_at >= target:
            outcome_state = ledger.record_outcome(prediction_id, outcome, observed_at)
            outcomes = 1 if outcome_state == "stored" else 0
        else:
            outcome_state = "not_mature"
    evaluation = None
    if registry is not None and outcome_state == "stored":
        evaluation = evaluate_settled(ledger, channel)
        payload = _registry_payload(strategy, prediction_id, evaluation, decision_time, horizon_days)
        for kind in ("experiment", "hypothesis", "evidence"):
            registry.remember(kind, payload)
    return {
        "status": "ok",
        "channel": channel,
        "prediction_id": prediction_id,
        "write": state,
        "predictions_stored": 1 if state == "stored" else 0,
        "outcomes_stored": outcomes,
        "outcome": outcome_state,
        "evaluation": evaluation,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description="Один цикл shadow mode без торговли")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--registry", default="")
    parser.add_argument("--channel", default="forward", choices=["forward", "reconstruction", "synthetic"])
    parser.add_argument("--fixture", default="")
    parser.add_argument("--live-source", default="")
    parser.add_argument("--market-fixture", default="")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--fetch-live", action="store_true")
    args = parser.parse_args(argv)
    ledger = PredictionLedger(args.ledger)
    registry = ForwardRegistry(args.registry) if args.registry else None
    try:
        if args.fetch_live or args.market_fixture:
            if args.channel != "forward":
                result = {"status": "forward_only", "predictions_stored": 0, "outcomes_stored": 0}
            else:
                from wolfai.research.forward_mode import run_shadow_market

                result = run_shadow_market(
                    ledger=ledger,
                    fixture_path=args.market_fixture,
                    fetch_live=args.fetch_live and not args.market_fixture,
                    commit=args.commit,
                )
        elif args.channel == "forward" and args.live_source != "verified-local":
            result = {"status": "source_unavailable", "predictions_stored": 0, "outcomes_stored": 0}
        elif not args.fixture:
            result = {"status": "market_data_missing", "predictions_stored": 0, "outcomes_stored": 0}
        else:
            payload = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
            result = run_shadow_once(
                ledger=ledger,
                registry=registry,
                channel=args.channel,
                now=_time(payload["now"]),
                strategy=ConstantHalfStrategy(),
                features={key: float(value) for key, value in payload["features"].items()},
                feature_available_at={
                    key: _time(value) for key, value in payload["feature_available_at"].items()
                },
                decision_time=_time(payload["decision_time"]),
                horizon_days=int(payload["horizon_days"]),
                outcome=None if payload.get("outcome") is None else int(payload["outcome"]),
                observed_at=None if payload.get("observed_at") is None else _time(payload["observed_at"]),
                live_source_verified=args.live_source == "verified-local",
                dataset_version=str(payload.get("dataset_version") or "fixture"),
            )
    finally:
        ledger.close()
        if registry is not None:
            registry.close()
    print(json.dumps(result, ensure_ascii=False, default=str))
    return result


def _registry_payload(strategy: ResearchStrategy, prediction_id: str, evaluation: dict[str, object], decision_time: datetime, horizon_days: int) -> dict[str, object]:
    return {
        "name": strategy.strategy_id,
        "model_id": strategy.strategy_id,
        "model_version": strategy.version,
        "dataset_version": "fixture",
        "code_version": CODE_VERSION,
        "horizon_days": horizon_days,
        "decision_time": decision_time.isoformat(),
        "prediction_id": prediction_id,
        "claim_status": evaluation.get("claim_status"),
        "role": evaluation.get("role"),
        "brier": evaluation.get("brier"),
        "observations": evaluation.get("observations"),
    }


def _time(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("время фикстуры должно быть в UTC")
    return parsed


if __name__ == "__main__":
    main()
