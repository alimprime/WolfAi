"""Связка проверенного источника свечей с журналом forward.

Первый проход по умолчанию ничего не пишет. Реальная строка появляется
только при `commit=True`. Реконструкция и исторический CSV в этот канал
не переносятся. Ежедневный планировщик не запускается.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from wolfai.data.binance import validate_daily_candles
from wolfai.data.spot_live import (
    SOURCE_UNAVAILABLE,
    SpotCandle,
    SpotClient,
    SpotSnapshot,
    interpret_market,
)
from wolfai.research.features import (
    FEATURE_SCHEMA_VERSION,
    LOOKBACK,
    features_from_closed_window,
)
from wolfai.research.prediction_ledger import PredictionLedger
from wolfai.research.strategy import ConstantHalfStrategy, FeatureView, ResearchStrategy

CODE_VERSION = "2.6.0"
LIVE_ORIGIN = "live_spot"
ALLOWED_MODES = frozenset({"network", "verified_fixture"})
HORIZONS = frozenset({1, 3, 7})


def run_forward_cycle(
    *,
    ledger: PredictionLedger,
    strategy: ResearchStrategy,
    snapshot: SpotSnapshot,
    horizon_days: int = 1,
    commit: bool = False,
    origin: str = LIVE_ORIGIN,
) -> dict[str, object]:
    """Один прогноз от последней закрытой свечи. Без commit это dry-run."""
    refused = _refuse_unlive(origin, snapshot, horizon_days)
    if refused is not None:
        return refused
    if snapshot.status != "ok" or not snapshot.closed or snapshot.received_at is None:
        status = snapshot.status if snapshot.status == SOURCE_UNAVAILABLE else snapshot.status or SOURCE_UNAVAILABLE
        if snapshot.status != "ok":
            status = snapshot.status
        return _empty(status if status != "ok" else SOURCE_UNAVAILABLE, snapshot)
    received_at = snapshot.received_at
    prepared = _prepare(snapshot)
    if isinstance(prepared, str):
        return _empty(prepared, snapshot)
    view, decision_time = prepared
    guard_available(view)
    target_time = decision_time + timedelta(days=horizon_days)
    if received_at < decision_time:
        return _empty("clock_skew", snapshot, decision_time=decision_time)
    if received_at >= target_time:
        return _empty("backtest_refused", snapshot, decision_time=decision_time)
    probability = strategy.predict(view)
    versions = _versions(strategy, snapshot)
    payload: dict[str, object] = {
        "status": "dry_run",
        "predictions_stored": 0,
        "outcomes_stored": 0,
        "validated_law": False,
        "claim_status": "not_confirmed",
        "api_first_seen_proven": False,
        "channel": "forward",
        "origin": origin,
        "probability_up": probability,
        "decision_time": decision_time,
        "target_time": target_time,
        "open_time": snapshot.closed[-1].open_time,
        "close_time": snapshot.closed[-1].close_time,
        "exchange_server_time": snapshot.exchange_server_time,
        "received_at": received_at,
        "available_at": decision_time,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        **versions,
    }
    if not commit:
        return payload
    prediction_id, state = ledger.append(
        model_id=strategy.strategy_id,
        model_version=strategy.version,
        created_at=received_at,
        decision_time=decision_time,
        horizon_days=horizon_days,
        probability_up=probability,
        features=view.features,
        feature_available_at=view.feature_available_at,
        dataset_version=str(versions["dataset_version"]),
        code_version=CODE_VERSION,
        channel="forward",
        baseline_probability=0.5,
    )
    payload["status"] = "ok"
    payload["prediction_id"] = prediction_id
    payload["write"] = state
    payload["predictions_stored"] = 1 if state == "stored" else 0
    return payload


def features_visible_at(candles: Sequence[SpotCandle], decision_time: datetime) -> dict[str, float]:
    """Признаки только из свечей с available_at <= decision_time.

    Более поздняя свеча в ряд не добавляется, даже если она передана рядом.
    """
    _aware(decision_time, "decision_time")
    visible = [candle for candle in candles if candle.available_at <= decision_time]
    if any(candle.available_at > decision_time for candle in visible):
        raise ValueError("available_at позже решения")
    if len(visible) < LOOKBACK + 1:
        raise ValueError("недостаточно истории для признаков")
    ordered = sorted(visible, key=lambda candle: candle.open_time)
    raw = [candle.as_candle() for candle in ordered]
    validate_daily_candles(raw)
    return features_from_closed_window(raw, len(raw) - 1)


def guard_available(view: FeatureView) -> None:
    """Отказывает снимку, в котором признак известен позже решения."""
    if set(view.feature_available_at) != set(view.features):
        raise ValueError("у каждого признака должно быть своё время доступности")
    for name, moment in view.feature_available_at.items():
        _aware(moment, name)
        if moment > view.decision_time:
            raise ValueError(f"признак {name} доступен позже decision_time")


def run_shadow_market(
    *,
    ledger: PredictionLedger,
    fixture_path: str = "",
    fetch_live: bool = False,
    commit: bool = False,
    strategy: ResearchStrategy | None = None,
) -> dict[str, object]:
    """Точка входа shadow для рыночного снимка. Сеть включается только явным флагом."""
    chosen = strategy or ConstantHalfStrategy()
    if fixture_path:
        snapshot = snapshot_from_fixture(fixture_path)
    elif fetch_live:
        snapshot = SpotClient().fetch()
    else:
        return _empty(SOURCE_UNAVAILABLE, _blank())
    return run_forward_cycle(
        ledger=ledger,
        strategy=chosen,
        snapshot=snapshot,
        commit=commit,
        origin=LIVE_ORIGIN,
    )


def snapshot_from_fixture(path: str | Path) -> SpotSnapshot:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    received = _parse_utc(str(payload["received_at"]))
    time_body = str(payload["time_body"]).encode()
    klines_body = str(payload["klines_body"]).encode()
    return interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )


def _prepare(snapshot: SpotSnapshot) -> tuple[FeatureView, datetime] | str:
    candles = tuple(sorted(snapshot.closed, key=lambda candle: candle.open_time))
    if len(candles) < LOOKBACK + 1:
        return "context_incomplete"
    for candle in candles:
        if candle.api_first_seen_proven or candle.api_observation_grade != "unknown":
            return "unproven_publication"
        if candle.availability_grade != "verified":
            return "context_incomplete"
        if candle.available_at != candle.open_time + timedelta(days=1):
            return "invalid_timestamp"
    decision_time = candles[-1].available_at
    if any(candle.available_at > decision_time for candle in candles):
        return "context_incomplete"
    try:
        features = features_visible_at(candles, decision_time)
    except ValueError:
        return "context_incomplete"
    view = FeatureView(
        decision_time=decision_time,
        features=features,
        feature_available_at={name: decision_time for name in features},
    )
    return view, decision_time


def _refuse_unlive(origin: str, snapshot: SpotSnapshot, horizon_days: int) -> dict[str, object] | None:
    if horizon_days not in HORIZONS:
        return _empty("invalid_horizon", snapshot)
    if origin != LIVE_ORIGIN or snapshot.source_mode not in ALLOWED_MODES:
        return _empty("backtest_refused", snapshot)
    return None


def _versions(strategy: ResearchStrategy, snapshot: SpotSnapshot) -> dict[str, object]:
    return {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "code_version": CODE_VERSION,
        "dataset_version": (
            f"binance-spot-btcusdt-1d:mode:{snapshot.source_mode}:"
            f"klines:{snapshot.klines_sha256}:time:{snapshot.time_sha256}:"
            f"features:{FEATURE_SCHEMA_VERSION}"
        ),
        "source_mode": snapshot.source_mode,
        "klines_sha256": snapshot.klines_sha256,
        "time_sha256": snapshot.time_sha256,
    }


def _empty(
    status: str,
    snapshot: SpotSnapshot,
    *,
    decision_time: datetime | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "predictions_stored": 0,
        "outcomes_stored": 0,
        "validated_law": False,
        "claim_status": "not_confirmed",
        "api_first_seen_proven": False,
        "channel": "forward",
        "decision_time": decision_time,
        "received_at": snapshot.received_at,
        "exchange_server_time": snapshot.exchange_server_time,
        "source_mode": snapshot.source_mode,
    }


def _blank() -> SpotSnapshot:
    return SpotSnapshot(
        SOURCE_UNAVAILABLE,
        SOURCE_UNAVAILABLE,
        None,
        "network",
        None,
        None,
        None,
        None,
        (),
        (),
    )


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _aware(parsed, "received_at")
    return parsed


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} должен быть в UTC")
