"""Этап 2.6: живой источник, forward и жизненный цикл гипотез.

Фикстуры ниже не являются прогнозом Bitcoin. Они проверяют правила.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from urllib.error import URLError

import pytest

from wolfai.data.spot_live import (
    SOURCE_UNAVAILABLE,
    HttpResponse,
    LiveMarketStore,
    SpotClient,
    interpret_market,
)
from wolfai.research.arena_gate import ArenaGate
from wolfai.research.formula import FormulaError
from wolfai.research.forward_mode import (
    CODE_VERSION,
    features_visible_at,
    guard_available,
    run_forward_cycle,
)
from wolfai.research.hypothesis_engine import (
    MIN_OBSERVATIONS,
    RULE_TEXT,
    ContradictionAssessment,
    HypothesisLifecycleError,
    HypothesisObservation,
    HypothesisStore,
    accept_proposal,
    review,
)
from wolfai.research.hypothesis_generator import (
    CATALOG,
    DeterministicHypothesisGenerator,
    GenerationContext,
    StructuredHypothesis,
)
from wolfai.research.prediction_ledger import PredictionImmutable, PredictionLedger, snapshot_hash
from wolfai.research.shadow import main
from wolfai.research.strategy import ConstantHalfStrategy, FeatureView

UTC = timezone.utc
START = datetime(2026, 9, 1, tzinfo=UTC)


class DownTransport:
    def __init__(self) -> None:
        self.calls = 0

    def request(self, url: str, timeout: float) -> HttpResponse:
        self.calls += 1
        raise URLError("no route")


class StatusTransport:
    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.calls = 0

    def request(self, url: str, timeout: float) -> HttpResponse:
        self.calls += 1
        return HttpResponse(self.status, self.headers, self.body, url)


@dataclass(frozen=True)
class NamedHalf:
    strategy_id: str
    version: str = "1"

    def predict(self, view: FeatureView) -> float:
        del view
        return 0.5


class Sneaky:
    strategy_id = "sneaky"
    version = "1"
    evaluated = False

    def predict(self, view: FeatureView) -> float:
        del view
        return 0.5

    def evaluate(self, ledger: PredictionLedger) -> dict[str, object]:
        del ledger
        self.evaluated = True
        return {"claim_status": "supported", "validated_law": True}


class FixtureTransport:
    def __init__(self, time_body: bytes, klines_body: bytes) -> None:
        self.time_body = time_body
        self.klines_body = klines_body
        self.urls: list[str] = []

    def request(self, url: str, timeout: float) -> HttpResponse:
        self.urls.append(url)
        if url.endswith("/api/v3/time"):
            return HttpResponse(200, {}, self.time_body, url)
        if "symbol=BTCUSDT" in url and "interval=1d" in url and "/api/v3/klines?" in url:
            return HttpResponse(200, {"X-MBX-USED-WEIGHT-1M": "3"}, self.klines_body, url)
        raise AssertionError(url)


def test_live_source_reads_a_local_fixture_without_network(tmp_path) -> None:
    time_body, klines_body, received, _rows = _bodies()
    transport = FixtureTransport(time_body, klines_body)
    fetched = SpotClient(transport).fetch(received_at=received)
    assert fetched.status == "ok"
    assert fetched.requests_made == 2
    assert fetched.source_mode == "network"
    assert fetched.base_url == "https://data-api.binance.vision"
    assert len(transport.urls) == 2
    assert all(url.startswith("https://data-api.binance.vision/api/v3/") for url in transport.urls)
    snapshot = interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert snapshot.status == "ok"
    assert snapshot.api_first_seen_proven is False
    assert snapshot.rejected[-1].reason == "unclosed"
    last = snapshot.closed[-1]
    assert last.api_first_seen_proven is False
    assert last.api_observation_grade == "unknown"
    assert last.availability_grade == "verified"
    assert len({last.open_time, last.close_time, last.exchange_server_time, last.received_at}) == 4
    assert last.available_at == last.open_time + timedelta(days=1)
    assert last.available_at != last.received_at
    assert last.available_at != last.close_time
    store = LiveMarketStore(tmp_path / "market.sqlite")
    store.save_snapshot(snapshot, time_body, klines_body)
    digest, state = store.save_body(klines_body, received)
    assert state == "duplicate"
    assert digest == hashlib.sha256(klines_body).hexdigest()
    assert store.candle_count() == len(snapshot.closed)
    store.close()


def test_unclosed_candle_is_not_treated_as_complete() -> None:
    time_body, klines_body, received, _rows = _bodies(closed_days=1, extra_open=False, server_after=timedelta(hours=-2))
    snapshot = interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert snapshot.status == SOURCE_UNAVAILABLE
    assert snapshot.reason == "no_closed_candle"
    assert snapshot.closed == ()
    assert snapshot.rejected[0].reason == "unclosed"
    incomplete_time, incomplete_rows, incomplete_received, _rows = _bodies(trades=0, extra_open=False)
    incomplete = interpret_market(
        incomplete_time,
        incomplete_rows,
        incomplete_received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert incomplete.status == "incomplete_candle"
    assert incomplete.closed == ()


def test_wrong_timestamp_and_bad_ohlcv_are_rejected() -> None:
    def shift(rows: list[list[object]]) -> None:
        rows[0][0] = int(rows[0][0]) + 60_000

    time_body, klines_body, received, _rows = _bodies(mutate=shift)
    shifted = interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert shifted.status == "invalid_timestamp"
    assert shifted.closed == ()

    def cross(rows: list[list[object]]) -> None:
        rows[0][2] = "1.00"
        rows[0][3] = "80.00"

    bad_time, bad_rows, bad_received, _rows = _bodies(mutate=cross)
    broken = interpret_market(
        bad_time,
        bad_rows,
        bad_received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert broken.status == "invalid_ohlcv"
    assert broken.closed == ()


def test_api_unavailable_does_not_retry_through_a_block(tmp_path) -> None:
    down = DownTransport()
    missed = SpotClient(down).fetch(received_at=datetime(2026, 11, 4, tzinfo=UTC))
    assert missed.status == SOURCE_UNAVAILABLE
    assert missed.reason == "connection_error"
    assert missed.closed == ()
    assert down.calls == 2
    restricted = StatusTransport(451)
    blocked = SpotClient(restricted).fetch(received_at=datetime(2026, 11, 4, tzinfo=UTC))
    assert blocked.status == SOURCE_UNAVAILABLE
    assert blocked.reason == "access_restricted"
    assert restricted.calls == 1
    limited = StatusTransport(429, headers={"Retry-After": "30"})
    paced = SpotClient(limited).fetch(received_at=datetime(2026, 11, 4, tzinfo=UTC))
    assert paced.reason == "rate_limited"
    assert limited.calls == 1
    heavy = StatusTransport(200, b'{"serverTime": 1}', {"X-MBX-USED-WEIGHT-1M": "5800"})
    backed = SpotClient(heavy).fetch(received_at=datetime(2026, 11, 4, tzinfo=UTC))
    assert backed.reason == "rate_limited"
    assert heavy.calls == 1
    assert not list(tmp_path.iterdir())


def test_repeat_fetch_is_duplicate_and_conflict_keeps_the_first_row(tmp_path) -> None:
    time_body, klines_body, received, _rows = _bodies()
    snapshot = interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    store = LiveMarketStore(tmp_path / "market.sqlite")
    first = store.save_snapshot(snapshot, time_body, klines_body)
    second = store.save_snapshot(snapshot, time_body, klines_body)
    assert first["stored"] == len(snapshot.closed)
    assert second["duplicate"] == len(snapshot.closed)
    assert store.candle_count() == len(snapshot.closed)
    original = snapshot.closed[0]
    changed = replace(original, close=original.close + 50, raw_ohlcv_sha256="ab" * 32)
    assert store.save_candle(changed) == "conflict"
    assert store.candle_count() == len(snapshot.closed)
    assert store.conflict_count() == 1
    store.close()
    checked = sqlite3.connect(tmp_path / "market.sqlite")
    row = checked.execute(
        "SELECT close, raw_ohlcv_sha256 FROM spot_candles WHERE open_time = ?",
        (original.open_time.isoformat(),),
    ).fetchone()
    checked.close()
    assert row[0] == original.close
    assert row[1] == original.raw_ohlcv_sha256


def test_available_at_hides_a_future_candle() -> None:
    time_body, klines_body, received, _rows = _bodies(closed_days=36, extra_open=False)
    snapshot = interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )
    assert snapshot.status == "ok"
    assert len(snapshot.closed) == 36
    decision = snapshot.closed[-2].available_at
    without_future = features_visible_at(snapshot.closed[:-1], decision)
    with_future = features_visible_at(snapshot.closed, decision)
    assert with_future == without_future
    later = features_visible_at(snapshot.closed, snapshot.closed[-1].available_at)
    assert later != with_future
    view = FeatureView(
        decision_time=decision,
        features={"log_return": 0.1},
        feature_available_at={"log_return": decision + timedelta(minutes=1)},
    )
    with pytest.raises(ValueError, match="decision_time"):
        guard_available(view)


def test_hypothesis_is_stored_once_and_lineage_cannot_cycle(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    created = datetime(2026, 11, 4, tzinfo=UTC)
    first, state = _create(store, CATALOG[0], "импульс", created_at=created)
    assert state == "stored"
    again, duplicate = _create(store, CATALOG[0], "то же под другим именем", created_at=created)
    assert duplicate == "duplicate"
    assert again == first
    assert store.count() == 1
    saved = store.get(first)
    assert saved.creation_reason
    assert saved.data_hash == "synthetic-not-btc"
    assert saved.claim_status == "not_confirmed"
    child, child_state = _create(
        store,
        CATALOG[1],
        "короткий импульс",
        created_at=created,
        parent_hypothesis_id=first,
    )
    assert child_state == "stored"
    assert store.get(child).version == 2
    assert store.get(child).parent_hypothesis_id == first
    assert store.derived_ids(first) == [child]
    with pytest.raises(HypothesisLifecycleError, match="цикли"):
        store.link_derived(child, first)
    assert store.derived_ids(child) == []
    experiment_id, experiment_state = store.add_experiment(
        first,
        {"note": "синтетическая проверка", "research_memory_key": "not-opened"},
        when=created,
    )
    assert experiment_state == "stored"
    assert store.experiment_count(first) == 1
    assert store.get(first).status == "testing"
    assert experiment_id
    store.close()


def test_generator_is_deterministic_and_does_not_search_history() -> None:
    generator = DeterministicHypothesisGenerator()
    context = GenerationContext(task_id="task-stable")
    assert generator.propose(context) == generator.propose(context)
    assert len(CATALOG) == 2
    assert generator.propose(context).expression in {item.expression for item in CATALOG}
    before = RULE_TEXT
    generator.propose(context)
    assert RULE_TEXT == before
    with pytest.raises(PermissionError):
        generator.propose(GenerationContext(task_id="holdout", touches_holdout=True))
    with pytest.raises(PermissionError):
        generator.propose(GenerationContext(task_id="late", reads_closed_history_after_2021=True))


def test_sustained_failure_is_recorded_and_one_miss_is_not(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    created = datetime(2026, 11, 4, tzinfo=UTC)
    single_id, _state = _create(store, CATALOG[0], "одна ошибка", created_at=created)
    expression = store.get(single_id).expression
    single = review(store, single_id, [_observation(0, hit=False)], when=created)
    assert single.kind == "single_error"
    assert single.create_contradiction is False
    assert store.get(single_id).status == "proposed"
    assert store.get(single_id).expression == expression
    assert store.contradictions_for(single_id) == []
    bad_id, _state = _create(store, CATALOG[1], "устойчивое ухудшение", created_at=created)
    before = store.get(bad_id).expression
    sustained = review(
        store,
        bad_id,
        [_observation(index, hit=False) for index in range(MIN_OBSERVATIONS)],
        when=created,
    )
    assert sustained.kind == "systematic_deterioration"
    assert sustained.reject is True
    assert float(sustained.detail["high"]) < 0.0
    saved = store.get(bad_id)
    assert saved.status == "rejected"
    assert saved.expression == before
    contradictions = store.contradictions_for(bad_id)
    assert len(contradictions) == 1
    assert contradictions[0]["expression"] == before
    tasks = store.tasks_for(bad_id)
    assert len(tasks) == 1
    assert "expression" not in tasks[0]
    regime_id, _state = _create(
        store,
        CATALOG[0],
        "режим",
        created_at=created,
        applicability={"market": "synthetic", "regimes": ["low_volatility"], "btc_claim": False},
        name="regime-shift",
    )
    regime = review(
        store,
        regime_id,
        [_observation(index, hit=True, regime="high_volatility") for index in range(20)],
        when=created,
    )
    assert regime.kind == "regime_shift"
    assert regime.reject is False
    assert store.get(regime_id).status == "proposed"
    assert store.get(regime_id).expression == CATALOG[0].expression
    assert store.contradictions_for(regime_id)[0]["kind"] == "regime_shift"
    store.close()
    assert isinstance(sustained, ContradictionAssessment)


def test_rejected_hypothesis_is_not_deleted(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    created = datetime(2026, 11, 4, tzinfo=UTC)
    hypothesis_id, _state = _create(store, CATALOG[0], "сохранить отказ", created_at=created)
    review(
        store,
        hypothesis_id,
        [_observation(index, hit=False) for index in range(MIN_OBSERVATIONS)],
        when=created,
    )
    assert store.count() == 1
    assert store.get(hypothesis_id).status == "rejected"
    expression = store.get(hypothesis_id).expression
    store.transition(hypothesis_id, "archived", "архив после отказа", when=created)
    assert store.count() == 1
    assert store.get(hypothesis_id).expression == expression
    assert store.get(hypothesis_id).status == "archived"
    assert not hasattr(store, "delete")
    store.close()


def test_validated_law_and_supported_are_forbidden(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    created = datetime(2026, 11, 4, tzinfo=UTC)
    hypothesis_id, _state = _create(store, CATALOG[0], "кандидат", created_at=created)
    store.add_experiment(hypothesis_id, {"note": "ещё не доказано"}, when=created)
    store.mark_candidate(hypothesis_id, "перспективно, подтверждения нет", when=created)
    saved = store.get(hypothesis_id)
    assert saved.status == "candidate"
    assert saved.claim_status == "not_confirmed"
    assert store.confirms_law(hypothesis_id) is False
    with pytest.raises(HypothesisLifecycleError, match="Validated"):
        store.transition(hypothesis_id, "validated_law", "нельзя", when=created)
    with pytest.raises(HypothesisLifecycleError, match="Validated"):
        store.transition(hypothesis_id, "Validated Law", "нельзя", when=created)
    with pytest.raises(HypothesisLifecycleError, match="Validated"):
        store.transition(hypothesis_id, "supported", "нельзя", when=created)
    with pytest.raises(HypothesisLifecycleError):
        store.add_experiment(hypothesis_id, {"claim_status": "supported"}, when=created)
    assert store.get(hypothesis_id).status == "candidate"
    assert store.confirms_law(hypothesis_id) is False
    store.close()


def test_strategies_cannot_cross_their_rights(tmp_path) -> None:
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    gate = ArenaGate(store)
    alpha = NamedHalf("alpha")
    beta = NamedHalf("beta")
    assert gate.register(alpha) == "registered"
    assert gate.register(alpha) == "duplicate"
    assert gate.register(beta) == "registered"
    created = datetime(2026, 11, 4, tzinfo=UTC)
    own, _state = gate.propose(alpha.strategy_id, _proposal(alpha.strategy_id, CATALOG[0]), created_at=created, data_hash="synthetic-not-btc")
    other, _state = gate.propose(beta.strategy_id, _proposal(beta.strategy_id, CATALOG[1]), created_at=created, data_hash="synthetic-not-btc")
    with pytest.raises(PermissionError):
        gate.propose(alpha.strategy_id, _proposal(beta.strategy_id, CATALOG[1]), created_at=created, data_hash="synthetic-not-btc")
    store.add_experiment(own, {"note": "только alpha"}, when=created)
    alpha_history = gate.own_history(alpha.strategy_id)
    beta_history = gate.own_history(beta.strategy_id)
    assert [item.hypothesis_id for item in alpha_history["hypotheses"]] == [own]
    assert [item.hypothesis_id for item in beta_history["hypotheses"]] == [other]
    assert alpha_history["experiments"]
    assert beta_history["experiments"] == []
    snapshot = _snapshot()
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    first = gate.submit(alpha.strategy_id, ledger, snapshot, commit=True)
    second = gate.submit(beta.strategy_id, ledger, snapshot, commit=True)
    assert first["predictions_stored"] == 1
    assert second["strategy_id"] == "beta"
    assert ledger.get(str(first["prediction_id"])).model_id == "alpha"
    features = features_visible_at(snapshot.closed, snapshot.closed[-1].available_at)
    with pytest.raises(PredictionImmutable, match="чужой"):
        ledger.append(
            model_id="beta",
            model_version="1",
            created_at=snapshot.received_at,
            decision_time=snapshot.closed[-1].available_at,
            horizon_days=1,
            probability_up=0.2,
            features=features,
            feature_available_at={name: snapshot.closed[-1].available_at for name in features},
            dataset_version=str(first["dataset_version"]),
            code_version=CODE_VERSION,
            channel="forward",
            replaces_prediction_id=str(first["prediction_id"]),
        )
    with pytest.raises(PermissionError):
        gate.submit("missing", ledger, snapshot)
    sneaky = Sneaky()
    gate.register(sneaky)
    report = gate.evaluation(ledger, "forward")
    assert sneaky.evaluated is False
    assert report["claim_status"] != "supported"
    assert report.get("validated_law") is not True
    ledger.close()
    store.close()


def test_generator_cannot_execute_arbitrary_code(tmp_path) -> None:
    sentinel = tmp_path / "pwned.txt"
    malicious = f"__import__('os').system('touch {sentinel}')"
    with pytest.raises(FormulaError):
        DeterministicHypothesisGenerator().propose(
            GenerationContext(task_id="danger", requested_expression=malicious)
        )
    assert not sentinel.exists()
    store = HypothesisStore(tmp_path / "hypotheses.sqlite")
    proposal = StructuredHypothesis(
        author_id="deterministic-catalog",
        name="bad",
        description="запрещённый текст",
        creation_reason="проверка отказа",
        features=("momentum_7",),
        horizon_days=1,
        expression=malicious,
        applicability={"market": "synthetic", "regimes": ["any"], "btc_claim": False},
    )
    with pytest.raises(FormulaError):
        accept_proposal(
            store,
            proposal,
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
            data_hash="synthetic-not-btc",
        )
    assert not sentinel.exists()
    assert store.count() == 0
    assert store.attempt_count() == 1
    store.close()


def test_verified_fixture_can_commit_a_forward_prediction(tmp_path) -> None:
    time_body, klines_body, received, _rows = _bodies()
    fixture = tmp_path / "market.json"
    fixture.write_text(json.dumps({
        "received_at": received.isoformat(),
        "time_body": time_body.decode(),
        "klines_body": klines_body.decode(),
    }), encoding="utf-8")
    ledger_path = tmp_path / "ledger.sqlite"
    dry = main(["--ledger", str(ledger_path), "--channel", "forward", "--market-fixture", str(fixture)])
    assert dry["status"] == "dry_run"
    assert dry["predictions_stored"] == 0
    assert dry["validated_law"] is False
    assert dry["api_first_seen_proven"] is False
    opened = PredictionLedger(ledger_path)
    assert opened.count() == 0
    opened.close()
    first = main([
        "--ledger", str(ledger_path),
        "--channel", "forward",
        "--market-fixture", str(fixture),
        "--commit",
    ])
    second = main([
        "--ledger", str(ledger_path),
        "--channel", "forward",
        "--market-fixture", str(fixture),
        "--commit",
    ])
    assert first["status"] == "ok"
    assert first["predictions_stored"] == 1
    assert first["source_mode"] == "verified_fixture"
    assert first["code_version"] == CODE_VERSION
    assert first["strategy_id"] == "constant-half"
    assert first["strategy_version"] == "1"
    assert FEATURE_SCHEMA_IN(str(first["dataset_version"]))
    assert hashlib.sha256(klines_body).hexdigest() in str(first["dataset_version"])
    assert second["write"] == "duplicate"
    ledger = PredictionLedger(ledger_path)
    stored = ledger.get(str(first["prediction_id"]))
    assert stored.channel == "forward"
    assert stored.model_version == "1"
    assert stored.code_version == CODE_VERSION
    snapshot = _snapshot()
    features = features_visible_at(snapshot.closed, snapshot.closed[-1].available_at)
    assert stored.feature_snapshot_hash == snapshot_hash(features)
    assert snapshot.rejected[-1].reason == "unclosed"
    assert stored.decision_time == snapshot.closed[-1].available_at
    assert stored.created_at == received
    ledger.close()


def test_backtest_cannot_be_labeled_as_a_live_forecast(tmp_path) -> None:
    snapshot = _snapshot()
    ledger = PredictionLedger(tmp_path / "ledger.sqlite")
    historical = run_forward_cycle(
        ledger=ledger,
        strategy=ConstantHalfStrategy(),
        snapshot=replace(snapshot, source_mode="historical"),
        commit=True,
    )
    reconstructed = run_forward_cycle(
        ledger=ledger,
        strategy=ConstantHalfStrategy(),
        snapshot=snapshot,
        commit=True,
        origin="reconstruction",
    )
    late = run_forward_cycle(
        ledger=ledger,
        strategy=ConstantHalfStrategy(),
        snapshot=replace(snapshot, received_at=snapshot.closed[-1].available_at + timedelta(days=2)),
        commit=True,
    )
    assert historical["status"] == "backtest_refused"
    assert reconstructed["status"] == "backtest_refused"
    assert late["status"] == "backtest_refused"
    assert historical["predictions_stored"] == 0
    assert ledger.count() == 0
    broken = tmp_path / "not-market.json"
    broken.write_text(json.dumps({
        "received_at": "2021-06-01T00:00:00+00:00",
        "time_body": "not-json",
        "klines_body": "[]",
    }), encoding="utf-8")
    refused = main([
        "--ledger", str(tmp_path / "other.sqlite"),
        "--channel", "forward",
        "--market-fixture", str(broken),
        "--commit",
    ])
    assert refused["predictions_stored"] == 0
    assert refused["status"] != "ok"
    other = PredictionLedger(tmp_path / "other.sqlite")
    assert other.count() == 0
    other.close()
    ledger.close()


def _bodies(
    *,
    closed_days: int = 35,
    extra_open: bool = True,
    server_after: timedelta = timedelta(hours=1),
    trades: int = 8,
    mutate=None,
) -> tuple[bytes, bytes, datetime, list[list[object]]]:
    count = closed_days + (1 if extra_open else 0)
    rows = [
        _row(START + timedelta(days=index), 100 + index, trades=trades)
        for index in range(count)
    ]
    if mutate is not None:
        mutate(rows)
    available = START + timedelta(days=closed_days)
    server_time = available + server_after
    received = server_time + timedelta(seconds=2)
    time_body = json.dumps({"serverTime": _millis(server_time)}).encode()
    klines_body = json.dumps(rows).encode()
    return time_body, klines_body, received, rows


def _snapshot():
    time_body, klines_body, received, _rows = _bodies()
    return interpret_market(
        time_body,
        klines_body,
        received,
        base_url="fixture",
        source_mode="verified_fixture",
    )


def _row(open_time: datetime, price: float, *, trades: int) -> list[object]:
    open_ms = _millis(open_time)
    return [
        open_ms,
        f"{price:.2f}",
        f"{price + 1:.2f}",
        f"{price - 1:.2f}",
        f"{price + 0.40:.2f}",
        "12.50",
        open_ms + 86_400_000 - 1,
        "100.00",
        trades,
        "1.00",
        "1.00",
        "0",
    ]


def _millis(moment: datetime) -> int:
    if moment.microsecond:
        raise AssertionError("метка должна попадать на миллисекунду без дробного хвоста")
    return int(moment.timestamp()) * 1000


def _create(
    store: HypothesisStore,
    item,
    reason: str,
    *,
    created_at: datetime,
    parent_hypothesis_id: str | None = None,
    applicability: dict[str, object] | None = None,
    name: str | None = None,
) -> tuple[str, str]:
    return store.create(
        author_id="deterministic-catalog",
        name=name or item.name,
        description=item.description,
        creation_reason=reason,
        features=item.features,
        horizon_days=item.horizon_days,
        expression=item.expression,
        applicability=item.applicability if applicability is None else applicability,
        created_at=created_at,
        data_hash="synthetic-not-btc",
        parent_hypothesis_id=parent_hypothesis_id,
    )


def _proposal(author_id: str, item) -> StructuredHypothesis:
    return StructuredHypothesis(
        author_id=author_id,
        name=item.name,
        description=item.description,
        creation_reason=item.creation_reason,
        features=item.features,
        horizon_days=item.horizon_days,
        expression=item.expression,
        applicability=dict(item.applicability),
    )


def _observation(index: int, *, hit: bool, regime: str = "low_volatility") -> HypothesisObservation:
    return HypothesisObservation(
        probability=0.1 if hit else 0.9,
        label=1 if hit else 0,
        regime=regime,
        in_domain=True,
        decision_date=date(2024, 1, 1) + timedelta(days=index),
    )


def FEATURE_SCHEMA_IN(dataset_version: str) -> bool:
    return "features:ohlcv-daily-v1" in dataset_version
