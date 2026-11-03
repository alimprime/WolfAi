"""Phase 3: время публикации, киты, ETF, аналоги, отказ источников и Qwen."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data.candles import Candle
from wolfai.intelligence.align import SeriesIndex, visible_exact_count
from wolfai.intelligence.brief import build_qwen_brief
from wolfai.intelligence.frame import build_market_rows, standardize
from wolfai.intelligence.institutions.etf import US_SPOT_BTC_ETF_START, etf_value, load_etf_csv
from wolfai.intelligence.institutions.farside import fetch_farside
from wolfai.intelligence.market.derivatives import FUNDING_SERIES, OI_SERIES, parse_funding_csv, parse_open_interest_csv
from wolfai.intelligence.market.macro import parse_fred_csv, parse_nyfed_effr
from wolfai.intelligence.market.refused import fetch_coinglass, parse_gdelt_doc
from wolfai.intelligence.memory.analogs import nearest_analogs
from wolfai.intelligence.memory.situation import Situation
from wolfai.intelligence.qwen_run import _parse, build_messages, run_phase3
from wolfai.intelligence.schema import USE_DESCRIPTIVE, USE_STRICT, Observation, SourceUnavailable
from wolfai.intelligence.whales.coinmetrics import flow_anomalies, parse_community_page
from wolfai.intelligence.whales.paid import fetch_paid
from wolfai.intelligence.whales.transfers import WalletTransfer, classify_transfer, net_exchange_btc
from wolfai.research.ollama import OllamaUnavailable, build_chat_body

UTC = timezone.utc
EXCHANGES = frozenset({"binance"})


def test_revision_published_later_is_hidden_and_ingest_does_not_rewrite_it() -> None:
    early = _obs(
        "funding",
        datetime(2022, 6, 1, tzinfo=UTC),
        published=datetime(2022, 6, 1, tzinfo=UTC),
        ingested=datetime(2022, 6, 2, tzinfo=UTC),
        value=0.01,
        version="v1",
        strict=True,
    )
    rewritten = _obs(
        "funding",
        datetime(2022, 6, 1, tzinfo=UTC),
        published=datetime(2022, 6, 1, tzinfo=UTC),
        ingested=datetime(2026, 1, 1, tzinfo=UTC),
        value=0.99,
        version="v2",
        strict=True,
    )
    late = _obs(
        "funding",
        datetime(2022, 6, 1, tzinfo=UTC),
        published=datetime(2023, 1, 1, tzinfo=UTC),
        ingested=datetime(2023, 1, 2, tzinfo=UTC),
        value=5,
        version="v3",
        strict=True,
    )
    chosen = SeriesIndex([late, rewritten, early]).latest_strict("funding", datetime(2022, 7, 1, tzinfo=UTC))
    assert chosen is early
    assert chosen.value == 0.01
    assert SeriesIndex([late]).latest_strict("funding", datetime(2022, 7, 1, tzinfo=UTC)) is None


def test_coinmetrics_restatement_is_not_a_strict_history() -> None:
    payload = {
        "data": [{
            "asset": "btc",
            "time": "2019-01-01T00:00:00.000000000Z",
            "FlowInExNtv": "10.5",
            "FlowInExNtv-status": "flash",
            "FlowInExNtv-status-time": "2026-04-03T22:40:42.988812000Z",
            "HashRate": "100",
            "FeeTotNtv": "1.5",
            "IssTotNtv": "2",
        }]
    }
    observations = parse_community_page(
        payload,
        source_version="abc",
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    flow = next(item for item in observations if item.series_id == "coinmetrics.btc.FlowInExNtv")
    hashrate = next(item for item in observations if item.series_id == "coinmetrics.btc.HashRate")
    revenue = next(item for item in observations if item.series_id == "coinmetrics.btc.miner_revenue_ntv")
    assert flow.strict_backtest is False
    assert flow.published_at == datetime(2026, 4, 3, 22, 40, 42, 988812, tzinfo=UTC)
    assert flow.value == 10.5
    assert "не доказан" not in flow.notes
    assert "не означает продажу" in flow.notes
    assert hashrate.published_at is None and hashrate.strict_backtest is False
    assert revenue.value == pytest.approx(3.5)
    as_of = datetime(2020, 1, 1, tzinfo=UTC)
    assert SeriesIndex(observations).latest_strict(flow.series_id, as_of) is None
    assert visible_exact_count(observations, as_of, "exchange_flow") == 0
    assert flow_anomalies(observations, as_of, series_id=flow.series_id) == []


def test_missing_metric_is_not_filled_with_zero() -> None:
    observations = parse_community_page(
        {"data": [{"asset": "btc", "time": "2019-01-02T00:00:00.000000000Z", "TxCnt": "3"}]},
        source_version="abc",
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert [item.series_id for item in observations] == ["coinmetrics.btc.TxCnt"]
    assert observations[0].value == 3


def test_late_wallet_label_and_internal_transfer() -> None:
    internal = _transfer(
        datetime(2021, 1, 1, tzinfo=UTC),
        label_at=datetime(2021, 1, 1, tzinfo=UTC),
        from_entity="binance",
        to_entity="binance",
        from_address="a",
        to_address="b",
    )
    view = classify_transfer(internal, datetime(2021, 2, 1, tzinfo=UTC), EXCHANGES)
    assert view.role == "internal"
    assert view.counts_toward_exchange_flow is False
    assert view.is_sale is False and view.is_purchase is False
    late = _transfer(
        datetime(2021, 1, 1, tzinfo=UTC),
        label_at=datetime(2024, 1, 1, tzinfo=UTC),
        from_entity="unknown-fund",
        to_entity="binance",
    )
    hidden = classify_transfer(late, datetime(2022, 1, 1, tzinfo=UTC), EXCHANGES)
    assert hidden.role == "unattributed"
    assert hidden.counts_toward_exchange_flow is False
    visible = classify_transfer(late, datetime(2024, 2, 1, tzinfo=UTC), EXCHANGES)
    assert visible.role == "exchange_inflow"
    assert visible.is_sale is False
    assert visible.strict_backtest is False
    assert net_exchange_btc([internal], datetime(2021, 2, 1, tzinfo=UTC), EXCHANGES) is None
    external = _transfer(
        datetime(2021, 1, 1, tzinfo=UTC),
        label_at=datetime(2021, 1, 1, tzinfo=UTC),
        from_entity="fund",
        to_entity="binance",
        amount=2,
    )
    assert net_exchange_btc(
        [internal, external],
        datetime(2021, 2, 1, tzinfo=UTC),
        EXCHANGES,
    ) == pytest.approx(2)


def test_etf_before_launch_is_unavailable_not_zero() -> None:
    assert US_SPOT_BTC_ETF_START == datetime(2024, 1, 11, tzinfo=UTC)
    assert etf_value(datetime(2021, 6, 1, tzinfo=UTC), 0.0) is None
    assert etf_value(datetime(2021, 6, 1, tzinfo=UTC), None) is None
    assert etf_value(datetime(2024, 2, 1, tzinfo=UTC), 0.0) == 0.0


def test_etf_csv_rejects_prelaunch_and_missing_publication(tmp_path) -> None:
    early = tmp_path / "early.csv"
    early.write_text(
        "date,net_flow_usd_million,published_at\n"
        "2020-01-01,0,2020-01-02T00:00:00+00:00\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceUnavailable, match="раньше запуска"):
        load_etf_csv(early, ingested_at=datetime(2026, 1, 1, tzinfo=UTC), source="local")
    undated = tmp_path / "undated.csv"
    undated.write_text("date,net_flow_usd_million\n2024-02-01,3\n", encoding="utf-8")
    with pytest.raises(SourceUnavailable, match="published_at"):
        load_etf_csv(undated, ingested_at=datetime(2026, 1, 1, tzinfo=UTC), source="local")


def test_paid_sources_and_farside_refuse_even_with_a_key() -> None:
    for name in ("arkham", "glassnode", "cryptoquant"):
        with pytest.raises(SourceUnavailable):
            fetch_paid(name, api_key="secret")
    with pytest.raises(SourceUnavailable):
        fetch_farside()
    with pytest.raises(SourceUnavailable):
        fetch_coinglass(api_key="secret")
    with pytest.raises(SourceUnavailable, match="seendate"):
        parse_gdelt_doc({"articles": [{"title": "Bitcoin"}]}, http_status=200)
    with pytest.raises(SourceUnavailable, match="429"):
        parse_gdelt_doc({"articles": []}, http_status=429)


def test_macro_without_release_clock_is_not_strict() -> None:
    fred = parse_fred_csv(
        "observation_date,DFF\n2019-01-02,2.4\n2019-01-03,.\n",
        series_id="DFF",
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert len(fred) == 1
    assert fred[0].strict_backtest is False
    assert fred[0].published_at is None
    assert fred[0].value == pytest.approx(2.4)
    nyfed = parse_nyfed_effr(
        {"refRates": [{"effectiveDate": "2019-01-02", "type": "EFFR", "percentRate": 2.4, "revisionIndicator": ""}]},
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
        source="https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json",
    )
    assert nyfed[0].strict_backtest is False
    assert SeriesIndex(fred + nyfed).latest_strict("nyfed.effr", datetime(2024, 1, 1, tzinfo=UTC)) is None


def test_funding_and_open_interest_keep_their_exchange_timestamp() -> None:
    funding = parse_funding_csv(
        "calc_time,funding_interval_hours,last_funding_rate\n1577836800000,8,0\n",
        source="binance",
        source_version="abc",
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert funding[0].value == 0.0
    assert funding[0].strict_backtest is True
    assert funding[0].published_at == funding[0].event_at
    text = (
        "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
        "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2021-08-01 00:05:00,BTCUSDT,10,1,1,1,1,1\n"
        "2021-08-01 23:55:00,BTCUSDT,12,1,1,1,1,1\n"
        "2021-08-02 00:00:00,BTCUSDT,99,1,1,1,1,1\n"
    )
    snapshot = parse_open_interest_csv(
        text,
        day=datetime(2021, 8, 1, tzinfo=UTC),
        source="binance",
        source_version="abc",
        ingested_at=datetime(2026, 10, 26, tzinfo=UTC),
    )
    assert snapshot is not None
    assert snapshot.value == 12
    assert snapshot.event_at == datetime(2021, 8, 1, 23, 55, tzinfo=UTC)


def test_incomplete_open_interest_is_not_interpolated() -> None:
    candles = _candles(date(2021, 8, 1), 40)
    observations = []
    for offset in range(40):
        if offset == 35:
            continue
        day = datetime(2021, 8, 1, tzinfo=UTC) + timedelta(days=offset)
        observations.append(_obs(
            OI_SERIES,
            day + timedelta(hours=23, minutes=55),
            published=day + timedelta(hours=23, minutes=55),
            ingested=datetime(2026, 1, 1, tzinfo=UTC),
            value=100 + offset,
            version=f"oi-{offset}",
            strict=True,
            category="derivatives",
            unit="BTC",
        ))
    rows = build_market_rows(candles, observations)
    missed = [row for row in rows if row.decision_date == date(2021, 9, 5)]
    assert missed
    assert missed[0].raw["oi_log_change_1d"] is None
    assert missed[0].raw["etf_net_flow_usd_million"] is None
    assert all(row.raw["effr"] is None for row in rows)


def test_standardization_ignores_the_future() -> None:
    rows = []
    for offset in range(80):
        day = date(2021, 1, 1) + timedelta(days=offset)
        rows.append(_market_row(day, {"log_return": 0.01}))
    base = standardize(rows, ("log_return",))
    mutated = list(rows)
    last = mutated[-1]
    mutated[-1] = _market_row(last.decision_date, {"log_return": 50})
    altered = standardize(mutated, ("log_return",))
    assert altered[0].standardized["log_return"] == base[0].standardized["log_return"]
    assert altered[0].decision_date == base[0].decision_date


def test_analogs_ignore_future_outcomes_and_repeat() -> None:
    history = _situations(80)
    query = history[-1]
    first = nearest_analogs(query, history[:-1], feature_names=("log_return", "volatility_7"))
    second = nearest_analogs(query, history[:-1], feature_names=("log_return", "volatility_7"))
    assert first.sufficient
    assert first == second
    changed = []
    for item in history[:-1]:
        changed.append(Situation(
            decision_date=item.decision_date,
            as_of=item.as_of,
            features=dict(item.features),
            outcome_1d=None if item.outcome_1d is None else item.outcome_1d + 10,
            outcome_7d=None if item.outcome_7d is None else item.outcome_7d - 4,
            outcome_1d_available_at=item.outcome_1d_available_at,
            outcome_7d_available_at=item.outcome_7d_available_at,
        ))
    rerun = nearest_analogs(query, changed, feature_names=("log_return", "volatility_7"))
    assert [item.decision_date for item in rerun.neighbors] == [item.decision_date for item in first.neighbors]
    leaked = Situation(
        decision_date=query.decision_date + timedelta(days=3),
        as_of=query.as_of + timedelta(days=3),
        features={"log_return": query.features["log_return"], "volatility_7": query.features["volatility_7"]},
        outcome_1d=0.5,
        outcome_7d=0.5,
        outcome_1d_available_at=query.as_of + timedelta(days=5),
        outcome_7d_available_at=query.as_of + timedelta(days=12),
    )
    without_future = nearest_analogs(query, [*history[:-1], leaked], feature_names=("log_return", "volatility_7"))
    assert leaked.decision_date not in {item.decision_date for item in without_future.neighbors}
    short = nearest_analogs(history[2], history[:2], feature_names=("log_return",))
    assert short.sufficient is False
    assert short.neighbors == ()
    assert "недостаточно" in short.reason


def test_brief_stops_before_development_validation_and_qwen_cannot_invent_analogs() -> None:
    candles = _candles(date(2020, 9, 1), 900)
    observations = _derivatives(datetime(2020, 9, 1, tzinfo=UTC), 900)
    rows = build_market_rows(candles, observations)
    brief = build_qwen_brief(rows, observations)
    assert brief["decision_date"] <= "2022-12-31"
    assert brief["query_outcome_included"] is False
    assert "brier" not in json.dumps(brief).lower()
    neighbor_dates = [item["decision_date"] for item in brief["analogs"]["neighbors"]]
    assert all(item <= "2022-12-31" for item in neighbor_dates)
    messages = build_messages(brief)
    assert "tools" not in messages[1]["content"]
    body = build_chat_body("qwen3.5:4b", messages)
    assert "tools" not in body
    assert body["options"]["num_ctx"] <= 4096
    rejected, error = _parse(
        json.dumps({
            "terms": [{"feature": "log_return", "transform": "identity"}],
            "analogs_used": ["2017-12-17"],
            "change": "тест",
            "problem": "тест",
            "expected_improvement": "тест",
            "explanation": "тест",
            "falsification": {"reference": "constant_0.5", "split": "inner_2022"},
        }),
        brief,
    )
    assert rejected is None
    assert error is not None and "аналог" in error


def test_qwen_unavailable_does_not_invent_a_hypothesis() -> None:
    class _Down:
        def health(self) -> object:
            raise OllamaUnavailable("Ollama недоступна")

        def complete(self, messages: list[dict[str, str]], schema: dict[str, object] | None = None) -> object:
            raise AssertionError("модель не должна вызываться")

    result = run_phase3(_Down(), _candles(date(2021, 1, 1), 10), [])
    assert result["status"] == "ollama_unavailable"
    assert result["hypothesis"] is None


def test_ablation_blocks_missing_categories_and_ignores_a_future_revision(tmp_path) -> None:
    from wolfai.intelligence.ablation import run_ablation

    candles = _candles(date(2020, 9, 1), 700)
    observations = _derivatives(datetime(2020, 9, 1, tzinfo=UTC), 700)
    first = run_ablation(candles, observations, include_ranges=False)
    assert first["configurations"]["B"]["status"] == "blocked"
    assert first["configurations"]["C"]["status"] == "blocked"
    assert first["configurations"]["A"]["sample"] == "matched"
    assert first["configurations"]["D"]["status"] == "scored"
    assert first["holdout"]["metrics"] is None
    assert "etf_net_flow_usd_million" not in first["configurations"]["D"]["features"]
    assert first["sample"]["matched_rows"] == first["sample"]["matched_rows"]
    late = _obs(
        FUNDING_SERIES,
        datetime(2022, 1, 15, 8, tzinfo=UTC),
        published=datetime(2023, 6, 1, tzinfo=UTC),
        ingested=datetime(2023, 6, 2, tzinfo=UTC),
        value=0.2,
        version="future-revision",
        strict=True,
        category="derivatives",
        unit="rate_per_interval",
    )
    second = run_ablation(candles, [*observations, late], include_ranges=False)
    assert _inner_brier(first) == _inner_brier(second)
    third = run_ablation(candles, [*observations, late], include_ranges=False)
    assert _inner_brier(second) == _inner_brier(third)
    del tmp_path


def _inner_brier(report: dict[str, object]) -> float:
    configuration = report["configurations"]["D"]
    assert isinstance(configuration, dict)
    periods = configuration["periods"]
    assert isinstance(periods, dict)
    inner = periods["inner_2022"]
    assert isinstance(inner, dict)
    return float(inner["brier"])


def _candles(start: date, count: int) -> list[Candle]:
    opened = datetime(start.year, start.month, start.day, tzinfo=UTC)
    candles = []
    for index in range(count):
        close = 100 + (index % 19) * 0.7 + index * 0.05
        candles.append(Candle(opened + timedelta(days=index), close, close, close, close, 10 + (index % 5)))
    return candles


def _derivatives(start: datetime, days: int) -> list[Observation]:
    observations: list[Observation] = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        for hour in (0, 8, 16):
            stamp = day + timedelta(hours=hour)
            observations.append(_obs(
                FUNDING_SERIES,
                stamp,
                published=stamp,
                ingested=datetime(2026, 1, 1, tzinfo=UTC),
                value=((offset + hour) % 7 - 3) / 10000,
                version=f"funding-{offset}-{hour}",
                strict=True,
                category="derivatives",
                unit="rate_per_interval",
            ))
        snapshot = day + timedelta(hours=23, minutes=55)
        observations.append(_obs(
            OI_SERIES,
            snapshot,
            published=snapshot,
            ingested=datetime(2026, 1, 1, tzinfo=UTC),
            value=1000 + offset,
            version=f"oi-{offset}",
            strict=True,
            category="derivatives",
            unit="BTC",
        ))
    return observations


def _market_row(day: date, raw: dict[str, float]) -> object:
    from wolfai.intelligence.frame import MarketRow

    return MarketRow(
        decision_date=day,
        target_date=day + timedelta(days=1),
        as_of=datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1),
        y_up=1,
        baseline_up_frequency=0.5,
        raw=raw,
        outcome_1d=None,
        outcome_7d=None,
        outcome_1d_available_at=None,
        outcome_7d_available_at=None,
    )


def _situations(count: int) -> list[Situation]:
    start = date(2021, 1, 1)
    situations = []
    for index in range(count):
        day = start + timedelta(days=index)
        as_of = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
        mature = index <= count - 10
        situations.append(Situation(
            decision_date=day,
            as_of=as_of,
            features={
                "log_return": ((index % 11) - 5) / 100,
                "volatility_7": 0.01 + (index % 5) / 1000,
            },
            outcome_1d=(index % 3 - 1) / 100 if mature else None,
            outcome_7d=(index % 4 - 2) / 50 if mature else None,
            outcome_1d_available_at=as_of + timedelta(days=1) if mature else None,
            outcome_7d_available_at=as_of + timedelta(days=7) if mature else None,
        ))
    return situations


def _obs(
    series_id: str,
    event_at: datetime,
    *,
    published: datetime | None,
    ingested: datetime,
    value: float | None,
    version: str,
    strict: bool,
    category: str = "derivatives",
    unit: str = "BTC",
) -> Observation:
    return Observation(
        series_id=series_id,
        event_at=event_at,
        published_at=published,
        ingested_at=ingested,
        source="test",
        source_version=version,
        confidence=0.9,
        value=value,
        unit=unit,
        category=category,
        use_class=USE_STRICT if strict else USE_DESCRIPTIVE,
        publication_time_exact=published is not None,
        strict_backtest=strict,
        notes="тестовое наблюдение с явным временем",
    )


def _transfer(
    event_at: datetime,
    *,
    label_at: datetime | None,
    from_entity: str | None,
    to_entity: str | None,
    from_address: str = "from",
    to_address: str = "to",
    amount: float = 1,
) -> WalletTransfer:
    return WalletTransfer(
        event_at=event_at,
        published_at=event_at,
        label_published_at=label_at,
        from_address=from_address,
        to_address=to_address,
        from_entity=from_entity,
        to_entity=to_entity,
        amount_btc=amount,
        source="test",
        source_version="v1",
    )
