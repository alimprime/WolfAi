"""Community API Coin Metrics.

Документация: https://docs.coinmetrics.io/api/v4/
Корень без ключа: https://community-api.coinmetrics.io/v4
Лицензия на странице community data: Creative Commons.
Лимит: 10 запросов за 6 секунд на IP.

Платный корень https://api.coinmetrics.io/v4 без ключа отвечает 401 и здесь не используется.
Point-in-time метрики FlowInExPitNtv и SplyExPitNtv в community-каталоге отсутствуют
и на пробном запросе отвечают 403. Стандартные Flow* и SplyEx* пересчитываются,
когда поставщик позже узнаёт новые адреса. Их status-time для дат до 2024 года
лежит в 2026-04, поэтому в строгий backtest они не входят.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from wolfai.intelligence.schema import (
    USE_DESCRIPTIVE,
    Observation,
    SourceUnavailable,
    parse_utc,
)

COMMUNITY_ROOT = "https://community-api.coinmetrics.io/v4"
CATALOG_URL = f"{COMMUNITY_ROOT}/catalog/assets?assets=btc"
PIT_PROBE_METRIC = "FlowInExPitNtv"
RESEARCH_END = "2024-12-31"

COMMUNITY_METRICS = (
    "FlowInExNtv",
    "FlowOutExNtv",
    "SplyExNtv",
    "HashRate",
    "FeeTotNtv",
    "IssTotNtv",
    "TxCnt",
    "AdrActCnt",
    "BlkCnt",
)


@dataclass(frozen=True)
class _Metric:
    metric: str
    unit: str
    category: str
    confidence: float
    label_dependent: bool
    entity: str | None


_METRICS: dict[str, _Metric] = {
    "FlowInExNtv": _Metric("FlowInExNtv", "BTC", "exchange_flow", 0.35, True, "all_exchanges"),
    "FlowOutExNtv": _Metric("FlowOutExNtv", "BTC", "exchange_flow", 0.35, True, "all_exchanges"),
    "SplyExNtv": _Metric("SplyExNtv", "BTC", "exchange_reserve", 0.35, True, "all_exchanges"),
    "HashRate": _Metric("HashRate", "TH/s", "miner", 0.45, False, None),
    "FeeTotNtv": _Metric("FeeTotNtv", "BTC", "miner", 0.55, False, None),
    "IssTotNtv": _Metric("IssTotNtv", "BTC", "miner", 0.55, False, None),
    "TxCnt": _Metric("TxCnt", "count", "network", 0.55, False, None),
    "AdrActCnt": _Metric("AdrActCnt", "addresses", "network", 0.55, False, None),
    "BlkCnt": _Metric("BlkCnt", "blocks", "network", 0.55, False, None),
}


def timeseries_url(start: str = "2009-01-09", end: str = RESEARCH_END) -> str:
    if end >= "2025-01-01":
        raise ValueError("выгрузка исследования не запрашивает даты holdout")
    metrics = ",".join(COMMUNITY_METRICS)
    return (
        f"{COMMUNITY_ROOT}/timeseries/asset-metrics?assets=btc&metrics={metrics}"
        f"&frequency=1d&start_time={start}&end_time={end}&page_size=10000"
    )


def pit_probe_url() -> str:
    return (
        f"{COMMUNITY_ROOT}/timeseries/asset-metrics?assets=btc&metrics={PIT_PROBE_METRIC}"
        "&frequency=1d&start_time=2024-01-01&end_time=2024-01-03&page_size=3"
    )


def parse_community_page(
    payload: dict[str, object],
    *,
    source_version: str,
    ingested_at: datetime,
) -> list[Observation]:
    """Строит наблюдения из одной страницы timeseries. Пропуск метрики не становится нулём."""
    data = payload.get("data")
    if not isinstance(data, list):
        raise SourceUnavailable("Coin Metrics не вернула список data")
    observations: list[Observation] = []
    for row in data:
        if not isinstance(row, dict):
            raise SourceUnavailable("строка Coin Metrics не объект")
        observations.extend(_row_observations(row, source_version=source_version, ingested_at=ingested_at))
    observations.extend(_miner_revenue(observations))
    return observations


def parse_catalog_coverage(payload: dict[str, object]) -> list[dict[str, object]]:
    """Границы community-каталога. Это метаданные, не значения ряда."""
    data = payload.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise SourceUnavailable("каталог Coin Metrics пуст")
    metrics = data[0].get("metrics")
    if not isinstance(metrics, list):
        raise SourceUnavailable("в каталоге нет metrics")
    coverage: list[dict[str, object]] = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        frequencies = item.get("frequencies")
        if not isinstance(frequencies, list) or not frequencies:
            continue
        daily = next(
            (freq for freq in frequencies if isinstance(freq, dict) and freq.get("frequency") == "1d"),
            None,
        )
        if not isinstance(daily, dict):
            continue
        coverage.append({
            "metric": item.get("metric"),
            "min_time": daily.get("min_time"),
            "max_time": daily.get("max_time"),
            "community": daily.get("community"),
        })
    return coverage


def payload_version(payloads: list[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(payload)
    return digest.hexdigest()


def _row_observations(
    row: dict[str, object],
    *,
    source_version: str,
    ingested_at: datetime,
) -> list[Observation]:
    if row.get("asset") not in (None, "btc"):
        raise SourceUnavailable("ожидался актив btc")
    event_at = parse_utc(str(row.get("time")))
    found: list[Observation] = []
    for metric, spec in _METRICS.items():
        if metric not in row or row[metric] is None:
            continue
        value = float(str(row[metric]))
        status_time = row.get(f"{metric}-status-time")
        published_at = parse_utc(str(status_time)) if isinstance(status_time, str) and status_time else None
        if spec.label_dependent:
            notes = (
                "Стандартная метрика Coin Metrics использует известные сегодня адреса бирж. "
                "Point-in-time ряд в community API недоступен. Приток не означает продажу, "
                "агрегат не разделяет внутренние переводы. "
                f"status={row.get(f'{metric}-status')!s}."
            )
            exact = published_at is not None
            found.append(Observation(
                series_id=f"coinmetrics.btc.{metric}",
                event_at=event_at,
                published_at=published_at,
                ingested_at=ingested_at,
                source=COMMUNITY_ROOT,
                source_version=source_version,
                confidence=spec.confidence,
                value=value,
                unit=spec.unit,
                category=spec.category,
                use_class=USE_DESCRIPTIVE,
                publication_time_exact=exact,
                strict_backtest=False,
                notes=notes,
                entity=spec.entity,
            ))
            continue
        found.append(Observation(
            series_id=f"coinmetrics.btc.{metric}",
            event_at=event_at,
            published_at=None,
            ingested_at=ingested_at,
            source=COMMUNITY_ROOT,
            source_version=source_version,
            confidence=spec.confidence,
            value=value,
            unit=spec.unit,
            category=spec.category,
            use_class=USE_DESCRIPTIVE,
            publication_time_exact=False,
            strict_backtest=False,
            notes=(
                "Суточное начало интервала Coin Metrics. Точное историческое время первой "
                "публикации этого значения в ответе нет: AssetEODCompletionTime у старых дат "
                "совпадает с пакетной отметкой, а не с утром того дня. В строгий backtest не входит."
            ),
            entity=spec.entity,
        ))
    return found


def _miner_revenue(observations: list[Observation]) -> list[Observation]:
    """Сумма комиссий и эмиссии. Это не платная метрика RevNtv и не строгий ряд."""
    fees = {
        item.event_at: item
        for item in observations
        if item.series_id == "coinmetrics.btc.FeeTotNtv" and item.value is not None
    }
    issued = {
        item.event_at: item
        for item in observations
        if item.series_id == "coinmetrics.btc.IssTotNtv" and item.value is not None
    }
    derived: list[Observation] = []
    for event_at, fee in fees.items():
        issuance = issued.get(event_at)
        if issuance is None or fee.value is None or issuance.value is None:
            continue
        derived.append(Observation(
            series_id="coinmetrics.btc.miner_revenue_ntv",
            event_at=event_at,
            published_at=None,
            ingested_at=fee.ingested_at,
            source="derived:coinmetrics-community FeeTotNtv+IssTotNtv",
            source_version=fee.source_version,
            confidence=min(fee.confidence, issuance.confidence),
            value=fee.value + issuance.value,
            unit="BTC",
            category="miner",
            use_class=USE_DESCRIPTIVE,
            publication_time_exact=False,
            strict_backtest=False,
            notes=(
                "Сумма двух community-метрик. Не является идентификацией кошельков майнеров "
                "и не заменяет платную RevNtv. Время первой публикации неизвестно."
            ),
        ))
    return derived


def flow_anomalies(
    observations: list[Observation],
    as_of: datetime,
    *,
    series_id: str,
    window: int = 30,
    z_min: float = 3.0,
) -> list[dict[str, object]]:
    """Z-оценка последнего строгого значения по предыдущему окну.

    Ряды без strict_backtest, включая пересчитанные биржевые агрегаты, не участвуют.
    Если строгого окна нет, список пуст, а не заполнен оценкой.
    """
    if window < 5:
        raise ValueError("окно аномалии короче 5")
    points = [
        item for item in observations
        if item.series_id == series_id
        and item.strict_backtest
        and item.publication_time_exact
        and item.published_at is not None
        and item.published_at <= as_of
        and item.event_at <= as_of
        and item.value is not None
    ]
    points.sort(key=lambda item: (item.event_at, item.published_at or item.event_at))
    if len(points) <= window:
        return []
    history = [item.value for item in points[-window - 1:-1] if item.value is not None]
    current = points[-1].value
    if current is None or len(history) < window:
        return []
    mean = sum(history) / len(history)
    variance = sum((value - mean) ** 2 for value in history) / (len(history) - 1)
    deviation = variance ** 0.5
    if deviation < 1e-12:
        return []
    score = (current - mean) / deviation
    if abs(score) < z_min:
        return []
    return [{
        "series_id": series_id,
        "event_at": points[-1].event_at.isoformat(),
        "z": score,
        "note": "отклонение строгого ряда; не продажа и не покупка",
    }]


def assert_research_window(end: datetime) -> None:
    if end >= datetime(2025, 1, 1, tzinfo=timezone.utc):
        raise ValueError("окно Coin Metrics захватывает holdout")
    if end - datetime(2009, 1, 1, tzinfo=timezone.utc) < timedelta(days=1):
        raise ValueError("окно Coin Metrics пусто")
