"""Проверенные ограничения источников. Сеть на этом этапе не вызывается.

Доступность считается доказанной только локальной проверкой архива или
явным отказом адаптера. Платный ключ здесь не подставляется.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

CUTOFF_TEXT = "2022-01-01"


@dataclass(frozen=True)
class SourceFinding:
    name: str
    checked: bool
    api: str
    access: str
    history_2019_2021: str
    timestamp_note: str
    availability_note: str
    point_in_time: str
    gaps: str
    leakage_risk: str
    forecast_ready: bool
    reason: str


def documented_findings() -> tuple[SourceFinding, ...]:
    """Каталог до чтения локальной базы. Ни один платный источник не включён."""
    return (
        SourceFinding(
            name="binance_spot_ohlcv",
            checked=True,
            api="локальный CSV Binance Vision, без сети в этом этапе",
            access="бесплатный публичный архив",
            history_2019_2021="дневные свечи с 2019-01-01 по 2021-12-31 есть в data/btcusdt_1d_2019-01_2026-09.csv; строки с 2022-01-01 отбрасываются до разбора цен",
            timestamp_note="timestamp — 00:00 UTC открытия бара",
            availability_note="close известен на следующей полуночи UTC",
            point_in_time="да, для уже закрытого бара",
            gaps="пропуск суток отвергает ряд, а не заполняется",
            leakage_risk="если читать close бара, который ещё не закрыт, или свечи с 2022-01-01",
            forecast_ready=True,
            reason="время доступности дневного close зафиксировано в daily_bar_available_at",
        ),
        SourceFinding(
            name="binance_um_funding_open_interest",
            checked=False,
            api="https://data.binance.vision, месячные ZIP funding и дневные metrics",
            access="бесплатный публичный архив",
            history_2019_2021="funding USD-M BTCUSDT с 2020-01, open interest с 2020-09-01; 2019 в этих рядах нет",
            timestamp_note="funding: calc_time; open interest: create_time снимка",
            availability_note="для строгого ряда available_at равен published_at только потому, что биржа ставит значение в этот момент",
            point_in_time="да, если strict_backtest и published_at не позже решения",
            gaps="до старта ряда признак отсутствует и не заполняется",
            leakage_risk="повтор сравнения 2022 года ради другого Brier; использование снимка с create_time после решения",
            forecast_ready=False,
            reason="локальный архив ещё не проверен этим процессом",
        ),
        SourceFinding(
            name="coinmetrics_community",
            checked=True,
            api="https://community-api.coinmetrics.io/v4",
            access="бесплатный community, без ключа",
            history_2019_2021="сетевые ряды и биржевые агрегаты в локальной базе есть",
            timestamp_note="у сетевых метрик published_at пуст; у Flow* published_at — апрель 2026",
            availability_note="пустое время публикации не заменяется датой события",
            point_in_time="нет: Flow*Pit в community отвечал 403, стандартный ряд переписывается новыми адресами бирж",
            gaps="пропуск не заполняется нулём",
            leakage_risk="сегодняшняя карта адресов биржи, применённая к 2019 году",
            forecast_ready=False,
            reason="нет точки во времени, известной рынку в 2019–2021",
        ),
        SourceFinding(
            name="cryptoquant",
            checked=True,
            api="https://api.cryptoquant.com/v1/btc/exchange-flows/reserve",
            access="on-chain не входит в бесплатный Basic",
            history_2019_2021="не загружалась",
            timestamp_note="не проверялась на живом ответе",
            availability_note="неизвестно",
            point_in_time="не доказана",
            gaps="нет ряда",
            leakage_risk="обход платного тарифа",
            forecast_ready=False,
            reason="адаптер отказывается без ключа и не создаёт значения",
        ),
        SourceFinding(
            name="glassnode",
            checked=True,
            api="исторический API платный",
            access="платный",
            history_2019_2021="не загружалась; Light API Advanced — около 14 дней",
            timestamp_note="не проверялась",
            availability_note="неизвестно",
            point_in_time="не доказана",
            gaps="нет ряда",
            leakage_risk="обход подписки",
            forecast_ready=False,
            reason="Glassnode не вызывался",
        ),
        SourceFinding(
            name="arkham",
            checked=True,
            api="https://api.arkm.com/transfers",
            access="платные кредиты",
            history_2019_2021="не загружалась",
            timestamp_note="не проверялась",
            availability_note="неизвестно",
            point_in_time="не доказана",
            gaps="переводов крупных держателей в локальной базе 0",
            leakage_risk="поздняя метка кошелька, применённая к старому переводу",
            forecast_ready=False,
            reason="нет ключа; перевод сам по себе не считается сделкой",
        ),
        SourceFinding(
            name="fred_graph_and_nyfed",
            checked=True,
            api="график FRED без ключа и JSON EFFR Нью-Йоркского ФРБ",
            access="бесплатные последние винтажности",
            history_2019_2021="числа в локальной базе есть",
            timestamp_note="в строке нет vintage date и часа релиза",
            availability_note="published_at пуст, поэтому ряд не forecast_ready",
            point_in_time="нет: ALFRED с ключом не вызывался, пересмотры CPI могли выйти позже",
            gaps="выходные FRED не являются пропуском биткоин-суток и не заполняются вперёд",
            leakage_risk="последняя винтажность вместо первой публикации",
            forecast_ready=False,
            reason="нет часа первой публикации",
        ),
        SourceFinding(
            name="official_bls_fed_releases",
            checked=True,
            api="первоисточники BLS и ФРС отдельно от графика FRED не загружались",
            access="публичные релизы, календарь не снимался",
            history_2019_2021="календарь CPI и FOMC в этот этап не вносился",
            timestamp_note="час релиза не записан",
            availability_note="без часа релиза публикацию нельзя ставить на ось решения",
            point_in_time="не доказана",
            gaps="нет календаря",
            leakage_risk="датировать релиз полуночью дня наблюдения",
            forecast_ready=False,
            reason="официальный календарь релизов не подключён",
        ),
    )


def audit_local(database: str | Path | None) -> tuple[SourceFinding, ...]:
    """Уточняет каталог по локальной SQLite, не читая value после 2022-01-01."""
    findings = list(documented_findings())
    if database is None or not Path(database).exists():
        return tuple(findings)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        ready = connection.execute(
            """
            SELECT COUNT(*) FROM observations
            WHERE series_id = ?
              AND strict_backtest = 1
              AND published_at IS NOT NULL
              AND published_at < ?
              AND event_at < ?
            """,
            ("binance_vision.um.BTCUSDT.funding_rate", CUTOFF_TEXT, CUTOFF_TEXT),
        ).fetchone()
        oi = connection.execute(
            """
            SELECT COUNT(*) FROM observations
            WHERE series_id = ?
              AND strict_backtest = 1
              AND published_at IS NOT NULL
              AND published_at < ?
              AND event_at < ?
            """,
            ("binance_vision.um.BTCUSDT.open_interest", CUTOFF_TEXT, CUTOFF_TEXT),
        ).fetchone()
        flow_pub = connection.execute(
            """
            SELECT MIN(published_at) FROM observations
            WHERE series_id = 'coinmetrics.btc.FlowInExNtv' AND event_at < ?
            """,
            (CUTOFF_TEXT,),
        ).fetchone()
    finally:
        connection.close()
    funding_n = int(ready[0]) if ready else 0
    oi_n = int(oi[0]) if oi else 0
    flow_min = None if flow_pub is None else flow_pub[0]
    checked = []
    for item in findings:
        if item.name == "binance_um_funding_open_interest":
            ok = funding_n > 0 and oi_n > 0
            checked.append(SourceFinding(
                name=item.name,
                checked=True,
                api=item.api,
                access=item.access,
                history_2019_2021=f"до 2022-01-01 в базе {funding_n} ставок funding и {oi_n} снимков open interest",
                timestamp_note=item.timestamp_note,
                availability_note=item.availability_note,
                point_in_time=item.point_in_time,
                gaps=item.gaps,
                leakage_risk=item.leakage_risk,
                forecast_ready=ok,
                reason="локальные strict-строки с published_at раньше 2022-01-01" if ok else "в базе нет строгого ряда до 2022-01-01",
            ))
        elif item.name == "coinmetrics_community" and isinstance(flow_min, str):
            checked.append(SourceFinding(
                name=item.name,
                checked=True,
                api=item.api,
                access=item.access,
                history_2019_2021=item.history_2019_2021,
                timestamp_note=item.timestamp_note,
                availability_note=f"ранний published_at притока для событий до 2022-01-01: {flow_min}",
                point_in_time=item.point_in_time,
                gaps=item.gaps,
                leakage_risk=item.leakage_risk,
                forecast_ready=False,
                reason="published_at биржевого потока позже окна прогноза",
            ))
        else:
            checked.append(item)
    return tuple(checked)
