"""Загрузка только подтверждённых публичных рядов.

Holdout с 2025-01-01 не запрашивается. Платные API и страница Farside не вызываются.
Пропуск архива остаётся пропуском.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wolfai.data.binance import read_sha256_checksum, verify_sha256
from wolfai.intelligence.institutions.farside import fetch_farside
from wolfai.intelligence.market.derivatives import (
    FUNDING_ROOT,
    OI_ROOT,
    parse_funding_csv,
    parse_open_interest_csv,
    read_zip_text,
)
from wolfai.intelligence.market.macro import (
    FRED_GRAPH,
    FRED_UNITS,
    NYFED_EFFR,
    parse_fred_csv,
    parse_nyfed_effr,
)
from wolfai.intelligence.market.refused import fetch_coinglass
from wolfai.intelligence.schema import Observation, SourceUnavailable
from wolfai.intelligence.store import ObservationStore
from wolfai.intelligence.whales.coinmetrics import (
    CATALOG_URL,
    parse_catalog_coverage,
    parse_community_page,
    payload_version,
    pit_probe_url,
    timeseries_url,
)
from wolfai.intelligence.whales.paid import PROFILES, fetch_paid

HOLDOUT_START = date(2025, 1, 1)
OI_START = date(2020, 9, 1)
FUNDING_START = (2020, 1)
USER_AGENT = "WolfAi-research/0.1"


def acquire(directory: str | Path, *, end: date = date(2024, 12, 31)) -> dict[str, object]:
    """Скачивает согласованные ряды и складывает их в SQLite. Holdout не запрашивает."""
    if end >= HOLDOUT_START:
        raise ValueError("выгрузка исследования не включает holdout")
    root = Path(directory)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ingested_at = datetime.now(timezone.utc)
    observations: list[Observation] = []
    manifest: dict[str, object] = {
        "ingested_at": ingested_at.isoformat(),
        "requested_end": end.isoformat(),
        "holdout_requested": False,
    }
    manifest["coinmetrics"] = _capture(
        "coinmetrics",
        lambda: _coinmetrics(raw / "coinmetrics", observations, ingested_at),
    )
    manifest["funding"] = _capture(
        "funding",
        lambda: _funding(raw / "funding", observations, ingested_at, end),
    )
    manifest["open_interest"] = _capture(
        "open_interest",
        lambda: _open_interest(raw / "open_interest", observations, ingested_at, end),
    )
    manifest["fred"] = _capture("fred", lambda: _fred(raw / "fred", observations, ingested_at))
    manifest["nyfed"] = _capture("nyfed", lambda: _nyfed(raw / "nyfed", observations, ingested_at, end))
    manifest["refused"] = _refusals()
    kept = _before_holdout(observations)
    manifest["dropped_holdout_events"] = len(observations) - len(kept)
    store = ObservationStore(root / "observations.sqlite")
    try:
        inserted = store.append(kept)
        manifest["stored_rows"] = inserted
        manifest["store_counts"] = store.counts_by_series()
    finally:
        store.close()
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _capture(name: str, function: object) -> dict[str, object]:
    try:
        result = function()
    except (SourceUnavailable, OSError, json.JSONDecodeError, UnicodeError, ValueError) as error:
        print(f"{name} unavailable: {error}", flush=True)
        return {"status": "unavailable", "reason": str(error)}
    if isinstance(result, dict):
        result.setdefault("status", "ok")
        return result
    raise SourceUnavailable(f"{name} вернул не объект")


def _coinmetrics(directory: Path, observations: list[Observation], ingested_at: datetime) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    catalog_body = _get_bytes(CATALOG_URL)
    (directory / "catalog.json").write_bytes(catalog_body)
    catalog = json.loads(catalog_body)
    pages: list[bytes] = []
    page_number = 0
    for year in range(2009, 2025):
        start = "2009-01-09" if year == 2009 else f"{year:04d}-01-01"
        url: str | None = timeseries_url(start=start, end=f"{year:04d}-12-31")
        while url:
            if not url.startswith("https://community-api.coinmetrics.io/"):
                raise SourceUnavailable("next_page_url ведёт вне community API")
            body = _get_bytes(url)
            page_number += 1
            (directory / f"page-{page_number}.json").write_bytes(body)
            pages.append(body)
            payload = json.loads(body)
            nxt = payload.get("next_page_url")
            url = str(nxt) if isinstance(nxt, str) and nxt else None
        time.sleep(0.7)
    if not pages:
        raise SourceUnavailable("Coin Metrics не вернула ни одной страницы")
    version = payload_version(pages)
    for body in pages:
        observations.extend(parse_community_page(
            json.loads(body),
            source_version=version,
            ingested_at=ingested_at,
        ))
    pit_status, pit_body = _get_status(pit_probe_url())
    return {
        "source_version": version,
        "pages": len(pages),
        "catalog_metrics": len(parse_catalog_coverage(catalog)),
        "pit_probe_metric": "FlowInExPitNtv",
        "pit_http_status": pit_status,
        "pit_available": pit_status == 200,
        "pit_excerpt": pit_body[:240].decode("utf-8", "replace"),
    }


def _funding(
    directory: Path,
    observations: list[Observation],
    ingested_at: datetime,
    end: date,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    months = _months(FUNDING_START, (end.year, end.month))
    verified = 0
    missing: list[str] = []
    for year, month in months:
        name = f"BTCUSDT-fundingRate-{year:04d}-{month:02d}.zip"
        try:
            record = _verified_zip(f"{FUNDING_ROOT}/{name}", directory / name)
        except SourceUnavailable as error:
            missing.append(f"{name}: {error}")
            continue
        if record["status"] != "verified" or not isinstance(record["sha256"], str):
            missing.append(name)
            continue
        source = f"{FUNDING_ROOT}/{name}"
        observations.extend(_before_holdout(parse_funding_csv(
            read_zip_text(directory / name),
            source=source,
            source_version=record["sha256"],
            ingested_at=ingested_at,
        )))
        verified += 1
    return {"verified_months": verified, "missing": missing}


def _open_interest(
    directory: Path,
    observations: list[Observation],
    ingested_at: datetime,
    end: date,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    days = _days(OI_START, end)
    statuses: dict[date, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_oi_day, directory, day): day
            for day in days
        }
        done = 0
        for future in as_completed(futures):
            day = futures[future]
            statuses[day] = future.result()
            done += 1
            if done % 200 == 0:
                print(f"open interest {done}/{len(days)}", flush=True)
    missing: list[str] = []
    parsed = 0
    for day in days:
        record = statuses[day]
        if record["status"] != "verified" or not isinstance(record["sha256"], str):
            missing.append(day.isoformat())
            continue
        path = directory / f"BTCUSDT-metrics-{day.isoformat()}.zip"
        snapshot = parse_open_interest_csv(
            read_zip_text(path),
            day=datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
            source=f"{OI_ROOT}/{path.name}",
            source_version=record["sha256"],
            ingested_at=ingested_at,
        )
        if snapshot is None:
            missing.append(day.isoformat())
            continue
        observations.append(snapshot)
        parsed += 1
    return {"verified_days": parsed, "missing_days": len(missing), "missing_sample": missing[:12]}


def _fred(directory: Path, observations: list[Observation], ingested_at: datetime) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for series_id in FRED_UNITS:
        body = _get_bytes(FRED_GRAPH.format(series_id=series_id))
        (directory / f"{series_id}.csv").write_bytes(body)
        parsed = parse_fred_csv(body.decode("utf-8-sig"), series_id=series_id, ingested_at=ingested_at)
        observations.extend(parsed)
        counts[series_id] = len(parsed)
    return {"rows": counts, "strict_backtest": False}


def _nyfed(
    directory: Path,
    observations: list[Observation],
    ingested_at: datetime,
    end: date,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    url = NYFED_EFFR.format(start="2019-01-01", end=end.isoformat())
    body = _get_bytes(url)
    (directory / "effr.json").write_bytes(body)
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise SourceUnavailable("NY Fed вернул не объект")
    parsed = _before_holdout(parse_nyfed_effr(payload, ingested_at=ingested_at, source=url))
    observations.extend(parsed)
    events = [item.event_at for item in parsed]
    return {
        "rows": len(parsed),
        "strict_backtest": False,
        "event_start": min(events).date().isoformat() if events else None,
        "event_end": max(events).date().isoformat() if events else None,
    }


def _refusals() -> list[dict[str, str]]:
    refused: list[dict[str, str]] = []
    for name in PROFILES:
        try:
            fetch_paid(name)
        except SourceUnavailable as error:
            refused.append({"source": name, "reason": str(error)})
    for source, call in (("farside", fetch_farside), ("coinglass", fetch_coinglass)):
        try:
            call()
        except SourceUnavailable as error:
            refused.append({"source": source, "reason": str(error)})
    return refused


def _fetch_oi_day(directory: Path, day: date) -> dict[str, object]:
    name = f"BTCUSDT-metrics-{day.isoformat()}.zip"
    try:
        return _verified_zip(f"{OI_ROOT}/{name}", directory / name)
    except SourceUnavailable as error:
        return {"status": "error", "sha256": None, "name": name, "error": str(error)}


def _verified_zip(url: str, destination: Path) -> dict[str, object]:
    checksum_path = destination.with_name(destination.name + ".CHECKSUM")
    record: dict[str, object] = {"status": "missing", "sha256": None, "name": destination.name}
    if not _download(url, destination):
        record["status"] = "missing_zip"
        return record
    if not _download(url + ".CHECKSUM", checksum_path):
        record["status"] = "missing_checksum"
        return record
    try:
        expected = read_sha256_checksum(checksum_path)
        verify_sha256(destination, expected)
    except ValueError as error:
        record["status"] = "checksum_mismatch"
        record["error"] = str(error)
        return record
    record["status"] = "verified"
    record["sha256"] = expected
    return record


def _download(url: str, destination: Path) -> bool:
    if destination.exists() and destination.stat().st_size > 0 and destination.suffix != ".CHECKSUM":
        checksum = destination.with_name(destination.name + ".CHECKSUM")
        if checksum.exists():
            try:
                verify_sha256(destination, read_sha256_checksum(checksum))
            except ValueError:
                pass
            else:
                return True
    status, body = _get_status(url)
    if status == 404:
        return False
    if status != 200:
        raise SourceUnavailable(f"{url} ответил {status}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return True


def _get_bytes(url: str) -> bytes:
    status, body = _get_status(url)
    if status != 200:
        raise SourceUnavailable(f"{url} ответил {status}")
    return body


def _get_status(url: str) -> tuple[int, bytes]:
    delay = 1.0
    last_error: Exception | None = None
    for _attempt in range(4):
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/csv,*/*"})
        try:
            with urlopen(request, timeout=90) as response:
                return response.status, response.read()
        except HTTPError as error:
            body = error.read()
            if error.code in {429, 500, 502, 503, 504}:
                last_error = error
                time.sleep(delay)
                delay *= 2
                continue
            return error.code, body
        except URLError as error:
            last_error = error
            time.sleep(delay)
            delay *= 2
    raise SourceUnavailable(f"не удалось скачать {url}") from last_error


def _before_holdout(observations: list[Observation]) -> list[Observation]:
    return [item for item in observations if item.event_at.date() < HOLDOUT_START]


def _months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    year, month = start
    months: list[tuple[int, int]] = []
    while (year, month) <= end:
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def _days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days
