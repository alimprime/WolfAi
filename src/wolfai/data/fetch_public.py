"""Загрузка публичных архивов Binance Vision.

Этот модуль единственный, кто обращается к сети за свечами. Исследовательский
цикл его не импортирует: прогноз читает только уже проверенные локальные файлы.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wolfai.data.binance import (
    load_binance_spot_daily_archives,
    read_sha256_checksum,
    save_normalized_candles,
    verify_sha256,
)
from wolfai.data.provenance import DataProvenance

VISION_ROOT = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d"
DEFAULT_START = (2019, 1)
DEFAULT_END = (2026, 9)


def iter_months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    if start > end:
        raise ValueError("начало диапазона позже конца")
    year, month = start
    months: list[tuple[int, int]] = []
    while (year, month) <= end:
        if not 1 <= month <= 12:
            raise ValueError(f"некорректный месяц {year:04d}-{month:02d}")
        months.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def archive_name(year: int, month: int) -> str:
    return f"BTCUSDT-1d-{year:04d}-{month:02d}.zip"


def contiguous_prefix(
    expected: list[tuple[int, int]],
    present: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Месяцы от начала диапазона до первой дыры. Дыра не заполняется."""
    prefix: list[tuple[int, int]] = []
    for month in expected:
        if month not in present:
            break
        prefix.append(month)
    return prefix


def fetch_history(
    directory: str | Path,
    *,
    start: tuple[int, int] = DEFAULT_START,
    end: tuple[int, int] = DEFAULT_END,
    normalized_csv: str | Path | None = None,
) -> dict[str, object]:
    """Скачивает ZIP и .CHECKSUM, сверяет SHA-256 и собирает непрерывный префикс."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    expected = iter_months(start, end)
    statuses = _download_all(target, expected)
    present = {
        item["month"]
        for item in statuses
        if item["status"] == "verified" and isinstance(item["month"], tuple)
    }
    prefix = contiguous_prefix(expected, present)
    missing = [item for item in statuses if item["status"] != "verified"]
    merged_rows = 0
    first_open = None
    last_open = None
    csv_path = None
    merge_error = None
    if prefix:
        try:
            candles = load_binance_spot_daily_archives(
                [target / archive_name(*month) for month in prefix]
            )
        except ValueError as error:
            merge_error = str(error)
        else:
            merged_rows = len(candles)
            first_open = candles[0].timestamp.isoformat()
            last_open = candles[-1].timestamp.isoformat()
            if normalized_csv is not None:
                csv_path = str(normalized_csv)
                save_normalized_candles(candles, normalized_csv)
    ingested_at = datetime.now(timezone.utc)
    provenances = [
        _archive_provenance(item, ingested_at)
        for item in statuses
        if item["status"] == "verified"
    ]
    manifest = {
        "source": VISION_ROOT,
        "asset": "BTCUSDT",
        "market": "spot",
        "interval": "1d",
        "requested_start": f"{start[0]:04d}-{start[1]:02d}",
        "requested_end": f"{end[0]:04d}-{end[1]:02d}",
        "ingested_at": ingested_at.isoformat(),
        "bar_availability_rule": "open timestamp + 1 day UTC",
        "verified_months": [
            _public_month(item) for item in statuses if item["status"] == "verified"
        ],
        "missing_or_rejected": [_public_month(item) for item in missing],
        "merged_prefix": [f"{year:04d}-{month:02d}" for year, month in prefix],
        "merged_rows": merged_rows,
        "merge_error": merge_error,
        "first_open": first_open,
        "last_open": last_open,
        "normalized_csv": csv_path,
        "archive_provenances": [
            {
                "event_time": item.event_time.isoformat(),
                "published_at": item.published_at.isoformat(),
                "ingested_at": item.ingested_at.isoformat(),
                "source": item.source,
                "source_version": item.source_version,
                "confidence": item.confidence,
            }
            for item in provenances
        ],
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _download_all(directory: Path, months: list[tuple[int, int]]) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_fetch_month, directory, year, month): (year, month)
            for year, month in months
        }
        for future in as_completed(futures):
            record = future.result()
            print(f"{record['name']} {record['status']}", flush=True)
            statuses.append(record)
    statuses.sort(key=lambda item: item["name"])
    return statuses


def _fetch_month(directory: Path, year: int, month: int) -> dict[str, object]:
    name = archive_name(year, month)
    zip_path = directory / name
    checksum_path = directory / f"{name}.CHECKSUM"
    record: dict[str, object] = {
        "month": (year, month),
        "name": name,
        "status": "missing",
        "sha256": None,
    }
    if not _download(f"{VISION_ROOT}/{name}", zip_path):
        record["status"] = "missing_zip"
        return record
    if not _download(f"{VISION_ROOT}/{name}.CHECKSUM", checksum_path):
        record["status"] = "missing_checksum"
        return record
    try:
        expected = read_sha256_checksum(checksum_path)
        verify_sha256(zip_path, expected)
    except ValueError as error:
        record["status"] = "checksum_mismatch"
        record["error"] = str(error)
        return record
    record["status"] = "verified"
    record["sha256"] = expected
    return record


def _download(url: str, destination: Path) -> bool:
    request = Request(url, headers={"User-Agent": "WolfAi-data/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            destination.write_bytes(response.read())
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    except URLError:
        raise
    return True


def _archive_provenance(item: dict[str, object], ingested_at: datetime) -> DataProvenance:
    month = item["month"]
    if not isinstance(month, tuple):
        raise ValueError("в статусе архива нет месяца")
    year, month_number = month
    if month_number == 12:
        available = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        event = datetime(year, 12, 1, tzinfo=timezone.utc)
    else:
        available = datetime(year, month_number + 1, 1, tzinfo=timezone.utc)
        event = datetime(year, month_number, 1, tzinfo=timezone.utc)
    digest = item["sha256"]
    if not isinstance(digest, str):
        raise ValueError("у проверенного архива нет SHA-256")
    return DataProvenance(
        event_time=event,
        published_at=available,
        ingested_at=ingested_at,
        source=f"{VISION_ROOT}/{item['name']}",
        source_version=digest,
        confidence=1.0,
    )


def _public_month(item: dict[str, object]) -> dict[str, object]:
    month = item["month"]
    label = None
    if isinstance(month, tuple):
        label = f"{month[0]:04d}-{month[1]:02d}"
    return {
        "month": label,
        "name": item["name"],
        "status": item["status"],
        "sha256": item.get("sha256"),
        "error": item.get("error"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Скачать BTCUSDT Spot 1d с Binance Vision")
    parser.add_argument("--directory", default="data/raw/binance")
    parser.add_argument("--csv", default="data/btcusdt_1d_2019-01_2026-09.csv")
    args = parser.parse_args()
    manifest = fetch_history(args.directory, normalized_csv=args.csv)
    print(
        f"verified={len(manifest['verified_months'])} "
        f"missing={len(manifest['missing_or_rejected'])} "
        f"rows={manifest['merged_rows']} "
        f"span={manifest['first_open']}..{manifest['last_open']}"
    )
    if manifest["missing_or_rejected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
