"""Текущие дневные свечи BTCUSDT Spot с официального публичного API Binance.

Документация Binance Spot REST (market data, security NONE):
`GET /api/v3/time` — вес 1, ответ `serverTime` в миллисекундах.
`GET /api/v3/klines` — вес 2, параметры symbol, interval, limit.
Для публичных рыночных данных указан хост `https://data-api.binance.vision`.
Общий хост `https://api.binance.com` — тот же документированный API.
`uiKlines` не используется: документация называет его изменённым рядом для графиков.

Ключ, подпись и торговый счёт не отправляются. Региональный отказ и лимит
запросов не обходятся. Недокументированных путей нет.

Четыре времени хранятся раздельно и не подменяют друг друга:
открытие свечи, закрытие свечи по полю биржи, серверное время биржи,
время получения ответа WolfAi. Момент первого появления свечи в API
не измеряется и не считается доказанным.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wolfai.data.binance import _open_time
from wolfai.data.candles import Candle
from wolfai.research.provenance_grade import DAILY_BAR_BASIS, grade_timing

SYMBOL = "BTCUSDT"
INTERVAL = "1d"
KLINE_LIMIT = 40
TIME_WEIGHT = 1
KLINES_WEIGHT = 2
LOCAL_WEIGHT_BUDGET = 10
MAX_HTTP_CALLS = 4
TIMEOUT_SECONDS = 10.0
MAX_BODY_BYTES = 2_000_000
WEIGHT_BACKOFF = 5_500
DAY_MS = 86_400_000
SOURCE_UNAVAILABLE = "source_unavailable"
MARKET_DATA_BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
ALLOWED_HOSTS = frozenset({"data-api.binance.vision", "api.binance.com"})
TIME_PATH = "/api/v3/time"
KLINES_PATH = "/api/v3/klines"
USER_AGENT = "WolfAi-spot/2.6"


class MarketTransport(Protocol):
    def request(self, url: str, timeout: float) -> "HttpResponse":
        """Один HTTP GET. Исключение транспорта означает, что ответа нет."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str


@dataclass(frozen=True)
class SpotCandle:
    """Одна дневная свеча. Четыре часа не равны друг другу по смыслу."""

    symbol: str
    interval: str
    open_time: datetime
    close_time: datetime
    exchange_server_time: datetime
    received_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    available_at: datetime
    raw_ohlcv_sha256: str
    response_sha256: str
    availability_grade: str
    api_observation_grade: str = "unknown"

    @property
    def api_first_seen_proven(self) -> bool:
        return False

    def as_candle(self) -> Candle:
        return Candle(self.open_time, self.open, self.high, self.low, self.close, self.volume)


@dataclass(frozen=True)
class RejectedBar:
    reason: str
    open_time: datetime | None


@dataclass(frozen=True)
class SpotSnapshot:
    status: str
    reason: str
    base_url: str | None
    source_mode: str
    exchange_server_time: datetime | None
    received_at: datetime | None
    klines_sha256: str | None
    time_sha256: str | None
    closed: tuple[SpotCandle, ...]
    rejected: tuple[RejectedBar, ...]
    requests_made: int = 0

    @property
    def api_first_seen_proven(self) -> bool:
        return False


class UrllibTransport:
    """Синхронный GET. Проверка TLS не отключается, ключ не добавляется."""

    def request(self, url: str, timeout: float) -> HttpResponse:
        host = url.split("/")[2]
        if host not in ALLOWED_HOSTS:
            raise ValueError("хост вне документированных баз Binance")
        request = Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                final_host = final_url.split("/")[2]
                if final_host not in ALLOWED_HOSTS:
                    raise ValueError("перенаправление ушло с документированного хоста")
                body = response.read(MAX_BODY_BYTES + 1)
                if len(body) > MAX_BODY_BYTES:
                    raise ValueError("ответ больше допустимого размера")
                headers = {key: value for key, value in response.headers.items()}
                return HttpResponse(int(response.status), headers, body, final_url)
        except HTTPError as error:
            raw = error.read(MAX_BODY_BYTES + 1) if error.fp is not None else b""
            headers = {key: value for key, value in error.headers.items()} if error.headers else {}
            return HttpResponse(int(error.code), headers, raw[:MAX_BODY_BYTES], url)


class _Stop(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason


class SpotClient:
    """Два документированных вызова: время сервера и дневные klines BTCUSDT."""

    def __init__(
        self,
        transport: MarketTransport | None = None,
        *,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.transport = transport or UrllibTransport()
        self.timeout = timeout
        self.calls = 0
        self.weight_spent = 0
        self.last_time_body: bytes | None = None
        self.last_klines_body: bytes | None = None

    def fetch(self, received_at: datetime | None = None) -> SpotSnapshot:
        last_reason = "connection_error"
        for base in MARKET_DATA_BASES:
            try:
                time_response = self._get(f"{base}{TIME_PATH}", TIME_WEIGHT)
            except _Stop as stopped:
                return _unavailable(stopped.reason, _stamp(received_at), self.calls)
            except (TimeoutError, URLError, OSError, ValueError):
                last_reason = "connection_error"
                continue
            if time_response.status in (429, 418):
                return _unavailable("rate_limited", _stamp(received_at), self.calls)
            if time_response.status in (401, 403, 451):
                return _unavailable("access_restricted", _stamp(received_at), self.calls)
            if time_response.status != 200:
                last_reason = f"http_{time_response.status}"
                continue
            if _header_weight(time_response.headers) >= WEIGHT_BACKOFF:
                return _unavailable("rate_limited", _stamp(received_at), self.calls)
            self.last_time_body = time_response.body
            try:
                server_time = _server_time(time_response.body)
            except ValueError:
                return _unavailable("unexpected_payload", _stamp(received_at), self.calls)
            try:
                klines_response = self._get(
                    f"{base}{KLINES_PATH}?symbol={SYMBOL}&interval={INTERVAL}&limit={KLINE_LIMIT}",
                    KLINES_WEIGHT,
                )
            except _Stop as stopped:
                return _unavailable(stopped.reason, _stamp(received_at), self.calls)
            except (TimeoutError, URLError, OSError, ValueError):
                last_reason = "connection_error"
                continue
            if klines_response.status in (429, 418):
                return _unavailable("rate_limited", _stamp(received_at), self.calls)
            if klines_response.status in (401, 403, 451):
                return _unavailable("access_restricted", _stamp(received_at), self.calls)
            if klines_response.status != 200:
                last_reason = f"http_{klines_response.status}"
                continue
            self.last_klines_body = klines_response.body
            moment = _stamp(received_at)
            snapshot = interpret_market(
                time_response.body,
                klines_response.body,
                moment,
                base_url=base,
                source_mode="network",
            )
            return SpotSnapshot(
                status=snapshot.status,
                reason=snapshot.reason,
                base_url=base,
                source_mode="network",
                exchange_server_time=server_time,
                received_at=moment,
                klines_sha256=snapshot.klines_sha256,
                time_sha256=snapshot.time_sha256,
                closed=snapshot.closed,
                rejected=snapshot.rejected,
                requests_made=self.calls,
            )
        return _unavailable(last_reason, _stamp(received_at), self.calls)

    def _get(self, url: str, weight: int) -> HttpResponse:
        if self.calls >= MAX_HTTP_CALLS or self.weight_spent + weight > LOCAL_WEIGHT_BUDGET:
            raise _Stop("rate_limited")
        self.calls += 1
        self.weight_spent += weight
        try:
            response = self.transport.request(url, self.timeout)
        except HTTPError as error:
            if error.code in (429, 418, 401, 403, 451):
                raise _Stop("rate_limited" if error.code in (429, 418) else "access_restricted") from error
            raise
        if response.status in (429, 418):
            raise _Stop("rate_limited")
        if response.status in (401, 403, 451):
            raise _Stop("access_restricted")
        return response


def interpret_market(
    time_body: bytes,
    klines_body: bytes,
    received_at: datetime,
    *,
    base_url: str | None,
    source_mode: str,
) -> SpotSnapshot:
    """Разбирает уже полученные тела. Сеть не вызывает.

    Незакрытая свеча в закрытые не попадает. Плохая метка или OHLCV
    бракует весь ответ: дырявый ряд не молча укорачивается.
    """
    _require_utc(received_at, "received_at")
    time_hash = hashlib.sha256(time_body).hexdigest()
    klines_hash = hashlib.sha256(klines_body).hexdigest()
    try:
        server_time = _server_time(time_body)
        rows = _kline_rows(klines_body)
    except ValueError as error:
        reason = str(error) or "unexpected_payload"
        return SpotSnapshot(
            "invalid_payload" if reason != SOURCE_UNAVAILABLE else SOURCE_UNAVAILABLE,
            reason,
            base_url,
            source_mode,
            None,
            received_at,
            klines_hash,
            time_hash,
            (),
            (),
        )
    closed: list[SpotCandle] = []
    rejected: list[RejectedBar] = []
    seen_unclosed = False
    for row in rows:
        try:
            parsed = _parse_row(row, server_time, received_at, klines_hash)
        except ValueError as error:
            return SpotSnapshot(
                str(error),
                str(error),
                base_url,
                source_mode,
                server_time,
                received_at,
                klines_hash,
                time_hash,
                (),
                (),
            )
        if parsed is None:
            seen_unclosed = True
            rejected.append(RejectedBar("unclosed", _maybe_open(row)))
            continue
        if seen_unclosed:
            return SpotSnapshot(
                "invalid_timestamp",
                "invalid_timestamp",
                base_url,
                source_mode,
                server_time,
                received_at,
                klines_hash,
                time_hash,
                (),
                tuple(rejected),
            )
        closed.append(parsed)
    if not closed:
        return SpotSnapshot(
            SOURCE_UNAVAILABLE,
            "no_closed_candle",
            base_url,
            source_mode,
            server_time,
            received_at,
            klines_hash,
            time_hash,
            (),
            tuple(rejected),
        )
    return SpotSnapshot(
        "ok",
        "ok",
        base_url,
        source_mode,
        server_time,
        received_at,
        klines_hash,
        time_hash,
        tuple(closed),
        tuple(rejected),
    )


class LiveMarketStore:
    """Сырой ответ и свечи только добавляются. Первая строка свечи не переписывается."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_responses (
                body_sha256 TEXT PRIMARY KEY,
                received_at TEXT NOT NULL,
                body BLOB NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS spot_candles (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                exchange_server_time TEXT NOT NULL,
                received_at TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                raw_ohlcv_sha256 TEXT NOT NULL,
                response_sha256 TEXT NOT NULL,
                available_at TEXT NOT NULL,
                api_first_seen_proven INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS spot_conflicts (
                conflict_id TEXT PRIMARY KEY,
                open_time TEXT NOT NULL,
                previous_sha TEXT NOT NULL,
                new_sha TEXT NOT NULL,
                received_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save_body(self, body: bytes, received_at: datetime) -> tuple[str, str]:
        _require_utc(received_at, "received_at")
        digest = hashlib.sha256(body).hexdigest()
        existing = self._connection.execute(
            "SELECT body_sha256 FROM raw_responses WHERE body_sha256 = ?",
            (digest,),
        ).fetchone()
        if existing is not None:
            return digest, "duplicate"
        self._connection.execute(
            "INSERT INTO raw_responses (body_sha256, received_at, body) VALUES (?, ?, ?)",
            (digest, received_at.isoformat(), body),
        )
        self._connection.commit()
        return digest, "stored"

    def save_candle(self, candle: SpotCandle) -> str:
        if candle.api_first_seen_proven:
            raise ValueError("время первого появления через API не доказано")
        row = self._connection.execute(
            """
            SELECT raw_ohlcv_sha256 FROM spot_candles
            WHERE symbol = ? AND interval = ? AND open_time = ?
            """,
            (candle.symbol, candle.interval, candle.open_time.isoformat()),
        ).fetchone()
        if row is not None:
            if row[0] == candle.raw_ohlcv_sha256:
                return "duplicate"
            conflict_id = hashlib.sha256(
                f"{candle.open_time.isoformat()}:{row[0]}:{candle.raw_ohlcv_sha256}".encode()
            ).hexdigest()
            self._connection.execute(
                """
                INSERT OR IGNORE INTO spot_conflicts (
                    conflict_id, open_time, previous_sha, new_sha, received_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    conflict_id,
                    candle.open_time.isoformat(),
                    row[0],
                    candle.raw_ohlcv_sha256,
                    candle.received_at.isoformat(),
                ),
            )
            self._connection.commit()
            return "conflict"
        self._connection.execute(
            """
            INSERT INTO spot_candles (
                symbol, interval, open_time, close_time, exchange_server_time, received_at,
                open, high, low, close, volume, raw_ohlcv_sha256, response_sha256,
                available_at, api_first_seen_proven
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                candle.symbol,
                candle.interval,
                candle.open_time.isoformat(),
                candle.close_time.isoformat(),
                candle.exchange_server_time.isoformat(),
                candle.received_at.isoformat(),
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.raw_ohlcv_sha256,
                candle.response_sha256,
                candle.available_at.isoformat(),
            ),
        )
        self._connection.commit()
        return "stored"

    def save_snapshot(self, snapshot: SpotSnapshot, time_body: bytes, klines_body: bytes) -> dict[str, int]:
        if snapshot.received_at is None:
            raise ValueError("нет времени получения")
        self.save_body(time_body, snapshot.received_at)
        self.save_body(klines_body, snapshot.received_at)
        counts = {"stored": 0, "duplicate": 0, "conflict": 0}
        for candle in snapshot.closed:
            state = self.save_candle(candle)
            counts[state] += 1
        return counts

    def candle_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM spot_candles").fetchone()
        return int(row[0])

    def conflict_count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM spot_conflicts").fetchone()
        return int(row[0])


def _stamp(received_at: datetime | None) -> datetime:
    """Время получения. Для живого вызова это момент уже прочитанного ответа."""
    value = received_at if received_at is not None else datetime.now(timezone.utc)
    _require_utc(value, "received_at")
    return value


def _unavailable(reason: str, received_at: datetime, calls: int) -> SpotSnapshot:
    return SpotSnapshot(
        SOURCE_UNAVAILABLE,
        reason,
        None,
        "network",
        None,
        received_at,
        None,
        None,
        (),
        (),
        calls,
    )


def _server_time(body: bytes) -> datetime:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError("unexpected_payload") from error
    if not isinstance(payload, dict) or "serverTime" not in payload:
        raise ValueError("unexpected_payload")
    return _decode_millis(payload["serverTime"], "serverTime")


def _kline_rows(body: bytes) -> list[list[object]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError("unexpected_payload") from error
    if not isinstance(payload, list) or not payload or len(payload) > KLINE_LIMIT:
        raise ValueError("unexpected_payload")
    rows: list[list[object]] = []
    for item in payload:
        if not isinstance(item, list) or len(item) != 12:
            raise ValueError("incomplete_candle")
        rows.append(item)
    return rows


def _parse_row(
    row: list[object],
    server_time: datetime,
    received_at: datetime,
    response_sha256: str,
) -> SpotCandle | None:
    open_time = _decode_millis(row[0], "open_time")
    close_time = _decode_millis(row[6], "close_time")
    if not _is_utc_midnight(open_time):
        raise ValueError("invalid_timestamp")
    expected_close = open_time + timedelta(days=1) - timedelta(milliseconds=1)
    if close_time != expected_close:
        raise ValueError("invalid_timestamp")
    prices = _price_strings(row)
    trades = _trades(row[8])
    if server_time <= close_time:
        return None
    available_at = open_time + timedelta(days=1)
    grade = grade_timing(
        event_time=open_time,
        published_at=available_at,
        available_at=available_at,
        ingested_at=received_at,
        basis=DAILY_BAR_BASIS,
    )
    identity = json.dumps(row[:6], ensure_ascii=False, separators=(",", ":"))
    return SpotCandle(
        symbol=SYMBOL,
        interval=INTERVAL,
        open_time=open_time,
        close_time=close_time,
        exchange_server_time=server_time,
        received_at=received_at,
        open=prices[0],
        high=prices[1],
        low=prices[2],
        close=prices[3],
        volume=prices[4],
        trades=trades,
        available_at=available_at,
        raw_ohlcv_sha256=hashlib.sha256(identity.encode()).hexdigest(),
        response_sha256=response_sha256,
        availability_grade=grade,
    )


def _price_strings(row: list[object]) -> tuple[float, float, float, float, float]:
    labels = ("open", "high", "low", "close", "volume")
    numbers: list[float] = []
    for index, label in zip((1, 2, 3, 4, 5), labels, strict=True):
        value = row[index]
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError("invalid_ohlcv")
        try:
            number = float(value)
        except ValueError as error:
            raise ValueError("invalid_ohlcv") from error
        if label == "volume":
            if number < 0 or number != number or number in (float("inf"), float("-inf")):
                raise ValueError("invalid_ohlcv")
        elif number <= 0 or number != number or number in (float("inf"), float("-inf")):
            raise ValueError("invalid_ohlcv")
        numbers.append(number)
    open_, high, low, close, volume = numbers
    if high < low or high < max(open_, close) or low > min(open_, close):
        raise ValueError("invalid_ohlcv")
    quote = row[7]
    if isinstance(quote, bool) or not isinstance(quote, str):
        raise ValueError("incomplete_candle")
    try:
        quote_volume = float(quote)
    except ValueError as error:
        raise ValueError("incomplete_candle") from error
    if quote_volume < 0 or quote_volume != quote_volume or quote_volume == float("inf"):
        raise ValueError("incomplete_candle")
    return open_, high, low, close, volume


def _trades(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("incomplete_candle")
    return value


def _decode_millis(value: object, name: str) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid_timestamp" if name != "serverTime" else "unexpected_payload")
    try:
        return _open_time(value)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("invalid_timestamp") from error


def _maybe_open(row: list[object]) -> datetime | None:
    try:
        return _decode_millis(row[0], "open_time")
    except ValueError:
        return None


def _header_weight(headers: Mapping[str, str]) -> int:
    for key, value in headers.items():
        if key.lower().startswith("x-mbx-used-weight-"):
            try:
                return int(value)
            except ValueError:
                return WEIGHT_BACKOFF
    return 0


def _is_utc_midnight(value: datetime) -> bool:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        return False
    return not any((value.hour, value.minute, value.second, value.microsecond))


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} должен быть в UTC")
