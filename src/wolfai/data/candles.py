"""Свечи OHLCV и проверка исторического ряда.

Метка времени хранится в UTC. Перевод из другого пояса сохраняет мгновение
и не подменяет его местной календарной датой. Пропуск дневной свечи — ошибка
для ряда на полуночной сетке UTC; вымышленная свеча на это место не ставится.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class DataIssue:
    """Одна находка проверки. error запрещает ряд, warning его не выбрасывает."""

    severity: str
    code: str
    message: str
    index: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[DataIssue, ...] = ()

    @property
    def errors(self) -> tuple[DataIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[DataIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == SEVERITY_WARNING)


@dataclass(frozen=True)
class CandleRead:
    candles: tuple[Candle, ...]
    report: ValidationReport


def to_utc(timestamp: datetime) -> datetime:
    """Переводит осведомлённую метку в UTC, не сдвигая мгновение.

    Наивная метка не считается UTC: у неё нет пояса, который можно сохранить.
    """
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp
    return timestamp.astimezone(timezone.utc)


def parse_utc_timestamp(value: str) -> tuple[datetime, bool]:
    """Разбирает ISO-8601 и возвращает (метка UTC, был ли исходный пояс не UTC).

    Наивная строка возвращается без пояса. Второй флаг тогда False.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed, False
    shifted = parsed.utcoffset() != timedelta(0)
    return parsed.astimezone(timezone.utc), shifted


def load_candles(path: str | Path) -> list[Candle]:
    """Читает CSV `timestamp,open,high,low,close,volume` и проверяет ряд."""
    loaded = read_candles(path)
    if loaded.report.errors:
        raise ValueError(loaded.report.errors[0].message)
    return list(loaded.candles)


def read_candles(path: str | Path) -> CandleRead:
    """Читает CSV, приводит метки к UTC и отдаёт свечи вместе с замечаниями.

    Критическая ошибка остаётся в отчёте. Предупреждение о поясе или нулевом
    объёме ряд не отклоняет и пропуск не заполняет.
    """
    file_path = Path(path)
    with file_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{file_path}: файл не содержит свечей")
    expected = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [name for name in expected if name not in rows[0]]
    if missing:
        raise ValueError(f"{file_path}: нет колонок {', '.join(missing)}")
    candles: list[Candle] = []
    notes: list[DataIssue] = []
    for index, row in enumerate(rows):
        try:
            candle, shifted = _parse_row(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"строка {index + 2}: не удалось прочитать свечу") from error
        if shifted:
            notes.append(DataIssue(
                SEVERITY_WARNING,
                "normalized_timezone",
                f"свеча {index}: метка приведена к UTC, мгновение сохранено",
                index,
            ))
        candles.append(candle)
    report = inspect_candles(candles)
    return CandleRead(tuple(candles), ValidationReport(tuple(notes) + report.issues))


def inspect_candles(candles: list[Candle], *, daily: bool = False) -> ValidationReport:
    """Собирает критические ошибки и предупреждения, не исправляя ряд.

    `daily=True` дополнительно требует полночь UTC и шаг ровно в одни сутки.
    Дыра на уже полуночной сетке — ошибка и при `daily=False`.
    """
    if not candles:
        return ValidationReport((DataIssue(SEVERITY_ERROR, "empty", "ряд свечей пуст", None),))
    issues: list[DataIssue] = []
    previous: datetime | None = None
    for index, candle in enumerate(candles):
        issues.extend(_candle_issues(index, candle, daily=daily))
        if previous is not None and _is_aware(previous) and _is_aware(candle.timestamp):
            issues.extend(_pair_issues(index, previous, candle.timestamp, daily=daily))
        previous = candle.timestamp
    return ValidationReport(tuple(issues))


def validate_candles(candles: list[Candle]) -> None:
    """Проверяет цены, объём, порядок и непрерывность полуночных суток UTC."""
    report = inspect_candles(candles)
    if report.errors:
        raise ValueError(report.errors[0].message)


def _parse_row(row: dict[str, str]) -> tuple[Candle, bool]:
    timestamp, shifted = parse_utc_timestamp(row["timestamp"])
    candle = Candle(
        timestamp=timestamp,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
    )
    return candle, shifted


def _candle_issues(index: int, candle: Candle, *, daily: bool) -> list[DataIssue]:
    issues: list[DataIssue] = []
    if not _is_aware(candle.timestamp):
        issues.append(_error(
            index,
            "missing_timezone",
            f"свеча {index}: timestamp без часового пояса",
        ))
    prices = {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
    }
    for name, value in prices.items():
        if not math.isfinite(value):
            issues.append(_error(
                index,
                "non_finite_price",
                f"свеча {index}: {name} должна быть конечной положительной",
            ))
        elif value <= 0:
            issues.append(_error(
                index,
                "non_positive_price",
                f"свеча {index}: {name} должна быть конечной положительной",
            ))
    if not math.isfinite(candle.volume):
        issues.append(_error(
            index,
            "non_finite_volume",
            f"свеча {index}: volume должен быть конечным и неотрицательным",
        ))
    elif candle.volume < 0:
        issues.append(_error(
            index,
            "negative_volume",
            f"свеча {index}: volume должен быть конечным и неотрицательным",
        ))
    if all(math.isfinite(value) for value in prices.values()):
        if candle.high < candle.low:
            issues.append(_error(index, "high_below_low", f"свеча {index}: high меньше low"))
        if candle.high < max(candle.open, candle.close):
            issues.append(_error(
                index,
                "high_below_body",
                f"свеча {index}: high ниже open или close",
            ))
        if candle.low > min(candle.open, candle.close):
            issues.append(_error(
                index,
                "low_above_body",
                f"свеча {index}: low выше open или close",
            ))
    if daily and _is_aware(candle.timestamp) and not _is_utc_midnight(candle.timestamp):
        issues.append(_error(
            index,
            "not_utc_midnight",
            f"свеча {index}: требуется начало суток UTC",
        ))
    if math.isfinite(candle.volume) and candle.volume == 0:
        issues.append(DataIssue(
            SEVERITY_WARNING,
            "zero_volume",
            f"свеча {index}: объём равен нулю",
            index,
        ))
    return issues


def _pair_issues(index: int, previous: datetime, current: datetime, *, daily: bool) -> list[DataIssue]:
    if current == previous:
        return [_error(
            index,
            "duplicate_timestamp",
            f"свеча {index}: timestamp дублируется и не возрастает",
        )]
    if current < previous:
        return [_error(index, "unordered_timestamp", f"свеча {index}: timestamp не возрастает")]
    delta = current - previous
    if _is_utc_midnight(previous) and _is_utc_midnight(current) and delta != timedelta(days=1):
        return [_error(index, "missing_daily_bar", _gap_message(index))]
    if daily and delta != timedelta(days=1):
        return [_error(index, "missing_daily_bar", _gap_message(index))]
    if _utc_gap_days(previous, current) > 1:
        return [DataIssue(
            SEVERITY_WARNING,
            "missing_daily_bar",
            f"свеча {index}: пропуск дневных свечей не заполняется",
            index,
        )]
    return []


def _gap_message(index: int) -> str:
    return f"свеча {index}: пропуск или нарушение дневного шага"


def _error(index: int, code: str, message: str) -> DataIssue:
    return DataIssue(SEVERITY_ERROR, code, message, index)


def _is_aware(timestamp: datetime) -> bool:
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _is_utc_midnight(timestamp: datetime) -> bool:
    if not _is_aware(timestamp) or timestamp.utcoffset() != timedelta(0):
        return False
    return not any((
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
        timestamp.microsecond,
    ))


def _utc_gap_days(left: datetime, right: datetime) -> int:
    return (right.astimezone(timezone.utc).date() - left.astimezone(timezone.utc).date()).days
