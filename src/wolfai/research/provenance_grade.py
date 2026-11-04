"""Степень достоверности времени источника.

Четыре метки хранятся раздельно: событие, публикация, доступность и загрузка.
`calc_time` и `create_time` биржи — заявление источника, а не проверка,
что исторический API отдавал это значение в тот момент.
В строгую историческую проверку проходит только категория `verified`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

GRADES = ("verified", "source_declared", "estimated", "unknown")
DAILY_BAR_BASIS = "daily_bar_next_midnight"
DECLARED_CLOCKS = frozenset({"exchange_calc_time", "exchange_create_time"})


def grade_timing(
    *,
    event_time: datetime,
    published_at: datetime | None,
    available_at: datetime | None,
    ingested_at: datetime,
    basis: str,
) -> str:
    """Не подставляет одну метку вместо другой. Пустая публикация — `unknown`."""
    _aware(event_time, "event_time")
    _aware(ingested_at, "ingested_at")
    if published_at is None or available_at is None or not basis.strip():
        return "unknown"
    _aware(published_at, "published_at")
    _aware(available_at, "available_at")
    if available_at < published_at:
        return "unknown"
    if basis in DECLARED_CLOCKS:
        return "source_declared"
    if basis == "estimated_lag":
        return "estimated"
    if (
        basis == DAILY_BAR_BASIS
        and published_at == available_at
        and available_at == event_time + timedelta(days=1)
    ):
        return "verified"
    return "unknown"


def admits_strict_history(grade: str) -> bool:
    if grade not in GRADES:
        raise ValueError("неизвестная категория достоверности")
    return grade == "verified"


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} должен быть в UTC")
