"""Активность рынка из уже закрытой свечи.

Соотношение объёма и хода цены использует только log1p-изменение объёма
и лог-доходность этого бара. Будущий бар в знаменатель не входит.
"""

from __future__ import annotations

MOVE_FLOOR = 1e-4


def volume_to_move(volume_change: float, log_return: float) -> float:
    """|изменение объёма| / max(|лог-доходность|, 1e-4). Порог фиксирован, не подбирается по 2021."""
    return abs(volume_change) / max(abs(log_return), MOVE_FLOOR)
