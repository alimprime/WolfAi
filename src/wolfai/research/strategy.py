"""Минимальный контракт будущей арены. Агенты, турниры и Qwen здесь не запускаются.

Стратегия видит только переданный снимок признаков и возвращает вероятность.
Журнал ей не передаётся, поэтому чужой прогноз она изменить не может.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class FeatureView:
    decision_time: datetime
    features: dict[str, float]
    feature_available_at: dict[str, datetime]


class ResearchStrategy(Protocol):
    strategy_id: str
    version: str

    def predict(self, view: FeatureView) -> float:
        """Вероятность роста. Будущая метка в view не входит."""


@dataclass(frozen=True)
class ConstantHalfStrategy:
    """Контроль для проверки интерфейса. Всегда возвращает 0.5."""

    strategy_id: str = "constant-half"
    version: str = "1"

    def predict(self, view: FeatureView) -> float:
        del view
        return 0.5
