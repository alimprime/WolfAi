"""Ситуация на дату решения. Исход цены лежит отдельно от признаков поиска."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

PREDICTIVE_FEATURES = (
    "log_return",
    "momentum_7",
    "momentum_30",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "range_position",
    "funding_rate",
    "oi_log_change_1d",
)

OUTCOME_FIELDS = frozenset({
    "outcome_1d",
    "outcome_7d",
    "y_up",
    "baseline_up_frequency",
})


@dataclass(frozen=True)
class Situation:
    decision_date: date
    as_of: datetime
    features: dict[str, float | None]
    outcome_1d: float | None
    outcome_7d: float | None
    outcome_1d_available_at: datetime | None
    outcome_7d_available_at: datetime | None

    def __post_init__(self) -> None:
        leaked = OUTCOME_FIELDS.intersection(self.features)
        if leaked:
            raise ValueError(f"исход попал в признаки ситуации: {sorted(leaked)}")
        unknown = set(self.features) - set(PREDICTIVE_FEATURES)
        if unknown:
            raise ValueError(f"неизвестный признак ситуации: {sorted(unknown)}")
