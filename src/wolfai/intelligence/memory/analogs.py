"""Ближайшие прошлые ситуации. Исход соседа в расстояние не входит.

Сосед годится только если его семидневная реакция уже была известна к as_of
запроса. Иначе поиск сообщал бы о ещё не случившемся движении.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from wolfai.intelligence.memory.situation import Situation

MIN_ANALOGS = 5
DEFAULT_K = 5
STD_FLOOR = 1e-8


@dataclass(frozen=True)
class Analog:
    decision_date: date
    distance: float
    similarity: float
    outcome_1d: float
    outcome_7d: float


@dataclass(frozen=True)
class AnalogSearch:
    sufficient: bool
    reason: str
    feature_names: tuple[str, ...]
    neighbors: tuple[Analog, ...]


def nearest_analogs(
    query: Situation,
    history: list[Situation],
    *,
    feature_names: tuple[str, ...],
    k: int = DEFAULT_K,
    min_count: int = MIN_ANALOGS,
) -> AnalogSearch:
    if k < 1 or min_count < 1:
        raise ValueError("k и min_count должны быть положительными")
    usable = tuple(
        name for name in feature_names
        if _finite(query.features.get(name))
    )
    candidates = [item for item in history if _eligible(item, query, usable)]
    if len(usable) == 0 or len(candidates) < min_count:
        return AnalogSearch(
            False,
            "недостаточно исторических аналогов",
            usable,
            (),
        )
    moments = _moments(candidates, usable)
    active = tuple(name for name in usable if moments[name][1] >= STD_FLOOR)
    if not active:
        return AnalogSearch(False, "недостаточно исторических аналогов", usable, ())
    ranked: list[tuple[float, str, Situation]] = []
    query_vector = _vector(query, active, moments)
    for item in candidates:
        distance = _distance(query_vector, _vector(item, active, moments))
        ranked.append((distance, item.decision_date.isoformat(), item))
    ranked.sort(key=lambda item: (item[0], item[1]))
    neighbors: list[Analog] = []
    for distance, _label, item in ranked[:k]:
        if item.outcome_1d is None or item.outcome_7d is None:
            raise RuntimeError("в соседи попала ситуация без уже известного исхода")
        neighbors.append(Analog(
            decision_date=item.decision_date,
            distance=distance,
            similarity=1 / (1 + distance),
            outcome_1d=item.outcome_1d,
            outcome_7d=item.outcome_7d,
        ))
    if len(neighbors) < min_count:
        return AnalogSearch(False, "недостаточно исторических аналогов", active, ())
    return AnalogSearch(True, "аналоги посчитаны по прошлым признакам", active, tuple(neighbors))


def _eligible(item: Situation, query: Situation, names: tuple[str, ...]) -> bool:
    if item.decision_date >= query.decision_date:
        return False
    if item.outcome_1d is None or item.outcome_7d is None:
        return False
    if item.outcome_1d_available_at is None or item.outcome_7d_available_at is None:
        return False
    if item.outcome_1d_available_at > query.as_of or item.outcome_7d_available_at > query.as_of:
        return False
    return all(_finite(item.features.get(name)) for name in names)


def _moments(
    candidates: list[Situation],
    names: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    moments: dict[str, tuple[float, float]] = {}
    for name in names:
        values = [float(item.features[name]) for item in candidates if item.features[name] is not None]
        mean = sum(values) / len(values)
        if len(values) < 2:
            deviation = 0.0
        else:
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            deviation = math.sqrt(variance)
        moments[name] = (mean, deviation)
    return moments


def _vector(
    situation: Situation,
    names: tuple[str, ...],
    moments: dict[str, tuple[float, float]],
) -> list[float]:
    vector: list[float] = []
    for name in names:
        value = situation.features[name]
        mean, deviation = moments[name]
        if value is None or deviation < STD_FLOOR:
            vector.append(0.0)
            continue
        vector.append((value - mean) / deviation)
    return vector


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
