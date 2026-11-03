"""Короткий отчёт для Qwen. Сырой ряд и метрики 2023–2024 сюда не входят."""

from __future__ import annotations

import math
from datetime import date

from wolfai.intelligence.align import visible_exact_count
from wolfai.intelligence.frame import DERIVATIVE_FEATURES, MACRO_ETF_FEATURES, PRICE_FEATURES, MarketRow
from wolfai.intelligence.memory.analogs import nearest_analogs
from wolfai.intelligence.memory.situation import PREDICTIVE_FEATURES, Situation
from wolfai.intelligence.schema import Observation

BRIEF_LAST_DECISION = date(2022, 12, 31)
FIT_END = date(2021, 12, 31)


def build_qwen_brief(rows: list[MarketRow], observations: list[Observation]) -> dict[str, object]:
    """Состояние на последнем решении не позже 2022-12-31."""
    eligible = [row for row in rows if row.decision_date <= BRIEF_LAST_DECISION]
    if any(row.decision_date > BRIEF_LAST_DECISION for row in eligible):
        raise ValueError("бриф захватывает период после внутренней проверки")
    if not eligible:
        raise ValueError("нет решения не позже 2022-12-31")
    if eligible[-1].decision_date > BRIEF_LAST_DECISION:
        raise ValueError("последнее решение брифа позже 2022-12-31")
    situations = [_situation(row) for row in eligible]
    query = situations[-1]
    active = tuple(name for name in PREDICTIVE_FEATURES if _finite(query.features.get(name)))
    search = nearest_analogs(query, situations[:-1], feature_names=active)
    allowed = _allowed(eligible)
    brief: dict[str, object] = {
        "decision_date": query.decision_date.isoformat(),
        "as_of": query.as_of.isoformat(),
        "features": {name: query.features.get(name) for name in allowed},
        "allowed_features": allowed,
        "unavailable": [name for name in (*DERIVATIVE_FEATURES, *MACRO_ETF_FEATURES) if name not in allowed],
        "etf_before_launch": "недоступно, не ноль",
        "whale_transfers_visible": visible_exact_count(observations, query.as_of, "whale_transfer"),
        "exchange_flow_visible": visible_exact_count(observations, query.as_of, "exchange_flow"),
        "exchange_flow_stored": sum(item.category == "exchange_flow" for item in observations),
        "limitations": (
            "Биржевой приток не означает продажу. Агрегаты с поздней меткой адреса "
            "в признаки не входят. До запуска спотовых ETF США поток недоступен."
        ),
        "analogs": {
            "sufficient": search.sufficient,
            "reason": search.reason,
            "feature_names": list(search.feature_names),
            "neighbors": [
                {
                    "decision_date": item.decision_date.isoformat(),
                    "distance": item.distance,
                    "similarity": item.similarity,
                    "outcome_1d": item.outcome_1d,
                    "outcome_7d": item.outcome_7d,
                }
                for item in search.neighbors
            ],
        },
        "query_outcome_included": False,
    }
    serialized = str(brief)
    if "brier" in serialized.lower() or "holdout" in serialized.lower():
        raise RuntimeError("бриф содержит метрику или holdout")
    return brief


def _situation(row: MarketRow) -> Situation:
    features = {name: row.raw.get(name) for name in PREDICTIVE_FEATURES if name in row.raw}
    return Situation(
        decision_date=row.decision_date,
        as_of=row.as_of,
        features=features,
        outcome_1d=row.outcome_1d,
        outcome_7d=row.outcome_7d,
        outcome_1d_available_at=row.outcome_1d_available_at,
        outcome_7d_available_at=row.outcome_7d_available_at,
    )


def _allowed(rows: list[MarketRow]) -> list[str]:
    allowed = list(PRICE_FEATURES)
    fit_rows = [row for row in rows if row.decision_date <= FIT_END]
    for name in DERIVATIVE_FEATURES:
        observed = sum(_finite(row.raw.get(name)) for row in fit_rows)
        if observed >= 100:
            allowed.append(name)
    return allowed


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
