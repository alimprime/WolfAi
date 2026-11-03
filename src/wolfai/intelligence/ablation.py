"""Сравнение A/B/C/D на одном наборе дат.

B и C не оцениваются, если в их категориях нет строгого признака.
Пропуск не заполняется. Holdout не читается. Флаг надёжности смотрит только
на заранее заданное окно 2022 года и не использует 2023–2024.
"""

from __future__ import annotations

import math
from datetime import date

from wolfai.data.candles import Candle
from wolfai.forecasting.baseline import HistoricalQuantileBaseline, WalkForwardResult
from wolfai.forecasting.evaluate import evaluate_split
from wolfai.forecasting.random_walk import RandomWalkBaseline
from wolfai.forecasting.splits import (
    COEFFICIENT_FIT_BOUNDS,
    INNER_CHECK_BOUNDS,
    Period,
    period_bounds,
)
from wolfai.intelligence.frame import (
    DERIVATIVE_FEATURES,
    MACRO_ETF_FEATURES,
    PRICE_FEATURES,
    MarketRow,
    ScaledRow,
    between,
    build_market_rows,
    finite_feature_names,
    rows_with_features,
    standardize,
)
from wolfai.intelligence.schema import Observation
from wolfai.intelligence.store import individual_whale_events
from wolfai.research.features import assert_research_series
from wolfai.research.logistic import (
    LogisticError,
    fit_l2_logistic,
    predict_probability,
    score_probabilities,
)

MIN_FIT = 100
MIN_BLOCK = 20
PRIMARY = "inner_2022"


def run_ablation(
    candles: list[Candle],
    observations: list[Observation],
    *,
    include_ranges: bool = True,
) -> dict[str, object]:
    assert_research_series(candles)
    rows = build_market_rows(candles, observations)
    derivative = finite_feature_names(rows, DERIVATIVE_FEATURES)
    macro = finite_feature_names(rows, MACRO_ETF_FEATURES)
    whale_events = individual_whale_events(observations)
    exchange_rows = sum(item.category == "exchange_flow" for item in observations)
    exchange_strict = sum(
        item.category == "exchange_flow" and item.strict_backtest for item in observations
    )
    blocked_b = _blocked(
        "B",
        "нет строгого on-chain или whale-признака: биржевые агрегаты Coin Metrics "
        "пересчитаны поздними метками, point-in-time ряд недоступен, отдельных переводов нет",
        {
            "individual_whale_events": whale_events,
            "exchange_flow_rows": exchange_rows,
            "exchange_flow_strict_rows": exchange_strict,
        },
    )
    if macro:
        blocked_c: dict[str, object] = {
            "status": "unexpected_strict_macro",
            "features": list(macro),
        }
    else:
        blocked_c = _blocked(
            "C",
            "нет строгого макро- или ETF-признака: у FRED и EFFR нет точного времени "
            "публикации, таблица Farside не скачивалась, до 2024-01-11 поток ETF недоступен",
            {"macro_etf_features": []},
        )
    configurations: dict[str, object] = {"B": blocked_b, "C": blocked_c}
    matched_rows: list[MarketRow] = []
    if not derivative and not macro:
        configurations["D"] = _blocked(
            "D",
            "нет строгого признака сверх цены; отсутствующие категории не заполнялись",
            {"omitted": ["whale", "etf", "macro", "derivatives"]},
        )
        configurations["A"] = _score_model(rows, PRICE_FEATURES, reference_model=None)
        configurations["A"].pop("probabilities", None)
        configurations["A"]["sample"] = "all_price_rows"
        comparison = None
    else:
        matched_names = PRICE_FEATURES + derivative + macro
        matched_rows = rows_with_features(rows, matched_names)
        configurations["A"] = _score_model(matched_rows, PRICE_FEATURES, reference_model=None)
        price_probabilities = configurations["A"].pop("probabilities", None)
        configurations["A"]["features"] = list(PRICE_FEATURES)
        configurations["A"]["sample"] = "matched"
        reference = price_probabilities if isinstance(price_probabilities, dict) else None
        configurations["D"] = _score_model(matched_rows, matched_names, reference_model=reference)
        configurations["D"].pop("probabilities", None)
        configurations["D"]["features"] = list(matched_names)
        configurations["D"]["omitted_categories"] = _omitted(derivative, macro)
        configurations["D"]["sample"] = "matched"
        comparison = {"A": configurations["A"], "D": configurations["D"], "same_rows": True}
    return {
        "holdout": {"metrics": None, "reason": "закрытый период не читался и не оценивался"},
        "primary_window": PRIMARY,
        "multiple_comparison_note": (
            "Интервалы HAC не поправлены на семейство проверок. "
            "Флаг надёжности требует, чтобы на внутреннем 2022 годе улучшение D против "
            "ценовой логистической модели и против константы 0.5 имело нижнюю границу выше нуля, "
            "и чтобы каждый квартал 2022 года с не менее чем 20 решениями показал меньший Brier, "
            "чем ценовая модель и чем константа 0.5. "
            "Окно 2023–2024 в флаг не входит."
        ),
        "statistically_reliable_improvement": _reliable(comparison),
        "configurations": configurations,
        "comparison": comparison,
        "sample": {
            "price_rows": len(rows),
            "matched_rows": len(matched_rows),
            "dropped_for_missing_features": len(rows) - len(matched_rows),
        },
        "price_ranges": _ranges(candles) if include_ranges else None,
        "price_ranges_note": (
            "Диапазоны цены — прежние baseline по свечам. Внешние ряды в них не входят "
            "и с Brier направления не сравниваются."
        ),
    }


def _score_model(
    rows: list[MarketRow],
    names: tuple[str, ...],
    *,
    reference_model: dict[str, list[float]] | None,
) -> dict[str, object]:
    if len(rows) < MIN_FIT:
        return {"status": "insufficient_history", "observations": len(rows), "features": list(names)}
    scaled = standardize(rows, names)
    fit_rows = between(scaled, *COEFFICIENT_FIT_BOUNDS)
    if len(fit_rows) < MIN_FIT:
        return {
            "status": "insufficient_history",
            "fit_observations": len(fit_rows),
            "features": list(names),
        }
    try:
        beta = fit_l2_logistic(
            [_design(row, names) for row in fit_rows],
            [row.y_up for row in fit_rows],
        )
    except LogisticError as error:
        return {"status": "fit_failed", "reason": str(error), "features": list(names)}
    periods: dict[str, object] = {}
    stored_probabilities: dict[str, dict[str, float]] = {}
    windows = (
        (PRIMARY, INNER_CHECK_BOUNDS),
        ("development_validation_2023_2024", period_bounds(Period.VALIDATION)),
    )
    for label, bounds in windows:
        chosen = between(scaled, *bounds)
        if len(chosen) < MIN_BLOCK:
            periods[label] = {"status": "insufficient_period", "observations": len(chosen)}
            continue
        probabilities = [predict_probability(beta, _design(row, names)) for row in chosen]
        references = {
            "constant_0.5": [0.5] * len(chosen),
            "historical_frequency": [row.baseline_up_frequency for row in chosen],
        }
        if reference_model is not None:
            aligned = [reference_model[label][row.decision_date.isoformat()] for row in chosen]
            references["price_only"] = aligned
        metrics = score_probabilities(probabilities, [row.y_up for row in chosen], references)
        metrics["status"] = "scored"
        reference_dates = None if reference_model is None else reference_model.get(label)
        metrics["blocks"] = _blocks(
            chosen,
            names,
            beta,
            reference_dates if isinstance(reference_dates, dict) else None,
        )
        periods[label] = metrics
        stored_probabilities[label] = {
            row.decision_date.isoformat(): probability
            for row, probability in zip(chosen, probabilities, strict=True)
        }
    return {
        "status": "scored",
        "features": list(names),
        "fit_observations": len(fit_rows),
        "coefficients": {
            "intercept": beta[0],
            "weights": [
                {"feature": name, "weight": weight}
                for name, weight in zip(names, beta[1:], strict=True)
            ],
        },
        "periods": periods,
        "probabilities": stored_probabilities,
    }


def _blocks(
    rows: list[ScaledRow],
    names: tuple[str, ...],
    beta: list[float],
    reference: dict[str, float] | None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[ScaledRow]] = {}
    for row in rows:
        grouped.setdefault(_quarter(row.decision_date), []).append(row)
    blocks: list[dict[str, object]] = []
    for label in sorted(grouped):
        chosen = grouped[label]
        probabilities = [predict_probability(beta, _design(row, names)) for row in chosen]
        labels = [row.y_up for row in chosen]
        constant = [0.5] * len(chosen)
        block = {
            "block": label,
            "observations": len(chosen),
            "brier": _mean_brier(probabilities, labels),
            "constant_brier": _mean_brier(constant, labels),
            "log_loss": _mean_log_loss(probabilities, labels),
        }
        if reference is not None:
            block["price_only_brier"] = _mean_brier(
                [reference[row.decision_date.isoformat()] for row in chosen],
                labels,
            )
        blocks.append(block)
    return blocks


def _reliable(comparison: dict[str, object] | None) -> bool:
    if comparison is None:
        return False
    model = comparison.get("D")
    if not isinstance(model, dict):
        return False
    periods = model.get("periods")
    if not isinstance(periods, dict):
        return False
    primary = periods.get(PRIMARY)
    if not isinstance(primary, dict) or primary.get("status") != "scored":
        return False
    skill = primary.get("skill")
    blocks = primary.get("blocks")
    if not isinstance(skill, dict) or not isinstance(blocks, list):
        return False
    usable = [block for block in blocks if isinstance(block, dict) and int(block["observations"]) >= MIN_BLOCK]
    if not usable:
        return False
    beats_price_blocks = all(
        "price_only_brier" in block and float(block["brier"]) < float(block["price_only_brier"])
        for block in usable
    )
    beats_constant_blocks = all(float(block["brier"]) < float(block["constant_brier"]) for block in usable)
    return (
        _clears(skill.get("price_only"))
        and _clears(skill.get("constant_0.5"))
        and beats_price_blocks
        and beats_constant_blocks
    )


def _clears(skill: object) -> bool:
    if not isinstance(skill, dict):
        return False
    low = skill.get("improvement_low")
    return isinstance(low, (int, float)) and float(low) > 0


def _blocked(name: str, reason: str, extra: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"configuration": name, "status": "blocked", "reason": reason, "metrics": None}
    payload.update(extra)
    return payload


def _omitted(derivative: tuple[str, ...], macro: tuple[str, ...]) -> list[str]:
    omitted = []
    if not derivative:
        omitted.append("derivatives")
    omitted.extend(["whale", "etf", "macro"] if not macro else [])
    if macro:
        omitted.append("whale")
    return omitted


def _design(row: ScaledRow, names: tuple[str, ...]) -> list[float]:
    return [row.standardized[name] for name in names]


def _quarter(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def _mean_brier(probabilities: list[float], labels: list[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(labels)


def _mean_log_loss(probabilities: list[float], labels: list[int]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        clipped = min(1 - 1e-12, max(1e-12, probability))
        total += -math.log(clipped if label == 1 else 1 - clipped)
    return total / len(labels)


def _ranges(candles: list[Candle]) -> dict[str, object]:
    periods: dict[str, object] = {}
    for period in (Period.TRAINING, Period.VALIDATION):
        models: dict[str, object] = {}
        for name, factory in (
            ("historical_quantile", HistoricalQuantileBaseline),
            ("random_walk", RandomWalkBaseline),
        ):
            horizons = {}
            for horizon in (1, 7):
                horizons[str(horizon)] = _range_dict(evaluate_split(candles, horizon, period, factory()))
            models[name] = horizons
        periods[period.value] = models
    return {"uses_external_features": False, "periods": periods}


def _range_dict(result: WalkForwardResult) -> dict[str, object]:
    return {
        "observations": result.observations,
        "mean_absolute_error": result.mean_absolute_error,
        "mean_absolute_percentage_error": result.mean_absolute_percentage_error,
        "directional_accuracy": result.directional_accuracy,
        "interval_coverage": result.interval_coverage,
        "mean_interval_width": result.mean_interval_width,
        "mean_interval_score": result.mean_interval_score,
    }
