"""Оценка созревших прогнозов. Один удачный период не получает статус supported.

Роли не смешиваются: диагностический backtest, forward-тест и синтетическая
проверка считаются отдельно. Подтверждённая гипотеза отсюда не назначается.
"""

from __future__ import annotations

from wolfai.research.logistic import score_probabilities
from wolfai.research.prediction_ledger import PredictionLedger
from wolfai.research.quality_contract import (
    MIN_RELIABILITY_OBSERVATIONS,
    STATUS_INSUFFICIENT,
    STATUS_REJECTED,
    dependence_block_length,
    linear_percentile,
    moving_block_means,
    point_brier,
)

ROLES = {
    "forward": "forward_test",
    "reconstruction": "diagnostic_backtest",
    "synthetic": "synthetic_check",
}


def evaluate_settled(ledger: PredictionLedger, channel: str) -> dict[str, object]:
    if channel not in ROLES:
        raise ValueError("неизвестный канал оценки")
    pairs = ledger.settled(channel)
    probabilities = [item[0].probability_up for item in pairs]
    labels = [item[1].label for item in pairs]
    baselines = [item[0].baseline_probability for item in pairs]
    dates = [item[0].decision_time.date() for item in pairs]
    horizon = pairs[0][0].horizon_days if pairs else 1
    if any(item[0].horizon_days != horizon for item in pairs):
        raise ValueError("в одной оценке смешаны разные горизонты")
    if not pairs:
        return {
            "role": ROLES[channel],
            "channel": channel,
            "observations": 0,
            "claim_status": STATUS_INSUFFICIENT,
            "note": "нет созревших прогнозов",
        }
    scored = score_probabilities(
        [_clip(value) for value in probabilities],
        labels,
        {
            "constant_half": [0.5] * len(labels),
            "frozen_baseline": baselines,
        },
    )
    coin_gaps = [
        point_brier(0.5, label) - point_brier(probability, label)
        for probability, label in zip(probabilities, labels, strict=True)
    ]
    base_gaps = [
        point_brier(baseline, label) - point_brier(probability, label)
        for probability, baseline, label in zip(probabilities, baselines, labels, strict=True)
    ]
    block, overlapping = dependence_block_length(dates, horizon) if len(dates) >= 2 else (horizon, False)
    uncertainty = None
    status = STATUS_INSUFFICIENT
    if len(pairs) >= MIN_RELIABILITY_OBSERVATIONS and block <= len(pairs):
        uncertainty = {
            "block_length": block,
            "overlapping": overlapping,
            "vs_coin": _interval(coin_gaps, block),
            "vs_baseline": _interval(base_gaps, block),
        }
        coin = uncertainty["vs_coin"]
        base = uncertainty["vs_baseline"]
        assert isinstance(coin, dict) and isinstance(base, dict)
        if coin["high"] < 0.0 and base["high"] < 0.0:
            status = STATUS_REJECTED
    elif len(pairs) < MIN_RELIABILITY_OBSERVATIONS:
        status = STATUS_INSUFFICIENT
    return {
        "role": ROLES[channel],
        "channel": channel,
        "observations": len(pairs),
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "base_rate": scored["base_rate"],
        "mean_probability": scored["mean_probability"],
        "min_probability": min(probabilities),
        "max_probability": max(probabilities),
        "calibration_occupied": sum(
            1 for item in scored["calibration"] if isinstance(item, dict) and item["count"]
        ),
        "delta_vs_coin": None if not coin_gaps else sum(coin_gaps) / len(coin_gaps),
        "delta_vs_baseline": None if not base_gaps else sum(base_gaps) / len(base_gaps),
        "uncertainty": uncertainty,
        "claim_status": status,
        "validated_law": False,
    }


def _interval(values: list[float], block: int) -> dict[str, float]:
    means = moving_block_means(values, block)
    return {
        "mean": sum(values) / len(values),
        "low": linear_percentile(means, 0.025),
        "high": linear_percentile(means, 0.975),
    }


def _clip(probability: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, probability))
