"""Контрольные модели на горизонтах 1, 3 и 7. PySR здесь не запускается.

2021 — development_only. Статус supported на этом окне не назначается.
Масштаб и логистическая регрессия считаются только по меткам,
созревшим до 2021 года. Внутреннего перебора гиперпараметров нет.
"""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from wolfai.data.candles import Candle
from wolfai.research.horizon_labels import AlignedRow, build_aligned_rows, complete_rows, rows_by_role
from wolfai.research.logistic import LOGISTIC_L2, fit_l2_logistic, predict_probability, score_probabilities
from wolfai.research.ohlcv_features import ALL_FEATURES, OHLCV_FEATURES, PRICE_FEATURES
from wolfai.research.quality_contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIAGNOSTIC_HORIZONS,
    FORBIDDEN_FROM,
    linear_percentile,
    moving_block_means,
)
from wolfai.research.reliability import assess_reliability
from wolfai.research.research_memory import ResearchMemory
from wolfai.research.stage_data import load_stage_candles
from wolfai.research.time_alignment import LABEL_RULE, TIMESTAMP_RULE

FAMILY_SIZE = 9
SEED = BOOTSTRAP_SEED
FEATURE_SETS = (
    ("logistic_price", PRICE_FEATURES),
    ("logistic_ohlcv", OHLCV_FEATURES),
    ("logistic_price_ohlcv", ALL_FEATURES),
)
PROBABILITY_CLIP = 1e-6


def run_diagnostic(path: str | Path, memory_path: str | Path | None = None) -> dict[str, object]:
    """Один проход по CSV, обрезанному до 2022-01-01 до разбора цен."""
    started = time.perf_counter()
    candles = load_stage_candles(path)
    if any(candle.timestamp.date() >= FORBIDDEN_FROM for candle in candles):
        raise ValueError("в диагностику попала свеча 2022 года")
    memory = None
    if memory_path is not None:
        memory = ResearchMemory(memory_path)
    try:
        report = _study(candles, memory)
    finally:
        if memory is not None:
            memory.close()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["max_rss"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report["last_open"] = candles[-1].timestamp.date().isoformat()
    report["candles"] = len(candles)
    report["dataset_fingerprint"] = fingerprint_candles(candles)
    report["pysr_ran"] = False
    report["optuna_ran"] = False
    report["qwen_ran"] = False
    return report


def evaluate_candles(candles: list[Candle]) -> dict[str, object]:
    """Тот же расчёт на уже обрезанном ряде. Память не обязательна."""
    if any(candle.timestamp.date() >= FORBIDDEN_FROM for candle in candles):
        raise ValueError("в диагностику попала свеча 2022 года")
    return _study(candles, None)


def fingerprint_candles(candles: list[Candle]) -> str:
    digest = hashlib.sha256()
    for candle in candles:
        if candle.timestamp.date() >= FORBIDDEN_FROM:
            break
        digest.update(
            (
                f"{candle.timestamp.isoformat()}|{candle.open}|{candle.high}|"
                f"{candle.low}|{candle.close}|{candle.volume}\n"
            ).encode()
        )
    return digest.hexdigest()


def sensitivity_blocks(values: list[float], horizon_days: int) -> list[dict[str, object]]:
    """Интервалы при длине блока h, max(7, h) и max(14, h). Это не отбор модели."""
    specs = (
        ("h", horizon_days),
        ("max_7_h", max(7, horizon_days)),
        ("max_14_h", max(14, horizon_days)),
    )
    rows: list[dict[str, object]] = []
    mean = sum(values) / len(values) if values else None
    for name, length in specs:
        if len(values) < 2 or length > len(values):
            rows.append({
                "name": name,
                "block_length": length,
                "mean": mean,
                "low": None,
                "high": None,
                "note": "ряд короче блока",
            })
            continue
        means = moving_block_means(values, length, BOOTSTRAP_REPLICATES, SEED)
        rows.append({
            "name": name,
            "block_length": length,
            "mean": mean,
            "low": linear_percentile(means, 0.025),
            "high": linear_percentile(means, 0.975),
            "note": "чувствительность, мощность ограничена; длина блока не выбиралась по 2021",
        })
    return rows


def _study(candles: list[Candle], memory: ResearchMemory | None) -> dict[str, object]:
    horizons: dict[str, object] = {}
    stored = 0
    duplicates = 0
    for horizon in DIAGNOSTIC_HORIZONS:
        built = build_aligned_rows(candles, horizon)
        usable = complete_rows(built)
        train = rows_by_role(usable, "train")
        dev = rows_by_role(usable, "dev")
        summary = _horizon_summary(built, usable, train, dev, horizon, candles, memory)
        stored += int(summary.pop("stored"))
        duplicates += int(summary.pop("duplicates"))
        horizons[str(horizon)] = summary
    return {
        "seed": SEED,
        "hypotheses_fitted": FAMILY_SIZE,
        "hypothesis_note": (
            "Подобраны 9 логистических моделей: 3 набора признаков и 3 горизонта. "
            "Константа, замороженная частота и расширяющаяся частота заданы заранее. "
            "K=192 прошлого PySR — это слоты популяции, а не число гипотез этого этапа."
        ),
        "inner_cv": "не используется: один проход 2019–2020 без подбора по 2021",
        "timestamp_rule": TIMESTAMP_RULE,
        "label_rule": LABEL_RULE,
        "regularization": LOGISTIC_L2,
        "code_version": _code_version(),
        "memory_stored": stored,
        "memory_duplicates": duplicates,
        "horizons": horizons,
    }


def _horizon_summary(
    built: list[AlignedRow],
    usable: list[AlignedRow],
    train: list[AlignedRow],
    dev: list[AlignedRow],
    horizon: int,
    candles: list[Candle],
    memory: ResearchMemory | None,
) -> dict[str, object]:
    reasons: dict[str, int] = {}
    for row in built:
        if row.exclusion is not None:
            reasons[row.exclusion] = reasons.get(row.exclusion, 0) + 1
    frozen = _rate([row.y_up for row in train])
    expanding = _expanding(usable, dev)
    models: list[dict[str, object]] = [
        _score_model(
            "constant_half",
            (),
            [0.5] * len(dev),
            dev,
            train,
            frozen,
            horizon,
            candles,
            memory,
            coefficients=None,
        ),
        _score_model(
            "frozen_train_frequency",
            (),
            [_clip(frozen)] * len(dev),
            dev,
            train,
            frozen,
            horizon,
            candles,
            memory,
            coefficients=None,
        ),
        _score_model(
            "expanding_matured_frequency",
            (),
            [_clip(value) for value in expanding],
            dev,
            train,
            frozen,
            horizon,
            candles,
            memory,
            coefficients=None,
        ),
    ]
    for name, columns in FEATURE_SETS:
        probabilities, coefficients = _logistic(train, dev, columns)
        models.append(_score_model(
            name,
            columns,
            probabilities,
            dev,
            train,
            frozen,
            horizon,
            candles,
            memory,
            coefficients=coefficients,
        ))
    counts = {item["name"]: item["count"] for item in models}
    if len(set(counts.values())) != 1:
        raise ValueError("модели горизонта посчитаны на разных датах")
    return {
        "train_count": len(train),
        "dev_count": len(dev),
        "built_count": len(built),
        "excluded_count": len(built) - len(usable),
        "exclusion_reasons": reasons,
        "gap_count": sum(1 for row in usable if row.role is None),
        "train_decision_first": _iso(train, "decision_time", 0),
        "train_decision_last": _iso(train, "decision_time", -1),
        "train_target_last": _iso(train, "target_time", -1),
        "dev_decision_first": _iso(dev, "decision_time", 0),
        "dev_decision_last": _iso(dev, "decision_time", -1),
        "dev_target_last": _iso(dev, "target_time", -1),
        "frozen_train_frequency": frozen,
        "models": models,
        "stored": sum(1 for item in models if item["memory"] == "stored"),
        "duplicates": sum(1 for item in models if item["memory"] == "duplicate"),
    }


def _logistic(
    train: list[AlignedRow],
    dev: list[AlignedRow],
    columns: tuple[str, ...],
) -> tuple[list[float], list[float]]:
    train_x = [_row_values(row, columns) for row in train]
    means, scales = _train_scale(train_x)
    beta = fit_l2_logistic([_apply(row, means, scales) for row in train_x], [row.y_up for row in train])
    probabilities = [
        predict_probability(beta, _apply(_row_values(row, columns), means, scales))
        for row in dev
    ]
    return probabilities, beta


def _score_model(
    name: str,
    columns: tuple[str, ...],
    probabilities: list[float],
    dev: list[AlignedRow],
    train: list[AlignedRow],
    frozen: float,
    horizon: int,
    candles: list[Candle],
    memory: ResearchMemory | None,
    coefficients: list[float] | None,
) -> dict[str, object]:
    labels = [row.y_up for row in dev]
    frequency = [frozen] * len(dev)
    scored = score_probabilities(
        probabilities,
        labels,
        {"historical_frequency": frequency, "constant_half": [0.5] * len(labels)},
    )
    reliability = assess_reliability(
        [row.decision_time.date() for row in dev],
        probabilities,
        labels,
        frequency,
        horizon_days=horizon,
        complexity=max(1, len(columns)),
        hypotheses_examined=FAMILY_SIZE,
        selection_separated=True,
        evidence_role="development_only",
    )
    coin_gaps = [
        (0.5 - label) ** 2 - (probability - label) ** 2
        for probability, label in zip(probabilities, labels, strict=True)
    ]
    payload = _payload(
        name,
        columns,
        horizon,
        train,
        dev,
        candles,
        reliability.status,
        reliability.provisional,
        scored,
        coefficients,
    )
    memory_state = "not_requested"
    if memory is not None:
        memory_state = memory.record(payload)
    return {
        "name": name,
        "features": list(columns),
        "count": len(dev),
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "base_rate": scored["base_rate"],
        "frequency_brier": reliability.frequency_brier,
        "coin_brier": reliability.coin_brier,
        "delta_vs_frequency": reliability.uncertainty.mean_improvement_vs_frequency if reliability.uncertainty else None,
        "delta_vs_coin": reliability.uncertainty.mean_improvement_vs_coin if reliability.uncertainty else None,
        "low_vs_frequency": reliability.uncertainty.low_vs_frequency if reliability.uncertainty else None,
        "high_vs_frequency": reliability.uncertainty.high_vs_frequency if reliability.uncertainty else None,
        "low_vs_coin": reliability.uncertainty.low_vs_coin if reliability.uncertainty else None,
        "high_vs_coin": reliability.uncertainty.high_vs_coin if reliability.uncertainty else None,
        "block_length": reliability.uncertainty.block_length if reliability.uncertainty else None,
        "sensitivity_vs_coin": sensitivity_blocks(coin_gaps, horizon),
        "claim_status": reliability.status,
        "provisional": reliability.provisional,
        "evidence_role": reliability.evidence_role,
        "coefficients": coefficients,
        "calibration": [
            {"bin": item.bin_index, "count": item.count}
            for item in reliability.calibration
        ],
        "memory": memory_state,
    }


def _payload(
    name: str,
    columns: tuple[str, ...],
    horizon: int,
    train: list[AlignedRow],
    dev: list[AlignedRow],
    candles: list[Candle],
    status: str,
    provisional: bool,
    scored: dict[str, object],
    coefficients: list[float] | None,
) -> dict[str, object]:
    config = {
        "model": name,
        "features": list(columns),
        "horizon_days": horizon,
        "l2": LOGISTIC_L2,
        "scaler": "train_zscore",
        "seed": SEED,
        "label": LABEL_RULE,
        "family_size": FAMILY_SIZE,
    }
    return {
        "model_name": name,
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "dataset_fingerprint": fingerprint_candles(candles),
        "features": list(columns),
        "label_definition": LABEL_RULE,
        "horizon_days": horizon,
        "train_start": _iso(train, "decision_time", 0),
        "train_end": _iso(train, "target_time", -1),
        "dev_start": _iso(dev, "decision_time", 0),
        "dev_end": _iso(dev, "target_time", -1),
        "seed": SEED,
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "observations": len(dev),
        "hypotheses_examined": FAMILY_SIZE,
        "evidence_role": "development_only",
        "claim_status": status,
        "provisional": provisional,
        "limitations": "DEV_ONLY 2021; supported запрещён; PySR не перезапускался",
        "code_version": _code_version(),
        "coefficients": coefficients,
    }


def _expanding(usable: list[AlignedRow], dev: list[AlignedRow]) -> list[float]:
    ordered = sorted(usable, key=lambda row: (row.decision_time, row.index))
    pointer = 0
    ups = 0
    count = 0
    rates: dict[datetime, float] = {}
    for row in ordered:
        while pointer < len(ordered) and ordered[pointer].target_time <= row.decision_time:
            ups += ordered[pointer].y_up
            count += 1
            pointer += 1
        if row.role == "dev":
            rates[row.decision_time] = 0.5 if count == 0 else ups / count
    return [rates[row.decision_time] for row in dev]


def _train_scale(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    means = []
    scales = []
    count = len(rows)
    for column in range(width):
        values = [row[column] for row in rows]
        mean = sum(values) / count
        if count < 2:
            scale = 0.0
        else:
            variance = sum((value - mean) ** 2 for value in values) / (count - 1)
            scale = variance ** 0.5
        means.append(mean)
        scales.append(scale)
    return means, scales


def _apply(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    scaled = []
    for value, mean, scale in zip(row, means, scales, strict=True):
        if scale < 1e-8:
            scaled.append(0.0)
        else:
            scaled.append((value - mean) / scale)
    return scaled


def _row_values(row: AlignedRow, columns: tuple[str, ...]) -> list[float]:
    values = []
    for name in columns:
        value = row.features[name]
        if value is None:
            raise ValueError(f"пустой признак {name} попал в общую выборку")
        values.append(float(value))
    return values


def _rate(labels: list[int]) -> float:
    if not labels:
        raise ValueError("в обучении нет меток")
    return sum(labels) / len(labels)


def _clip(probability: float) -> float:
    return min(1.0 - PROBABILITY_CLIP, max(PROBABILITY_CLIP, probability))


def _iso(rows: list[AlignedRow], field: str, index: int) -> str:
    if not rows:
        return ""
    moment = getattr(rows[index], field)
    if not isinstance(moment, datetime):
        raise ValueError("в строке нет метки времени")
    return moment.isoformat()


def _code_version() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = completed.stdout.strip()
    return text or "unknown"
