"""Сравнение групп информации на 2019–2020 и диагностическом 2021.

Новый поиск PySR, Optuna и Qwen не запускаются. Пропуск внешнего ряда
не заполняется. Группы без доказанного времени доступности не оцениваются.
Результат funding и open interest за 2022 год не пересчитывается.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

from wolfai.data.candles import Candle
from wolfai.research.external_join import derivative_features, load_strict_derivatives
from wolfai.research.horizon_labels import AlignedRow, build_aligned_rows, complete_rows, rows_by_role
from wolfai.research.logistic import LOGISTIC_L2, fit_l2_logistic, predict_probability, score_probabilities
from wolfai.research.market_activity import volume_to_move
from wolfai.research.ohlcv_features import OHLCV_FEATURES, PRICE_FEATURES
from wolfai.research.quality_contract import DIAGNOSTIC_HORIZONS, FORBIDDEN_FROM
from wolfai.research.reliability import assess_reliability
from wolfai.research.research_memory import Stage24Memory
from wolfai.research.source_audit import audit_local
from wolfai.research.source_record import CUTOFF, MetricRecord
from wolfai.research.stage_data import load_stage_candles
from wolfai.research.time_alignment import LABEL_RULE

MIN_TRAIN = 60
MIN_DEV = 30
ACTIVITY_EXTRA = ("volume_to_move",)
DERIVATIVE_FEATURES = ("funding_rate", "funding_change", "oi_log_change")
PRICE_OHLCV = PRICE_FEATURES + OHLCV_FEATURES
PRESERVED_2022 = {
    "window": "2022",
    "decisions": 362,
    "constant_brier": 0.25000,
    "frequency_brier": 0.25287,
    "price_brier": 0.24970,
    "price_log_loss": 0.69255,
    "price_plus_funding_oi_brier": 0.25091,
    "price_plus_funding_oi_log_loss": 0.69500,
    "note": (
        "Сохранённый результат docs/INTELLIGENCE.md от 2026-10-26. "
        "Этот этап его не пересчитывает и не заменяет диагностикой 2021 года."
    ),
}


def run_expansion(
    csv_path: str | Path,
    observations_db: str | Path | None,
    memory_path: str | Path,
    cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Один проход. Свечи с датой открытия 2022-01-01 не разбираются."""
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    candles = load_stage_candles(csv_path)
    _reject_prices(candles)
    derivatives = load_strict_derivatives(str(observations_db)) if observations_db else []
    findings = audit_local(None if observations_db is None else str(observations_db))
    report = evaluate_groups(candles, derivatives)
    report["sources"] = [finding.__dict__ for finding in findings]
    report["preserved_2022"] = dict(PRESERVED_2022)
    report["pysr_ran"] = False
    report["optuna_ran"] = False
    report["qwen_ran"] = False
    report["derivative_records_before_2022"] = len(derivatives)
    memory = Stage24Memory(memory_path)
    try:
        report["memory"] = _store(memory, report)
    finally:
        memory.close()
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    cpu_ended = resource.getrusage(resource.RUSAGE_SELF)
    report["cpu_seconds"] = round(
        (cpu_ended.ru_utime + cpu_ended.ru_stime) - (cpu_started.ru_utime + cpu_started.ru_stime),
        3,
    )
    report["max_rss"] = int(cpu_ended.ru_maxrss)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def evaluate_groups(candles: list[Candle], derivatives: list[MetricRecord]) -> dict[str, object]:
    """Сравнение без записи на диск. Все логистики считаются после правила покрытия."""
    _reject_prices(candles)
    for record in derivatives:
        if record.event_time >= CUTOFF or (record.available_at is not None and record.available_at >= CUTOFF):
            raise ValueError("в признаки попала метка 2022-01-01 или позже")
    prepared = []
    for horizon in DIAGNOSTIC_HORIZONS:
        rows = complete_rows(build_aligned_rows(candles, horizon))
        enriched = [_with_activity(row) for row in rows]
        train = rows_by_role(enriched, "train")
        dev = rows_by_role(enriched, "dev")
        derivative_train = [row for row in train if _derivatives(derivatives, row) is not None]
        derivative_dev = [row for row in dev if _derivatives(derivatives, row) is not None]
        prepared.append({
            "horizon": horizon,
            "train": train,
            "dev": dev,
            "derivative_train": derivative_train,
            "derivative_dev": derivative_dev,
            "derivatives_eligible": len(derivative_train) >= MIN_TRAIN and len(derivative_dev) >= MIN_DEV,
        })
    planned = 0
    for item in prepared:
        if len(item["train"]) >= MIN_TRAIN and len(item["dev"]) >= MIN_DEV:
            planned += 2
        if item["derivatives_eligible"]:
            planned += 2
    horizons = [_score_horizon(item, derivatives, planned) for item in prepared]
    return {
        "seed": 1729,
        "hypotheses_fitted": planned,
        "regularization": LOGISTIC_L2,
        "label_rule": LABEL_RULE,
        "groups": {
            "A_market_activity": "оценивается, если в окне достаточно строк",
            "B_derivatives": "только на датах, где funding и open interest уже известны; сравнение 2022 не повторяется",
            "C_onchain": "не оценивается: нет point-in-time ряда",
            "D_macro": "не оценивается: нет часа первой публикации",
            "E_all": "не оценивается: группы C и D нельзя заполнить нулями",
        },
        "horizons": horizons,
    }


def _score_horizon(item: dict[str, object], derivatives: list[MetricRecord], planned: int) -> dict[str, object]:
    horizon = int(item["horizon"])
    train = item["train"]
    dev = item["dev"]
    assert isinstance(train, list) and isinstance(dev, list)
    models: list[dict[str, object]] = []
    if len(train) >= MIN_TRAIN and len(dev) >= MIN_DEV:
        frozen = sum(row.y_up for row in train) / len(train)
        models.extend(_pair(
            train,
            dev,
            frozen,
            horizon,
            planned,
            left_name="logistic_price_ohlcv",
            left_columns=PRICE_OHLCV,
            right_name="logistic_market_activity",
            right_columns=PRICE_OHLCV + ACTIVITY_EXTRA,
            sample="ohlcv",
        ))
    derivative_note = "покрытие меньше порога: модель не подгонялась и пропуски не заполнялись"
    if item["derivatives_eligible"]:
        d_train = item["derivative_train"]
        d_dev = item["derivative_dev"]
        assert isinstance(d_train, list) and isinstance(d_dev, list)
        attached_train = [_attach(row, derivatives) for row in d_train]
        attached_dev = [_attach(row, derivatives) for row in d_dev]
        frozen = sum(row.y_up for row in attached_train) / len(attached_train)
        models.extend(_pair(
            attached_train,
            attached_dev,
            frozen,
            horizon,
            planned,
            left_name="logistic_price_ohlcv_matched",
            left_columns=PRICE_OHLCV,
            right_name="logistic_price_derivatives",
            right_columns=PRICE_OHLCV + DERIVATIVE_FEATURES,
            sample="derivatives_intersection",
        ))
        derivative_note = "сравнение только на пересечении дат"
    counts = {model["count"] for model in models if model["sample"] == "ohlcv"}
    if len(counts) > 1:
        raise ValueError("модели группы A посчитаны на разных датах")
    matched = {model["count"] for model in models if model["sample"] == "derivatives_intersection"}
    if len(matched) > 1:
        raise ValueError("модели группы B посчитаны на разных датах")
    return {
        "horizon_days": horizon,
        "train_count": len(train),
        "dev_count": len(dev),
        "derivative_train_count": len(item["derivative_train"]),
        "derivative_dev_count": len(item["derivative_dev"]),
        "derivative_coverage_dev": 0.0 if not dev else len(item["derivative_dev"]) / len(dev),
        "derivative_note": derivative_note,
        "onchain": "not_evaluated",
        "macro": "not_evaluated",
        "all_groups": "not_evaluated",
        "models": models,
    }


def _pair(
    train: list[AlignedRow],
    dev: list[AlignedRow],
    frozen: float,
    horizon: int,
    planned: int,
    *,
    left_name: str,
    left_columns: tuple[str, ...],
    right_name: str,
    right_columns: tuple[str, ...],
    sample: str,
) -> list[dict[str, object]]:
    frequency = [_clip(frozen)] * len(dev)
    labels = [row.y_up for row in dev]
    dates = [row.decision_time.date() for row in dev]
    rows = [
        _score(left_name, left_columns, train, dev, dates, labels, frequency, horizon, planned, sample),
        _score(right_name, right_columns, train, dev, dates, labels, frequency, horizon, planned, sample),
        _reference("constant_half", [0.5] * len(dev), dates, labels, frequency, horizon, planned, sample),
        _reference("frozen_train_frequency", frequency, dates, labels, frequency, horizon, planned, sample),
    ]
    if len({item["count"] for item in rows}) != 1:
        raise ValueError("пара моделей разошлась по числу дат")
    return rows


def _score(
    name: str,
    columns: tuple[str, ...],
    train: list[AlignedRow],
    dev: list[AlignedRow],
    dates: list,
    labels: list[int],
    frequency: list[float],
    horizon: int,
    planned: int,
    sample: str,
) -> dict[str, object]:
    probabilities, coefficients = _logistic(train, dev, columns)
    return _pack(name, columns, probabilities, coefficients, dates, labels, frequency, horizon, planned, sample)


def _reference(
    name: str,
    probabilities: list[float],
    dates: list,
    labels: list[int],
    frequency: list[float],
    horizon: int,
    planned: int,
    sample: str,
) -> dict[str, object]:
    return _pack(name, (), probabilities, None, dates, labels, frequency, horizon, planned, sample)


def _pack(
    name: str,
    columns: tuple[str, ...],
    probabilities: list[float],
    coefficients: list[float] | None,
    dates: list,
    labels: list[int],
    frequency: list[float],
    horizon: int,
    planned: int,
    sample: str,
) -> dict[str, object]:
    scored = score_probabilities(
        probabilities,
        labels,
        {"historical_frequency": frequency, "constant_half": [0.5] * len(labels)},
    )
    reliability = assess_reliability(
        dates,
        probabilities,
        labels,
        frequency,
        horizon_days=horizon,
        complexity=max(1, len(columns)),
        hypotheses_examined=planned,
        selection_separated=True,
        evidence_role="development_only",
    )
    uncertainty = reliability.uncertainty
    return {
        "name": name,
        "sample": sample,
        "features": list(columns),
        "count": len(labels),
        "brier": scored["brier"],
        "log_loss": scored["log_loss"],
        "base_rate": scored["base_rate"],
        "delta_vs_coin": None if uncertainty is None else uncertainty.mean_improvement_vs_coin,
        "low_vs_coin": None if uncertainty is None else uncertainty.low_vs_coin,
        "high_vs_coin": None if uncertainty is None else uncertainty.high_vs_coin,
        "delta_vs_frequency": None if uncertainty is None else uncertainty.mean_improvement_vs_frequency,
        "low_vs_frequency": None if uncertainty is None else uncertainty.low_vs_frequency,
        "high_vs_frequency": None if uncertainty is None else uncertainty.high_vs_frequency,
        "block_length": None if uncertainty is None else uncertainty.block_length,
        "claim_status": reliability.status,
        "provisional": reliability.provisional,
        "evidence_role": reliability.evidence_role,
        "calibration_occupied": sum(1 for item in reliability.calibration if item.count),
        "coefficients": coefficients,
    }


def _logistic(
    train: list[AlignedRow],
    dev: list[AlignedRow],
    columns: tuple[str, ...],
) -> tuple[list[float], list[float]]:
    train_x = [_values(row, columns) for row in train]
    means, scales = _scale(train_x)
    beta = fit_l2_logistic([_apply(row, means, scales) for row in train_x], [row.y_up for row in train])
    probabilities = [
        predict_probability(beta, _apply(_values(row, columns), means, scales))
        for row in dev
    ]
    return probabilities, beta


def _with_activity(row: AlignedRow) -> AlignedRow:
    features = dict(row.features)
    volume_change = features["volume_change"]
    log_return = features["log_return"]
    if volume_change is None or log_return is None:
        features["volume_to_move"] = None
    else:
        features["volume_to_move"] = volume_to_move(float(volume_change), float(log_return))
    return AlignedRow(
        index=row.index,
        open_time=row.open_time,
        decision_time=row.decision_time,
        target_open_time=row.target_open_time,
        target_time=row.target_time,
        horizon_days=row.horizon_days,
        y_up=row.y_up,
        features=features,
        exclusion=row.exclusion,
        role=row.role,
    )


def _derivatives(records: list[MetricRecord], row: AlignedRow) -> dict[str, float] | None:
    return derivative_features(records, row.decision_time)


def _attach(row: AlignedRow, records: list[MetricRecord]) -> AlignedRow:
    extra = derivative_features(records, row.decision_time)
    if extra is None:
        raise ValueError("в пересечение попала строка без деривативов")
    features = dict(row.features)
    features.update(extra)
    return AlignedRow(
        index=row.index,
        open_time=row.open_time,
        decision_time=row.decision_time,
        target_open_time=row.target_open_time,
        target_time=row.target_time,
        horizon_days=row.horizon_days,
        y_up=row.y_up,
        features=features,
        exclusion=row.exclusion,
        role=row.role,
    )


def _values(row: AlignedRow, columns: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for name in columns:
        value = row.features.get(name)
        if value is None:
            raise ValueError(f"пустой признак {name}")
        values.append(float(value))
    return values


def _scale(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    width = len(rows[0])
    count = len(rows)
    means: list[float] = []
    scales: list[float] = []
    for column in range(width):
        values = [row[column] for row in rows]
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        means.append(mean)
        scales.append(variance ** 0.5)
    return means, scales


def _apply(row: list[float], means: list[float], scales: list[float]) -> list[float]:
    scaled = []
    for value, mean, scale in zip(row, means, scales, strict=True):
        scaled.append(0.0 if scale < 1e-8 else (value - mean) / scale)
    return scaled


def _clip(probability: float) -> float:
    return min(1.0 - 1e-6, max(1e-6, probability))


def _reject_prices(candles: list[Candle]) -> None:
    for candle in candles:
        if candle.timestamp.date() >= FORBIDDEN_FROM:
            raise ValueError("в диагностику попала свеча 2022 года")


def _store(memory: Stage24Memory, report: dict[str, object]) -> dict[str, int]:
    stored = 0
    duplicates = 0
    preserved = _preserved_payload()
    for layer in ("research", "candidate"):
        state = memory.remember(layer, preserved)
        if state == "stored":
            stored += 1
        else:
            duplicates += 1
    horizons = report["horizons"]
    assert isinstance(horizons, list)
    for horizon in horizons:
        assert isinstance(horizon, dict)
        for model in horizon["models"]:
            assert isinstance(model, dict)
            if not str(model["name"]).startswith("logistic"):
                continue
            payload = _model_payload(horizon, model, report)
            for layer in ("research", "candidate"):
                state = memory.remember(layer, payload)
                if state == "stored":
                    stored += 1
                else:
                    duplicates += 1
    return {"stored": stored, "duplicates": duplicates}


def _preserved_payload() -> dict[str, object]:
    return {
        "model_name": "preserved_2022_price_plus_funding_oi",
        "config_hash": "intelligence-2026-10-26",
        "dataset_fingerprint": "not-recomputed",
        "features": ["price", "funding_rate", "oi_log_change_1d"],
        "label_definition": LABEL_RULE,
        "horizon_days": 1,
        "train_start": "2020-10",
        "train_end": "2021-12-31",
        "dev_start": "2022-01-01",
        "dev_end": "2022-12-31",
        "seed": 1729,
        "source_version": "docs/INTELLIGENCE.md",
        "feature_group": "preserved_2022",
        "brier": PRESERVED_2022["price_plus_funding_oi_brier"],
        "log_loss": PRESERVED_2022["price_plus_funding_oi_log_loss"],
        "evidence_role": "prior_window_2022",
        "claim_status": "insufficient_evidence",
        "limitations": PRESERVED_2022["note"],
        "hypotheses_examined": 1,
    }


def _model_payload(horizon: dict[str, object], model: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    return {
        "model_name": model["name"],
        "config_hash": f"l2={LOGISTIC_L2}|seed=1729|sample={model['sample']}",
        "dataset_fingerprint": "stage-candles-before-2022",
        "features": model["features"],
        "label_definition": LABEL_RULE,
        "horizon_days": horizon["horizon_days"],
        "train_start": "2019-2020",
        "train_end": "target_time<=2020-12-31T23:59:59Z",
        "dev_start": "2021-01-01",
        "dev_end": "target_time<2022-01-01",
        "seed": 1729,
        "source_version": "local-observation-store-before-2022",
        "feature_group": model["sample"],
        "brier": model["brier"],
        "log_loss": model["log_loss"],
        "observations": model["count"],
        "delta_vs_coin": model["delta_vs_coin"],
        "low_vs_coin": model["low_vs_coin"],
        "high_vs_coin": model["high_vs_coin"],
        "block_length": model["block_length"],
        "evidence_role": "development_only",
        "claim_status": model["claim_status"],
        "provisional": model["provisional"],
        "hypotheses_examined": report["hypotheses_fitted"],
        "limitations": "DEV_ONLY 2021; supported запрещён; группы C, D и E не заполнялись",
    }
