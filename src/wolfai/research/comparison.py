"""Сравнение вариантов на одних и тех же окнах 2019–2021.

Один Brier одного запуска победителя не назначает. 2022, 2023–2024 и
holdout в этот расчёт не входят.
"""

from __future__ import annotations

import resource
import time
from datetime import datetime, timezone
from pathlib import Path

from wolfai.data.candles import Candle
from wolfai.research.efficiency import (
    CheckView,
    Proposal,
    SearchBudget,
    TrainView,
    research_cycle,
)
from wolfai.research.engine import ExperimentMemory, ExperimentRow, build_experiment_rows
from wolfai.research.feature_quality import audit_features
from wolfai.research.formula import evaluate, node_count, parse_formula
from wolfai.research.logistic import score_probabilities
from wolfai.research.periods import SEARCH_CHECK, SEARCH_TRAIN, chronological_rows
from wolfai.research.quality_contract import (
    FORBIDDEN_FROM,
    STATUS_SUPPORTED,
    TimedRow,
    stage_window,
)
from wolfai.research.regimes import label_regimes, regime_report
from wolfai.research.reliability import assess_reliability
from wolfai.research.symbolic_strategy import examination_count

CREATED_AT = datetime(2026, 11, 4, tzinfo=timezone.utc)
SIGNAL_EXPRESSION = "sigmoid(-4 * log_return)"


def split_stage(
    candles: list[Candle],
    horizon_days: int,
) -> tuple[list[ExperimentRow], list[ExperimentRow]]:
    """Подбор 2019–2020 и проверка 2021. Более поздняя свеча отклоняется."""
    for candle in candles:
        if candle.timestamp.date() >= FORBIDDEN_FROM:
            raise ValueError("свеча этапа не может начинаться с 2022-01-01")
    built = build_experiment_rows(candles, horizon_days)
    fit = chronological_rows(built, SEARCH_TRAIN[0], SEARCH_TRAIN[1], horizon_days)
    check = chronological_rows(built, SEARCH_CHECK[0], SEARCH_CHECK[1], horizon_days)
    return fit, check


def to_timed(rows: list[ExperimentRow]) -> list[TimedRow]:
    timed: list[TimedRow] = []
    for row in rows:
        if stage_window(row.decision_date) != stage_window(row.target_date):
            continue
        timed.append(TimedRow(
            decision_date=row.decision_date,
            target_date=row.target_date,
            features=dict(row.features),
            y_up=row.y_up,
            baseline_frequency=row.baseline_frequency,
        ))
    return timed


def train_view(rows: list[ExperimentRow], expression: str) -> TrainView:
    """Brier только на переданном окне. Это внутренний фильтр, не итог."""
    if len(rows) < 2:
        raise ValueError("для внутреннего фильтра слишком мало точек")
    tree = parse_formula(expression)
    probabilities = [evaluate(tree, row.features) for row in rows]
    labels = [row.y_up for row in rows]
    frequency = [row.baseline_frequency for row in rows]
    scored = score_probabilities(
        probabilities,
        labels,
        {
            "historical_frequency": frequency,
            "constant_half": [0.5] * len(labels),
        },
    )
    return TrainView(
        brier=_number(scored, "brier"),
        frequency_brier=_skill(scored, "historical_frequency"),
        coin_brier=_skill(scored, "constant_half"),
    )


def score_expression(
    rows: list[ExperimentRow],
    expression: str,
    *,
    horizon_days: int,
    hypotheses_examined: int,
    selection_separated: bool = True,
):
    """Надёжность готовой формулы. Новую формулу функция не подбирает."""
    tree = parse_formula(expression)
    return assess_reliability(
        [row.decision_date for row in rows],
        [evaluate(tree, row.features) for row in rows],
        [row.y_up for row in rows],
        [row.baseline_frequency for row in rows],
        horizon_days=horizon_days,
        complexity=node_count(tree),
        hypotheses_examined=hypotheses_examined,
        selection_separated=selection_separated,
    )


def score_constant(
    rows: list[ExperimentRow],
    probability: float,
    *,
    horizon_days: int,
    hypotheses_examined: int,
):
    return assess_reliability(
        [row.decision_date for row in rows],
        [probability] * len(rows),
        [row.y_up for row in rows],
        [row.baseline_frequency for row in rows],
        horizon_days=horizon_days,
        complexity=1,
        hypotheses_examined=hypotheses_examined,
        selection_separated=True,
    )


def known_pattern_status(candles: list[Candle]) -> dict[str, str]:
    """Сильный синтетический знак и константа 0.5. PySR здесь не запускается."""
    _fit, check = split_stage(candles, 1)
    signal = score_expression(
        check,
        SIGNAL_EXPRESSION,
        horizon_days=1,
        hypotheses_examined=1,
    )
    noise = score_constant(check, 0.5, horizon_days=1, hypotheses_examined=1)
    return {"signal": signal.status, "noise": noise.status}


def run_limited_btc(path: str | Path) -> dict[str, object]:
    """Один процесс: сначала исходный PySR, затем улучшенный, тот же бюджет.

    Признаки и режимы не запускают второй поиск. Рабочая модель не назначается.
    """
    from wolfai.research.pysr_engine import discover_report
    from wolfai.research.quality_contract import AUDITED_FEATURES
    from wolfai.research.stage_data import load_stage_candles

    started = time.perf_counter()
    candles = load_stage_candles(path)
    fit, check = split_stage(candles, 1)
    timed = to_timed(fit + check)
    audit = audit_features(timed, horizon_days=1)
    regimes = regime_report(timed, None, horizon_days=1)
    feature_rows = [row.features for row in fit]
    labels = [row.y_up for row in fit]
    common = {
        "feature_names": AUDITED_FEATURES,
        "horizon_days": 1,
        "dataset_version": "btc-stage-232",
        "created_at": CREATED_AT,
        "seed": 1729,
        "max_candidates": 5,
    }
    baseline_started = time.perf_counter()
    baseline = discover_report(feature_rows, labels, strategy="baseline", **common)
    baseline_seconds = time.perf_counter() - baseline_started
    improved_started = time.perf_counter()
    improved = discover_report(feature_rows, labels, strategy="improved", **common)
    improved_seconds = time.perf_counter() - improved_started
    memory = ExperimentMemory(Path(path).with_suffix(".stage232.sqlite"))
    try:
        baseline_cycle = _cycle(baseline.hypotheses, fit, check, baseline.examined, "baseline", memory)
        improved_cycle = _cycle(improved.hypotheses, fit, check, improved.examined, "improved", memory)
        repeated = _cycle(baseline.hypotheses, fit, check, baseline.examined, "baseline", memory)
    finally:
        memory.close()
    regime_notes = _regime_notes(check, baseline.hypotheses, baseline.examined, regimes.thresholds)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "winner": None,
        "winner_note": "Одного запуска недостаточно, чтобы назвать победителя.",
        "candles": len(candles),
        "last_candle": candles[-1].timestamp.date().isoformat(),
        "fit_rows": len(fit),
        "check_rows": len(check),
        "features": [
            {
                "name": item.name,
                "kept": item.kept,
                "predictive_claim": item.usefulness.predictive_claim,
                "brier": item.usefulness.brier,
                "log_loss": item.usefulness.log_loss,
                "improvement_vs_frequency": item.usefulness.brier_improvement_vs_frequency,
                "improvement_vs_coin": item.usefulness.brier_improvement_vs_coin,
                "anomalies": item.distribution.anomalies,
                "check_anomalies": item.check_anomalies,
                "stability_shift": item.stability.standardized_shift,
                "missing": item.distribution.missing,
                "non_finite": item.distribution.non_finite,
            }
            for item in audit.features
        ],
        "redundancy": [
            {"left": item.left, "right": item.right, "rank_agreement": item.rank_agreement}
            for item in audit.redundancy
        ],
        "dropped_features": list(audit.dropped_features),
        "regimes": {
            "status": regimes.status,
            "separate_model": regimes.separate_model,
            "thresholds": {
                "volatility_low": regimes.thresholds.volatility_low,
                "volatility_high": regimes.thresholds.volatility_high,
                "trend_band": regimes.thresholds.trend_band,
                "fit_observations": regimes.thresholds.fit_observations,
            },
            "check_slices": [
                {"state": item.state, "observations": item.observations, "status": item.status}
                for item in regimes.slices
                if item.window == "check"
            ],
            "authorization": regime_notes,
        },
        "baseline": _variant(baseline, baseline_cycle, baseline_seconds),
        "improved": _variant(improved, improved_cycle, improved_seconds),
        "cache_hits_on_repeat": repeated.cache_hits,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_rss": int(rss),
        "examination_note": (
            "K — слоты популяций PySR, если их больше, чем возвращённых уравнений. "
            "Это осторожная поправка, не p-value."
        ),
    }


def _cycle(hypotheses, fit, check, examined, config_id, memory):
    proposals = [
        Proposal(item.expression, "pysr", None, config_id)
        for item in hypotheses
    ]
    if not proposals:
        return research_cycle(
            (),
            budget=SearchBudget(1, 30),
            dataset_checksum="btc-stage-232",
            horizon_days=1,
            config_id=config_id,
            memory=memory,
            train_metrics=lambda expression: train_view(fit, expression),
            check_metrics=lambda expression: _check_view(check, expression, examination_count(examined)),
        )
    return research_cycle(
        proposals,
        budget=SearchBudget(max(len(proposals), 1), 60),
        dataset_checksum="btc-stage-232",
        horizon_days=1,
        config_id=config_id,
        memory=memory,
        train_metrics=lambda expression: train_view(fit, expression),
        check_metrics=lambda expression: _check_view(check, expression, examination_count(examined)),
    )


def _check_view(rows, expression: str, examined: int) -> CheckView:
    report = score_expression(
        rows,
        expression,
        horizon_days=1,
        hypotheses_examined=examined,
    )
    return CheckView(
        brier=float(report.brier or 0.0),
        log_loss=float(report.log_loss or 0.0),
        status=report.status,
    )


def _variant(assembly, cycle, seconds: float) -> dict[str, object]:
    return {
        "selector": assembly.selector,
        "examined_equations": assembly.examined,
        "hypotheses_for_reliability": examination_count(assembly.examined),
        "kept": [item.expression for item in assembly.hypotheses],
        "rejected": [{"equation": item.equation, "reason": item.reason} for item in assembly.rejected],
        "unique_structures": assembly.unique_structures,
        "pareto_size": assembly.pareto_size,
        "scored": [
            {"expression": expression, "brier": view.brier, "log_loss": view.log_loss, "status": view.status}
            for expression, view in cycle.scored
        ],
        "lineage": [record.outcome for record in cycle.lineage],
        "elapsed_seconds": round(seconds, 3),
    }


def _regime_notes(check, hypotheses, examined, thresholds) -> dict[str, object]:
    """Срез режима не становится отдельной моделью без статуса supported."""
    if not hypotheses:
        return {"separate_model": False, "supported_states": []}
    expression = hypotheses[0].expression
    timed = to_timed(check)
    points = label_regimes(timed, thresholds)
    tree = parse_formula(expression)
    probabilities = [evaluate(tree, row.features) for row in timed]
    supported: list[str] = []
    details: list[dict[str, object]] = []
    examined_regimes = examination_count(examined) * 5
    for state in ("high_volatility", "low_volatility", "uptrend", "downtrend", "sideways"):
        chosen = [
            index
            for index, point in enumerate(points)
            if point.volatility_state == state or point.trend_state == state
        ]
        if len(chosen) < 40:
            details.append({"state": state, "observations": len(chosen), "status": "insufficient_evidence"})
            continue
        report = assess_reliability(
            [timed[index].decision_date for index in chosen],
            [probabilities[index] for index in chosen],
            [timed[index].y_up for index in chosen],
            [timed[index].baseline_frequency for index in chosen],
            horizon_days=1,
            complexity=node_count(tree),
            hypotheses_examined=examined_regimes,
            selection_separated=True,
        )
        details.append({
            "state": state,
            "observations": len(chosen),
            "status": report.status,
            "brier": report.brier,
        })
        if report.status == STATUS_SUPPORTED:
            supported.append(state)
    return {
        "separate_model": False,
        "supported_states": supported,
        "slices": details,
        "note": "Даже supported-срез не назначается рабочей моделью.",
    }


def _number(scored: dict[str, object], key: str) -> float:
    value = scored[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("метрика вероятности не собрана")
    return float(value)


def _skill(scored: dict[str, object], name: str) -> float:
    skill = scored["skill"]
    if not isinstance(skill, dict):
        raise ValueError("метрика вероятности не собрана")
    block = skill[name]
    if not isinstance(block, dict):
        raise ValueError("метрика вероятности не собрана")
    value = block["reference_brier"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("метрика вероятности не собрана")
    return float(value)
