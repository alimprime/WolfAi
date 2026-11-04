"""Сравнение Qwen, PySR и Optuna. Итог 2023–2024 считается только у замороженных.

Победитель не выбирается по одному Brier. Рабочую модель модуль не назначает.
2022 остаётся ограниченной обратной связью, а не независимым тестом.
"""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass

from wolfai.data.candles import Candle
from wolfai.data.history import HistoricalStore
from wolfai.research.engine import (
    MIN_DECISION_ROWS,
    ExperimentMemory,
    ExperimentResult,
    build_experiment_rows,
    run_experiment,
    score_direction,
)
from wolfai.research.formula import FormulaError, evaluate, node_count, parse_formula
from wolfai.research.periods import FEEDBACK, FEEDBACK_NOTE, FINAL, SEARCH_CHECK, chronological_rows
from wolfai.research.spec import ResearchHypothesis, as_engine

MAX_FROZEN = 3
_STABILITY_LIMIT = 0.05


@dataclass(frozen=True)
class RankedCandidate:
    hypothesis: ResearchHypothesis
    check_metrics: dict[str, object] | None
    feedback_metrics: dict[str, object] | None
    stability_gap: float | None
    complexity: int
    eligible: bool
    reason: str
    research_score: float
    final_evaluation: ExperimentResult | None


@dataclass(frozen=True)
class CompetitionResult:
    ranked: tuple[RankedCandidate, ...]
    frozen: tuple[RankedCandidate, ...]
    n_examined: int
    feedback_independent: bool
    feedback_note: str
    elapsed_seconds: float
    max_rss_bytes: int


def selection_margin(n_examined: int) -> float:
    if type(n_examined) is not int or n_examined < 1:
        raise ValueError("число проверенных гипотез должно быть положительным")
    return min(0.05, 0.01 * _log1p(n_examined))


def research_score(
    check: dict[str, object],
    feedback: dict[str, object],
    *,
    complexity: int,
    n_examined: int,
    stability_gap: float | None,
) -> float:
    """Чем хуже окно, тем ниже оценка. Один удачный период победу не даёт."""
    margin = selection_margin(n_examined)
    worst = min(_improvement(check), _improvement(feedback))
    score = worst - 0.002 * complexity - margin
    if stability_gap is not None:
        score -= 0.5 * stability_gap
    return score


def eligibility(
    check: dict[str, object],
    feedback: dict[str, object],
    *,
    n_examined: int,
    stability_gap: float | None,
) -> tuple[bool, str]:
    margin = selection_margin(n_examined)
    if _improvement(check) <= margin:
        return False, "внутреннее окно 2021 не даёт устойчивого преимущества"
    if _improvement(feedback) <= margin:
        return False, "обратная связь 2022 не подтверждает кандидата"
    if stability_gap is not None and stability_gap > _STABILITY_LIMIT:
        return False, "нестабильна на половинах окна обратной связи"
    return True, "кандидат заморожен для одной итоговой оценки"


def compete(
    candidates: list[ResearchHypothesis],
    candles: list[Candle],
    *,
    n_examined: int,
    store: HistoricalStore | None = None,
    source: str | None = None,
    memory: ExperimentMemory | None = None,
) -> CompetitionResult:
    started = time.perf_counter()
    _final_window_closed()
    by_horizon: dict[int, list] = {}
    ranked: list[RankedCandidate] = []
    seen: set[tuple[str, int]] = set()
    for hypothesis in candidates:
        key = (hypothesis.expression, hypothesis.horizon_days)
        if key in seen:
            ranked.append(_plain(hypothesis, eligible=False, reason="дубликат формулы", score=0.0))
            continue
        seen.add(key)
        if hypothesis.status == "active" or hypothesis.origin not in {"qwen", "pysr", "optuna"}:
            ranked.append(_plain(hypothesis, eligible=False, reason="источник не может назначить рабочую модель", score=0.0))
            continue
        try:
            tree = parse_formula(hypothesis.expression)
            rows = by_horizon.setdefault(
                hypothesis.horizon_days,
                build_experiment_rows(candles, hypothesis.horizon_days),
            )
            check_rows = chronological_rows(rows, SEARCH_CHECK[0], SEARCH_CHECK[1], hypothesis.horizon_days)
            feedback_rows = chronological_rows(rows, FEEDBACK[0], FEEDBACK[1], hypothesis.horizon_days)
            check = _metrics(check_rows, tree)
            feedback = _metrics(feedback_rows, tree)
            gap = _stability(feedback_rows, tree)
        except (FormulaError, ValueError) as error:
            ranked.append(_plain(hypothesis, eligible=False, reason=str(error)[:200], score=0.0))
            continue
        complexity = node_count(tree)
        ok, reason = eligibility(check, feedback, n_examined=n_examined, stability_gap=gap)
        score = research_score(
            check,
            feedback,
            complexity=complexity,
            n_examined=n_examined,
            stability_gap=gap,
        )
        ranked.append(RankedCandidate(
            hypothesis=hypothesis,
            check_metrics=check,
            feedback_metrics=feedback,
            stability_gap=gap,
            complexity=complexity,
            eligible=ok,
            reason=reason,
            research_score=score,
            final_evaluation=None,
        ))
    ranked.sort(key=lambda item: item.research_score, reverse=True)
    frozen_ids: list[str] = []
    finalized: list[RankedCandidate] = []
    frozen_count = 0
    for item in ranked:
        if not item.eligible:
            finalized.append(item)
            continue
        if frozen_count >= MAX_FROZEN:
            finalized.append(_copy(item, eligible=False, reason="лимит итоговой оценки", final=None))
            continue
        final = None
        if store is not None and source is not None:
            final = run_experiment(store, source, as_engine(item.hypothesis), memory)
            if final.promoted:
                raise RuntimeError("сравнение не назначает рабочую модель")
        frozen_ids.append(item.hypothesis.id)
        finalized.append(_copy(item, eligible=True, reason=item.reason, final=final))
        frozen_count += 1
    frozen = tuple(item for item in finalized if item.hypothesis.id in frozen_ids)
    return CompetitionResult(
        ranked=tuple(finalized),
        frozen=frozen,
        n_examined=n_examined,
        feedback_independent=False,
        feedback_note=FEEDBACK_NOTE,
        elapsed_seconds=round(time.perf_counter() - started, 6),
        max_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )


def _metrics(rows, tree) -> dict[str, object]:
    if len(rows) < MIN_DECISION_ROWS:
        raise ValueError("в окне сравнения слишком мало точек")
    probabilities = [evaluate(tree, row.features) for row in rows]
    scored = score_direction(
        probabilities,
        [row.y_up for row in rows],
        [row.baseline_frequency for row in rows],
    )
    skill = scored["skill"]
    if not isinstance(skill, dict):
        raise ValueError("сравнение с ориентиром не посчитано")
    return {
        "observations": int(scored["observations"]),
        "brier": float(scored["brier"]),
        "log_loss": float(scored["log_loss"]),
        "frequency_brier": float(skill["historical_frequency"]["reference_brier"]),
        "coin_brier": float(skill["constant_half"]["reference_brier"]),
        "calibration_bins": len(scored["calibration"]) if isinstance(scored["calibration"], list) else 0,
    }


def _stability(rows, tree) -> float | None:
    if len(rows) < 20:
        return None
    middle = len(rows) // 2
    halves = (rows[:middle], rows[middle:])
    if any(len(half) < 10 for half in halves):
        return None
    try:
        scores = []
        for half in halves:
            if len(half) >= MIN_DECISION_ROWS:
                metrics = _metrics(half, tree)
            else:
                metrics = _loose_brier(half, tree)
            scores.append(float(metrics["brier"]))
        return abs(scores[0] - scores[1])
    except (FormulaError, ValueError):
        return None


def _loose_brier(rows, tree) -> dict[str, object]:
    probabilities = [evaluate(tree, row.features) for row in rows]
    labels = [row.y_up for row in rows]
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(rows)
    return {"brier": brier}


def _improvement(metrics: dict[str, object]) -> float:
    return min(
        float(metrics["frequency_brier"]) - float(metrics["brier"]),
        float(metrics["coin_brier"]) - float(metrics["brier"]),
    )


def _log1p(value: int) -> float:
    return __import__("math").log1p(value)


def _plain(hypothesis: ResearchHypothesis, *, eligible: bool, reason: str, score: float) -> RankedCandidate:
    return RankedCandidate(
        hypothesis=hypothesis,
        check_metrics=None,
        feedback_metrics=None,
        stability_gap=None,
        complexity=0,
        eligible=eligible,
        reason=reason,
        research_score=score,
        final_evaluation=None,
    )


def _copy(
    item: RankedCandidate,
    *,
    eligible: bool,
    reason: str,
    final: ExperimentResult | None,
) -> RankedCandidate:
    return RankedCandidate(
        hypothesis=item.hypothesis,
        check_metrics=item.check_metrics,
        feedback_metrics=item.feedback_metrics,
        stability_gap=item.stability_gap,
        complexity=item.complexity,
        eligible=eligible,
        reason=reason,
        research_score=item.research_score,
        final_evaluation=final,
    )


def _final_window_closed() -> None:
    if FINAL[0].year != 2023:
        raise RuntimeError("граница итоговой оценки сдвинута")
