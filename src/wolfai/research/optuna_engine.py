"""Подбор коэффициентов. Структура формулы остаётся прежней.

Потеря — log loss вероятности на 2019–2020. Выбор испытания — по log loss
окна 2021. 2022 и 2023–2024 в испытания не входят. Optuna используется,
если пакет уже установлен; иначе детерминированная сетка с тем же бюджетом.
По умолчанию не больше 20 испытаний.
"""

from __future__ import annotations

import importlib.util
import random
import resource
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from wolfai.data.candles import Candle
from wolfai.research.engine import (
    MIN_DECISION_ROWS,
    ExperimentMemory,
    build_experiment_rows,
    score_direction,
)
from wolfai.research.formula import FormulaError, evaluate, parse_formula
from wolfai.research.formula_edit import list_coefficients, render, replace_coefficients, skeleton
from wolfai.research.periods import SEARCH_CHECK, SEARCH_TRAIN, assert_search_window, chronological_rows
from wolfai.research.spec import DISCOVERY_VERSION, ORIGIN_OPTUNA, ResearchHypothesis, bind_hypothesis

DEFAULT_TRIALS = 20
MAX_TRIALS = 20
DEFAULT_SEED = 1729
DEFAULT_TIMEOUT_SECONDS = 30.0
_GRID = (-10.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 10.0)


@dataclass(frozen=True)
class TrialRecord:
    number: int
    parameters: tuple[float, ...]
    state: str
    reason: str
    train_log_loss: float | None
    check_log_loss: float | None


@dataclass(frozen=True)
class OptimizationResult:
    hypothesis: ResearchHypothesis
    trials: tuple[TrialRecord, ...]
    backend: str
    elapsed_seconds: float
    max_rss_bytes: int
    reused: bool
    period_errors: dict[str, object]
    reason: str


class ProposalPlan:
    def propose(self, index: int) -> tuple[float, ...] | None:
        raise NotImplementedError


class GridPlan(ProposalPlan):
    def __init__(self, start: tuple[float, ...], n_trials: int, seed: int) -> None:
        self.points = _grid_points(start, n_trials, seed)

    def propose(self, index: int) -> tuple[float, ...] | None:
        if index >= len(self.points):
            return None
        return self.points[index]


def optimize_coefficients(
    candles: list[Candle],
    expression: str,
    features: tuple[str, ...] | list[str],
    *,
    horizon_days: int = 1,
    dataset_version: str,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory: ExperimentMemory | None = None,
    plan: ProposalPlan | None = None,
    created_at: datetime | None = None,
) -> OptimizationResult:
    if type(n_trials) is not int or not 1 <= n_trials <= MAX_TRIALS:
        raise ValueError("на гипотезу допускается от 1 до 20 испытаний")
    if type(seed) is not int:
        raise ValueError("seed должен быть целым")
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    names = tuple(features)
    tree = parse_formula(expression)
    start = list_coefficients(tree)
    backend = "grid" if plan is None and not _optuna_installed() else "optuna"
    if plan is not None:
        backend = "grid"
    settings = _settings(backend, timeout_seconds)
    key = _key(expression, names, horizon_days, dataset_version, seed, n_trials, settings)
    if memory is not None:
        cached = memory.find_discovery(key)
        if cached is not None:
            return _from_payload(cached, reused=True)
    if not start:
        moment = created_at or datetime.now(timezone.utc)
        hypothesis = bind_hypothesis(
            id=f"optuna-{uuid.uuid4().hex[:12]}",
            origin=ORIGIN_OPTUNA,
            expression=expression,
            features=names,
            horizon_days=horizon_days,
            dataset_version=dataset_version,
            created_at=moment,
            source_expression=expression,
        )
        result = OptimizationResult(
            hypothesis=hypothesis,
            trials=(),
            backend=backend,
            elapsed_seconds=0.0,
            max_rss_bytes=_rss(),
            reused=False,
            period_errors={},
            reason="нет настраиваемых коэффициентов",
        )
        _store(memory, key, result)
        return result
    rows = build_experiment_rows(candles, horizon_days)
    started = time.perf_counter()
    if plan is None and _optuna_installed():
        plan = _OptunaPlan(len(start), seed)
        backend = "optuna"
    elif plan is None:
        plan = GridPlan(start, n_trials, seed)
        backend = "grid"
    trials: list[TrialRecord] = []
    best_check: float | None = None
    best_expression = expression
    best_parameters = start
    best_periods: dict[str, object] = {}
    while len(trials) < n_trials:
        if trials and time.perf_counter() - started >= timeout_seconds:
            break
        point = start if not trials else plan.propose(len(trials))
        if point is None:
            break
        record, periods, rendered = _trial(tree, point, rows, len(trials))
        trials.append(record)
        if record.state == "complete" and record.check_log_loss is not None and (
            best_check is None or record.check_log_loss < best_check
        ):
            best_check = record.check_log_loss
            best_expression = rendered
            best_parameters = record.parameters
            best_periods = periods
        tell = getattr(plan, "tell", None)
        if tell is not None and point is not start:
            tell(record.train_log_loss if record.state == "complete" else None)
    if not any(trial.state == "complete" for trial in trials):
        raise ValueError(trials[-1].reason if trials else "подбор не выполнен")
    moment = created_at or datetime.now(timezone.utc)
    hypothesis = bind_hypothesis(
        id=f"optuna-{uuid.uuid4().hex[:12]}",
        origin=ORIGIN_OPTUNA,
        expression=best_expression,
        features=names,
        horizon_days=horizon_days,
        dataset_version=dataset_version,
        created_at=moment,
        parameters=best_parameters,
        source_expression=expression,
    )
    improved = tuple(round(value, 4) for value in list_coefficients(parse_formula(best_expression))) != tuple(
        round(value, 4) for value in start
    )
    result = OptimizationResult(
        hypothesis=hypothesis,
        trials=tuple(trials),
        backend=backend,
        elapsed_seconds=round(time.perf_counter() - started, 6),
        max_rss_bytes=_rss(),
        reused=False,
        period_errors=best_periods,
        reason="коэффициенты обновлены по окну 2021" if improved else "исходные коэффициенты оставлены",
    )
    _store(memory, key, result)
    return result


def _trial(
    tree,
    point: tuple[float, ...],
    rows,
    number: int,
) -> tuple[TrialRecord, dict[str, object], str]:
    try:
        replaced = replace_coefficients(tree, point)
        if skeleton(replaced) != skeleton(tree):
            raise FormulaError("структура выражения изменилась")
        rendered = render(replaced)
        parsed = parse_formula(rendered)
        periods = {
            "search_train": _window_metrics(rows, parsed, SEARCH_TRAIN, _horizon_of(rows)),
            "search_check": _window_metrics(rows, parsed, SEARCH_CHECK, _horizon_of(rows)),
        }
    except (FormulaError, ValueError) as error:
        return (
            TrialRecord(number, point, "failed", str(error)[:200], None, None),
            {},
            "",
        )
    train = periods["search_train"]
    check = periods["search_check"]
    return (
        TrialRecord(
            number,
            tuple(list_coefficients(parsed)),
            "complete",
            "посчитано",
            float(train["log_loss"]),
            float(check["log_loss"]),
        ),
        periods,
        rendered,
    )


def _window_metrics(rows, tree, bounds: tuple, horizon: int) -> dict[str, object]:
    assert_search_window(bounds[0], bounds[1])
    chosen = chronological_rows(rows, bounds[0], bounds[1], horizon)
    if len(chosen) < MIN_DECISION_ROWS:
        raise ValueError("в окне подбора слишком мало точек")
    probabilities = [evaluate(tree, row.features) for row in chosen]
    labels = [row.y_up for row in chosen]
    frequency = [row.baseline_frequency for row in chosen]
    scored = score_direction(probabilities, labels, frequency)
    skill = scored["skill"]
    if not isinstance(skill, dict):
        raise ValueError("сравнение с ориентиром не посчитано")
    return {
        "observations": int(scored["observations"]),
        "brier": round(float(scored["brier"]), 6),
        "log_loss": round(float(scored["log_loss"]), 6),
        "frequency_brier": round(float(skill["historical_frequency"]["reference_brier"]), 6),
        "coin_brier": round(float(skill["constant_half"]["reference_brier"]), 6),
        "calibration_bins": len(scored["calibration"]) if isinstance(scored["calibration"], list) else 0,
        "first_decision": chosen[0].decision_date.isoformat(),
        "last_decision": chosen[-1].decision_date.isoformat(),
        "last_target": chosen[-1].target_date.isoformat(),
        "last_label_available": chosen[-1].label_available.isoformat(),
    }


def _horizon_of(rows: Sequence) -> int:
    if not rows:
        raise ValueError("нет строк эксперимента")
    return (rows[0].target_date - rows[0].decision_date).days


def _grid_points(start: tuple[float, ...], n_trials: int, seed: int) -> list[tuple[float, ...]]:
    points = [start]
    rng = random.Random(seed)
    for index in range(len(start)):
        for value in _GRID:
            candidate = list(start)
            candidate[index] = value
            point = tuple(candidate)
            if point not in points:
                points.append(point)
            if len(points) >= n_trials:
                return points[:n_trials]
    while len(points) < n_trials:
        point = tuple(round(rng.uniform(-10, 10), 4) for _ in start)
        if point not in points:
            points.append(point)
    return points[:n_trials]


class _OptunaPlan(ProposalPlan):
    def __init__(self, width: int, seed: int) -> None:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna = optuna
        self.width = width
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self._trial = None

    def propose(self, index: int) -> tuple[float, ...] | None:
        self._trial = self.study.ask()
        return tuple(self._trial.suggest_float(f"c{name}", -10.0, 10.0) for name in range(self.width))

    def tell(self, value: float | None) -> None:
        if self._trial is None:
            return
        if value is None:
            self.study.tell(self._trial, state=self.optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self._trial, value)


def _optuna_installed() -> bool:
    return importlib.util.find_spec("optuna") is not None


def _settings(backend: str, timeout_seconds: float) -> dict[str, object]:
    return {
        "loss": "log_loss",
        "backend": backend,
        "timeout_seconds": timeout_seconds,
        "train_window": [SEARCH_TRAIN[0].isoformat(), SEARCH_TRAIN[1].isoformat()],
        "check_window": [SEARCH_CHECK[0].isoformat(), SEARCH_CHECK[1].isoformat()],
        "bounds": [-10, 10],
        "code_version": DISCOVERY_VERSION,
    }


def _key(
    expression: str,
    features: tuple[str, ...],
    horizon_days: int,
    dataset_version: str,
    seed: int,
    n_trials: int,
    settings: dict[str, object],
) -> str:
    import json

    payload = {
        "origin": ORIGIN_OPTUNA,
        "source_expression": expression,
        "features": list(features),
        "horizon_days": horizon_days,
        "dataset_checksum": dataset_version,
        "seed": seed,
        "n_trials": n_trials,
        "settings": settings,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _store(memory: ExperimentMemory | None, key: str, result: OptimizationResult) -> None:
    if memory is None:
        return
    memory.save_discovery(key, _payload(result), origin=ORIGIN_OPTUNA, dataset_checksum=result.hypothesis.dataset_version)


def _payload(result: OptimizationResult) -> dict[str, object]:
    hypothesis = result.hypothesis
    return {
        "origin": hypothesis.origin,
        "id": hypothesis.id,
        "source_expression": hypothesis.source_expression,
        "expression": hypothesis.expression,
        "features": list(hypothesis.features),
        "horizon_days": hypothesis.horizon_days,
        "parameters": list(hypothesis.parameters),
        "dataset_checksum": hypothesis.dataset_version,
        "created_at": hypothesis.created_at.isoformat(),
        "backend": result.backend,
        "reason": result.reason,
        "elapsed_seconds": result.elapsed_seconds,
        "max_rss_bytes": result.max_rss_bytes,
        "period_errors": result.period_errors,
        "complexity": node_count_of(hypothesis.expression),
        "n_trials": len(result.trials),
        "seed_note": "seed входит в ключ поиска",
        "dependencies": _dependencies(),
        "code_version": DISCOVERY_VERSION,
        "trials": [
            {
                "number": trial.number,
                "parameters": list(trial.parameters),
                "state": trial.state,
                "reason": trial.reason,
                "train_log_loss": trial.train_log_loss,
                "check_log_loss": trial.check_log_loss,
            }
            for trial in result.trials
        ],
    }


def node_count_of(expression: str) -> int:
    from wolfai.research.formula import node_count

    return node_count(parse_formula(expression))


def _dependencies() -> dict[str, str | None]:
    return {
        "optuna": _package_version("optuna"),
        "pysr": _package_version("pysr"),
        "julia": None,
    }


def _package_version(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return None
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _from_payload(payload: dict[str, object], *, reused: bool) -> OptimizationResult:
    created = datetime.fromisoformat(str(payload["created_at"]))
    hypothesis = bind_hypothesis(
        id=str(payload["id"]),
        origin=ORIGIN_OPTUNA,
        expression=str(payload["expression"]),
        features=tuple(payload["features"]),  # type: ignore[arg-type]
        horizon_days=int(payload["horizon_days"]),
        dataset_version=str(payload["dataset_checksum"]),
        created_at=created,
        parameters=tuple(float(value) for value in payload["parameters"]),  # type: ignore[union-attr]
        source_expression=str(payload["source_expression"]) if payload.get("source_expression") else None,
    )
    trials = tuple(
        TrialRecord(
            number=int(item["number"]),
            parameters=tuple(float(value) for value in item["parameters"]),
            state=str(item["state"]),
            reason=str(item["reason"]),
            train_log_loss=None if item["train_log_loss"] is None else float(item["train_log_loss"]),
            check_log_loss=None if item["check_log_loss"] is None else float(item["check_log_loss"]),
        )
        for item in payload["trials"]  # type: ignore[union-attr]
    )
    periods = payload["period_errors"]
    return OptimizationResult(
        hypothesis=hypothesis,
        trials=trials,
        backend=str(payload["backend"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        max_rss_bytes=int(payload["max_rss_bytes"]),
        reused=reused,
        period_errors=periods if isinstance(periods, dict) else {},
        reason=str(payload["reason"]),
    )


def _rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value)
