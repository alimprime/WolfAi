"""Символический поиск PySR. Пакет и Julia не устанавливаются сами.

Внутренняя потеря PySR — логарифмическая потеря логита. Это число не
сравнивается с Brier Score WolfAi: метрики лаборатории считаются отдельно.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from wolfai.research.budget import exclusive_search
from wolfai.research.formula import FEATURES, FormulaError, node_count, parse_formula
from wolfai.research.formula_edit import skeleton
from wolfai.research.spec import ORIGIN_PYSR, ResearchHypothesis, bind_hypothesis
from wolfai.research.translate import equation_to_expression

# prediction — логит, target — метка 0 или 1. Результат оборачивается в sigmoid.
LOGISTIC_LOSS_JULIA = """
function loss(prediction, target)
    z = clamp(prediction, -20.0, 20.0)
    p = 1.0 / (1.0 + exp(-z))
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return -(target * log(p) + (1.0 - target) * log(1.0 - p))
end
"""
PYSR_LOSS_NOTE = (
    "Потеря PySR — log loss логита. Её значение не сравнивается с Brier Score WolfAi."
)
LIGHT_LIMITS = {
    "niterations": 4,
    "populations": 3,
    "population_size": 16,
    "maxsize": 12,
    "maxdepth": 5,
    "timeout_seconds": 40,
    "procs": 1,
    "parallelism": "serial",
}
MAX_CANDIDATES = 5


class DiscoveryUnavailable(RuntimeError):
    """Julia или PySR нет. Поиск не запускался и ничего не скачивалось."""


def discovery_status() -> dict[str, object]:
    julia = shutil.which("julia")
    version = _julia_version(julia) if julia else None
    return {
        "julia": bool(julia),
        "julia_version": version,
        "pysr": importlib.util.find_spec("pysr") is not None,
        "loss_note": PYSR_LOSS_NOTE,
    }


def discover(
    rows: list[dict[str, float]],
    labels: list[int],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    horizon_days: int = 1,
    dataset_version: str,
    created_at: datetime | None = None,
    seed: int = 1729,
    max_candidates: int = MAX_CANDIDATES,
    strategy: str = "baseline",
) -> tuple[ResearchHypothesis, ...]:
    """Реальный поиск. Без Julia и PySR завершается до запуска модели."""
    return discover_report(
        rows,
        labels,
        feature_names=feature_names,
        horizon_days=horizon_days,
        dataset_version=dataset_version,
        created_at=created_at,
        seed=seed,
        max_candidates=max_candidates,
        strategy=strategy,
    ).hypotheses


def discover_report(
    rows: list[dict[str, float]],
    labels: list[int],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    horizon_days: int = 1,
    dataset_version: str,
    created_at: datetime | None = None,
    seed: int = 1729,
    max_candidates: int = MAX_CANDIDATES,
    strategy: str = "baseline",
):
    """Тот же лёгкий бюджет. Улучшенная стратегия не добавляет итераций."""
    from wolfai.research.symbolic_strategy import (
        STRATEGY_BASELINE,
        STRATEGY_IMPROVED,
        RejectedFormula,
        SearchAssembly,
        screen_equations,
    )

    if strategy not in {STRATEGY_BASELINE, STRATEGY_IMPROVED}:
        raise ValueError("стратегия поиска должна быть baseline или improved")
    if not _runtime_ready():
        raise DiscoveryUnavailable(
            "PySR или Julia недоступны. Поиск не запускался и компоненты не устанавливались."
        )
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    if len(rows) != len(labels) or len(rows) < 20:
        raise ValueError("для символического поиска слишком мало строк")
    with exclusive_search("pysr"):
        records = _fit_table(rows, labels, feature_names, seed, strategy)
    moment = created_at or datetime.now(timezone.utc)
    equations = [record.equation for record in records]
    if strategy == STRATEGY_BASELINE:
        accepted, rejected = classify_equations(
            equations,
            feature_names=feature_names,
            horizon_days=horizon_days,
            dataset_version=dataset_version,
            created_at=moment,
            max_candidates=max_candidates,
        )
        structures = {skeleton(parse_formula(item.expression)) for item in accepted}
        return SearchAssembly(
            hypotheses=accepted,
            rejected=tuple(RejectedFormula(item.equation, item.reason) for item in rejected),
            unique_structures=len(structures),
            screened=len(accepted),
            pareto_size=len(accepted),
            examined=len(equations),
            selector="baseline-spread",
        )
    losses = None if any(record.loss is None for record in records) else [
        float(record.loss) for record in records if record.loss is not None
    ]
    return screen_equations(
        equations,
        losses,
        feature_names=feature_names,
        horizon_days=horizon_days,
        dataset_version=dataset_version,
        created_at=moment,
        max_candidates=max_candidates,
    )


@dataclass(frozen=True)
class RejectedEquation:
    equation: str
    reason: str


def classify_equations(
    equations: list[str],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    horizon_days: int,
    dataset_version: str,
    created_at: datetime,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[tuple[ResearchHypothesis, ...], tuple[RejectedEquation, ...]]:
    """Принятые формулы и причины отклонения. Отклонённый текст не исполняется."""
    if max_candidates < 1:
        raise ValueError("нужен хотя бы один кандидат")
    accepted: list[ResearchHypothesis] = []
    rejected: list[RejectedEquation] = []
    seen: set[tuple[object, ...]] = set()
    for index, equation in enumerate(equations):
        try:
            expression = equation_to_expression(equation, feature_names)
            tree = parse_formula(expression)
        except FormulaError as error:
            rejected.append(RejectedEquation(equation, str(error)[:200]))
            continue
        key = skeleton(tree)
        if key in seen:
            rejected.append(RejectedEquation(equation, "повтор структуры"))
            continue
        seen.add(key)
        accepted.append(bind_hypothesis(
            id=f"pysr-{index}",
            origin=ORIGIN_PYSR,
            expression=expression,
            features=tuple(sorted(feature_names_used(tree))),
            horizon_days=horizon_days,
            dataset_version=dataset_version,
            created_at=created_at,
            source_expression=equation.strip(),
        ))
    accepted.sort(key=lambda item: node_count(parse_formula(item.expression)))
    return tuple(_spread(accepted, max_candidates)), tuple(rejected)


def select_candidates(
    equations: list[str],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    horizon_days: int,
    dataset_version: str,
    created_at: datetime,
    max_candidates: int = MAX_CANDIDATES,
) -> tuple[ResearchHypothesis, ...]:
    """Оставляет разные по структуре формулы, а не одну самую сложную."""
    accepted, _rejected = classify_equations(
        equations,
        feature_names=feature_names,
        horizon_days=horizon_days,
        dataset_version=dataset_version,
        created_at=created_at,
        max_candidates=max_candidates,
    )
    return accepted


def feature_names_used(tree) -> set[str]:
    from wolfai.research.formula import features_used

    return features_used(tree)


def _spread(items: list[ResearchHypothesis], limit: int) -> list[ResearchHypothesis]:
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    chosen: list[ResearchHypothesis] = []
    seen: set[int] = set()
    last = len(items) - 1
    for step in range(limit):
        index = round(step * last / (limit - 1))
        if index in seen:
            continue
        seen.add(index)
        chosen.append(items[index])
    return chosen


def _runtime_ready() -> bool:
    status = discovery_status()
    return bool(status["julia"] and status["pysr"])


def _julia_version(executable: str) -> str | None:
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr).strip()
    return text or None


@dataclass(frozen=True)
class EquationRecord:
    equation: str
    loss: float | None


def regressor_settings(strategy: str, seed: int) -> dict[str, object]:
    """Общие пределы поиска. Улучшение не поднимает итерации, популяции и процессы."""
    if strategy not in {"baseline", "improved"}:
        raise ValueError("стратегия поиска должна быть baseline или improved")
    settings: dict[str, object] = {
        "niterations": LIGHT_LIMITS["niterations"],
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["abs", "tanh"],
        "maxsize": LIGHT_LIMITS["maxsize"],
        "maxdepth": LIGHT_LIMITS["maxdepth"],
        "populations": LIGHT_LIMITS["populations"],
        "population_size": LIGHT_LIMITS["population_size"],
        "procs": LIGHT_LIMITS["procs"],
        "parallelism": LIGHT_LIMITS["parallelism"],
        "random_state": seed,
        "timeout_in_seconds": LIGHT_LIMITS["timeout_seconds"],
        "deterministic": True,
        "temp_equation_file": True,
        "elementwise_loss": LOGISTIC_LOSS_JULIA,
    }
    if strategy == "improved":
        from wolfai.research.symbolic_strategy import extra_regressor_options

        settings.update(extra_regressor_options())
    if (
        settings["niterations"] != LIGHT_LIMITS["niterations"]
        or settings["populations"] != LIGHT_LIMITS["populations"]
        or settings["population_size"] != LIGHT_LIMITS["population_size"]
        or settings["procs"] != 1
        or float(settings["timeout_in_seconds"]) > float(LIGHT_LIMITS["timeout_seconds"])
    ):
        raise ValueError("бюджет PySR нельзя увеличивать")
    return settings


def _fit_pysr(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: tuple[str, ...],
    seed: int,
) -> list[str]:
    """Вызывается только при установленном PySR. Потоки и процессы ограничены."""
    return [
        record.equation
        for record in _fit_table(rows, labels, feature_names, seed, "baseline")
    ]


def _fit_table(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: tuple[str, ...],
    seed: int,
    strategy: str,
) -> list[EquationRecord]:
    import os

    os.environ["JULIA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from pysr import PySRRegressor

    model = PySRRegressor(**regressor_settings(strategy, seed))
    frame = [[row[name] for name in feature_names] for row in rows]
    model.fit(frame, labels, variable_names=list(feature_names))
    table = model.equations_
    equations = [str(value) for value in table["equation"].tolist()]
    if "loss" in getattr(table, "columns", ()):
        losses = [float(value) for value in table["loss"].tolist()]
    else:
        losses = [None] * len(equations)
    return [
        EquationRecord(equation, loss)
        for equation, loss in zip(equations, losses, strict=True)
    ]
