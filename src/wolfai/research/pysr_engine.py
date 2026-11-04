"""Символический поиск PySR. Пакет и Julia не устанавливаются сами.

Внутренняя потеря PySR — логарифмическая потеря логита. Это число не
сравнивается с Brier Score WolfAi: метрики лаборатории считаются отдельно.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
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
) -> tuple[ResearchHypothesis, ...]:
    """Реальный поиск. Без Julia и PySR завершается до запуска модели."""
    if not _runtime_ready():
        raise DiscoveryUnavailable(
            "PySR или Julia недоступны. Поиск не запускался и компоненты не устанавливались."
        )
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    if len(rows) != len(labels) or len(rows) < 20:
        raise ValueError("для символического поиска слишком мало строк")
    with exclusive_search("pysr"):
        equations = _fit_pysr(rows, labels, feature_names, seed)
    moment = created_at or datetime.now(timezone.utc)
    return select_candidates(
        equations,
        feature_names=feature_names,
        horizon_days=horizon_days,
        dataset_version=dataset_version,
        created_at=moment,
        max_candidates=max_candidates,
    )


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
    if max_candidates < 1:
        raise ValueError("нужен хотя бы один кандидат")
    accepted: list[ResearchHypothesis] = []
    seen: set[tuple[object, ...]] = set()
    for index, equation in enumerate(equations):
        try:
            expression = equation_to_expression(equation, feature_names)
            tree = parse_formula(expression)
        except FormulaError:
            continue
        key = skeleton(tree)
        if key in seen:
            continue
        seen.add(key)
        hypothesis = bind_hypothesis(
            id=f"pysr-{index}",
            origin=ORIGIN_PYSR,
            expression=expression,
            features=tuple(sorted(feature_names_used(tree))),
            horizon_days=horizon_days,
            dataset_version=dataset_version,
            created_at=created_at,
            source_expression=equation.strip(),
        )
        accepted.append(hypothesis)
    accepted.sort(key=lambda item: node_count(parse_formula(item.expression)))
    return tuple(_spread(accepted, max_candidates))


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


def _fit_pysr(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: tuple[str, ...],
    seed: int,
) -> list[str]:
    """Вызывается только при установленном PySR. Потоки и процессы ограничены."""
    import os

    os.environ["JULIA_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=LIGHT_LIMITS["niterations"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "tanh"],
        maxsize=LIGHT_LIMITS["maxsize"],
        maxdepth=LIGHT_LIMITS["maxdepth"],
        populations=LIGHT_LIMITS["populations"],
        population_size=LIGHT_LIMITS["population_size"],
        procs=LIGHT_LIMITS["procs"],
        parallelism=LIGHT_LIMITS["parallelism"],
        random_state=seed,
        timeout_in_seconds=LIGHT_LIMITS["timeout_seconds"],
        deterministic=True,
        temp_equation_file=True,
        elementwise_loss=LOGISTIC_LOSS_JULIA,
    )
    frame = [[row[name] for name in feature_names] for row in rows]
    model.fit(frame, labels, variable_names=list(feature_names))
    table = model.equations_
    return [str(value) for value in table["equation"].tolist()]
