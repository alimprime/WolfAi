"""Повторные и бесперспективные гипотезы не считаются заново.

Кэш совпадает только при тех же данных и той же конфигурации.
Ранняя остановка смотрит на окно подбора, не на 2021 и не на закрытые годы.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from wolfai.research.engine import ExperimentMemory
from wolfai.research.formula import FormulaError
from wolfai.research.quality_contract import STAGE
from wolfai.research.symbolic_strategy import canonical_expression

MAX_GENERATION_ROUNDS = 8
MAX_PROPOSALS = 500
HOPELESS_NOTE = (
    "На окне подбора Brier хуже и частоты прошлых ростов, и константы 0.5. "
    "Дальше эта ветка не считается."
)


class BudgetExhausted(RuntimeError):
    """Лимит гипотез или времени исчерпан. Новый круг не начинается."""


@dataclass(frozen=True)
class Proposal:
    expression: str
    origin: str
    parent: str | None
    transform: str


@dataclass(frozen=True)
class TrainView:
    brier: float
    frequency_brier: float
    coin_brier: float


@dataclass(frozen=True)
class CheckView:
    brier: float
    log_loss: float
    status: str


@dataclass(frozen=True)
class LineageRecord:
    expression: str
    canonical: str | None
    origin: str
    parent: str | None
    transform: str
    outcome: str


@dataclass(frozen=True)
class CycleResult:
    scored: tuple[tuple[str, CheckView], ...]
    lineage: tuple[LineageRecord, ...]
    examined: int
    cache_hits: int
    elapsed_seconds: float
    stopped: str


@dataclass
class SearchBudget:
    max_hypotheses: int
    max_seconds: float

    def __post_init__(self) -> None:
        if type(self.max_hypotheses) is not int or self.max_hypotheses < 1:
            raise ValueError("бюджет гипотез должен быть положительным целым")
        if not isinstance(self.max_seconds, (int, float)) or isinstance(self.max_seconds, bool):
            raise ValueError("бюджет времени должен быть положительным")
        if float(self.max_seconds) <= 0:
            raise ValueError("бюджет времени должен быть положительным")
        self.max_seconds = float(self.max_seconds)
        self.started = time.perf_counter()
        self.examined = 0

    def open(self) -> bool:
        return (
            self.examined < self.max_hypotheses
            and time.perf_counter() - self.started < self.max_seconds
        )

    def consume(self) -> None:
        if not self.open():
            raise BudgetExhausted("вычислительный бюджет исчерпан")
        self.examined += 1


def same_budget(left: SearchBudget, right: SearchBudget) -> None:
    if left.max_hypotheses != right.max_hypotheses or left.max_seconds != right.max_seconds:
        raise ValueError("сравнение требует одинаковый бюджет гипотез и времени")


def collect_rounds(
    generate: Callable[[int], Sequence[Proposal]],
    *,
    max_rounds: int = MAX_GENERATION_ROUNDS,
) -> tuple[Proposal, ...]:
    """Конечное число кругов. Пустой круг останавливает генерацию."""
    if type(max_rounds) is not int or not 1 <= max_rounds <= MAX_GENERATION_ROUNDS:
        raise ValueError("число кругов генерации вне допустимого")
    collected: list[Proposal] = []
    for index in range(max_rounds):
        batch = list(generate(index))
        if not batch:
            break
        collected.extend(batch)
        if len(collected) > MAX_PROPOSALS:
            raise ValueError("генерация превысила предел предложений")
    return tuple(collected)


def hopeless_on_train(train: TrainView) -> bool:
    """Хуже обоих контролей уже на подборе. Это не порог значимости."""
    return train.brier > train.frequency_brier and train.brier > train.coin_brier


def research_cycle(
    proposals: Sequence[Proposal],
    *,
    budget: SearchBudget,
    dataset_checksum: str,
    horizon_days: int,
    config_id: str,
    memory: ExperimentMemory | None,
    train_metrics: Callable[[str], TrainView],
    check_metrics: Callable[[str], CheckView],
) -> CycleResult:
    """Одинаковая каноническая формула считается один раз."""
    if horizon_days not in (1, 7):
        raise ValueError("горизонт должен быть 1 или 7 дней")
    started = time.perf_counter()
    seen: set[str] = set()
    lineage: list[LineageRecord] = []
    scored: list[tuple[str, CheckView]] = []
    cache_hits = 0
    stopped = "completed"
    for proposal in proposals:
        if not budget.open():
            stopped = "budget"
            break
        budget.consume()
        try:
            canonical = canonical_expression(proposal.expression)
        except FormulaError:
            lineage.append(_line(proposal, None, "invalid"))
            continue
        if canonical in seen:
            lineage.append(_line(proposal, canonical, "duplicate"))
            continue
        seen.add(canonical)
        cached = _read_cache(memory, canonical, dataset_checksum, horizon_days, config_id)
        if cached is not None:
            cache_hits += 1
            lineage.append(_line(proposal, canonical, "cached"))
            if cached[0] is not None:
                scored.append((canonical, cached[0]))
            continue
        try:
            train = train_metrics(canonical)
        except FormulaError:
            lineage.append(_line(proposal, canonical, "invalid"))
            continue
        if hopeless_on_train(train):
            _write_cache(
                memory,
                canonical,
                dataset_checksum,
                horizon_days,
                config_id,
                proposal.origin,
                train,
                None,
            )
            lineage.append(_line(proposal, canonical, "hopeless"))
            continue
        try:
            check = check_metrics(canonical)
        except FormulaError:
            lineage.append(_line(proposal, canonical, "invalid"))
            continue
        _write_cache(
            memory,
            canonical,
            dataset_checksum,
            horizon_days,
            config_id,
            proposal.origin,
            train,
            check,
        )
        lineage.append(_line(proposal, canonical, "scored"))
        scored.append((canonical, check))
    return CycleResult(
        scored=tuple(scored),
        lineage=tuple(lineage),
        examined=budget.examined,
        cache_hits=cache_hits,
        elapsed_seconds=time.perf_counter() - started,
        stopped=stopped,
    )


def _line(proposal: Proposal, canonical: str | None, outcome: str) -> LineageRecord:
    return LineageRecord(
        expression=proposal.expression,
        canonical=canonical,
        origin=proposal.origin,
        parent=proposal.parent,
        transform=proposal.transform,
        outcome=outcome,
    )


def _cache_key(
    canonical: str,
    dataset_checksum: str,
    horizon_days: int,
    config_id: str,
) -> str:
    return json.dumps(
        {
            "stage": STAGE,
            "canonical": canonical,
            "dataset": dataset_checksum,
            "horizon": horizon_days,
            "config": config_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _read_cache(
    memory: ExperimentMemory | None,
    canonical: str,
    dataset_checksum: str,
    horizon_days: int,
    config_id: str,
) -> tuple[CheckView | None] | None:
    if memory is None:
        return None
    payload = memory.find_discovery(
        _cache_key(canonical, dataset_checksum, horizon_days, config_id)
    )
    if payload is None:
        return None
    raw = payload.get("check")
    if raw is None:
        return (None,)
    if not isinstance(raw, dict):
        raise ValueError("кэш проверки повреждён")
    return (
        CheckView(
            brier=float(raw["brier"]),
            log_loss=float(raw["log_loss"]),
            status=str(raw["status"]),
        ),
    )


def _write_cache(
    memory: ExperimentMemory | None,
    canonical: str,
    dataset_checksum: str,
    horizon_days: int,
    config_id: str,
    origin: str,
    train: TrainView,
    check: CheckView | None,
) -> None:
    if memory is None:
        return
    payload: dict[str, object] = {
        "canonical": canonical,
        "train_brier": train.brier,
        "hopeless": check is None,
        "check": None
        if check is None
        else {"brier": check.brier, "log_loss": check.log_loss, "status": check.status},
        "code_version": STAGE,
    }
    memory.save_discovery(
        _cache_key(canonical, dataset_checksum, horizon_days, config_id),
        payload,
        origin=origin,
        dataset_checksum=dataset_checksum,
    )
