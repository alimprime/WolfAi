"""Кэш, одинаковые формулы и предел генерации."""

from pathlib import Path

import pytest

from wolfai.research.efficiency import (
    MAX_GENERATION_ROUNDS,
    CheckView,
    Proposal,
    SearchBudget,
    TrainView,
    collect_rounds,
    research_cycle,
    same_budget,
)
from wolfai.research.engine import ExperimentMemory


def _proposal(expression: str, transform: str = "proposed") -> Proposal:
    return Proposal(expression, "pysr", None, transform)


def test_equivalent_formula_is_not_scored_twice(tmp_path: Path) -> None:
    calls: list[str] = []

    def train(expression: str) -> TrainView:
        calls.append(f"train:{expression}")
        return TrainView(0.20, 0.25, 0.25)

    def check(expression: str) -> CheckView:
        calls.append(f"check:{expression}")
        return CheckView(0.22, 0.60, "insufficient_evidence")

    memory = ExperimentMemory(tmp_path / "cache.sqlite")
    proposals = (
        _proposal("sigmoid(log_return + momentum_7)"),
        _proposal("sigmoid(momentum_7 + log_return)", "reordered"),
    )
    first = research_cycle(
        proposals,
        budget=SearchBudget(10, 5),
        dataset_checksum="series",
        horizon_days=1,
        config_id="unit",
        memory=memory,
        train_metrics=train,
        check_metrics=check,
    )
    second = research_cycle(
        proposals[:1],
        budget=SearchBudget(10, 5),
        dataset_checksum="series",
        horizon_days=1,
        config_id="unit",
        memory=memory,
        train_metrics=train,
        check_metrics=check,
    )
    memory.close()
    assert [item.outcome for item in first.lineage] == ["scored", "duplicate"]
    assert second.cache_hits == 1
    assert calls.count("train:" + calls[0].split(":", 1)[1]) == 1
    assert sum(item.startswith("check:") for item in calls) == 1


def test_hopeless_train_does_not_spend_the_check() -> None:
    checked: list[str] = []

    def train(expression: str) -> TrainView:
        return TrainView(0.80, 0.25, 0.25)

    def check(expression: str) -> CheckView:
        checked.append(expression)
        return CheckView(0.10, 0.10, "supported")

    result = research_cycle(
        (_proposal("sigmoid(log_return)"),),
        budget=SearchBudget(5, 5),
        dataset_checksum="series",
        horizon_days=1,
        config_id="unit",
        memory=None,
        train_metrics=train,
        check_metrics=check,
    )
    assert result.lineage[0].outcome == "hopeless"
    assert result.scored == ()
    assert checked == []


def test_budget_stops_and_rounds_cannot_run_forever() -> None:
    result = research_cycle(
        tuple(_proposal(f"sigmoid({name})") for name in ("log_return", "momentum_7", "volatility_7")),
        budget=SearchBudget(2, 5),
        dataset_checksum="series",
        horizon_days=1,
        config_id="unit",
        memory=None,
        train_metrics=lambda expression: TrainView(0.20, 0.25, 0.25),
        check_metrics=lambda expression: CheckView(0.24, 0.69, "insufficient_evidence"),
    )
    assert result.examined == 2
    assert result.stopped == "budget"
    assert len(result.lineage) == 2

    def always(index: int) -> tuple[Proposal, ...]:
        return (_proposal("sigmoid(log_return)", f"round-{index}"),)

    assert len(collect_rounds(always)) == MAX_GENERATION_ROUNDS
    with pytest.raises(ValueError, match="кругов"):
        collect_rounds(always, max_rounds=MAX_GENERATION_ROUNDS + 1)
    with pytest.raises(ValueError, match="бюджет"):
        same_budget(SearchBudget(2, 5), SearchBudget(3, 5))


def test_invalid_text_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    result = research_cycle(
        (_proposal(f"__import__('pathlib').Path('{marker}').write_text('x')"),),
        budget=SearchBudget(3, 5),
        dataset_checksum="series",
        horizon_days=1,
        config_id="unit",
        memory=None,
        train_metrics=lambda expression: TrainView(0.20, 0.25, 0.25),
        check_metrics=lambda expression: CheckView(0.20, 0.20, "insufficient_evidence"),
    )
    assert result.lineage[0].outcome == "invalid"
    assert not marker.exists()
