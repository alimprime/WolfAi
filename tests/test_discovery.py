"""Поиск формул и подбор коэффициентов без финальной валидации и без исполнения кода."""

import inspect
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from wolfai.data import Candle
from wolfai.data.history import HistoricalStore
from wolfai.research.budget import SearchBusy, exclusive_search
from wolfai.research.compete import (
    compete,
    eligibility,
    research_score,
    selection_margin,
)
from wolfai.research.engine import ExperimentMemory, build_experiment_rows, run_experiment
from wolfai.research.formula import FormulaError, parse_formula
from wolfai.research.formula_edit import list_coefficients, render, replace_coefficients, skeleton
from wolfai.research.optuna_engine import ProposalPlan, _trial, optimize_coefficients
from wolfai.research.periods import (
    FEEDBACK,
    SEARCH_CHECK,
    SEARCH_TRAIN,
    chronological_rows,
)
from wolfai.research.pysr_engine import (
    LIGHT_LIMITS,
    PYSR_LOSS_NOTE,
    DiscoveryUnavailable,
    discover,
    discovery_status,
    select_candidates,
)
from wolfai.research.spec import ORIGIN_OPTUNA, ORIGIN_QWEN, as_engine, bind_hypothesis
from wolfai.research.translate import equation_to_expression

UTC = timezone.utc
SOURCE = "binance-vision"


def _span(start: date, end: date, *, step: float = 1.0) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    price = 100.0
    while day <= end:
        close = price
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close + 1,
            low=max(close - 1, 0.5),
            close=close,
            volume=2,
        ))
        price += step
        day += timedelta(days=1)
    return candles


def _blocks(start: date, end: date) -> list[Candle]:
    candles: list[Candle] = []
    day = start
    price = 100.0
    index = 0
    while day <= end:
        close = price
        candles.append(Candle(
            timestamp=datetime(day.year, day.month, day.day, tzinfo=UTC),
            open=close,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=2,
        ))
        price = close * (1.05 if (index // 30) % 2 == 0 else 1 / 1.05)
        day += timedelta(days=1)
        index += 1
    return candles


def _hypothesis(expression: str, feature: str = "log_return"):
    return bind_hypothesis(
        id="candidate",
        origin=ORIGIN_QWEN,
        expression=expression,
        features=(feature,),
        horizon_days=1,
        dataset_version="series-v1",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )


class _Plan(ProposalPlan):
    def __init__(self, points: dict[int, tuple[float, ...]]) -> None:
        self.points = points

    def propose(self, index: int) -> tuple[float, ...] | None:
        return self.points.get(index)


def test_hypothesis_contract_rejects_code_and_active_status(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    with pytest.raises(FormulaError):
        bind_hypothesis(
            id="bad",
            origin=ORIGIN_QWEN,
            expression=f"sigmoid(__import__('pathlib').Path({str(marker)!r}).write_text('x'))",
            features=("log_return",),
            horizon_days=1,
            dataset_version="v",
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
        )
    assert not marker.exists()
    with pytest.raises(ValueError, match="не назначается"):
        bind_hypothesis(
            id="bad",
            origin=ORIGIN_OPTUNA,
            expression="sigmoid(0.5 * log_return)",
            features=("log_return",),
            horizon_days=1,
            dataset_version="v",
            created_at=datetime(2026, 11, 4, tzinfo=UTC),
            status="active",
        )
    bound = _hypothesis("sigmoid(0.5 * log_return)")
    engine = as_engine(bound)
    assert engine.status == "proposed"
    assert parse_formula(engine.expression)


def test_coefficients_keep_structure_and_roundtrip() -> None:
    tree = parse_formula("sigmoid(-2 * log_return)")
    updated = replace_coefficients(tree, (4,))
    assert skeleton(tree) == skeleton(updated)
    rendered = render(updated)
    parsed = parse_formula(rendered)
    assert list_coefficients(parsed) == (4,)
    assert skeleton(parsed) == skeleton(tree)


def test_equation_translation_is_not_executed(tmp_path: Path) -> None:
    marker = tmp_path / "pwned.txt"
    with pytest.raises(FormulaError):
        equation_to_expression(f"__import__('pathlib').Path('{marker}').write_text('x')")
    assert not marker.exists()
    with pytest.raises(FormulaError, match="неизвестная функция"):
        equation_to_expression("cos(log_return)")
    with pytest.raises(FormulaError, match="научная"):
        equation_to_expression("1e-3 * log_return")
    expression = equation_to_expression("0.4 * momentum_7 - abs(volatility_30)")
    tree = parse_formula(expression)
    assert expression.startswith("sigmoid(")
    assert {name for name in ("momentum_7", "volatility_30") if name in expression}


def test_pysr_keeps_diverse_candidates_and_does_not_run_without_julia(tmp_path: Path) -> None:
    moment = datetime(2026, 11, 4, tzinfo=UTC)
    chosen = select_candidates(
        [
            "log_return + momentum_7 + momentum_30 + volatility_7",
            "cos(log_return)",
            f"__import__('os').system('echo {tmp_path}')",
            "log_return",
            "log_return",
        ],
        horizon_days=1,
        dataset_version="v",
        created_at=moment,
        max_candidates=2,
    )
    assert len(chosen) == 2
    complexities = [len(item.expression) for item in chosen]
    assert min(complexities) < max(complexities)
    assert all(item.origin == "pysr" for item in chosen)
    assert all(item.status == "proposed" for item in chosen)
    assert "pysr_loss" not in inspect.signature(select_candidates).parameters
    assert "Brier" in PYSR_LOSS_NOTE
    assert LIGHT_LIMITS["procs"] == 1
    assert LIGHT_LIMITS["niterations"] <= 8
    assert LIGHT_LIMITS["timeout_seconds"] <= 60
    with patch("wolfai.research.pysr_engine.shutil.which", return_value=None), patch(
        "wolfai.research.pysr_engine.importlib.util.find_spec", return_value=None
    ):
        status = discovery_status()
        assert status["julia"] is False
        assert status["pysr"] is False
        with pytest.raises(DiscoveryUnavailable, match="не устанавливались"):
            discover([], [], dataset_version="v", horizon_days=1)


def test_windows_drop_overlapping_targets_and_holdout() -> None:
    candles = _span(date(2020, 1, 1), date(2022, 6, 1))
    rows = build_experiment_rows(candles, 7)
    check = chronological_rows(rows, SEARCH_CHECK[0], SEARCH_CHECK[1], 7)
    train = chronological_rows(rows, SEARCH_TRAIN[0], SEARCH_TRAIN[1], 7)
    assert check
    assert all(row.target_date.year == 2021 for row in check)
    assert all(row.label_available <= SEARCH_CHECK[1] for row in check)
    assert all(row.label_available > row.decision_available for row in check)
    gaps = [right.decision_date - left.decision_date for left, right in zip(check, check[1:])]
    assert gaps
    assert min(gaps) >= timedelta(days=7)
    assert date(2021, 12, 28) not in {row.decision_date for row in check}
    train_targets = {row.target_date for row in train}
    check_targets = {row.target_date for row in check}
    assert train_targets.isdisjoint(check_targets)
    with pytest.raises(ValueError, match="holdout"):
        chronological_rows(rows, date(2025, 1, 1), date(2025, 6, 1), 7)


def test_optuna_improves_coefficients_without_final_validation(tmp_path: Path) -> None:
    candles = _blocks(date(2019, 1, 1), date(2022, 6, 1))
    altered = list(candles)
    day = date(2022, 3, 1)
    for index, candle in enumerate(altered):
        if candle.timestamp.date() >= day:
            close = 50.0 + index
            altered[index] = Candle(candle.timestamp, close, close + 1, max(close - 1, 0.5), close, 2)
    memory = ExperimentMemory(tmp_path / "discovery.sqlite")
    with patch("wolfai.research.optuna_engine._trial", wraps=_trial) as scorer:
        first = optimize_coefficients(
            candles,
            "sigmoid(-2 * log_return)",
            ("log_return",),
            dataset_version="series-v1",
            memory=memory,
            n_trials=20,
            seed=1729,
        )
        calls = scorer.call_count
        second = optimize_coefficients(
            altered,
            "sigmoid(-2 * log_return)",
            ("log_return",),
            dataset_version="series-v1",
            memory=memory,
            n_trials=20,
            seed=1729,
        )
    assert scorer.call_count == calls
    assert second.reused is True
    assert second.hypothesis.expression == first.hypothesis.expression
    assert first.hypothesis.origin == ORIGIN_OPTUNA
    assert first.hypothesis.status == "proposed"
    assert skeleton(parse_formula(first.hypothesis.expression)) == skeleton(
        parse_formula("sigmoid(-2 * log_return)")
    )
    assert list_coefficients(parse_formula(first.hypothesis.expression))[0] > 0
    check = first.period_errors["search_check"]
    train = first.period_errors["search_train"]
    assert set(first.period_errors) == {"search_train", "search_check"}
    assert check["log_loss"] < first.trials[0].check_log_loss
    assert str(train["last_target"]) <= "2020-12-31"
    assert str(check["last_target"]) <= "2021-12-31"
    assert str(check["last_label_available"]) <= "2021-12-31"
    assert "2023" not in str(first.period_errors)
    assert first.backend == "grid"
    same = optimize_coefficients(
        candles,
        "sigmoid(-2 * log_return)",
        ("log_return",),
        dataset_version="series-v1",
        n_trials=20,
        seed=1729,
    )
    assert same.hypothesis.expression == first.hypothesis.expression
    failed = optimize_coefficients(
        candles,
        "sigmoid(-2 * log_return)",
        ("log_return",),
        dataset_version="other",
        n_trials=3,
        seed=1,
        plan=_Plan({1: (11.0,), 2: (4.0,)}),
    )
    assert any(trial.state == "failed" for trial in failed.trials)
    assert list_coefficients(parse_formula(failed.hypothesis.expression))[0] == 4
    with pytest.raises(ValueError, match="20"):
        optimize_coefficients(
            candles,
            "sigmoid(log_return)",
            ("log_return",),
            dataset_version="v",
            n_trials=21,
        )
    with pytest.raises(ValueError, match="пропуск"):
        optimize_coefficients(
            candles[:10] + candles[12:],
            "sigmoid(0.5 * log_return)",
            ("log_return",),
            dataset_version="v",
            n_trials=1,
        )
    with pytest.raises(ValueError, match="не заменяется"):
        memory.promote(first.hypothesis.id)
    memory.close()


def test_competition_does_not_pick_by_one_period_or_promote(tmp_path: Path) -> None:
    assert "final" not in inspect.signature(eligibility).parameters
    assert "final" not in inspect.signature(research_score).parameters
    strong = {"brier": 0.20, "frequency_brier": 0.25, "coin_brier": 0.25}
    weak = {"brier": 0.249, "frequency_brier": 0.25, "coin_brier": 0.25}
    only_feedback = eligibility(weak, strong, n_examined=1, stability_gap=0.0)
    assert only_feedback[0] is False
    tiny = eligibility(weak, weak, n_examined=1, stability_gap=0.0)
    assert tiny[0] is False
    balanced = {"brier": 0.21, "frequency_brier": 0.25, "coin_brier": 0.25}
    one_sided = {"brier": 0.05, "frequency_brier": 0.25, "coin_brier": 0.25}
    balanced_score = research_score(balanced, balanced, complexity=4, n_examined=1, stability_gap=0.0)
    sided_score = research_score(balanced, one_sided, complexity=20, n_examined=50, stability_gap=0.04)
    assert balanced_score > sided_score
    assert selection_margin(50) > selection_margin(1)

    candles = _span(date(2019, 1, 1), date(2022, 12, 31))
    store = HistoricalStore(tmp_path / "btc")
    manifest = store.import_candles(candles, source=SOURCE)
    hypothesis = bind_hypothesis(
        id="mono",
        origin=ORIGIN_QWEN,
        expression="sigmoid(0.5 * log_return)",
        features=("log_return",),
        horizon_days=1,
        dataset_version=manifest.content_sha256,
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    memory = ExperimentMemory(tmp_path / "memory.sqlite")
    with patch("wolfai.research.compete.run_experiment", wraps=run_experiment) as runner:
        result = compete(
            [hypothesis, hypothesis],
            candles,
            n_examined=5,
            store=store,
            source=SOURCE,
            memory=memory,
        )
    assert runner.call_count == 0
    assert result.frozen == ()
    assert result.feedback_independent is False
    assert "не является независимым" in result.feedback_note
    assert any(item.reason == "дубликат формулы" for item in result.ranked)
    with pytest.raises(ValueError, match="не заменяется"):
        memory.promote(hypothesis.id)
    memory.close()


def test_heavy_searches_do_not_overlap() -> None:
    with exclusive_search("qwen"):
        with pytest.raises(SearchBusy):
            with exclusive_search("pysr"):
                pass


def test_feedback_rows_do_not_include_final_validation() -> None:
    candles = _span(date(2021, 1, 1), date(2024, 3, 1))
    rows = build_experiment_rows(candles, 1)
    feedback = chronological_rows(rows, FEEDBACK[0], FEEDBACK[1], 1)
    assert feedback
    assert all(row.decision_date.year == 2022 for row in feedback)
    assert all(row.target_date.year == 2022 for row in feedback)
