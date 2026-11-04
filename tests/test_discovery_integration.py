"""Реальные PySR и Optuna. Обычный pytest пропускает их, пока пакетов нет."""

import shutil
from datetime import date, datetime, timedelta, timezone

import pytest

from wolfai.data import Candle
from wolfai.research.formula import parse_formula
from wolfai.research.formula_edit import list_coefficients, skeleton
from wolfai.research.optuna_engine import OptimizerUnavailable, optimize_coefficients
from wolfai.research.periods import SEARCH_CHECK, SEARCH_TRAIN
from wolfai.research.pysr_engine import LIGHT_LIMITS, classify_equations, discover

UTC = timezone.utc


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


def test_rejected_equations_keep_their_reason(tmp_path) -> None:
    marker = tmp_path / "pwned.txt"
    accepted, rejected = classify_equations(
        [
            "log_return",
            "cos(log_return)",
            f"__import__('pathlib').Path('{marker}').write_text('x')",
            "log_return",
        ],
        horizon_days=1,
        dataset_version="v",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    assert len(accepted) == 1
    assert parse_formula(accepted[0].expression)
    reasons = [item.reason for item in rejected]
    assert any("неизвестная функция" in reason for reason in reasons)
    assert any("запрещ" in reason for reason in reasons)
    assert any(reason == "повтор структуры" for reason in reasons)
    assert not marker.exists()


def test_real_optuna_tpe_stays_inside_search_windows() -> None:
    pytest.importorskip("optuna")
    candles = _blocks(date(2019, 1, 1), date(2022, 6, 1))
    grid = optimize_coefficients(
        candles,
        "sigmoid(-2 * log_return)",
        ("log_return",),
        dataset_version="integration",
        n_trials=20,
        seed=1729,
        backend="grid",
    )
    tpe = optimize_coefficients(
        candles,
        "sigmoid(-2 * log_return)",
        ("log_return",),
        dataset_version="integration-tpe",
        n_trials=20,
        seed=1729,
        backend="optuna",
    )
    assert tpe.backend == "optuna"
    assert len(tpe.trials) <= 20
    assert skeleton(parse_formula(tpe.hypothesis.expression)) == skeleton(
        parse_formula("sigmoid(-2 * log_return)")
    )
    assert set(tpe.period_errors) == {"search_train", "search_check"}
    assert tpe.period_errors["search_train"]["last_target"] <= SEARCH_TRAIN[1].isoformat()
    assert tpe.period_errors["search_check"]["last_target"] <= SEARCH_CHECK[1].isoformat()
    assert "2023" not in str(tpe.period_errors)
    assert tpe.hypothesis.status == "proposed"
    # TPE при 20 испытаниях не обязан обойти сетку. Сравнение только фиксируется.
    assert list_coefficients(parse_formula(grid.hypothesis.expression))
    assert list_coefficients(parse_formula(tpe.hypothesis.expression))


def test_optuna_backend_refuses_when_package_is_missing() -> None:
    import importlib.util

    if importlib.util.find_spec("optuna") is not None:
        pytest.skip("Optuna уже установлена")
    with pytest.raises(OptimizerUnavailable):
        optimize_coefficients(
            [],
            "sigmoid(0.5 * log_return)",
            ("log_return",),
            dataset_version="v",
            backend="optuna",
        )


def test_real_pysr_search_uses_the_light_budget() -> None:
    pytest.importorskip("pysr")
    if shutil.which("julia") is None:
        pytest.skip("Julia не установлена")
    candles = _blocks(date(2019, 1, 1), date(2019, 4, 1))
    rows = []
    labels = []
    from wolfai.research.engine import build_experiment_rows
    from wolfai.research.periods import chronological_rows

    built = chronological_rows(
        build_experiment_rows(candles, 1),
        SEARCH_TRAIN[0],
        SEARCH_TRAIN[1],
        1,
    )
    for row in built[:40]:
        rows.append(row.features)
        labels.append(row.y_up)
    found = discover(
        rows,
        labels,
        feature_names=("log_return", "momentum_7", "volatility_7"),
        horizon_days=1,
        dataset_version="synthetic",
        seed=1729,
        max_candidates=LIGHT_LIMITS["populations"],
    )
    assert LIGHT_LIMITS["procs"] == 1
    assert LIGHT_LIMITS["niterations"] <= 4
    for hypothesis in found:
        assert hypothesis.origin == "pysr"
        assert hypothesis.status == "proposed"
        parse_formula(hypothesis.expression)
