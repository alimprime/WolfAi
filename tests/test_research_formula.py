import math
from pathlib import Path

import pytest

from wolfai.research.formula import (
    FEATURES,
    EvalState,
    FormulaError,
    evaluate,
    parse_formula,
)
from wolfai.research.hypothesis import HypothesisSchemaError, parse_draft


def _features(**overrides: float) -> dict[str, float]:
    values = {name: 0.1 for name in FEATURES}
    values.update(overrides)
    return values


def test_formula_module_does_not_call_eval_or_exec() -> None:
    source = Path(parse_formula.__code__.co_filename).read_text(encoding="utf-8")
    assert "eval(" not in source
    assert "exec(" not in source


def test_parses_a_bounded_formula_and_rejects_code() -> None:
    tree = parse_formula("sigmoid(0.4 * momentum_7 - 0.2 * volatility_30)")
    value = evaluate(tree, _features(momentum_7=0.2, volatility_30=0.1))
    assert 0.0 < value < 1.0
    with pytest.raises(FormulaError):
        parse_formula("sigmoid(__import__('os'))")
    with pytest.raises(FormulaError, match="функция"):
        parse_formula("sigmoid(eval(momentum_7))")
    with pytest.raises(FormulaError, match="sigmoid"):
        parse_formula("momentum_7")
    with pytest.raises(FormulaError, match="коэффициент"):
        parse_formula("sigmoid(10.1 * momentum_7)")
    with pytest.raises(FormulaError, match="глубокое"):
        parse_formula("sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(momentum_7))))))")


def test_protected_division_and_non_finite_feature() -> None:
    tree = parse_formula("sigmoid(momentum_7 / 0)")
    state = EvalState()
    assert evaluate(tree, _features(), state) == pytest.approx(0.5)
    assert state.protected_divisions == 1
    with pytest.raises(FormulaError, match="конечным"):
        evaluate(parse_formula("sigmoid(momentum_7)"), _features(momentum_7=math.nan))
    with pytest.raises(FormulaError, match="конечным"):
        evaluate(parse_formula("sigmoid(log_return)"), _features(log_return=math.inf))


def test_same_formula_is_repeatable() -> None:
    tree = parse_formula("sigmoid(0.5 * momentum_7 + tanh(volume_change))")
    features = _features(momentum_7=-0.3, volume_change=0.2)
    assert evaluate(tree, features) == evaluate(tree, features)


def test_draft_schema_ignores_active_status_and_rejects_bad_json() -> None:
    draft = parse_draft(
        """```json
        {"expression":"sigmoid(0.2 * log_return)","features":["log_return"],
         "effects":[{"feature":"log_return","sign":"positive"}],
         "explanation":"краткий рост повышает вероятность","status":"active"}
        ```"""
    )
    assert draft.expression.startswith("sigmoid")
    assert not hasattr(draft, "status")
    with pytest.raises(HypothesisSchemaError):
        parse_draft("не json")
    with pytest.raises(HypothesisSchemaError):
        parse_draft('{"expression":"sigmoid(0.2 * log_return)","features":["price"],'
                    '"effects":[{"feature":"price","sign":"positive"}],"explanation":"x"}')
