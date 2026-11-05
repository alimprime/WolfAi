"""Безопасная грамматика формул. Сгенерированный текст не исполняется как код."""

from __future__ import annotations

import math
from dataclasses import dataclass

FEATURES = (
    "log_return",
    "momentum_7",
    "momentum_30",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "range_position",
)
FUNCTIONS = ("sigmoid", "tanh", "abs", "sign")
MAX_EXPRESSION_LENGTH = 300
MAX_DEPTH = 6
MAX_NODES = 20
MAX_COEFFICIENT = 10.0
DIVISION_EPSILON = 1e-6


class FormulaError(ValueError):
    """Формула не входит в грамматику или даёт непригодное число."""


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class Feature:
    name: str


@dataclass(frozen=True)
class Unary:
    op: str
    arg: "Expr"


@dataclass(frozen=True)
class Binary:
    op: str
    left: "Expr"
    right: "Expr"


Expr = Const | Feature | Unary | Binary


@dataclass
class EvalState:
    protected_divisions: int = 0


def parse_expression(expression: str) -> Expr:
    """Разбирает выражение без требования корня sigmoid.

    Результат не является вероятностью и не используется как прогноз.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise FormulaError("выражение длиннее 300 символов")
    parser = _Parser(expression)
    tree = parser.parse_expr()
    parser.skip()
    if parser.i != len(parser.text):
        raise FormulaError("лишние символы в выражении")
    return tree


def parse_formula(expression: str) -> Expr:
    """Разбирает формулу в дерево. Корень обязан быть sigmoid."""
    tree = parse_expression(expression)
    if not isinstance(tree, Unary) or tree.op != "sigmoid":
        raise FormulaError("корень формулы должен быть sigmoid")
    if tree_depth(tree) > MAX_DEPTH:
        raise FormulaError("слишком глубокое выражение")
    if node_count(tree) > MAX_NODES:
        raise FormulaError("слишком много операций")
    if not features_used(tree):
        raise FormulaError("в формуле нет признака")
    return tree


def evaluate(tree: Expr, features: dict[str, float], state: EvalState | None = None) -> float:
    """Считает вероятность. Деление на ноль не бросает исключение и не даёт inf."""
    if state is None:
        state = EvalState()
    value = _compute(tree, features, state)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise FormulaError("вероятность вне (0, 1) или не число")
    return value


def extract_linear_terms(tree: Expr) -> list[tuple[float, str]] | None:
    """Коэффициенты при признаках внутри sigmoid, если выражение линейное."""
    if not isinstance(tree, Unary) or tree.op != "sigmoid":
        return None
    flat = _flatten_linear(tree.arg, 1.0)
    if flat is None:
        return None
    return flat


def _flatten_linear(node: Expr, scale: float) -> list[tuple[float, str]] | None:
    if isinstance(node, Feature):
        return [(scale, node.name)]
    if isinstance(node, Unary) and node.op == "neg":
        return _flatten_linear(node.arg, -scale)
    if isinstance(node, Binary) and node.op == "add":
        left = _flatten_linear(node.left, scale)
        right = _flatten_linear(node.right, scale)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, Binary) and node.op == "sub":
        left = _flatten_linear(node.left, scale)
        right = _flatten_linear(node.right, -scale)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, Binary) and node.op == "mul":
        const, feature, factor = _const_feature(node)
        if feature is None:
            return None
        return [(scale * factor * const, feature)]
    return None


def _const_feature(node: Binary) -> tuple[float, str | None, float]:
    left, right = node.left, node.right
    if isinstance(left, Const) and isinstance(right, Feature):
        return left.value, right.name, 1.0
    if isinstance(right, Const) and isinstance(left, Feature):
        return right.value, left.name, 1.0
    if isinstance(left, Const) and isinstance(right, Unary) and right.op == "neg" and isinstance(right.arg, Feature):
        return left.value, right.arg.name, -1.0
    if isinstance(right, Const) and isinstance(left, Unary) and left.op == "neg" and isinstance(left.arg, Feature):
        return right.value, left.arg.name, -1.0
    return 0.0, None, 1.0


def features_used(tree: Expr) -> set[str]:
    found: set[str] = set()
    _collect(tree, found)
    return found


def tree_depth(tree: Expr) -> int:
    if isinstance(tree, (Const, Feature)):
        return 1
    if isinstance(tree, Unary):
        return 1 + tree_depth(tree.arg)
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))


def node_count(tree: Expr) -> int:
    if isinstance(tree, (Const, Feature)):
        return 1
    if isinstance(tree, Unary):
        return 1 + node_count(tree.arg)
    return 1 + node_count(tree.left) + node_count(tree.right)


def _collect(tree: Expr, found: set[str]) -> None:
    if isinstance(tree, Feature):
        found.add(tree.name)
    elif isinstance(tree, Unary):
        _collect(tree.arg, found)
    elif isinstance(tree, Binary):
        _collect(tree.left, found)
        _collect(tree.right, found)


def _compute(tree: Expr, features: dict[str, float], state: EvalState) -> float:
    if isinstance(tree, Const):
        return tree.value
    if isinstance(tree, Feature):
        try:
            value = features[tree.name]
        except KeyError as error:
            raise FormulaError(f"нет значения {tree.name}") from error
        if not math.isfinite(value):
            raise FormulaError(f"{tree.name} не является конечным числом")
        return value
    if isinstance(tree, Unary):
        value = _compute(tree.arg, features, state)
        if tree.op == "neg":
            return -value
        if tree.op == "abs":
            return abs(value)
        if tree.op == "sign":
            if value > 0:
                return 1.0
            if value < 0:
                return -1.0
            return 0.0
        if tree.op == "sigmoid":
            return _sigmoid(value)
        if tree.op == "tanh":
            return math.tanh(max(-20.0, min(20.0, value)))
        raise FormulaError(f"неизвестная операция {tree.op}")
    left = _compute(tree.left, features, state)
    right = _compute(tree.right, features, state)
    if tree.op == "add":
        return left + right
    if tree.op == "sub":
        return left - right
    if tree.op == "mul":
        return left * right
    if tree.op == "div":
        if abs(right) < DIVISION_EPSILON:
            state.protected_divisions += 1
            return 0.0
        return left / right
    raise FormulaError(f"неизвестная операция {tree.op}")


def _sigmoid(value: float) -> float:
    clipped = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.i = 0

    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while self.peek() in {"+", "-"}:
            op = self.pop()
            right = self.parse_term()
            node = Binary("add" if op == "+" else "sub", node, right)
        return node

    def parse_term(self) -> Expr:
        node = self.parse_unary()
        while self.peek() in {"*", "/"}:
            op = self.pop()
            right = self.parse_unary()
            node = Binary("mul" if op == "*" else "div", node, right)
        return node

    def parse_unary(self) -> Expr:
        if self.peek() == "-":
            self.pop()
            return Unary("neg", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        current = self.peek()
        if current == "":
            raise FormulaError("выражение обрывается")
        if current == "(":
            self.pop()
            node = self.parse_expr()
            if self.peek() != ")":
                raise FormulaError("ожидалась закрывающая скобка")
            self.pop()
            return node
        if current.isdigit():
            return Const(self.read_number())
        if current.isalpha():
            name = self.read_name()
            if self.peek() == "(":
                self.pop()
                if name not in FUNCTIONS:
                    raise FormulaError(f"неизвестная функция {name}")
                if self.peek() == ")":
                    raise FormulaError("пустой вызов функции")
                arg = self.parse_expr()
                if self.peek() != ")":
                    raise FormulaError("ожидалась закрывающая скобка")
                self.pop()
                return Unary(name, arg)
            if name not in FEATURES:
                raise FormulaError(f"неизвестный признак {name}")
            return Feature(name)
        raise FormulaError(f"неожиданный символ {current}")

    def read_number(self) -> float:
        start = self.i
        while self.i < len(self.text) and self.text[self.i].isdigit():
            self.i += 1
        if self.i < len(self.text) and self.text[self.i] == ".":
            if self.i + 1 < len(self.text) and self.text[self.i + 1].isdigit():
                self.i += 1
                while self.i < len(self.text) and self.text[self.i].isdigit():
                    self.i += 1
        token = self.text[start:self.i]
        value = float(token)
        if not math.isfinite(value) or abs(value) > MAX_COEFFICIENT:
            raise FormulaError("коэффициент вне [-10, 10]")
        return value

    def read_name(self) -> str:
        start = self.i
        while self.i < len(self.text) and (self.text[self.i].isalnum() or self.text[self.i] == "_"):
            self.i += 1
        return self.text[start:self.i]

    def peek(self) -> str:
        self.skip()
        if self.i >= len(self.text):
            return ""
        return self.text[self.i]

    def pop(self) -> str:
        current = self.peek()
        if current == "":
            raise FormulaError("выражение обрывается")
        self.i += 1
        return current

    def skip(self) -> None:
        while self.i < len(self.text) and self.text[self.i].isspace():
            self.i += 1
