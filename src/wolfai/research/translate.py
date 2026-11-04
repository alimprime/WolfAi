"""Перевод строки PySR в грамматику WolfAi. Строка не исполняется."""

from __future__ import annotations

from wolfai.research.formula import (
    FEATURES,
    MAX_DEPTH,
    MAX_NODES,
    Binary,
    Const,
    Expr,
    Feature,
    FormulaError,
    Unary,
    node_count,
    parse_formula,
    tree_depth,
)
from wolfai.research.formula_edit import render

_FUNCTIONS = {"sigmoid": "sigmoid", "tanh": "tanh", "abs": "abs", "sign": "sign"}
_FORBIDDEN = ("import", "exec", "eval", "open", "system", "subprocess", "__")


def equation_to_expression(equation: str, allowed: tuple[str, ...] = FEATURES) -> str:
    """Собирает `sigmoid(...)`. Незнакомая функция и код отклоняются."""
    if not isinstance(equation, str) or not equation.strip():
        raise FormulaError("пустое выражение")
    if len(equation) > 300:
        raise FormulaError("выражение слишком длинное")
    lowered = equation.lower()
    if any(token in lowered for token in _FORBIDDEN) or any(char in equation for char in "\"'`[];{}\\"):
        raise FormulaError("выражение содержит запрещённую конструкцию")
    parser = _Parser(equation.strip(), set(allowed))
    tree = parser.parse_expr()
    parser.skip()
    if parser.i != len(parser.text):
        raise FormulaError("лишние символы в выражении")
    if not isinstance(tree, Unary) or tree.op != "sigmoid":
        tree = Unary("sigmoid", tree)
    if tree_depth(tree) > MAX_DEPTH or node_count(tree) > MAX_NODES:
        raise FormulaError("выражение сложнее разрешённой грамматики")
    rendered = render(tree)
    parse_formula(rendered)
    return rendered


class _Parser:
    def __init__(self, text: str, allowed: set[str]) -> None:
        self.text = text
        self.allowed = allowed
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
                if name not in _FUNCTIONS:
                    raise FormulaError(f"неизвестная функция {name}")
                if self.peek() == ")":
                    raise FormulaError("пустой вызов функции")
                arg = self.parse_expr()
                if self.peek() != ")":
                    raise FormulaError("ожидалась закрывающая скобка")
                self.pop()
                return Unary(_FUNCTIONS[name], arg)
            if name not in self.allowed:
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
        if self.i < len(self.text) and self.text[self.i] in {"e", "E"}:
            raise FormulaError("научная запись коэффициента запрещена")
        value = float(self.text[start:self.i])
        if abs(value) > 10:
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
