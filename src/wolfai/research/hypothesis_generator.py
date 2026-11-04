"""Интерфейс генератора гипотез. Сейчас это детерминированный каталог.

Генератор возвращает описание, а не исполняемый Python. Выражение берётся
только из короткого заранее разрешённого списка и проверяется грамматикой
формул. Случайный перебор ради 2021 года здесь не делается. Qwen не вызывается.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from wolfai.research.formula import FormulaError, parse_formula
from wolfai.research.generator import expression_is_safe

GENERATOR_ID = "deterministic-catalog"
GENERATOR_VERSION = "1"


@dataclass(frozen=True)
class GenerationContext:
    task_id: str
    parent_hypothesis_id: str | None = None
    requested_expression: str | None = None
    touches_holdout: bool = False
    reads_closed_history_after_2021: bool = False


@dataclass(frozen=True)
class StructuredHypothesis:
    author_id: str
    name: str
    description: str
    creation_reason: str
    features: tuple[str, ...]
    horizon_days: int
    expression: str
    applicability: dict[str, object]
    parent_hypothesis_id: str | None = None


@dataclass(frozen=True)
class _CatalogItem:
    name: str
    expression: str
    features: tuple[str, ...]
    horizon_days: int
    description: str
    creation_reason: str
    applicability: dict[str, object]


_APPLICABILITY: dict[str, object] = {
    "market": "synthetic",
    "regimes": ["any"],
    "btc_claim": False,
}
_REASON = "заранее разрешённое выражение для синтетической проверки, не поиск по 2021 году"

CATALOG: tuple[_CatalogItem, ...] = (
    _CatalogItem(
        name="catalog-momentum-vol",
        expression="sigmoid(0.3 * momentum_30 - 0.2 * volatility_7)",
        features=("momentum_30", "volatility_7"),
        horizon_days=1,
        description="Синтетический каталог: импульс 30 дней минус короткая волатильность.",
        creation_reason=_REASON,
        applicability=dict(_APPLICABILITY),
    ),
    _CatalogItem(
        name="catalog-momentum-7",
        expression="sigmoid(0.2 * momentum_7)",
        features=("momentum_7",),
        horizon_days=1,
        description="Синтетический каталог: короткий импульс.",
        creation_reason=_REASON,
        applicability=dict(_APPLICABILITY),
    ),
)


class HypothesisGenerator(Protocol):
    generator_id: str
    version: str

    def propose(self, context: GenerationContext) -> StructuredHypothesis:
        """Структурированная гипотеза. Код, файлы и счёт отсюда не доступны."""


@dataclass(frozen=True)
class DeterministicHypothesisGenerator:
    generator_id: str = GENERATOR_ID
    version: str = GENERATOR_VERSION

    def propose(self, context: GenerationContext) -> StructuredHypothesis:
        if context.touches_holdout or context.reads_closed_history_after_2021:
            raise PermissionError(
                "генератор не читает закрытый holdout и не подбирает формулу после 2021 года"
            )
        if not context.task_id.strip():
            raise ValueError("пустая задача")
        chosen = CATALOG[_index(context.task_id)]
        if context.requested_expression is not None:
            if not expression_is_safe(context.requested_expression):
                raise FormulaError("выражение содержит запрещённый символ")
            if context.requested_expression.strip() != chosen.expression:
                raise FormulaError("выражение вне заранее разрешённого каталога")
        parse_formula(chosen.expression)
        return StructuredHypothesis(
            author_id=self.generator_id,
            name=chosen.name,
            description=chosen.description,
            creation_reason=chosen.creation_reason,
            features=chosen.features,
            horizon_days=chosen.horizon_days,
            expression=chosen.expression,
            applicability=dict(chosen.applicability),
            parent_hypothesis_id=context.parent_hypothesis_id,
        )


def _index(task_id: str) -> int:
    return hashlib.sha256(task_id.encode()).digest()[0] % len(CATALOG)
