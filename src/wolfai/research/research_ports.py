"""Интерфейсы исследовательских агентов.

Планировщик, ремонт синтаксиса и научный пересмотр не привязаны к HTTP.
Qwen — одна возможная реализация. Проверку выполняет другой компонент.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelTurn:
    status: str
    message: str
    content: str | None
    model_name: str = ""
    model_digest: str = ""
    elapsed_seconds: float = 0.0
    rss_bytes: int = 0
    seed: int = 0
    temperature: float = 0.0


@dataclass(frozen=True)
class PlanContext:
    market_brief: str
    untrusted_memory: dict[str, object]
    author_id: str
    strategy_id: str
    generator_version: str


@dataclass(frozen=True)
class GenerateContext:
    plan: Mapping[str, object]
    untrusted_memory: dict[str, object]
    author_id: str
    strategy_id: str
    generator_version: str


@dataclass(frozen=True)
class RepairContext:
    brief: Mapping[str, object]
    mode: str
    approved_plan: Mapping[str, object] | None
    parent_hypothesis_id: str | None
    untrusted_excerpt: str
    author_id: str
    strategy_id: str
    generator_version: str


@dataclass(frozen=True)
class RevisionContext:
    plan: Mapping[str, object]
    parent_hypothesis_id: str
    parent_expression: str
    scientific_feedback: Mapping[str, object]
    untrusted_memory: dict[str, object]
    author_id: str
    strategy_id: str
    generator_version: str


class ResearchPlanner(Protocol):
    def plan(self, context: PlanContext) -> ModelTurn:
        """Возвращает план. Не выбирает окно и не ставит оценку."""


class HypothesisGenerator(Protocol):
    def generate(self, context: GenerateContext) -> ModelTurn:
        """Возвращает одну формулу внутри уже принятого плана."""


class SyntaxRepairAgent(Protocol):
    def repair(self, context: RepairContext) -> ModelTurn:
        """Исправляет формат или грамматику. Не объявляет научный результат."""


class ScientificReviser(Protocol):
    def revise(self, context: RevisionContext) -> ModelTurn:
        """Предлагает новую версию. Не переписывает родительскую формулу."""


def invoke_model(
    client: object,
    messages: list[dict[str, str]],
    schema: dict[str, object],
    *,
    temperature: float,
) -> ModelTurn:
    """Один вызов переданного клиента. Сетевой протокол здесь не выбирается."""
    generate = getattr(client, "generate", None)
    if client is None or not callable(generate):
        return ModelTurn("ollama_unavailable", "клиент модели не подключён", None, temperature=temperature)
    try:
        result = generate(messages, schema, temperature=temperature)
    except (OSError, TimeoutError, ConnectionError) as error:
        return ModelTurn(
            "ollama_unavailable",
            f"{error.__class__.__name__}: {error}",
            None,
            temperature=temperature,
        )
    return ModelTurn(
        status=str(getattr(result, "status", "ollama_unavailable")),
        message=str(getattr(result, "message", "")),
        content=getattr(result, "content", None),
        model_name=str(getattr(result, "model_name", "") or ""),
        model_digest=str(getattr(result, "model_digest", "") or ""),
        elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
        rss_bytes=int(getattr(result, "rss_bytes", 0) or 0),
        seed=int(getattr(result, "seed", 0) or 0),
        temperature=float(getattr(result, "temperature", temperature) or temperature),
    )
