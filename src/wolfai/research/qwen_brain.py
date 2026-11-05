"""Qwen предлагает одну гипотезу. Проверку и статус выполняет другой компонент.

Модель получает только текст, который собрал Python. Журнал прогнозов
и закрытая выборка сюда не передаются.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from wolfai.research.formula import FEATURES
from wolfai.research.hypothesis_protocol import QWEN_HYPOTHESIS_SCHEMA, REGIMES
from wolfai.research.memory_context import MemoryExcerpt
from wolfai.research.ollama import RESEARCH_SEED

FEEDBACK_KEYS = frozenset({
    "hypothesis_id",
    "experiment_id",
    "horizon_days",
    "sample_count",
    "brier_score",
    "log_loss",
    "baseline_brier",
    "improvement",
    "uncertainty",
    "failure_reason",
    "limitations",
    "revision_allowed",
})

GENERATOR_ID = "qwen-research-brain"
GENERATOR_VERSION = "3.0.0"
MAX_BRIEF_CHARS = 500
MAX_USER_CHARS = 3500

SYSTEM_PROMPT = """Ты предлагаешь одну исследовательскую гипотезу и возвращаешь только JSON.
Это черновик для независимой проверки Python. Поля status, claim_status и validated law отсутствуют в схеме.
Критерии проверки, пороги и правила соревнования заданы заранее и из ответа не меняются.
Закрытая выборка и данные после 2021 года в контекст не входят. 2021 год — пометка DEV_ONLY, не окно подбора.
Корень expression — sigmoid. Внутри допустимы признаки из списка, числа от -10 до 10, скобки и операции + - * /.
Функции: sigmoid, tanh, abs, sign. Других функций нет.
horizon_days равен 1, 3 или 7. features совпадает с признаками внутри expression.
rationale объясняет, чем гипотеза отличается от уже проверенных.
expected_regime называет ожидаемое условие применимости из разрешённого списка.
test_plan описывает проверку на синтетике или на окне 2019–2020.
limitations называет неопределённость. Это не доказательство прогнозного преимущества.
hypothesis_id — техническая метка. Календарный год в ней не является рыночными данными.
description не является формулой и не выбирает ряд. Не объявляй свою формулу победителем.
"""


@dataclass(frozen=True)
class BrainContext:
    author_id: str
    market_brief: str
    excerpt: MemoryExcerpt
    touches_holdout: bool = False
    reads_after_2021: bool = False
    fit_on_2021: bool = False
    prediction_ledger: object | None = None
    feedback: dict[str, object] | None = None


@dataclass(frozen=True)
class BrainTurn:
    status: str
    message: str
    content: str | None
    model_name: str
    model_digest: str
    elapsed_seconds: float
    rss_bytes: int
    available_bytes: int | None
    author_id: str
    generator_version: str
    temperature: float
    seed: int
    prompt_chars: int
    model_calls: int
    parameter_size: str = ""
    ollama_version: str = ""
    attempts_used: int = 0


class QwenResearchBrain:
    """Один автор и одна версия генератора. Оценка результата здесь не считается."""

    generator_id = GENERATOR_ID
    version = GENERATOR_VERSION

    def __init__(self, author_id: str = GENERATOR_ID, *, temperature: float = 0.0, seed: int = RESEARCH_SEED) -> None:
        if not author_id.strip():
            raise ValueError("идентификатор автора пуст")
        self.author_id = author_id.strip()
        self.generator_id = author_id.strip()
        self.temperature = temperature
        self.seed = seed

    def propose(self, context: BrainContext, client: object) -> BrainTurn:
        if context.author_id != self.author_id:
            return self._turn("agent_isolation", "контекст принадлежит другому автору", calls=0)
        if context.prediction_ledger is not None:
            return self._turn("ledger_forbidden", "журнал прогнозов генератору не передаётся", calls=0)
        if context.touches_holdout or context.reads_after_2021:
            return self._turn("time_boundary", "закрытые и поздние данные в запрос не входят", calls=0)
        if context.fit_on_2021:
            return self._turn("time_boundary", "2021 год остаётся DEV_ONLY и не используется для подбора", calls=0)
        if not context.market_brief.strip() or len(context.market_brief) > MAX_BRIEF_CHARS:
            return self._turn("context_rejected", "описание ситуации пустое или длиннее бюджета", calls=0)
        if context.feedback is not None and set(context.feedback) != FEEDBACK_KEYS:
            return self._turn("context_rejected", "обратная связь содержит лишние поля", calls=0)
        messages = build_messages(context)
        prompt_chars = sum(len(item["content"]) for item in messages)
        generate = getattr(client, "generate", None)
        if not callable(generate):
            return self._turn("ollama_unavailable", "клиент модели не подключён", calls=0, prompt_chars=prompt_chars)
        result = generate(messages, QWEN_HYPOTHESIS_SCHEMA, temperature=self.temperature)
        return BrainTurn(
            status=str(getattr(result, "status", "ollama_unavailable")),
            message=str(getattr(result, "message", "")),
            content=getattr(result, "content", None),
            model_name=str(getattr(result, "model_name", "")),
            model_digest=str(getattr(result, "model_digest", "")),
            elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
            rss_bytes=int(getattr(result, "rss_bytes", 0) or 0),
            available_bytes=getattr(result, "available_bytes", None),
            author_id=self.author_id,
            generator_version=GENERATOR_VERSION,
            temperature=self.temperature,
            seed=self.seed,
            prompt_chars=prompt_chars,
            model_calls=1,
            parameter_size=str(getattr(result, "parameter_size", "") or ""),
            ollama_version=str(getattr(result, "ollama_version", "") or ""),
            attempts_used=int(getattr(result, "attempts_used", 0) or 0),
        )

    def _turn(self, status: str, message: str, *, calls: int, prompt_chars: int = 0) -> BrainTurn:
        return BrainTurn(
            status=status,
            message=message,
            content=None,
            model_name="",
            model_digest="",
            elapsed_seconds=0.0,
            rss_bytes=0,
            available_bytes=None,
            author_id=self.author_id,
            generator_version=GENERATOR_VERSION,
            temperature=self.temperature,
            seed=self.seed,
            prompt_chars=prompt_chars,
            model_calls=calls,
        )


def build_messages(context: BrainContext) -> list[dict[str, str]]:
    user = {
        "market_brief": context.market_brief,
        "allowed_features": list(FEATURES),
        "allowed_horizons": [1, 3, 7],
        "allowed_regimes": sorted(REGIMES),
        "checked_hypotheses": context.excerpt.as_dict()["hypotheses"],
        "experiments": context.excerpt.as_dict()["experiments"],
        "errors": list(context.excerpt.errors),
        "rejections": list(context.excerpt.rejections),
        "data_limitations": list(context.excerpt.limitations),
        "leak_suspicions": list(context.excerpt.leak_suspicions),
        "rules": {
            "verifier": "verification-desk",
            "validated_law": False,
            "fit_after_2021": False,
            "year_2021": "DEV_ONLY",
        },
    }
    if context.feedback is not None:
        user["feedback"] = context.feedback
        user["revision_rule"] = (
            "Верни новую гипотезу. parent_hypothesis_id равен feedback.hypothesis_id. "
            "Прежнюю формулу не заменяй. Причину изменения напиши в rationale."
        )
    text = json.dumps(user, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(text) > MAX_USER_CHARS:
        user["checked_hypotheses"] = []
        user["experiments"] = []
        text = json.dumps(user, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(text) > MAX_USER_CHARS:
        raise ValueError("контекст не помещается в бюджет промпта")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
