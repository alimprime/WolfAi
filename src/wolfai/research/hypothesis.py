"""Схема ответа Qwen. Статус и идентификатор назначает Python."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wolfai.research.formula import FEATURES

FeatureName = Literal[
    "log_return",
    "momentum_7",
    "momentum_30",
    "volatility_7",
    "volatility_30",
    "volume_change",
    "range_position",
]


class HypothesisSchemaError(ValueError):
    """Ответ модели не является допустимой гипотезой."""


class EffectDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature: FeatureName
    sign: Literal["positive", "negative"]


class TermDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature: FeatureName
    transform: Literal["identity", "abs", "tanh", "sign"]


class FalsificationDraft(BaseModel):
    """Условие опровержения только на внутреннем 2022 годе."""

    model_config = ConfigDict(extra="ignore")

    reference: Literal["constant_0.5", "historical_frequency", "logistic_all_features"]
    split: Literal["inner_2022"]


class AdaptiveDraft(BaseModel):
    """Структура формулы. Числовые веса назначает Python, не модель."""

    model_config = ConfigDict(extra="ignore")

    parent_id: str | None = None
    independent_reason: str | None = None
    terms: list[TermDraft] = Field(min_length=1, max_length=6)
    change: str = Field(min_length=1, max_length=300)
    problem: str = Field(min_length=1, max_length=300)
    expected_improvement: str = Field(min_length=1, max_length=300)
    falsification: FalsificationDraft
    explanation: str = Field(min_length=1, max_length=400)

    @model_validator(mode="after")
    def lineage_is_exclusive(self) -> "AdaptiveDraft":
        has_parent = bool(self.parent_id and self.parent_id.strip())
        has_reason = bool(self.independent_reason and self.independent_reason.strip())
        if has_parent == has_reason:
            raise ValueError("нужен parent_id либо independent_reason")
        if self.parent_id is not None and len(self.parent_id) > 64:
            raise ValueError("parent_id слишком длинный")
        keys = [(term.feature, term.transform) for term in self.terms]
        if len(keys) != len(set(keys)):
            raise ValueError("член формулы повторён")
        return self


class HypothesisDraft(BaseModel):
    """Черновик модели. Поле status, если оно пришло, игнорируется."""

    model_config = ConfigDict(extra="ignore")

    expression: str = Field(min_length=1, max_length=300)
    features: list[FeatureName] = Field(min_length=1, max_length=len(FEATURES))
    effects: list[EffectDraft] = Field(min_length=1, max_length=len(FEATURES))
    explanation: str = Field(min_length=1, max_length=400)
    based_on: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def effects_cover_features(self) -> "HypothesisDraft":
        names = list(self.features)
        effect_names = [effect.feature for effect in self.effects]
        if len(names) != len(set(names)):
            raise ValueError("признак указан дважды")
        if len(effect_names) != len(set(effect_names)):
            raise ValueError("эффект признака указан дважды")
        if set(effect_names) != set(names):
            raise ValueError("effects должны описывать ровно заявленные признаки")
        for item in self.based_on:
            if not item.strip() or len(item) > 64:
                raise ValueError("идентификатор based_on пуст или слишком длинный")
        return self


def parse_draft(payload: str) -> HypothesisDraft:
    return _validate(HypothesisDraft, payload)


def parse_adaptive(payload: str) -> AdaptiveDraft:
    return _validate(AdaptiveDraft, payload)


def _validate(model: type[HypothesisDraft] | type[AdaptiveDraft], payload: str):
    try:
        return model.model_validate(_extract_json(payload))
    except (HypothesisSchemaError, ValidationError) as error:
        raise HypothesisSchemaError(str(error)) from error


def _extract_json(payload: str) -> dict[str, object]:
    cleaned = re.sub(r"<think>.*?</think>", "", payload, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise HypothesisSchemaError("в ответе нет JSON-объекта")
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as error:
        raise HypothesisSchemaError("JSON не читается") from error
    if not isinstance(value, dict):
        raise HypothesisSchemaError("JSON должен быть объектом")
    return value
