"""Локальный клиент Ollama без инструментов сети, терминала и файлов."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL_NAME = "qwen3.5:4b"
BASE_URL = "http://127.0.0.1:11434"
TIMEOUT_SECONDS = 60
HTTP_ATTEMPTS = 3
NUM_CTX = 4096
NUM_PREDICT = 900
GENERATION = {
    "temperature": 0.6,
    "top_p": 0.9,
    "seed": 1729,
    "num_ctx": NUM_CTX,
    "num_predict": NUM_PREDICT,
}
ALLOWED_OPTIONS = set(GENERATION)
HYPOTHESIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "expression": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "effects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "feature": {"type": "string"},
                    "sign": {"type": "string", "enum": ["positive", "negative"]},
                },
                "required": ["feature", "sign"],
            },
        },
        "explanation": {"type": "string"},
        "based_on": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["expression", "features", "effects", "explanation"],
}


class OllamaUnavailable(RuntimeError):
    """Локальная модель не ответила. Эксперимент на этом останавливается."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    digest: str


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    model_name: str
    model_digest: str
    generation: dict[str, float | int]


def build_chat_body(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, object] | None = None,
) -> dict[str, object]:
    """Тело запроса. Инструменты не передаются, контекст обрезан."""
    options = {key: GENERATION[key] for key in ALLOWED_OPTIONS}
    options["num_ctx"] = min(int(options["num_ctx"]), NUM_CTX)
    options["num_predict"] = min(int(options["num_predict"]), NUM_PREDICT)
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": HYPOTHESIS_SCHEMA if schema is None else schema,
        "options": options,
    }
    if "tools" in body:
        raise RuntimeError("tools запрещены")
    return body


class OllamaClient:
    """Чат только с 127.0.0.1. Модель не получает tools."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        model: str = MODEL_NAME,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.model_name = model
        self.model_digest = ""
        self.generation = dict(GENERATION)

    def health(self) -> ModelInfo:
        try:
            payload = self._request("GET", "/api/tags", None, attempts=1, timeout=min(self.timeout, 3))
        except OllamaUnavailable:
            raise
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OllamaUnavailable("Ollama недоступна") from error
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaUnavailable("Ollama вернула неожиданный список моделей")
        for item in models:
            if isinstance(item, dict) and item.get("name") == self.model:
                digest = str(item.get("digest") or "")
                if not digest:
                    raise OllamaUnavailable("у локальной модели нет digest")
                self.model_digest = digest
                return ModelInfo(name=self.model, digest=digest)
        raise OllamaUnavailable(f"локальная модель {self.model} не найдена")

    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
    ) -> ChatCompletion:
        if not self.model_digest:
            self.health()
        body = build_chat_body(self.model, messages, schema)
        payload = self._request("POST", "/api/chat", body, attempts=HTTP_ATTEMPTS, timeout=self.timeout)
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaUnavailable("в ответе Ollama нет текста")
        return ChatCompletion(
            content=message["content"],
            model_name=self.model,
            model_digest=self.model_digest,
            generation=dict(GENERATION),
        )

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        attempts: int,
        timeout: float,
    ) -> dict[str, object]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for _ in range(attempts):
            request = Request(
                f"{self.base_url}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                with urlopen(request, timeout=timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                raise OllamaUnavailable(f"Ollama ответила {error.code}") from error
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                continue
            if not isinstance(parsed, dict):
                raise OllamaUnavailable("Ollama вернула не объект JSON")
            return parsed
        raise OllamaUnavailable("Ollama не ответила в отведённое время") from last_error
