"""Локальный клиент Ollama без инструментов сети, терминала и файлов."""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

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

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ModelRejectedOutput(RuntimeError):
    """Ответ нельзя принять. Вызов инструмента из него не выполняется."""


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


# Контур этапа 2.2. Имя модели берётся из локального списка, а не из догадки.
# Контекст 256k у установленной Qwen здесь не запрашивается.
PREFERRED_LOCAL_MODEL = "qwen3.5:4b"
RESEARCH_NUM_CTX = 4096
RESEARCH_NUM_PREDICT = 640
RESEARCH_TIMEOUT_SECONDS = 60.0
MAX_RESEARCH_TIMEOUT_SECONDS = 180.0
RESEARCH_HTTP_ATTEMPTS = 3
RESEARCH_SEED = 1729
MAX_RESPONSE_CHARS = 4000
_RESPONSE_BYTES = 65536
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise OllamaUnavailable("перенаправление Ollama не принимается")


_LOCAL_OPENER = build_opener(_NoRedirect())


def _open_without_redirect(request: Request, timeout: float):
    return _LOCAL_OPENER.open(request, timeout=timeout)


def assert_loopback(base_url: str) -> str:
    """Исследовательский запрос уходит только на локальный Ollama."""
    parsed = urllib.parse.urlparse(base_url.strip())
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "http"
        or host not in _LOOPBACK_HOSTS
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise OllamaUnavailable("исследовательская модель доступна только на локальном адресе")
    return base_url.rstrip("/")


def _is_remote_model(name: str) -> bool:
    if any(char.isspace() or ord(char) < 32 for char in name):
        return True
    lowered = name.lower()
    return "cloud" in lowered or "://" in lowered


def select_installed_model(installed: list[str], requested: str | None) -> str:
    """Выбирает уже установленную модель. Скачивание отсюда не запускается."""
    local = [name for name in installed if name and not _is_remote_model(name)]
    if requested is not None:
        requested = requested.strip()
    if requested:
        if _is_remote_model(requested) or requested not in local:
            raise OllamaUnavailable(
                f"локальная модель {requested} недоступна; скачивание не выполняется"
            )
        return requested
    if PREFERRED_LOCAL_MODEL in local:
        return PREFERRED_LOCAL_MODEL
    raise OllamaUnavailable(
        "локальная модель qwen3.5:4b не найдена; скачивание не выполняется"
    )


def schema_mode_for_version(version: str) -> str:
    """JSON Schema в поле format есть у Ollama с 0.5. Более ранние версии получают format=json."""
    cleaned = version.strip().lstrip("vV")
    parts: list[int] = []
    for piece in cleaned.split("."):
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            return "json"
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    if tuple(parts[:3]) >= (0, 5, 0):
        return "schema"
    return "json"


def clamp_token_budget(num_ctx: int, num_predict: int) -> tuple[int, int]:
    ctx = min(max(int(num_ctx), 256), RESEARCH_NUM_CTX)
    predict = min(max(int(num_predict), 64), RESEARCH_NUM_PREDICT)
    return ctx, predict


def build_research_chat_body(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, object] | None,
    *,
    schema_mode: str,
    num_ctx: int,
    num_predict: int,
    seed: int,
) -> dict[str, object]:
    """Тело чата без инструментов. Температура 0 и seed фиксируют генерацию."""
    ctx, predict = clamp_token_budget(num_ctx, num_predict)
    if schema_mode == "schema" and schema is not None:
        response_format: object = schema
    else:
        response_format = "json"
    body: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": response_format,
        "options": {
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "num_ctx": ctx,
            "num_predict": predict,
            "repeat_penalty": 1,
            "mirostat": 0,
        },
    }
    if "tools" in body:
        raise RuntimeError("tools запрещены")
    return body


class ResearchOllama:
    """Локальный чат для лаборатории. Модель не скачивается и не получает tools."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        model: str | None = None,
        timeout: float = RESEARCH_TIMEOUT_SECONDS,
        http_attempts: int = RESEARCH_HTTP_ATTEMPTS,
        num_ctx: int = RESEARCH_NUM_CTX,
        num_predict: int = RESEARCH_NUM_PREDICT,
        seed: int = RESEARCH_SEED,
        transport: Callable[..., object] | None = None,
    ) -> None:
        if type(http_attempts) is not int or not 1 <= http_attempts <= RESEARCH_HTTP_ATTEMPTS:
            raise ValueError("число попыток связи должно быть от 1 до 3")
        if type(seed) is not int:
            raise ValueError("seed генерации должен быть целым")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout должен быть числом")
        if not 0 < float(timeout) <= MAX_RESEARCH_TIMEOUT_SECONDS:
            raise ValueError("timeout должен быть в пределах 180 секунд")
        self.base_url = assert_loopback(base_url)
        self.requested_model = None if model is None or not model.strip() else model.strip()
        self.timeout = float(timeout)
        self.http_attempts = http_attempts
        self.num_ctx, self.num_predict = clamp_token_budget(num_ctx, num_predict)
        self.seed = seed
        self._transport = transport or _open_without_redirect
        self.model_name = ""
        self.model_digest = ""
        self.schema_mode = "json"
        self.generation = {
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": 1,
            "mirostat": 0,
        }

    def health(self) -> ModelInfo:
        version_payload = self._request(
            "GET",
            "/api/version",
            None,
            attempts=1,
            timeout=min(self.timeout, 5),
        )
        version = version_payload.get("version")
        if not isinstance(version, str) or not version.strip():
            raise OllamaUnavailable("Ollama не сообщила версию")
        self.schema_mode = schema_mode_for_version(version)
        payload = self._request(
            "GET",
            "/api/tags",
            None,
            attempts=1,
            timeout=min(self.timeout, 5),
        )
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaUnavailable("Ollama вернула неожиданный список моделей")
        installed: list[str] = []
        digests: dict[str, str] = {}
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"]
            installed.append(name)
            digest = item.get("digest")
            if isinstance(digest, str) and digest:
                digests[name] = digest
        selected = select_installed_model(installed, self.requested_model)
        digest = digests.get(selected, "")
        if not digest:
            raise OllamaUnavailable("у локальной модели нет digest")
        self.model_name = selected
        self.model_digest = digest
        return ModelInfo(name=selected, digest=digest)

    def complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
    ) -> ChatCompletion:
        if not self.model_digest:
            self.health()
        payload = self._complete_once(messages, schema)
        if payload is None:
            self.schema_mode = "json"
            payload = self._complete_once(messages, schema)
        if payload is None:
            raise OllamaUnavailable("Ollama не приняла формат ответа", status=400)
        message = payload.get("message")
        if not isinstance(message, dict):
            raise OllamaUnavailable("в ответе Ollama нет сообщения")
        if message.get("tool_calls"):
            raise ModelRejectedOutput("вызов инструмента не выполняется")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelRejectedOutput("в ответе нет текста гипотезы")
        if len(content) > MAX_RESPONSE_CHARS:
            raise ModelRejectedOutput("ответ длиннее допустимого")
        return ChatCompletion(
            content=content,
            model_name=self.model_name,
            model_digest=self.model_digest,
            generation=dict(self.generation),
        )

    def _complete_once(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None,
    ) -> dict[str, object] | None:
        body = build_research_chat_body(
            self.model_name,
            messages,
            schema,
            schema_mode=self.schema_mode,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
            seed=self.seed,
        )
        try:
            return self._request(
                "POST",
                "/api/chat",
                body,
                attempts=self.http_attempts,
                timeout=self.timeout,
            )
        except OllamaUnavailable as error:
            if error.status == 400 and self.schema_mode == "schema":
                return None
            raise

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        attempts: int,
        timeout: float,
    ) -> dict[str, object]:
        if path not in {"/api/version", "/api/tags", "/api/chat"}:
            raise OllamaUnavailable("этот запрос к Ollama не разрешён")
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
                with self._transport(request, timeout=timeout) as response:
                    raw = response.read(_RESPONSE_BYTES)
                    if response.read(1):
                        raise OllamaUnavailable("ответ Ollama слишком большой")
            except OllamaUnavailable:
                raise
            except HTTPError as error:
                status = error.code
                error.close()
                raise OllamaUnavailable(f"Ollama ответила {status}", status=status) from error
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                continue
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                last_error = error
                continue
            if not isinstance(parsed, dict):
                raise OllamaUnavailable("Ollama вернула не объект JSON")
            return parsed
        raise OllamaUnavailable("Ollama не ответила в отведённое время") from last_error
