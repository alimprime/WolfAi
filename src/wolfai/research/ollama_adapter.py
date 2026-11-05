"""Локальный адаптер Ollama для одного запроса Qwen.

Соединение только с 127.0.0.1. Модель не скачивается и не получает tools.
Сбой связи возвращает статус, а не необработанное исключение.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request

from wolfai.research.ollama import (
    BASE_URL,
    PREFERRED_LOCAL_MODEL,
    OllamaUnavailable,
    _open_without_redirect,
    assert_loopback,
    schema_mode_for_version,
    select_installed_model,
)

STAGE30_NUM_CTX = 2048
STAGE30_NUM_PREDICT = 512
STAGE30_TIMEOUT_SECONDS = 60.0
STAGE30_MAX_TIMEOUT_SECONDS = 180.0
STAGE30_ATTEMPTS = 3
STAGE30_SEED = 1729
MIN_AVAILABLE_BYTES = 4 * 1024 * 1024 * 1024
_RESPONSE_BYTES = 65536
_TRANSIENT_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_ALLOWED_PATHS = frozenset({"/api/version", "/api/tags", "/api/chat"})
MemoryGuard = Callable[[], str | None]


@dataclass(frozen=True)
class AdapterResult:
    status: str
    message: str
    content: str | None = None
    model_name: str = ""
    model_digest: str = ""
    parameter_size: str = ""
    ollama_version: str = ""
    elapsed_seconds: float = 0.0
    rss_bytes: int = 0
    available_bytes: int | None = None
    attempts_used: int = 0
    temperature: float = 0.0
    seed: int = STAGE30_SEED
    num_ctx: int = STAGE30_NUM_CTX
    num_predict: int = STAGE30_NUM_PREDICT


def process_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def available_bytes() -> int | None:
    """Оценка свободной памяти. Неизвестное значение не считается нехваткой."""
    if sys.platform == "darwin":
        return _darwin_available_bytes()
    try:
        page = int(os_sysconf("SC_PAGE_SIZE"))
        avail = int(os_sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None
    if page <= 0 or avail < 0:
        return None
    return page * avail


def os_sysconf(name: str) -> int:
    import os

    return int(os.sysconf(name))


def _darwin_available_bytes() -> int | None:
    try:
        completed = subprocess.run(
            ["vm_stat"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    page = 0
    counts: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if "page size of" in line:
            for token in line.replace(".", " ").split():
                if token.isdigit():
                    page = int(token)
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        digits = "".join(char for char in raw if char.isdigit())
        if digits:
            counts[name.strip().lower()] = int(digits)
    if page <= 0:
        return None
    pages = sum(counts.get(key, 0) for key in ("pages free", "pages inactive", "pages speculative"))
    if pages <= 0:
        return None
    return pages * page


def default_memory_guard() -> str | None:
    free = available_bytes()
    if free is not None and free < MIN_AVAILABLE_BYTES:
        return f"свободно {free} байт, для одной Qwen нужно не меньше {MIN_AVAILABLE_BYTES}"
    return None


def clamp_budget(num_ctx: int, num_predict: int) -> tuple[int, int]:
    ctx = min(max(int(num_ctx), 256), STAGE30_NUM_CTX)
    predict = min(max(int(num_predict), 64), STAGE30_NUM_PREDICT)
    return ctx, predict


def build_adapter_body(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, object] | None,
    *,
    schema_mode: str,
    temperature: float,
    seed: int,
    num_ctx: int,
    num_predict: int,
) -> dict[str, object]:
    """Тело чата без инструментов. Контекст и длина ответа ограничены."""
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("температура должна быть числом")
    if not 0.0 <= float(temperature) <= 1.0:
        raise ValueError("температура должна быть от 0 до 1")
    if type(seed) is not int:
        raise ValueError("seed должен быть целым")
    ctx, predict = clamp_budget(num_ctx, num_predict)
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
        "keep_alive": 0,
        "options": {
            "temperature": float(temperature),
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


class OllamaAdapter:
    """Один локальный вызов. Второй одновременный вызов получает статус busy."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        model: str = PREFERRED_LOCAL_MODEL,
        timeout: float = STAGE30_TIMEOUT_SECONDS,
        http_attempts: int = STAGE30_ATTEMPTS,
        num_ctx: int = STAGE30_NUM_CTX,
        num_predict: int = STAGE30_NUM_PREDICT,
        seed: int = STAGE30_SEED,
        transport: Callable[..., object] | None = None,
        memory_guard: MemoryGuard | None = None,
    ) -> None:
        if type(http_attempts) is not int or not 1 <= http_attempts <= STAGE30_ATTEMPTS:
            raise ValueError("число попыток связи должно быть от 1 до 3")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout должен быть числом")
        if not 0 < float(timeout) <= STAGE30_MAX_TIMEOUT_SECONDS:
            raise ValueError("timeout должен быть в пределах 180 секунд")
        if not model or not model.strip():
            raise ValueError("имя модели обязательно")
        self.base_url = assert_loopback(base_url)
        self.model = model.strip()
        self.timeout = float(timeout)
        self.http_attempts = http_attempts
        self.num_ctx, self.num_predict = clamp_budget(num_ctx, num_predict)
        self.seed = seed
        self._transport = transport or _open_without_redirect
        self._memory_guard = memory_guard or default_memory_guard
        self._call_lock = threading.Lock()
        self.logs: list[AdapterResult] = []
        self.model_name = ""
        self.model_digest = ""
        self.parameter_size = ""
        self.ollama_version = ""
        self.schema_mode = "json"

    def generate(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None = None,
        *,
        temperature: float = 0.0,
    ) -> AdapterResult:
        started = time.perf_counter()
        if not self._call_lock.acquire(blocking=False):
            return self._finish(
                "busy",
                "уже выполняется один запрос Qwen",
                started,
                attempts=0,
                temperature=0.0,
            )
        try:
            return self._generate(messages, schema, temperature=temperature, started=started)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, OllamaUnavailable) as error:
            return self._finish(
                "ollama_unavailable",
                f"Ollama недоступна: {error.__class__.__name__}",
                started,
                attempts=self.http_attempts,
                temperature=_temperature_or_zero(temperature),
            )
        finally:
            self._call_lock.release()

    def _generate(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None,
        *,
        temperature: float,
        started: float,
    ) -> AdapterResult:
        temperature_value = _checked_temperature(temperature)
        blocked = self._memory_guard()
        if blocked:
            return self._finish(
                "resource_limited",
                blocked,
                started,
                attempts=0,
                temperature=temperature_value,
            )
        if not messages or any(not isinstance(item.get("content"), str) or not item["content"].strip() for item in messages):
            return self._finish(
                "rejected",
                "пустой запрос к модели",
                started,
                attempts=0,
                temperature=temperature_value,
            )
        if self.model_digest:
            attempts = 0
            health = None
        else:
            health, attempts = self._health()
        if health is not None:
            timed = dataclasses.replace(
                health,
                elapsed_seconds=round(time.perf_counter() - started, 6),
                temperature=temperature_value,
            )
            self.logs.append(timed)
            return timed
        payload, problem, chat_attempts = self._chat(messages, schema, temperature_value)
        attempts += chat_attempts
        if problem == "schema_rejected" and self.schema_mode == "schema":
            self.schema_mode = "json"
            payload, problem, again = self._chat(messages, schema, temperature_value)
            attempts += again
        if problem == "timed_out":
            return self._finish(
                "timed_out",
                "Ollama не ответила в отведённое время",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        if problem is not None or payload is None:
            return self._finish(
                "ollama_unavailable",
                "Ollama не вернула ответ",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        message = payload.get("message")
        if not isinstance(message, dict):
            return self._finish(
                "rejected",
                "в ответе Ollama нет сообщения",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        if message.get("tool_calls"):
            return self._finish(
                "rejected",
                "вызов инструмента не выполняется",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return self._finish(
                "rejected",
                "в ответе нет текста гипотезы",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        if len(content) > 4000:
            return self._finish(
                "rejected",
                "ответ длиннее допустимого",
                started,
                attempts=attempts,
                temperature=temperature_value,
            )
        return self._finish(
            "ok",
            "ответ получен",
            started,
            attempts=attempts,
            temperature=temperature_value,
            content=content,
        )

    def _health(self) -> tuple[AdapterResult | None, int]:
        version_payload, problem, attempts = self._request("GET", "/api/version", None, timeout=min(self.timeout, 5))
        if problem is not None or version_payload is None:
            status = "timed_out" if problem == "timed_out" else "ollama_unavailable"
            return (
                AdapterResult(
                    status=status,
                    message="Ollama недоступна" if status == "ollama_unavailable" else "Ollama не ответила в отведённое время",
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    available_bytes=available_bytes(),
                    seed=self.seed,
                    num_ctx=self.num_ctx,
                    num_predict=self.num_predict,
                ),
                attempts,
            )
        version = version_payload.get("version")
        if not isinstance(version, str) or not version.strip():
            return (
                AdapterResult(
                    status="ollama_unavailable",
                    message="Ollama не сообщила версию",
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    seed=self.seed,
                ),
                attempts,
            )
        self.ollama_version = version.strip()
        self.schema_mode = schema_mode_for_version(self.ollama_version)
        tags, tag_problem, tag_attempts = self._request("GET", "/api/tags", None, timeout=min(self.timeout, 5))
        attempts += tag_attempts
        if tag_problem is not None or tags is None:
            status = "timed_out" if tag_problem == "timed_out" else "ollama_unavailable"
            return (
                AdapterResult(
                    status=status,
                    message="список локальных моделей недоступен",
                    ollama_version=self.ollama_version,
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    seed=self.seed,
                ),
                attempts,
            )
        models = tags.get("models")
        if not isinstance(models, list):
            return (
                AdapterResult(
                    status="ollama_unavailable",
                    message="Ollama вернула неожиданный список моделей",
                    ollama_version=self.ollama_version,
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    seed=self.seed,
                ),
                attempts,
            )
        installed: list[str] = []
        digests: dict[str, str] = {}
        sizes: dict[str, str] = {}
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            name = item["name"]
            installed.append(name)
            digest = item.get("digest")
            if isinstance(digest, str) and digest:
                digests[name] = digest
            details = item.get("details")
            if isinstance(details, dict) and isinstance(details.get("parameter_size"), str):
                sizes[name] = details["parameter_size"]
        try:
            selected = select_installed_model(installed, self.model)
        except OllamaUnavailable as error:
            return (
                AdapterResult(
                    status="model_missing",
                    message=str(error),
                    ollama_version=self.ollama_version,
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    available_bytes=available_bytes(),
                    seed=self.seed,
                    num_ctx=self.num_ctx,
                    num_predict=self.num_predict,
                ),
                attempts,
            )
        digest = digests.get(selected, "")
        if not digest:
            return (
                AdapterResult(
                    status="model_missing",
                    message="у локальной модели нет digest",
                    model_name=selected,
                    ollama_version=self.ollama_version,
                    attempts_used=attempts,
                    rss_bytes=process_rss_bytes(),
                    seed=self.seed,
                ),
                attempts,
            )
        self.model_name = selected
        self.model_digest = digest
        self.parameter_size = sizes.get(selected, "")
        return None, attempts

    def _chat(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, object] | None,
        temperature: float,
    ) -> tuple[dict[str, object] | None, str | None, int]:
        body = build_adapter_body(
            self.model_name,
            messages,
            schema,
            schema_mode=self.schema_mode,
            temperature=temperature,
            seed=self.seed,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        return self._request("POST", "/api/chat", body, timeout=self.timeout)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, object] | None,
        *,
        timeout: float,
    ) -> tuple[dict[str, object] | None, str | None, int]:
        if path not in _ALLOWED_PATHS:
            return None, "ollama_unavailable", 0
        data = None if body is None else json.dumps(body).encode("utf-8")
        last = "ollama_unavailable"
        attempts = 0
        for _ in range(self.http_attempts):
            attempts += 1
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
                        return None, "ollama_unavailable", attempts
                parsed = json.loads(raw.decode("utf-8"))
            except HTTPError as error:
                status = error.code
                error.close()
                if status == 400 and path == "/api/chat":
                    return None, "schema_rejected", attempts
                if status in _TRANSIENT_HTTP:
                    last = "timed_out" if status == 408 else "ollama_unavailable"
                    continue
                return None, "ollama_unavailable", attempts
            except OllamaUnavailable:
                raise
            except TimeoutError:
                last = "timed_out"
                continue
            except URLError as error:
                reason = getattr(error, "reason", None)
                last = "timed_out" if isinstance(reason, TimeoutError) else "ollama_unavailable"
                continue
            except (UnicodeDecodeError, json.JSONDecodeError, OSError):
                last = "ollama_unavailable"
                continue
            if not isinstance(parsed, dict):
                return None, "ollama_unavailable", attempts
            return parsed, None, attempts
        return None, last, attempts

    def _finish(
        self,
        status: str,
        message: str,
        started: float,
        *,
        attempts: int,
        temperature: float,
        content: str | None = None,
    ) -> AdapterResult:
        result = AdapterResult(
            status=status,
            message=message,
            content=content,
            model_name=self.model_name,
            model_digest=self.model_digest,
            parameter_size=self.parameter_size,
            ollama_version=self.ollama_version,
            elapsed_seconds=round(time.perf_counter() - started, 6),
            rss_bytes=process_rss_bytes(),
            available_bytes=available_bytes(),
            attempts_used=attempts,
            temperature=temperature,
            seed=self.seed,
            num_ctx=self.num_ctx,
            num_predict=self.num_predict,
        )
        self.logs.append(result)
        return result


def _checked_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("температура должна быть числом")
    if not 0.0 <= float(temperature) <= 1.0:
        raise ValueError("температура должна быть от 0 до 1")
    return float(temperature)


def _temperature_or_zero(temperature: float) -> float:
    try:
        return _checked_temperature(temperature)
    except ValueError:
        return 0.0
