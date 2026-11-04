"""Один тяжёлый поиск за раз. PySR и длинная генерация Qwen не совмещаются."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_LOCK = threading.Lock()


class SearchBusy(RuntimeError):
    """Второй тяжёлый поиск не запускается, пока первый не освободил память."""


@contextmanager
def exclusive_search(name: str) -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise SearchBusy(f"тяжёлый поиск уже выполняется, {name} отменён")
    try:
        yield
    finally:
        _LOCK.release()
