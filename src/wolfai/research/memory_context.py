"""Контекст Research Memory для одного запроса. Закрытая выборка не передаётся."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from wolfai.research.hypothesis_protocol import KnownFormula

MAX_ITEMS = 6
MAX_EXCERPT_CHARS = 1600
_FUTURE_MARKERS = ("holdout", "2022", "2023", "2024", "2025", "2026", "2027", "y_up", "next_close")
_BANNED_KEY_PARTS = ("holdout", "future", "2023", "2024", "2025", "2026", "2027")


@dataclass(frozen=True)
class MemoryExcerpt:
    hypotheses: tuple[dict[str, object], ...]
    experiments: tuple[dict[str, object], ...]
    errors: tuple[str, ...]
    rejections: tuple[str, ...]
    limitations: tuple[str, ...]
    leak_suspicions: tuple[str, ...]
    known: tuple[KnownFormula, ...]
    truncated: bool
    chars: int

    def as_dict(self) -> dict[str, object]:
        return {
            "hypotheses": [dict(item) for item in self.hypotheses],
            "experiments": [dict(item) for item in self.experiments],
            "errors": list(self.errors),
            "rejections": list(self.rejections),
            "limitations": list(self.limitations),
            "leak_suspicions": list(self.leak_suspicions),
            "truncated": self.truncated,
        }


def load_research_excerpt(
    memory: object | None,
    store: object | None,
    *,
    author_id: str,
    limit: int = MAX_ITEMS,
) -> tuple[MemoryExcerpt | None, str | None]:
    """Читает разрешённую часть памяти. Ошибка чтения возвращается текстом."""
    if type(limit) is not int or not 1 <= limit <= 32:
        return None, "лимит контекста должен быть от 1 до 32"
    try:
        raw_experiments = _read_memory(memory, limit)
        views = _read_store(store)
    except (sqlite3.Error, OSError) as error:
        return None, f"research memory недоступна: {error.__class__.__name__}"
    hypotheses: list[dict[str, object]] = []
    known: list[KnownFormula] = []
    for view in views:
        known.append(
            KnownFormula(
                expression=view.expression,
                features=tuple(view.features),
                horizon_days=view.horizon_days,
                status=view.status,
                hypothesis_id=view.hypothesis_id,
            )
        )
        item: dict[str, object] = {
            "hypothesis_id": view.hypothesis_id,
            "expression": view.expression,
            "features": list(view.features),
            "horizon_days": view.horizon_days,
            "status": view.status,
        }
        if view.author_id == author_id:
            item["description"] = _safe_text(view.description, 180) or ""
        else:
            item["author"] = "other"
        hypotheses.append(item)
    experiments: list[dict[str, object]] = []
    errors: list[str] = []
    rejections: list[str] = []
    limitations: list[str] = []
    leaks: list[str] = []
    for payload in raw_experiments:
        public = _public_experiment(payload)
        if public is None:
            continue
        experiments.append(public)
        _append_unique(errors, _safe_text(payload.get("error") or payload.get("operator_error")))
        _append_unique(rejections, _safe_text(payload.get("rejection_reason")))
        _append_unique(limitations, _safe_text(payload.get("limitations")))
        _append_unique(leaks, _safe_text(payload.get("leak_suspicion")))
    excerpt = _fit(
        hypotheses,
        experiments,
        errors,
        rejections,
        limitations,
        leaks,
        tuple(known),
    )
    return excerpt, None


def _read_memory(memory: object | None, limit: int) -> list[dict[str, object]]:
    if memory is None:
        return []
    reader = getattr(memory, "public_records", None)
    if not callable(reader):
        raise sqlite3.DatabaseError("у research memory нет чтения")
    rows = reader(limit=limit)
    if not isinstance(rows, list):
        raise sqlite3.DatabaseError("research memory вернула не список")
    found: list[dict[str, object]] = []
    for item in rows:
        if isinstance(item, dict):
            found.append(item)
    return found


def _read_store(store: object | None) -> Sequence[object]:
    if store is None:
        return []
    reader = getattr(store, "list_all", None)
    if not callable(reader):
        raise sqlite3.DatabaseError("журнал гипотез недоступен")
    rows = reader()
    if not isinstance(rows, list):
        raise sqlite3.DatabaseError("журнал гипотез вернул не список")
    return rows


def _public_experiment(payload: dict[str, object]) -> dict[str, object] | None:
    public: dict[str, object] = {}
    for key in ("expression", "features", "horizon_days", "claim_status", "evidence_role"):
        if any(part in key.lower() for part in _BANNED_KEY_PARTS):
            continue
        value = payload.get(key)
        compact = _compact(value)
        if compact is not None:
            public[key] = compact
    if not public:
        return None
    return public


def _compact(value: object) -> object | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and _has_closed_marker(value):
            return None
        return value
    if isinstance(value, list) and len(value) <= 8 and all(isinstance(item, (str, int, float)) for item in value):
        return value
    return None


def _safe_text(value: object, limit: int = 180) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if _has_closed_marker(value):
        return None
    return value.strip()[:limit]


def _has_closed_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _FUTURE_MARKERS)


def _append_unique(bucket: list[str], value: str | None) -> None:
    if value and value not in bucket and len(bucket) < MAX_ITEMS:
        bucket.append(value)


def _fit(
    hypotheses: list[dict[str, object]],
    experiments: list[dict[str, object]],
    errors: list[str],
    rejections: list[str],
    limitations: list[str],
    leaks: list[str],
    known: tuple[KnownFormula, ...],
) -> MemoryExcerpt:
    truncated = False
    while True:
        payload = {
            "hypotheses": hypotheses,
            "experiments": experiments,
            "errors": errors,
            "rejections": rejections,
            "limitations": limitations,
            "leak_suspicions": leaks,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(text) <= MAX_EXCERPT_CHARS and len(hypotheses) <= MAX_ITEMS and len(experiments) <= MAX_ITEMS:
            return MemoryExcerpt(
                hypotheses=tuple(hypotheses),
                experiments=tuple(experiments),
                errors=tuple(errors),
                rejections=tuple(rejections),
                limitations=tuple(limitations),
                leak_suspicions=tuple(leaks),
                known=known,
                truncated=truncated,
                chars=len(text),
            )
        truncated = True
        if len(hypotheses) > 1:
            hypotheses = hypotheses[1:]
            continue
        if hypotheses:
            hypotheses = []
            continue
        if experiments:
            experiments = experiments[1:]
            continue
        if limitations:
            limitations = limitations[1:]
            continue
        if rejections:
            rejections = rejections[1:]
            continue
        if errors:
            errors = errors[1:]
            continue
        if leaks:
            leaks = leaks[1:]
            continue
        return MemoryExcerpt(
            hypotheses=(),
            experiments=(),
            errors=(),
            rejections=(),
            limitations=(),
            leak_suspicions=(),
            known=known,
            truncated=True,
            chars=2,
        )
