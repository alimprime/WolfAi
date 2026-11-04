"""Слои доказательств рядом с Research Memory. Старые таблицы не меняются.

Experiment, Hypothesis и Evidence можно записать. Prediction живёт в журнале.
Validated Law на этом этапе не создаётся. Отрицательный результат сохраняется.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

KINDS = frozenset({"experiment", "hypothesis", "evidence"})
KIND_KEY = (
    "name",
    "model_id",
    "model_version",
    "dataset_version",
    "code_version",
    "horizon_days",
    "decision_time",
)


class ForwardRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS forward_records (
                kind TEXT NOT NULL,
                record_key TEXT NOT NULL,
                claim_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (kind, record_key)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def remember(self, kind: str, payload: dict[str, object]) -> str:
        if kind == "validated_law":
            raise ValueError("этап 2.5 не создаёт Validated Law")
        if kind not in KINDS:
            raise ValueError("запись должна быть experiment, hypothesis или evidence")
        if payload.get("claim_status") == "supported":
            raise ValueError("supported из forward-контура не назначается")
        for name in KIND_KEY:
            if name not in payload:
                raise ValueError(f"в записи нет поля {name}")
        key = _key(payload)
        existing = self.find(kind, key)
        stored = dict(payload)
        stored.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        if existing is not None:
            if _stable(existing) != _stable(stored):
                raise ValueError("результат не воспроизвёлся")
            return "duplicate"
        self._connection.execute(
            """
            INSERT INTO forward_records (kind, record_key, claim_status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (kind, key, str(stored.get("claim_status") or "insufficient_evidence"), json.dumps(stored, ensure_ascii=False, sort_keys=True)),
        )
        self._connection.commit()
        return "stored"

    def find(self, kind: str, key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM forward_records WHERE kind = ? AND record_key = ?",
            (kind, key),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("реестр forward повреждён")
        return payload

    def count(self, kind: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM forward_records WHERE kind = ?",
            (kind,),
        ).fetchone()
        return int(row[0])


def _key(payload: dict[str, object]) -> str:
    identity = {name: payload[name] for name in KIND_KEY}
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _stable(payload: dict[str, object]) -> dict[str, object]:
    copied = {key: value for key, value in payload.items() if key != "recorded_at"}
    return json.loads(json.dumps(copied, sort_keys=True))
