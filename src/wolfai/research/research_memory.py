"""Реестр экспериментов. Это не память подтверждённых закономерностей.

Одинаковый ключ не создаёт вторую запись. Отрицательный результат хранится
так же, как и любой другой. Роль development_only не может получить статус supported.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

IDENTITY = (
    "model_name",
    "config_hash",
    "dataset_fingerprint",
    "features",
    "label_definition",
    "horizon_days",
    "train_start",
    "train_end",
    "dev_start",
    "dev_end",
    "seed",
)


class ResearchMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_key TEXT PRIMARY KEY,
                evidence_role TEXT NOT NULL,
                claim_status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def record(self, payload: dict[str, object]) -> str:
        """`stored` для новой строки, `duplicate` для того же результата."""
        _reject_false_support(payload)
        stored = dict(payload)
        stored.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        key = experiment_key(stored)
        existing = self.find(key)
        if existing is not None:
            if _stable(existing) != _stable(stored):
                raise ValueError("результат не воспроизвёлся")
            return "duplicate"
        self._connection.execute(
            """
            INSERT INTO experiments (experiment_key, evidence_role, claim_status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                key,
                str(stored["evidence_role"]),
                str(stored["claim_status"]),
                json.dumps(stored, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._connection.commit()
        return "stored"

    def find(self, key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM experiments WHERE experiment_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("реестр эксперимента повреждён")
        return payload

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
        return int(row[0])


def experiment_key(payload: dict[str, object]) -> str:
    identity = {name: payload[name] for name in IDENTITY}
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _reject_false_support(payload: dict[str, object]) -> None:
    if payload.get("evidence_role") == "development_only" and payload.get("claim_status") == "supported":
        raise ValueError("development_only не может получить статус supported")
    for name in IDENTITY:
        if name not in payload:
            raise ValueError(f"в записи эксперимента нет поля {name}")


STAGE24_LAYERS = ("research", "candidate")
STAGE24_KEY = IDENTITY + ("source_version", "feature_group")


class Stage24Memory:
    """Три слоя. Validated Laws на этапе 2.4 не пополняется."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS stage24_records (
                layer TEXT NOT NULL,
                experiment_key TEXT NOT NULL,
                claim_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (layer, experiment_key)
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def remember(self, layer: str, payload: dict[str, object]) -> str:
        if layer == "validated":
            raise ValueError("этап 2.4 не повышает гипотезу в Validated Laws")
        if layer not in STAGE24_LAYERS:
            raise ValueError("слой должен быть research или candidate")
        if payload.get("evidence_role") == "development_only" and payload.get("claim_status") == "supported":
            raise ValueError("development_only не может получить статус supported")
        for name in STAGE24_KEY:
            if name not in payload:
                raise ValueError(f"в записи этапа 2.4 нет поля {name}")
        key = stage24_key(payload)
        existing = self.find(layer, key)
        stored = dict(payload)
        stored.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        if existing is not None:
            if _stable(existing) != _stable(stored):
                raise ValueError("результат не воспроизвёлся")
            return "duplicate"
        self._connection.execute(
            """
            INSERT INTO stage24_records (layer, experiment_key, claim_status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                layer,
                key,
                str(stored["claim_status"]),
                json.dumps(stored, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._connection.commit()
        return "stored"

    def find(self, layer: str, key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM stage24_records WHERE layer = ? AND experiment_key = ?",
            (layer, key),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        if not isinstance(payload, dict):
            raise ValueError("реестр этапа 2.4 повреждён")
        return payload

    def count(self, layer: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM stage24_records WHERE layer = ?",
            (layer,),
        ).fetchone()
        return int(row[0])


def stage24_key(payload: dict[str, object]) -> str:
    identity = {name: payload[name] for name in STAGE24_KEY}
    text = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _stable(payload: dict[str, object]) -> dict[str, object]:
    copied = {key: value for key, value in payload.items() if key != "recorded_at"}
    return json.loads(json.dumps(copied, sort_keys=True))
