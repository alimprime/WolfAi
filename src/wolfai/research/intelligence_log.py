"""Журнал этапа 3.2. Неподтверждённая формула остаётся записью, а не законом."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class IntelligenceLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS intelligence_events (
                event_key TEXT PRIMARY KEY,
                step TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.commit()
        self._sequence = self.count()

    def close(self) -> None:
        self._connection.close()

    def append(self, payload: dict[str, object]) -> str:
        self._sequence += 1
        stored = dict(payload)
        stored["sequence"] = self._sequence
        text = json.dumps(stored, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(text.encode()).hexdigest()
        self._connection.execute(
            """
            INSERT INTO intelligence_events (event_key, step, status, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (key, str(stored.get("step", "")), str(stored.get("status", "")), text),
        )
        self._connection.commit()
        return key

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM intelligence_events").fetchone()
        return int(row[0])

    def records(self) -> list[dict[str, object]]:
        rows = self._connection.execute(
            "SELECT payload_json FROM intelligence_events ORDER BY rowid"
        ).fetchall()
        found: list[dict[str, object]] = []
        for row in rows:
            payload = json.loads(row[0])
            if isinstance(payload, dict):
                found.append(payload)
        return found

    def rollback_open(self) -> None:
        self._connection.rollback()
