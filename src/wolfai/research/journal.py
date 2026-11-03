"""Журнал гипотез в SQLite. Строки не удаляются."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    id: str
    created_at: str
    data_version: str
    model_name: str | None
    model_digest: str | None
    generation: dict[str, object]
    baseline_metrics: dict[str, object] | None
    stop_reason: str | None
    max_hypotheses: int


@dataclass(frozen=True)
class HypothesisRecord:
    id: str
    run_id: str
    created_at: str
    expression: str
    features: list[str]
    effects: list[dict[str, str]]
    explanation: str
    version: int
    status: str
    parent_ids: list[str]
    data_version: str
    model_name: str | None
    model_digest: str | None
    generation: dict[str, object]
    rejection_reason: str | None
    error_type: str | None
    error_scale: float | None
    observations: int | None
    sufficient_observations: bool | None
    train_metrics: dict[str, object] | None
    validation_metrics: dict[str, object] | None
    raw_response: str | None
    kind: str = "fixed"
    coefficients: dict[str, object] | None = None
    inner_metrics: dict[str, object] | None = None
    dev_validation_metrics: dict[str, object] | None = None
    change_summary: str | None = None
    problem_addressed: str | None = None
    expected_improvement: str | None = None
    falsification: dict[str, object] | None = None
    independent_reason: str | None = None


class Journal:
    """Постоянная память экспериментов. Статус active через этот класс не ставится."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._create()

    def close(self) -> None:
        self._connection.close()

    def start_run(
        self,
        run_id: str,
        *,
        data_version: str,
        model_name: str | None,
        generation: dict[str, object],
        max_hypotheses: int,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO runs (
                id, created_at, data_version, model_name, model_digest,
                generation_json, baseline_metrics_json, stop_reason, max_hypotheses
            ) VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?)
            """,
            (run_id, utc_now(), data_version, model_name, _dump(generation), max_hypotheses),
        )
        self._connection.commit()

    def finish_run(
        self,
        run_id: str,
        *,
        model_digest: str | None,
        baseline_metrics: dict[str, object] | None,
        stop_reason: str,
    ) -> None:
        self._connection.execute(
            """
            UPDATE runs
            SET model_digest = ?, baseline_metrics_json = ?, stop_reason = ?
            WHERE id = ?
            """,
            (model_digest, _dump(baseline_metrics) if baseline_metrics is not None else None, stop_reason, run_id),
        )
        self._connection.commit()

    def add(self, record: HypothesisRecord) -> None:
        if record.status == "active":
            raise ValueError("статус active в Phase 2 не назначается")
        if record.status not in {"experimental", "archived"}:
            raise ValueError("неизвестный статус гипотезы")
        self._connection.execute(
            """
            INSERT INTO hypotheses (
                id, run_id, created_at, expression, features_json, effects_json,
                explanation, version, status, parent_ids_json, data_version,
                model_name, model_digest, generation_json, rejection_reason,
                error_type, error_scale, observations, sufficient_observations,
                train_metrics_json, validation_metrics_json, raw_response,
                hypothesis_kind, coefficients_json, inner_metrics_json, dev_validation_json,
                change_summary, problem_addressed, expected_improvement, falsification_json,
                independent_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.run_id,
                record.created_at,
                record.expression,
                _dump(record.features),
                _dump(record.effects),
                record.explanation,
                record.version,
                record.status,
                _dump(record.parent_ids),
                record.data_version,
                record.model_name,
                record.model_digest,
                _dump(record.generation),
                record.rejection_reason,
                record.error_type,
                record.error_scale,
                record.observations,
                None if record.sufficient_observations is None else int(record.sufficient_observations),
                _dump(record.train_metrics) if record.train_metrics is not None else None,
                _dump(record.validation_metrics) if record.validation_metrics is not None else None,
                record.raw_response,
                record.kind,
                _dump(record.coefficients) if record.coefficients is not None else None,
                _dump(record.inner_metrics) if record.inner_metrics is not None else None,
                _dump(record.dev_validation_metrics) if record.dev_validation_metrics is not None else None,
                record.change_summary,
                record.problem_addressed,
                record.expected_improvement,
                _dump(record.falsification) if record.falsification is not None else None,
                record.independent_reason,
            ),
        )
        self._connection.commit()

    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        row = self._connection.execute(
            "SELECT * FROM hypotheses WHERE id = ?",
            (hypothesis_id,),
        ).fetchone()
        return None if row is None else _hypothesis(row)

    def list_hypotheses(self) -> list[HypothesisRecord]:
        rows = self._connection.execute(
            "SELECT * FROM hypotheses ORDER BY created_at, id"
        ).fetchall()
        return [_hypothesis(row) for row in rows]

    def list_runs(self) -> list[RunRecord]:
        rows = self._connection.execute("SELECT * FROM runs ORDER BY created_at, id").fetchall()
        return [_run(row) for row in rows]

    def record_development_validation(self, hypothesis_id: str, metrics: dict[str, object]) -> None:
        """Пишет итоговую проверку 2023–2024, не меняя статус."""
        self._connection.execute(
            """
            UPDATE hypotheses
            SET dev_validation_json = ?, validation_metrics_json = ?
            WHERE id = ?
            """,
            (_dump(metrics), _dump(metrics), hypothesis_id),
        )
        self._connection.commit()

    def known_ids(self) -> set[str]:
        rows = self._connection.execute("SELECT id FROM hypotheses").fetchall()
        return {str(row["id"]) for row in rows}

    def _create(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                data_version TEXT NOT NULL,
                model_name TEXT,
                model_digest TEXT,
                generation_json TEXT NOT NULL,
                baseline_metrics_json TEXT,
                stop_reason TEXT,
                max_hypotheses INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expression TEXT NOT NULL,
                features_json TEXT NOT NULL,
                effects_json TEXT NOT NULL,
                explanation TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('experimental', 'active', 'archived')),
                parent_ids_json TEXT NOT NULL,
                data_version TEXT NOT NULL,
                model_name TEXT,
                model_digest TEXT,
                generation_json TEXT NOT NULL,
                rejection_reason TEXT,
                error_type TEXT,
                error_scale REAL,
                observations INTEGER,
                sufficient_observations INTEGER,
                train_metrics_json TEXT,
                validation_metrics_json TEXT,
                raw_response TEXT,
                hypothesis_kind TEXT DEFAULT 'fixed',
                coefficients_json TEXT,
                inner_metrics_json TEXT,
                dev_validation_json TEXT,
                change_summary TEXT,
                problem_addressed TEXT,
                expected_improvement TEXT,
                falsification_json TEXT,
                independent_reason TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            """
        )
        self._migrate()
        self._connection.commit()

    def _migrate(self) -> None:
        columns = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(hypotheses)").fetchall()
        }
        if not columns:
            return
        additions = {
            "hypothesis_kind": "TEXT DEFAULT 'fixed'",
            "coefficients_json": "TEXT",
            "inner_metrics_json": "TEXT",
            "dev_validation_json": "TEXT",
            "change_summary": "TEXT",
            "problem_addressed": "TEXT",
            "expected_improvement": "TEXT",
            "falsification_json": "TEXT",
            "independent_reason": "TEXT",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._connection.execute(f"ALTER TABLE hypotheses ADD COLUMN {name} {declaration}")
        self._connection.commit()


def _text(row: sqlite3.Row, name: str) -> str | None:
    if name not in row.keys():
        return None
    value = row[name]
    return None if value is None else str(value)


def _object_dict(row: sqlite3.Row, name: str) -> dict[str, object] | None:
    if name not in row.keys():
        return None
    loaded = _load(row[name])
    return loaded if isinstance(loaded, dict) else None


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load(value: str | None) -> object:
    if value is None:
        return None
    return json.loads(value)


def _run(row: sqlite3.Row) -> RunRecord:
    baseline = _load(row["baseline_metrics_json"])
    return RunRecord(
        id=row["id"],
        created_at=row["created_at"],
        data_version=row["data_version"],
        model_name=row["model_name"],
        model_digest=row["model_digest"],
        generation=_load(row["generation_json"]) or {},
        baseline_metrics=baseline if isinstance(baseline, dict) else None,
        stop_reason=row["stop_reason"],
        max_hypotheses=int(row["max_hypotheses"]),
    )


def _hypothesis(row: sqlite3.Row) -> HypothesisRecord:
    sufficient = row["sufficient_observations"]
    train = _load(row["train_metrics_json"])
    validation = _load(row["validation_metrics_json"])
    generation = _load(row["generation_json"])
    return HypothesisRecord(
        id=row["id"],
        run_id=row["run_id"],
        created_at=row["created_at"],
        expression=row["expression"],
        features=list(_load(row["features_json"]) or []),
        effects=list(_load(row["effects_json"]) or []),
        explanation=row["explanation"],
        version=int(row["version"]),
        status=row["status"],
        parent_ids=list(_load(row["parent_ids_json"]) or []),
        data_version=row["data_version"],
        model_name=row["model_name"],
        model_digest=row["model_digest"],
        generation=generation if isinstance(generation, dict) else {},
        rejection_reason=row["rejection_reason"],
        error_type=row["error_type"],
        error_scale=row["error_scale"],
        observations=row["observations"],
        sufficient_observations=None if sufficient is None else bool(sufficient),
        train_metrics=train if isinstance(train, dict) else None,
        validation_metrics=validation if isinstance(validation, dict) else None,
        raw_response=row["raw_response"],
        kind=_text(row, "hypothesis_kind") or "fixed",
        coefficients=_object_dict(row, "coefficients_json"),
        inner_metrics=_object_dict(row, "inner_metrics_json"),
        dev_validation_metrics=_object_dict(row, "dev_validation_json"),
        change_summary=_text(row, "change_summary"),
        problem_addressed=_text(row, "problem_addressed"),
        expected_improvement=_text(row, "expected_improvement"),
        falsification=_object_dict(row, "falsification_json"),
        independent_reason=_text(row, "independent_reason"),
    )
