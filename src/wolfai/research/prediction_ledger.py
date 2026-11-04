"""Неизменяемый журнал прогнозов.

Вероятность после вставки не переписывается. Исправление — новая строка
со ссылкой на прежнюю. Исход хранится отдельно и тоже не заменяется другим числом.

Каналы `synthetic` и `reconstruction` не являются прогнозом, сделанным в реальном времени.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

CHANNELS = frozenset({"forward", "reconstruction", "synthetic"})
HORIZONS = frozenset({1, 3, 7})
CLOSED_RECONSTRUCTION = datetime.fromisoformat("2022-01-01T00:00:00+00:00")


class PredictionImmutable(ValueError):
    """Попытка заменить уже записанную вероятность или исход."""


@dataclass(frozen=True)
class StoredPrediction:
    prediction_id: str
    model_id: str
    model_version: str
    created_at: datetime
    decision_time: datetime
    target_time: datetime
    horizon_days: int
    probability_up: float
    feature_snapshot_hash: str
    dataset_version: str
    code_version: str
    channel: str
    replaces_prediction_id: str | None
    baseline_probability: float


@dataclass(frozen=True)
class StoredEvidence:
    prediction_id: str
    label: int
    observed_at: datetime


class PredictionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decision_time TEXT NOT NULL,
                target_time TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                probability_up REAL NOT NULL,
                feature_snapshot_hash TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                code_version TEXT NOT NULL,
                channel TEXT NOT NULL,
                replaces_prediction_id TEXT,
                baseline_probability REAL NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_evidence (
                prediction_id TEXT PRIMARY KEY,
                label INTEGER NOT NULL,
                observed_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def append(
        self,
        *,
        model_id: str,
        model_version: str,
        created_at: datetime,
        decision_time: datetime,
        horizon_days: int,
        probability_up: float,
        features: dict[str, float],
        feature_available_at: dict[str, datetime],
        dataset_version: str,
        code_version: str,
        channel: str,
        baseline_probability: float = 0.5,
        replaces_prediction_id: str | None = None,
    ) -> tuple[str, str]:
        """Возвращает `(prediction_id, stored|duplicate)`."""
        target_time = _validate_new(
            model_id=model_id,
            model_version=model_version,
            created_at=created_at,
            decision_time=decision_time,
            horizon_days=horizon_days,
            probability_up=probability_up,
            features=features,
            feature_available_at=feature_available_at,
            dataset_version=dataset_version,
            code_version=code_version,
            channel=channel,
            baseline_probability=baseline_probability,
        )
        if replaces_prediction_id is not None:
            previous = self.get(replaces_prediction_id)
            if previous.model_id != model_id:
                raise PredictionImmutable("другая стратегия не может заменить чужой прогноз")
            if previous.channel != channel:
                raise PredictionImmutable("канал исправления должен совпадать с исходным")
        digest = snapshot_hash(features)
        identity = _identity(
            model_id=model_id,
            model_version=model_version,
            decision_time=decision_time,
            horizon_days=horizon_days,
            digest=digest,
            channel=channel,
            replaces=replaces_prediction_id,
        )
        existing = self._find_identity(identity)
        if existing is not None:
            if existing.probability_up != probability_up:
                raise PredictionImmutable("вероятность уже записана и не меняется")
            return existing.prediction_id, "duplicate"
        prediction_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._connection.execute(
            """
            INSERT INTO predictions (
                prediction_id, model_id, model_version, created_at, decision_time, target_time,
                horizon_days, probability_up, feature_snapshot_hash, dataset_version, code_version,
                channel, replaces_prediction_id, baseline_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                model_id,
                model_version,
                created_at.isoformat(),
                decision_time.isoformat(),
                target_time.isoformat(),
                horizon_days,
                probability_up,
                digest,
                dataset_version,
                code_version,
                channel,
                replaces_prediction_id,
                baseline_probability,
            ),
        )
        self._connection.commit()
        return prediction_id, "stored"

    def record_outcome(
        self,
        prediction_id: str,
        label: int,
        observed_at: datetime,
    ) -> str:
        prediction = self.get(prediction_id)
        _aware(observed_at, "observed_at")
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("исход должен быть 0 или 1")
        if observed_at < prediction.target_time:
            raise ValueError("горизонт ещё не созрел")
        current = self.evidence(prediction_id)
        if current is not None:
            if current.label != label or current.observed_at != observed_at:
                raise PredictionImmutable("фактический результат уже записан")
            return "duplicate"
        self._connection.execute(
            "INSERT INTO prediction_evidence (prediction_id, label, observed_at) VALUES (?, ?, ?)",
            (prediction_id, label, observed_at.isoformat()),
        )
        self._connection.commit()
        return "stored"

    def get(self, prediction_id: str) -> StoredPrediction:
        row = self._connection.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            raise KeyError(prediction_id)
        return _prediction(row)

    def evidence(self, prediction_id: str) -> StoredEvidence | None:
        row = self._connection.execute(
            "SELECT prediction_id, label, observed_at FROM prediction_evidence WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            return None
        return StoredEvidence(row[0], int(row[1]), _parse(row[2]))

    def settled(self, channel: str) -> list[tuple[StoredPrediction, StoredEvidence]]:
        rows = self._connection.execute(
            """
            SELECT p.*, e.label, e.observed_at
            FROM predictions p
            JOIN prediction_evidence e ON e.prediction_id = p.prediction_id
            WHERE p.channel = ?
            ORDER BY p.decision_time, p.prediction_id
            """,
            (channel,),
        ).fetchall()
        paired: list[tuple[StoredPrediction, StoredEvidence]] = []
        for row in rows:
            prediction = _prediction(row[:14])
            paired.append((
                prediction,
                StoredEvidence(prediction.prediction_id, int(row[14]), _parse(row[15])),
            ))
        return paired

    def count(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return int(row[0])

    def _find_identity(self, identity: dict[str, object]) -> StoredPrediction | None:
        row = self._connection.execute(
            """
            SELECT * FROM predictions
            WHERE model_id = ? AND model_version = ? AND decision_time = ?
              AND horizon_days = ? AND feature_snapshot_hash = ? AND channel = ?
              AND IFNULL(replaces_prediction_id, '') = ?
            """,
            (
                identity["model_id"],
                identity["model_version"],
                identity["decision_time"],
                identity["horizon_days"],
                identity["feature_snapshot_hash"],
                identity["channel"],
                identity["replaces"],
            ),
        ).fetchone()
        if row is None:
            return None
        return _prediction(row)


def snapshot_hash(features: dict[str, float]) -> str:
    if not features:
        raise ValueError("снимок признаков пуст")
    payload = {name: features[name] for name in sorted(features)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _validate_new(
    *,
    model_id: str,
    model_version: str,
    created_at: datetime,
    decision_time: datetime,
    horizon_days: int,
    probability_up: float,
    features: dict[str, float],
    feature_available_at: dict[str, datetime],
    dataset_version: str,
    code_version: str,
    channel: str,
    baseline_probability: float,
) -> datetime:
    if channel not in CHANNELS:
        raise ValueError("канал должен быть forward, reconstruction или synthetic")
    if horizon_days not in HORIZONS:
        raise ValueError("горизонт должен быть 1, 3 или 7 дней")
    _aware(created_at, "created_at")
    _aware(decision_time, "decision_time")
    if not model_id.strip() or not model_version.strip():
        raise ValueError("модель и версия обязательны")
    if not dataset_version.strip() or not code_version.strip():
        raise ValueError("версия данных и кода обязательны")
    if isinstance(probability_up, bool) or not isinstance(probability_up, (int, float)):
        raise ValueError("вероятность должна быть числом от 0 до 1")
    if not 0.0 <= float(probability_up) <= 1.0:
        raise ValueError("вероятность должна быть числом от 0 до 1")
    if not 0.0 <= float(baseline_probability) <= 1.0:
        raise ValueError("baseline должен быть числом от 0 до 1")
    target_time = decision_time + timedelta(days=horizon_days)
    if set(feature_available_at) != set(features):
        raise ValueError("у каждого признака должно быть своё время доступности")
    for name, moment in feature_available_at.items():
        _aware(moment, name)
        if moment > decision_time:
            raise ValueError(f"признак {name} доступен позже decision_time")
    if channel == "forward" and not decision_time <= created_at < target_time:
        raise ValueError("forward-прогноз записывается в окне от decision_time до target_time")
    if channel == "reconstruction" and (
        decision_time >= CLOSED_RECONSTRUCTION or target_time >= CLOSED_RECONSTRUCTION
    ):
        raise ValueError("реконструкция не открывает 2022 год и позже")
    return target_time


def _identity(
    *,
    model_id: str,
    model_version: str,
    decision_time: datetime,
    horizon_days: int,
    digest: str,
    channel: str,
    replaces: str | None,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_version": model_version,
        "decision_time": decision_time.isoformat(),
        "horizon_days": horizon_days,
        "feature_snapshot_hash": digest,
        "channel": channel,
        "replaces": replaces or "",
    }


def _prediction(row: tuple[object, ...]) -> StoredPrediction:
    return StoredPrediction(
        prediction_id=str(row[0]),
        model_id=str(row[1]),
        model_version=str(row[2]),
        created_at=_parse(str(row[3])),
        decision_time=_parse(str(row[4])),
        target_time=_parse(str(row[5])),
        horizon_days=int(row[6]),
        probability_up=float(row[7]),
        feature_snapshot_hash=str(row[8]),
        dataset_version=str(row[9]),
        code_version=str(row[10]),
        channel=str(row[11]),
        replaces_prediction_id=None if row[12] is None else str(row[12]),
        baseline_probability=float(row[13]),
    )


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} должен быть в UTC")
